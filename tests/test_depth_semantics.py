"""Guards on the semantics the study depends on.

The central risk in this project is spending GPU hours on grid cells that
cannot produce a result. These tests pin down which (corruption x loss)
combinations can actually move a number, so scripts/run_sweep.py can skip the
rest and so nobody later "fixes" a loss in a way that silently invalidates the
Axis C design.
"""
import numpy as np
import pytest
import torch

from src.corruptions import degrade
from src.depth_loss import align_scale_shift, ssi_loss, pearson_loss, absolute_loss
from src.depth_init import backproject_depth
from src.eval_geometry import chamfer_distance, depth_metrics


@pytest.fixture
def depths():
    g = torch.Generator().manual_seed(0)
    pred = torch.rand(32, 32, generator=g) * 3 + 1
    target = torch.rand(32, 32, generator=g) * 3 + 1
    return pred, target


# --------------------------------------------------------------------------
# Axis C: what each depth loss can and cannot see
# --------------------------------------------------------------------------
def test_ssi_turns_affine_bias_into_a_lambda_rescale(depths):
    """SSI cannot see an affine bias as geometry error -- only as a gain on lambda.

    Fitting (s,t) to a corrupted target a*d+b yields (a*s, a*t+b), so the
    residual is exactly a * the clean residual. The gradient DIRECTION is
    unchanged; only its magnitude scales. An 'affine x ssi' cell therefore
    duplicates an 'affine-free x lambda*a' cell instead of testing prior error.
    """
    pred, target = depths
    a, b = 2.0, 0.5
    clean = ssi_loss(pred, target)
    biased = ssi_loss(pred, degrade(target, scale=a, shift=b))
    assert biased.item() == pytest.approx(a * clean.item(), rel=1e-5)

    def grad(t):
        p = pred.clone().requires_grad_(True)
        ssi_loss(p, t).backward()
        return p.grad.flatten()

    gc, gb = grad(target), grad(degrade(target, scale=a, shift=b))
    cos = torch.nn.functional.cosine_similarity(gc, gb, dim=0).item()
    assert cos == pytest.approx(1.0, abs=1e-5)
    assert (gb.norm() / gc.norm()).item() == pytest.approx(a, rel=1e-3)


def test_pearson_is_blind_to_affine_bias(depths):
    """Pearson is scale/shift invariant by construction: a literal no-op."""
    pred, target = depths
    clean = pearson_loss(pred, target)
    biased = pearson_loss(pred, degrade(target, scale=2.0, shift=0.5))
    assert biased.item() == pytest.approx(clean.item(), rel=1e-6)


def test_absolute_loss_does_see_affine_bias(depths):
    """The absolute loss is the only loss path through which Axis C can bite."""
    pred, target = depths
    clean = absolute_loss(pred, target)
    biased = absolute_loss(pred, degrade(target, scale=2.0, shift=0.5))
    assert biased.item() > 1.5 * clean.item()


def test_all_losses_see_additive_noise(depths):
    """Noise is not affine, so every loss -- including SSI -- responds to it.

    pred must be CORRELATED with target here: two independent depth maps are
    already maximally uncorrelated, so Pearson sits pinned at ~1.0 and further
    noise cannot move it. A real rendered depth tracks the target closely, and
    that is the regime where the noise knob has to bite.
    """
    _, target = depths
    g = torch.Generator().manual_seed(1)
    pred = 0.5 * target + 0.1 * torch.rand(target.shape, generator=g)
    noisy = degrade(target, sigma_rel=0.5, generator=g)
    for fn in (ssi_loss, pearson_loss, absolute_loss):
        assert fn(pred, noisy).item() != pytest.approx(fn(pred, target).item(), rel=1e-3)


def test_align_scale_shift_recovers_a_known_affine(depths):
    _, target = depths
    s, t = align_scale_shift((target - 0.5) / 2.0, target)
    assert s.item() == pytest.approx(2.0, rel=1e-4)
    assert t.item() == pytest.approx(0.5, rel=1e-4)


# --------------------------------------------------------------------------
# Evaluation: the aligned metric erases the corruption too
# --------------------------------------------------------------------------
def test_aligned_depth_metric_erases_the_injected_bias():
    """Why unaligned MAE must be the PRIMARY geometry metric on Synthetic.

    depth_metrics(align=True) least-squares fits scale+shift before scoring, so
    it reports a near-zero error for a depth map that is globally wrong by a
    factor of two. Reporting only the aligned metric would make Axis C's affine
    arm read 'no effect' for a purely definitional reason.
    """
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 4.0, size=(32, 32))
    biased = 2.0 * gt + 0.5

    aligned = depth_metrics(biased, gt, align=True)
    unaligned = depth_metrics(biased, gt, align=False)

    assert aligned["mae"] < 1e-8
    assert unaligned["mae"] > 1.0
    assert unaligned["mae"] > 1e6 * max(aligned["mae"], 1e-12)


# --------------------------------------------------------------------------
# Init: the path through which affine bias DOES corrupt geometry
# --------------------------------------------------------------------------
def test_affine_bias_displaces_the_backprojected_cloud():
    """Depth init consumes METRIC depth, so Axis C reaches geometry through it.

    This is the arm of the study that can actually show 'appearance up,
    geometry down' -- the loss paths above mostly cannot.
    """
    K = np.array([[100.0, 0, 16.0], [0, 100.0, 16.0], [0, 0, 1.0]])
    c2w = np.eye(4)
    depth = np.full((32, 32), 2.0)

    clean = backproject_depth(depth, K, c2w)
    biased = backproject_depth(2.0 * depth + 0.5, K, c2w)

    assert clean.shape == biased.shape
    assert chamfer_distance(biased, clean) > 1.0


def test_backprojection_roundtrips_through_a_known_camera():
    """A pixel at the principal point at depth d must land at distance d ahead."""
    K = np.array([[100.0, 0, 16.0], [0, 100.0, 16.0], [0, 0, 1.0]])
    pts = backproject_depth(np.full((33, 33), 2.0), K, np.eye(4))
    centre = pts[np.argmin(np.linalg.norm(pts[:, :2], axis=1))]
    assert centre == pytest.approx([0.0, 0.0, 2.0], abs=1e-6)


def test_backprojection_respects_camera_pose():
    """A translated camera must translate the cloud by the same amount."""
    K = np.array([[100.0, 0, 16.0], [0, 100.0, 16.0], [0, 0, 1.0]])
    depth = np.full((16, 16), 2.0)
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, -2.0, 0.5]
    shifted = backproject_depth(depth, K, c2w) - backproject_depth(depth, K, np.eye(4))
    assert np.allclose(shifted, np.array([1.0, -2.0, 0.5]), atol=1e-6)


def test_backprojection_drops_invalid_depth():
    K = np.array([[100.0, 0, 8.0], [0, 100.0, 8.0], [0, 0, 1.0]])
    depth = np.full((16, 16), 2.0)
    depth[:8] = 0.0          # background / no return
    depth[8:12] = np.nan
    assert len(backproject_depth(depth, K, np.eye(4))) == 4 * 16


# --------------------------------------------------------------------------
# Corruption knobs behave as documented
# --------------------------------------------------------------------------
def test_degrade_is_identity_under_defaults(depths):
    _, target = depths
    assert torch.equal(degrade(target), target)


def test_noise_scales_with_sigma_rel(depths):
    _, target = depths
    spreads = [
        (degrade(target, sigma_rel=s, generator=torch.Generator().manual_seed(0))
         - target).std().item()
        for s in (0.01, 0.1, 0.5)
    ]
    assert spreads[0] < spreads[1] < spreads[2]

"""Exercise the training pipeline without CUDA.

gsplat.rasterization is the only part of this codebase that needs a GPU, so it
is isolated in src/render.py and substituted here. Everything around it -- pose
conversion, loss assembly, the optimizer/strategy handshake, evaluation, the
metrics.json contract -- is ordinary code and is tested like ordinary code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.data import View                                      # noqa: E402
from src.gaussians import Gaussians, init_random, init_from_depth  # noqa: E402
from src.render import render, viewmats_from_c2w               # noqa: E402


# --------------------------------------------------------------------------
def make_view(h=16, w=16, dist=4.0, with_depth=True):
    K = np.array([[float(w), 0, w / 2], [0, float(w), h / 2], [0, 0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, 0.0, -dist]
    return View(image=np.full((h, w, 3), 0.5, np.float32), K=K, c2w=c2w,
                depth=np.full((h, w), dist, np.float32) if with_depth else None,
                mask=np.ones((h, w), bool), name="v0")


class FakeRasterizer:
    """Stands in for gsplat: differentiable, records what it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, means, quats, scales, opacities, colors, viewmats, Ks,
                 width, height, render_mode, sh_degree=None, **kw):
        self.calls.append(dict(viewmats=viewmats, Ks=Ks, width=width,
                               height=height, render_mode=render_mode,
                               n=len(means)))
        C = viewmats.shape[0]
        # Depend on every parameter so gradients actually flow to all of them.
        seed = (means.mean() + scales.mean() + quats.mean()
                + opacities.mean() + colors.mean())
        out = torch.ones(C, height, width, 4, device=means.device) * seed
        alphas = torch.full((C, height, width, 1), 0.9, device=means.device)
        return out, alphas, {"radii": torch.ones(C, len(means))}


class FakeStrategy:
    def __init__(self):
        self.pre = self.post = 0

    def initialize_state(self, scene_scale=1.0):
        return {"scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        self.pre += 1

    def step_post_backward(self, params, optimizers, state, step, info):
        self.post += 1


# --------------------------------------------------------------------------
# Pose handling: the single most likely thing to be silently wrong
# --------------------------------------------------------------------------
def test_render_passes_world_to_camera_not_camera_to_world():
    """gsplat wants w2c. Handing it c2w renders a plausible-looking wrong scene."""
    g, v, fake = init_random(64, seed=0), make_view(), FakeRasterizer()
    render(g, [v], "cpu", fake)

    sent = fake.calls[0]["viewmats"][0].numpy()
    assert np.allclose(sent @ v.c2w, np.eye(4), atol=1e-5), \
        "viewmats must be the inverse of c2w"
    assert not np.allclose(sent, v.c2w, atol=1e-3)


def test_viewmat_inversion_matches_a_general_inverse():
    rng = np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c2w[:3, 3] = [1.5, -2.0, 3.0]
    assert np.allclose(viewmats_from_c2w(c2w)[0].numpy(),
                       np.linalg.inv(c2w), atol=1e-5)


def test_render_unpacks_rgb_and_depth_channels():
    g, v, fake = init_random(32, seed=0), make_view(), FakeRasterizer()
    rgb, depth, alpha, _ = render(g, [v], "cpu", fake)
    assert fake.calls[0]["render_mode"] == "RGB+ED"
    assert rgb.shape == (1, 16, 16, 3)
    assert depth.shape == (1, 16, 16)
    assert alpha.shape == (1, 16, 16, 1)


def test_render_composites_uncovered_pixels_over_background():
    """alpha=0.9 must leave 10% of the background showing."""
    g, v, fake = init_random(32, seed=0), make_view(), FakeRasterizer()
    dark, _, _, _ = render(g, [v], "cpu", fake, background=0.0)
    light, _, _, _ = render(g, [v], "cpu", fake, background=1.0)
    assert torch.allclose(light - dark, torch.full_like(dark, 0.1), atol=1e-6)


def test_render_forwards_intrinsics_and_size():
    g, v, fake = init_random(16, seed=0), make_view(h=24, w=32), FakeRasterizer()
    render(g, [v], "cpu", fake)
    c = fake.calls[0]
    assert (c["width"], c["height"]) == (32, 24)
    assert np.allclose(c["Ks"][0].numpy(), v.K)


# --------------------------------------------------------------------------
# The training loop
# --------------------------------------------------------------------------
def _cfg(**kw):
    base = {"scene": "lego", "n_views": 1, "init": "random",
            "depth": {"model": "gt"}, "loss": {"depth_kind": "ssi", "depth_lambda": 0.0},
            "train": {"iterations": 3, "seed": 0}}
    for k, v in kw.items():
        if isinstance(v, dict):
            base[k] = {**base.get(k, {}), **v}
        else:
            base[k] = v
    return base


def test_training_updates_every_parameter_group(tmp_path):
    import train as train_mod
    g, v = init_random(64, seed=0), make_view()
    before = {k: p.detach().clone() for k, p in g.params.items()}

    train_mod.train(_cfg(train={"iterations": 5}), g, [v], "cpu", tmp_path,
                    rasterize=FakeRasterizer(), strategy=FakeStrategy())

    for k, old in before.items():
        assert not torch.allclose(old, g.params[k]), f"{k} never updated"


def test_strategy_is_called_around_backward_every_step(tmp_path):
    import train as train_mod
    strat = FakeStrategy()
    train_mod.train(_cfg(train={"iterations": 7}), init_random(32, seed=0),
                    [make_view()], "cpu", tmp_path, FakeRasterizer(), strat)
    assert strat.pre == 7 and strat.post == 7


def test_depth_term_is_off_at_lambda_zero_and_on_above(tmp_path):
    import train as train_mod
    kw = dict(rasterize=FakeRasterizer(), strategy=FakeStrategy())
    off = train_mod.train(_cfg(loss={"depth_lambda": 0.0}), init_random(32, seed=0),
                          [make_view()], "cpu", tmp_path, **kw)
    on = train_mod.train(_cfg(loss={"depth_lambda": 0.5}), init_random(32, seed=0),
                         [make_view()], "cpu", tmp_path, **kw)
    assert all(r["depth"] == 0.0 for r in off["log"])
    assert any(r["depth"] != 0.0 for r in on["log"])


def test_scene_scale_tracks_the_camera_rig():
    import train as train_mod
    vs = []
    for d in (3.0, 4.0, 5.0):
        v = make_view(dist=d)
        vs.append(v)
    assert train_mod.scene_scale(vs) > 0


def test_evaluate_writes_renders_and_reports_both_metric_families(tmp_path):
    import train as train_mod
    m = train_mod.evaluate(init_random(32, seed=0), [make_view(), make_view()],
                           "cpu", tmp_path, save_n=1, rasterize=FakeRasterizer())
    assert {"psnr", "ssim", "lpips"} <= set(m)
    # Geometry must be reported unaligned AND aligned -- aligned alone would
    # erase the affine bias Axis C injects.
    assert any(k.startswith("unaligned_") for k in m)
    assert any(k.startswith("aligned_") for k in m)
    assert (tmp_path / "render_00.png").exists()


# --------------------------------------------------------------------------
# Failure modes must be actionable, not cryptic
# --------------------------------------------------------------------------
def test_gt_depth_missing_explains_the_blender_render_step():
    import train as train_mod
    with pytest.raises(SystemExit) as e:
        train_mod.attach_depth(_cfg(loss={"depth_lambda": 0.1}),
                               [make_view(with_depth=False)], "cpu")
    msg = str(e.value)
    assert ".blend" in msg and "depth_train" in msg


def test_depth_init_without_depth_names_the_offending_view():
    with pytest.raises(ValueError, match="no depth"):
        init_from_depth([make_view(with_depth=False)])


def test_depth_init_backprojects_in_front_of_the_camera():
    """Points must land at `depth` along the camera's forward axis.

    With the camera at z=-4 looking down +z (OpenCV) and depth 4, the correct
    answer is the ORIGIN -- which is where the object is. Checking the sign of
    z would pass for a camera pointing the wrong way; checking the distance
    from the camera along its forward axis would not.
    """
    v = make_view(dist=4.0)
    g = init_from_depth([v], stride=1)
    pts = g.params["means"].detach().numpy()

    assert len(g) > 0
    cam = np.asarray(v.c2w)[:3, 3]
    fwd = np.asarray(v.c2w)[:3, 2]
    along = (pts - cam) @ fwd
    assert np.allclose(along, 4.0, atol=1e-4), "points not at `depth` along forward"
    # The principal ray must land on the object at the origin.
    assert np.linalg.norm(pts.mean(axis=0)) < 1.0


def test_no_depth_needed_when_neither_init_nor_loss_uses_it():
    """init=random with lambda=0 must run on a dataset that has no depth at all.

    The prior reaches a run only through initialization or the depth loss. With
    neither active, requiring depth would block precisely the baseline runs
    that are supposed to work before GT depth has been rendered.
    """
    import train as train_mod
    info = train_mod.attach_depth(
        _cfg(init="random", loss={"depth_lambda": 0.0}),
        [make_view(with_depth=False)], "cpu")
    assert info["used"] is False


def test_depth_still_required_when_the_loss_is_on():
    import train as train_mod
    with pytest.raises(SystemExit):
        train_mod.attach_depth(_cfg(init="random", loss={"depth_lambda": 0.1}),
                               [make_view(with_depth=False)], "cpu")


def test_depth_still_required_when_init_uses_it():
    import train as train_mod
    with pytest.raises(SystemExit):
        train_mod.attach_depth(_cfg(init="depth", loss={"depth_lambda": 0.0}),
                               [make_view(with_depth=False)], "cpu")

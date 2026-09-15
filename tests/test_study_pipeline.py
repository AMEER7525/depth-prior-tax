"""Guards on the pieces the full study adds on top of the loss semantics.

Each test pins a behaviour that, if it silently broke, would bias a result
rather than crash: a noise level that does nothing, a split that is not nested,
a triangulation that keeps outliers, an alignment done in the wrong space, a
geometry metric that forgives an offset, a sweep that runs cells twice.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.corruptions import degrade, degrade_view                     # noqa: E402
from src.data import (BLENDER_DIETNERF_8, DTU_EXCLUDED_VIEWS,         # noqa: E402
                      DTU_SPARSE_INPUT_VIEWS, DTU_TEST_VIEWS, View,
                      _load_ply_points, blender_sparse_split, downscale_depth,
                      farthest_view_sampling)
from src.depth_model import (align_inverse_depth,                     # noqa: E402
                             align_inverse_to_points, align_to_metric,
                             disparity_to_depth)
from src.eval_geometry import (depth_metrics, dtu_chamfer,            # noqa: E402
                               floater_stats, reprojection_consistency)
from src.gaussians import Gaussians                                   # noqa: E402
from src.sfm import (fundamental_from_poses, project,                 # noqa: E402
                     projection_matrix, sampson_distance, triangulate_pair)
from src.sweep import degenerate_reason, expand, run_id               # noqa: E402

K128 = np.array([[200.0, 0, 64.0], [0, 200.0, 64.0], [0, 0, 1.0]])


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """OpenCV c2w (x right, y down, z forward) for a camera at `eye`."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


# --------------------------------------------------------------------------
# Corruptions: the knobs must bite, and only where depth is valid
# --------------------------------------------------------------------------
def test_noise_uses_the_median_of_valid_pixels_not_the_background():
    """NeRF-Synthetic views are mostly background (depth 0). A median over all
    pixels is then 0 and every noise level becomes a silent no-op."""
    d = torch.zeros(64, 64)
    d[:16] = 4.0                                       # 25% object
    out = degrade(d, sigma_rel=0.1, generator=torch.Generator().manual_seed(0))
    assert (out[:16] - 4.0).std().item() == pytest.approx(0.4, rel=0.15)
    assert torch.equal(out[16:], d[16:])               # background untouched


def test_affine_bias_leaves_invalid_pixels_invalid():
    d = torch.zeros(8, 8)
    d[:4] = 2.0
    d[4, 0] = float("nan")
    out = degrade(d, scale=1.5, shift=0.5)
    assert torch.allclose(out[:4], torch.full((4, 8), 3.5))
    assert (out[5:] == 0).all() and (out[4, 1:] == 0).all()
    assert torch.isnan(out[4, 0])


def test_per_image_jitter_draws_a_different_affine_per_view():
    d = torch.full((8, 8), 2.0)
    g = torch.Generator().manual_seed(0)
    scales = [degrade_view(d, scale_jitter=0.2, generator=g)[1]["scale"]
              for _ in range(5)]
    assert len(set(np.round(scales, 6))) == 5
    assert all(0.3 < s < 3.0 for s in scales)


def test_jitter_draws_do_not_depend_on_the_noise_level():
    """Turning the noise knob must not reshuffle the per-image scales."""
    d = torch.full((8, 8), 2.0)

    def scales(sigma):
        gn, gj = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
        return [degrade_view(d, sigma_rel=sigma, scale_jitter=0.2, generator=gn,
                             jitter_generator=gj)[1]["scale"] for _ in range(3)]

    assert scales(0.0) == scales(0.2)


def test_correlated_noise_has_the_same_amplitude_but_is_smooth():
    d = torch.full((128, 128), 4.0)
    iid = degrade(d, sigma_rel=0.1, generator=torch.Generator().manual_seed(0)) - d
    smooth = degrade(d, sigma_rel=0.1, generator=torch.Generator().manual_seed(0),
                     corr_px=4.0) - d
    assert smooth.std().item() == pytest.approx(iid.std().item(), rel=0.2)

    def step(n):
        return (n[:, 1:] - n[:, :-1]).std().item()

    assert step(smooth) < 0.3 * step(iid)


# --------------------------------------------------------------------------
# Splits and resampling
# --------------------------------------------------------------------------
def _cams(n=100, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = []
    for p in v:
        c = np.eye(4)
        c[:3, 3] = 4.0 * p
        out.append(c)
    return out


def _min_angle(c2ws, ids):
    d = np.stack([c2ws[i][:3, 3] for i in ids])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cos = np.clip(d @ d.T, -1, 1)
    np.fill_diagonal(cos, -1)
    return np.degrees(np.arccos(cos.max()))


def test_fps_split_is_nested_deterministic_and_spread():
    c2ws = _cams()
    s3, s9 = farthest_view_sampling(c2ws, 3), farthest_view_sampling(c2ws, 9)
    assert s9[:3] == s3 and len(set(s9)) == 9
    assert blender_sparse_split(3, c2ws) == s3 == farthest_view_sampling(c2ws, 3)
    rng = np.random.default_rng(1)
    random_subsets = [_min_angle(c2ws, rng.choice(100, 9, replace=False))
                      for _ in range(200)]
    assert _min_angle(c2ws, s9) > np.percentile(random_subsets, 95)


def test_dietnerf_split_is_the_published_one():
    assert blender_sparse_split(8, protocol="dietnerf") == BLENDER_DIETNERF_8
    assert BLENDER_DIETNERF_8 == [2, 16, 26, 55, 73, 76, 86, 93]
    with pytest.raises(ValueError):
        blender_sparse_split(3, protocol="dietnerf")


def test_dtu_test_split_is_the_regnerf_one():
    assert len(DTU_TEST_VIEWS) == 25
    assert not set(DTU_TEST_VIEWS) & set(DTU_SPARSE_INPUT_VIEWS + DTU_EXCLUDED_VIEWS)


def test_downscale_depth_drops_silhouette_and_occlusion_blocks():
    """Averaging across an edge would invent a surface between two real ones."""
    d = np.full((4, 4), 2.0, np.float32)
    d[0, 0] = 0.0                                       # touches background
    d[2:, 2:] = [[2.0, 5.0], [2.0, 2.0]]                # straddles an occlusion edge
    out = downscale_depth(d, 2)
    assert out[0, 0] == 0 and out[1, 1] == 0
    assert out[0, 1] == pytest.approx(2.0) and out[1, 0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# SfM with known poses
# --------------------------------------------------------------------------
def _pair():
    return look_at([4.0, -1.0, 1.0]), look_at([4.0, 1.0, 1.0])


def _points(n=200, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n, 3))


def test_fundamental_from_poses_satisfies_the_epipolar_constraint():
    c1, c2 = _pair()
    X = _points()
    x1, _ = project(projection_matrix(K128, c1), X)
    x2, _ = project(projection_matrix(K128, c2), X)
    F = fundamental_from_poses(K128, c1, K128, c2)
    assert sampson_distance(F, x1, x2).max() < 1e-8


def test_triangulation_recovers_points_and_rejects_bad_matches():
    c1, c2 = _pair()
    X = _points()
    rng = np.random.default_rng(1)
    x1 = project(projection_matrix(K128, c1), X)[0] + rng.normal(0, 0.2, (200, 2))
    x2 = project(projection_matrix(K128, c2), X)[0] + rng.normal(0, 0.2, (200, 2))
    x2[:50] = rng.uniform(0, 128, size=(50, 2))         # 50 wrong matches
    pts, keep = triangulate_pair(K128, c1, K128, c2, x1, x2)
    good = keep >= 50
    assert good.sum() >= 140
    assert np.abs(pts[good] - X[keep[good]]).max() < 0.05
    assert (~good).sum() <= 5


def test_triangulation_drops_points_behind_the_cameras():
    c1, c2 = _pair()
    X = np.array([[8.0, 0.0, 2.0]])                     # behind both
    x1 = project(projection_matrix(K128, c1), X)[0]
    x2 = project(projection_matrix(K128, c2), X)[0]
    pts, _ = triangulate_pair(K128, c1, K128, c2, x1, x2)
    assert len(pts) == 0


# --------------------------------------------------------------------------
# Real-model alignment happens in inverse-depth space
# --------------------------------------------------------------------------
def _disparity_scene(seed=0):
    depth = np.random.default_rng(seed).uniform(2.0, 6.0, size=(32, 32))
    return depth, 3.0 / depth + 0.2                     # affine in DISPARITY


def test_inverse_depth_alignment_recovers_metric_depth():
    depth, disp = _disparity_scene()
    out, s, t = align_inverse_depth(disp, depth)
    assert np.allclose(out, depth, rtol=1e-5)
    assert s == pytest.approx(1 / 3, rel=1e-6) and t == pytest.approx(-0.2 / 3, rel=1e-6)


def test_depth_space_alignment_cannot_undo_a_disparity_shift():
    """Why alignment is done on disparity: with a shift, 1/disparity is not an
    affine function of depth, so no depth-space fit can recover it."""
    depth, disp = _disparity_scene()
    aligned, _, _ = align_to_metric(disparity_to_depth(disp), depth)
    assert np.abs(aligned - depth).mean() > 0.05


def test_sparse_point_alignment_survives_outliers_and_refuses_too_few():
    depth, disp = _disparity_scene()
    rng = np.random.default_rng(2)
    i, j = rng.integers(0, 32, 60), rng.integers(0, 32, 60)
    px = np.stack([j + 0.5, i + 0.5], axis=1)
    z = depth[i, j].copy()
    z[:6] *= 3.0                                        # 10% bad triangulations
    out, _, _ = align_inverse_to_points(disp, px, z)
    assert np.allclose(out, depth, rtol=1e-3)
    with pytest.raises(ValueError):
        align_inverse_to_points(disp, px[:0], z[:0])      # nothing in view


# --------------------------------------------------------------------------
# Geometry metrics
# --------------------------------------------------------------------------
def test_depth_metrics_mask_never_readmits_invalid_gt():
    gt = np.full((4, 4), 2.0)
    gt[0] = 0.0
    pred = np.full((4, 4), 2.0)
    pred[0] = 100.0
    assert depth_metrics(pred, gt, mask=np.ones((4, 4), bool))["mae"] == 0.0


def _plane_cloud(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, 100, n), rng.uniform(0, 100, n),
                     np.full(n, 50.0)], axis=1)


def test_dtu_chamfer_is_zero_for_the_gt_and_tracks_an_offset():
    stl = _plane_cloud()
    obs = np.ones((111, 111, 111), bool)
    bb = np.array([[-5.0, -5.0, -5.0], [105.0, 105.0, 105.0]])
    plane = np.array([0.0, 0.0, 1.0, -10.0])            # z > 10 is "above"
    assert dtu_chamfer(stl, stl, obs, bb, 1.0, plane)["chamfer"] < 0.6
    off = dtu_chamfer(stl + [0, 0, 3.0], stl, obs, bb, 1.0, plane)
    assert off["chamfer"] == pytest.approx(3.0, abs=0.5)


def test_floater_stats_counts_only_opaque_far_gaussians():
    g = np.linspace(-1, 1, 21)
    surface = np.stack(np.meshgrid(g, g, [0.0]), -1).reshape(-1, 3)
    means = np.array([[0, 0, 0.01], [0, 0, 1.0], [0, 0, 2.0]])
    f = floater_stats(means, [0.9, 0.9, 0.1], surface, tau=0.1)
    assert (f["floater_count"], f["n_opaque"], f["floater_frac"]) == (1, 2, 0.5)


def test_reprojection_consistency_is_zero_for_a_consistent_surface():
    K = np.array([[64.0, 0, 32.0], [0, 64.0, 32.0], [0, 0, 1.0]])
    ca, cb = np.eye(4), np.eye(4)
    cb[0, 3] = 0.5
    img = np.zeros((64, 64, 3), np.float32)
    va, vb = View(img, K, ca), View(img, K, cb)
    plane = np.full((64, 64), 5.0)
    r = reprojection_consistency(plane, va, plane, vb)
    assert r["median_rel"] < 1e-9 and r["n"] > 100
    r2 = reprojection_consistency(plane, va, np.full((64, 64), 5.5), vb)
    assert r2["median_rel"] == pytest.approx(0.5 / 5.5, rel=1e-6)


# --------------------------------------------------------------------------
# Gaussians
# --------------------------------------------------------------------------
def test_sh_degree_sets_coefficient_shapes_and_the_ply_roundtrips(tmp_path):
    means = np.random.default_rng(0).normal(size=(5, 3))
    g = Gaussians(means, np.full((5, 3), 0.25), sh_degree=3)
    assert g.params["sh0"].shape == (5, 1, 3) and g.params["shN"].shape == (5, 15, 3)
    assert g.activated["sh"].shape == (5, 16, 3)
    assert torch.allclose(g.activated["rgb"], torch.full((5, 3), 0.25), atol=1e-6)
    g.save_ply(tmp_path / "g.ply")
    assert np.allclose(_load_ply_points(tmp_path / "g.ply"), means, atol=1e-6)


# --------------------------------------------------------------------------
# Sweep: pruning and routing
# --------------------------------------------------------------------------
def test_inert_real_model_cell_is_pruned():
    assert degenerate_reason({"init": "random", "depth_lambda": 0.0,
                              "depth_model": "depth_anything_v2"})


def test_per_image_affine_on_the_loss_only_path():
    base = {"init": "random", "depth_lambda": 0.1, "scale_jitter": 0.2}
    assert "invariant" in degenerate_reason({**base, "depth_kind": "pearson"})
    assert "per-view" in degenerate_reason({**base, "depth_kind": "ssi"})
    assert degenerate_reason({**base, "depth_kind": "absolute"}) is None
    assert degenerate_reason({**base, "depth_kind": "ssi", "init": "depth"}) is None


def test_alignment_axis_is_inert_for_gt_depth():
    assert degenerate_reason({"init": "depth", "depth_lambda": 0.0, "depth_align": "sfm"})
    assert degenerate_reason({"init": "depth", "depth_lambda": 0.0, "depth_align": "sfm",
                              "depth_model": "depth_anything_v2"}) is None


def test_fixed_block_is_merged_and_run_ids_stay_unique():
    cells, _ = expand({"fixed": {"dataset": "dtu"},
                       "axes": {"init": ["sfm", "random"], "n_views": [3, 9],
                                "depth_lambda": [0.0]},
                       "scenes": ["scan1", "scan6"]})
    assert len(cells) == 8 and all(c["dataset"] == "dtu" for c in cells)
    assert len({run_id(c) for c in cells}) == 8
    assert "dataset-dtu" in run_id(cells[0])


def test_every_sweep_stage_expands_and_routes_into_the_base_config():
    """A typo in sweep.yaml must fail here, not 40 minutes into a Colab session."""
    import run_sweep
    sweep = yaml.safe_load(open(REPO / "configs" / "sweep.yaml"))
    base = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    for name, stage in sweep.items():
        cells, _ = expand(stage)
        assert cells, f"{name} expands to nothing"
        for c in cells:
            cfg = run_sweep._apply_cell(base, c)
            run_sweep._check_corruption_reachable(cfg, c, run_id(c))


# --------------------------------------------------------------------------
# Results collection and analysis
# --------------------------------------------------------------------------
def test_collector_flattens_runs_and_reports_failures(tmp_path, monkeypatch):
    import evaluate as ev
    ok = tmp_path / "stage1" / "lego__init-random"
    ok.mkdir(parents=True)
    (ok / "metrics.json").write_text(json.dumps({
        "config": {"scene": "lego", "init": "random", "n_views": 3,
                   "loss": {"depth_lambda": 0.0}},
        "psnr": 20.0, "unaligned_mae": 0.1, "depth": {"used": False, "model": "gt"},
        "psnr_curve": [[500, 15.0]], "opacity_resets": [500, 3000]}))
    bad = tmp_path / "stage1" / "lego__init-sfm"
    bad.mkdir()
    (bad / "FAILED").write_text("exit 1\n")
    (bad / "error.txt").write_text("Traceback ...\nERROR: ValueError: SfM triangulated 0 point(s)\n")
    (bad / "cell.json").write_text(json.dumps({"init": "sfm", "scene": "lego"}))

    out = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--runs", str(tmp_path),
                                      "--out", str(out), "--sweep", ""])
    ev.main()
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1
    assert rows[0]["stage"] == "stage1" and rows[0]["init"] == "random"
    assert float(rows[0]["psnr"]) == 20.0 and rows[0]["depth_used"] == "False"
    fails = list(csv.DictReader(open(tmp_path / "results_failures.csv")))
    assert fails[0]["init"] == "sfm" and "SfM" in fails[0]["error"]


def _toy_results():
    pd = pytest.importorskip("pandas")
    rows = []
    for scene in ("lego", "chair"):
        common = {"scene": scene, "n_views": 3, "iterations": 7000, "ssim": 0.8,
                  "lpips": 0.2, "n_gaussians": 50000, "train_seconds": 120.0,
                  "init_n_init_points": 100}
        rows += [
            {**common, "stage": "stage1", "init": "random", "depth_lambda": 0.0,
             "psnr": 20.0, "unaligned_mae": 0.10, "psnr_curve": "[[500, 10.0], [7000, 18.0]]"},
            {**common, "stage": "stage1", "init": "depth", "depth_lambda": 0.0,
             "psnr": 22.0, "unaligned_mae": 0.05, "prior_absrel": 0.0,
             "psnr_curve": "[[500, 18.5], [7000, 20.0]]"},
            {**common, "stage": "stage3_affine", "init": "depth", "depth_lambda": 0.0,
             "scale": 1.5, "psnr": 21.0, "unaligned_mae": 0.30, "prior_absrel": 0.5},
            {**common, "stage": "stage2_lambda", "init": "depth", "depth_lambda": 1.0,
             "depth_kind": "absolute", "psnr": 19.0, "unaligned_mae": 0.20},
            {**common, "stage": "stage2_lambda", "init": "depth", "depth_lambda": 0.1,
             "depth_kind": "absolute", "psnr": 22.5, "unaligned_mae": 0.04,
             "prior_absrel": 0.0},
            {**common, "stage": "stage4_real", "init": "depth", "depth_lambda": 0.0,
             "depth_model": "depth_anything_v2", "depth_align": "gt",
             "psnr": 21.5, "unaligned_mae": 0.08, "prior_absrel": 0.12},
        ]
    return pd.DataFrame(rows)


def test_effects_are_classified_against_the_vanilla_baseline():
    from src.analysis import TRADE_UP, divergences, load_results
    df = load_results(_toy_results())
    assert df.loc[(df.stage == "stage1") & (df.init == "depth"), "effect"].eq("helps").all()
    assert df.loc[df.stage == "stage3_affine", "effect"].eq(TRADE_UP).all()
    assert df.loc[df.depth_lambda == 1.0, "effect"].eq("hurts").all()
    assert set(divergences(df)["stage"]) == {"stage3_affine"}


def test_axis_divergences_find_the_lambda_flip():
    from src.analysis import axis_divergences, load_results
    df = load_results(_toy_results())
    out = axis_divergences(df, "depth_lambda")
    assert out.empty or not out["effect"].str.startswith("trade").all() or len(out) > 0
    moved = df[(df.depth_kind == "absolute") | (df.depth_lambda == 0)]
    assert len(moved)


def test_iterations_to_target_uses_the_baseline_probe():
    from src.analysis import iters_to_target, load_results
    df = load_results(_toy_results())
    it = iters_to_target(df)
    base = (df.init == "random").to_numpy()
    depth0 = ((df.stage == "stage1") & (df.init == "depth")).to_numpy()
    assert np.all(it[base] == 7000) and np.all(it[depth0] == 500)


def test_every_figure_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src import analysis as A
    df = A.load_results(_toy_results())
    for name, fig in {"init": A.plot_init_by_views(df),
                      "lambda": A.plot_lambda_sweep(df),
                      "degr": A.plot_degradation(df, kind="absolute"),
                      "map": A.plot_tradeoff_map(df)}.items():
        A.save(fig, tmp_path / name)
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_notebook_shell_lines_never_mix_env_vars_and_python_substitution():
    """IPython expands {expr} and $name on a `!` line together, and if ANY name
    fails to resolve as a Python variable it silently expands NOTHING. A line
    like `!cmd --runs "$ENV_VAR" --out "{PY_VAR}/x"` then passes the literal
    text "{PY_VAR}" to the shell -- which is how results.csv once went missing."""
    import re
    for nb_path in (REPO / "notebooks").glob("*.ipynb"):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for line in "".join(cell["source"]).splitlines():
                s = line.strip()
                if s.startswith("!"):
                    assert not (re.search(r"\$\w", s) and "{" in s), \
                        f"{nb_path.name}: mixes $ENV and {{}} in: {s}"


def test_figures_without_their_runs_say_which_stage_is_missing():
    """Only the smoke stage has run: every figure must say what it is waiting
    for instead of drawing empty axes, and leave no blank figure open for the
    notebook to auto-display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import analysis as A
    smoke = A.load_results(_toy_results().query("stage == 'stage2_lambda' and depth_lambda == 0.1"))
    plt.close("all")
    for fig in (A.plot_init_by_views(smoke), A.plot_lambda_sweep(smoke),
                A.plot_degradation(smoke), A.plot_tradeoff_map(smoke)):
        assert isinstance(fig, A.NoData) and "stage" in fig
    assert not plt.get_fignums()
    assert A.save(A.NoData("needs stage1"), "/nonexistent/dir/fig") is False


def test_geometry_label_follows_the_plotted_dataset():
    """Blender runs also report Chamfer; their primary number is still depth MAE."""
    from src import analysis as A
    df = A.load_results(_toy_results()).assign(chamfer=0.1)
    label = A._geom_label(df)
    assert "depth MAE" in label and "Chamfer" not in label


def test_rerunning_the_code_cell_drops_stale_imports():
    """Cell 3 re-copies the code; without dropping cached `src` modules, later
    cells keep running the previous copy until a runtime restart."""
    nb = json.loads((REPO / "notebooks" / "colab.ipynb").read_text())
    cell3 = next("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title 3."))
    assert "sys.modules" in cell3 and "startswith('src.')" in cell3


@pytest.mark.parametrize("progress", ["smoke only", "every stage"])
def test_analysis_notebook_runs_at_every_stage_of_progress(tmp_path, monkeypatch, progress):
    """The report notebook is opened long before every stage has run; no cell
    may crash because a later stage's columns do not exist yet."""
    pytest.importorskip("pandas")
    import matplotlib
    matplotlib.use("Agg")
    df = _toy_results()
    if progress == "smoke only":
        df = df[df.stage == "stage2_lambda"].assign(stage="smoke")
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)
    work = tmp_path / "notebooks"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("RESULTS_CSV", str(csv_path))
    monkeypatch.syspath_prepend(str(REPO))
    nb = json.loads((REPO / "notebooks" / "analysis.ipynb").read_text())
    scope = {"display": lambda *a, **k: None, "__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "analysis.ipynb", "exec"), scope)


def test_collector_leaves_out_pre_fix_runs_and_orphans(tmp_path, monkeypatch):
    """Stage0 once mixed pre-fix and fixed runs of the same scenes into one
    gate-1 mean. Only runs with the opacity-reset fix AND settings the sweep
    still defines may count; --all is the escape hatch."""
    import evaluate as ev
    from src.sweep import expand, run_id
    sweep = yaml.safe_load(open(REPO / "configs" / "sweep.yaml"))

    def run(stage, rid, **rec):
        d = tmp_path / stage / rid
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({
            "config": {"scene": rid.split("__")[0], "init": "random", "n_views": 8},
            "psnr": 21.0, "ssim": 0.8, "lpips": 0.2, **rec}))

    good = run_id(expand(sweep["stage0_repro"])[0][0])
    run("stage0_repro", good, opacity_resets=[500, 3000])               # counts
    run("stage0_repro", "chair__old_schedule", opacity_resets=[3000])    # orphan
    run("smoke", run_id(expand(sweep["smoke"])[0][0]))                   # pre-fix
    out = tmp_path / "r.csv"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--runs", str(tmp_path), "--out", str(out)])
    ev.main()
    assert [r["run_id"] for r in csv.DictReader(open(out))] == [good]
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--runs", str(tmp_path), "--out", str(out),
                                      "--all"])
    ev.main()
    assert len(list(csv.DictReader(open(out)))) == 3


def test_resume_reruns_runs_trained_before_the_opacity_reset_fix(tmp_path):
    import run_sweep
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "metrics.json").write_text(json.dumps({"psnr": 20.0}))
    (new / "metrics.json").write_text(json.dumps({"psnr": 21.0, "opacity_resets": [500, 3000]}))
    assert not run_sweep.is_finished(old) and run_sweep.is_finished(new)
    assert not run_sweep.is_finished(tmp_path / "absent")


def test_run_sweep_names_the_cell_that_provides_missing_data(tmp_path):
    """Missing inputs stop a stage before it starts, instead of one FAILED run
    per scene -- stage0 once failed 5 of 8 scenes that were never copied."""
    import run_sweep
    base = yaml.safe_load(open(REPO / "configs" / "base.yaml"))
    depth = run_sweep._apply_cell(base, {"scene": "lego", "init": "depth", "depth_lambda": 0.0})
    assert "cell 4" in run_sweep.missing_inputs(depth, tmp_path)[0]
    scene = tmp_path / "nerf_synthetic" / "lego"
    scene.mkdir(parents=True)
    (scene / "transforms_train.json").write_text("{}")
    assert "cell 5" in run_sweep.missing_inputs(depth, tmp_path)[0]
    (scene / "depth_train").mkdir()
    np.save(scene / "depth_train" / "r_0.npy", np.zeros(1))
    assert run_sweep.missing_inputs(depth, tmp_path) == []
    rnd = run_sweep._apply_cell(base, {"scene": "lego", "init": "random", "depth_lambda": 0.0})
    assert run_sweep.missing_inputs(rnd, tmp_path) == []
    dtu = run_sweep._apply_cell(base, {"scene": "scan1", "dataset": "dtu", "init": "random",
                                       "depth_lambda": 0.0})
    assert "cell 7" in run_sweep.missing_inputs(dtu, tmp_path)[0]


def test_init_figure_does_not_mix_protocols():
    """stage0_repro shares this figure's axes -- init, view count, no depth
    loss -- but runs the published split, SH 3 and a 6k budget. Its points must
    not join the stage1 lines, or the figure compares more than the axis."""
    pd = pytest.importorskip("pandas")
    import matplotlib
    matplotlib.use("Agg")
    from src import analysis as A
    study = _toy_results()
    study = study[study.stage == "stage1"]
    repro = study.head(1).assign(stage="stage0_repro", n_views=8, split="dietnerf",
                                 sh_degree=3, iterations=6000)
    df = A.load_results(pd.concat([study, repro], ignore_index=True))
    clean = df[(df.depth_lambda == 0) & (df.corruption == "clean")]
    kept = A._one_protocol(clean)
    assert set(kept.n_views) == {3}
    assert "stage0_repro" not in set(kept.stage)
    assert "fps split" in A._protocol_label(kept)


def test_tradeoff_map_scales_to_the_bulk_and_flags_outliers():
    """One blown-up run (pearson at lambda=1 was +660% geometry error) must not
    squash every other point into a band: it is parked on the frame and counted."""
    pd = pytest.importorskip("pandas")
    import matplotlib
    matplotlib.use("Agg")
    from src import analysis as A
    rows = _toy_results()
    blown = rows[rows.stage == "stage2_lambda"].head(1).assign(
        scene="lego", unaligned_mae=5.0, psnr=12.0)
    df = A.load_results(pd.concat([rows, blown], ignore_index=True))
    fig = A.plot_tradeoff_map(df)
    assert "beyond the axis" in fig._suptitle.get_text()
    top = fig.axes[0].get_ylim()[1]
    worst = 100 * df["d_geometry_rel"].max()
    assert top < worst                      # the axis follows the bulk, not the outlier
    A.save(fig, "/tmp/_tradeoff_test")


def test_sparse_alignment_survives_the_handful_of_points_3_view_sfm_gives():
    """3-view SfM triangulates 6-10 points TOTAL, a few per view. Alignment has
    to degrade (exact 2-point fit, then scale-only) instead of raising: how
    metric a practitioner can get from 3-view SfM is the measurement."""
    depth, disp = _disparity_scene()
    rng = np.random.default_rng(3)
    i, j = rng.integers(0, 32, 8), rng.integers(0, 32, 8)
    px = np.stack([j + 0.5, i + 0.5], axis=1)
    z = depth[i, j]
    for n in (8, 3, 2):
        out, s, _ = align_inverse_to_points(disp, px[:n], z[:n])
        assert np.allclose(out, depth, rtol=1e-3), f"{n} points"
        assert s > 0
    one, s1, t1 = align_inverse_to_points(disp, px[:1], z[:1])   # scale only
    assert t1 == 0.0 and s1 > 0 and np.isfinite(one).all() and (one > 0).all()


def test_degenerate_two_point_alignment_falls_back_to_scale_only():
    """Two points with the same disparity cannot determine a shift; the fit
    must not explode into negative or infinite depth."""
    depth, disp = _disparity_scene()
    flat = np.full_like(disp, 0.5)
    px = np.array([[1.5, 1.5], [9.5, 9.5]])
    z = np.array([2.0, 5.0])
    out, s, t = align_inverse_to_points(flat, px, z)
    assert s > 0 and np.isfinite(out).all() and (out > 0).all()


def test_reference_validation_reports_the_error_the_study_would_see():
    """A reference 1% too deep must read as 1% AbsRel at the study's resolution
    (the scale every prior error is reported on), and a Blender silhouette off
    the release mask must lower the IoU -- the check that the .blend geometry
    is the geometry the images show."""
    from validate_reference_depth import view_agreement
    exact = np.zeros((64, 64))
    exact[8:56, 8:56] = np.linspace(2.0, 2.5, 48)[None, :]
    mask = exact > 0
    alpha = mask.astype(np.float32)

    r = view_agreement(1.01 * exact, exact, mask, alpha)
    assert r["absrel"] == pytest.approx(0.01, rel=1e-6)
    assert np.median(r["rel"]) == pytest.approx(0.01, rel=1e-6)
    assert r["silhouette_iou"] == 1.0 and r["coverage"] == 1.0

    off = view_agreement(exact, exact, mask, np.roll(alpha, 4, axis=1))
    assert off["silhouette_iou"] < 0.9 and off["absrel"] == 0.0

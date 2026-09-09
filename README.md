# Sparse-View Depth-Prior 3DGS — Study

Controlled study of when a monocular-depth prior helps vs. harms geometry in
sparse-view 3D Gaussian Splatting. Axes: how depth enters (init vs. loss),
number of views, and depth-prior quality.

## Structure
```
configs/     base config + the staged sweep
src/         OUR code (the contribution) — clearly ours
scripts/     drivers: setup, single run, sweep, evaluate
tests/       guards on the semantics the study depends on
third_party/ the original 3DGS repo (NOT ours; do not edit)
notebooks/   colab_bootstrap.ipynb (overflow host), analysis.ipynb (figures)
```

## Code / third-party boundary
Everything under `src/`, `scripts/` and `tests/` is written by us. The
rasterizer is used as an installed dependency (`gsplat`) and is never edited.
We chose gsplat over the INRIA reference implementation deliberately: its
`render_mode="RGB+ED"` exposes expected depth as a rendered channel, so the
depth loss is *ours* to define. The INRIA repo ships a depth-regularization
path (`-d`, `make_depth_scale.py`) but hard-codes the loss, which would force
us to patch third-party code — the thing this boundary exists to prevent.

## Where things run
| | Mac (Apple Silicon) | GPU host (4080 / Colab) |
|---|---|---|
| data loading, splits, corruption | yes | yes |
| depth inference (Depth Anything V2, MPS) | yes | yes |
| metrics, analysis notebook | yes | yes |
| **3DGS optimization** | **no — gsplat is CUDA-only** | yes |

```bash
# Mac / anywhere
pip install -r requirements.txt && pytest tests/ -q

# GPU host
bash scripts/setup_gpu.sh
```

Both hosts read `$DATA_ROOT` and write `$RUNS_ROOT`; no dataset path is
hard-coded anywhere. Point them at shared storage and the 4080 and Colab can
work the same grid — `--shard 0/2` and `--shard 1/2`, `--resume` on both.

## Reproduce a single run
```bash
python scripts/train.py --config configs/base.yaml --out runs/demo
```

## Reproduce the full study
The sweep is **staged**: each stage fixes the winner of the previous one. The
full Cartesian product of all four axes is ~550 runs (~140–275 GPU-hours);
staged it is 137 runs (~35–70 hours).

```bash
python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage1 --resume   # 27  init x views
python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage2 --resume   # 54  lambda x loss kind
python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage3_noise      # 32  Axis C: noise
python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage3_affine     # 24  Axis C: affine bias
python scripts/evaluate.py --runs runs --out results.csv
jupyter lab notebooks/analysis.ipynb
```
Set each stage's `init` / `depth_lambda` in `configs/sweep.yaml` to the
previous stage's winner before launching it.

## Two design facts this codebase encodes
Both are pinned by `tests/test_depth_semantics.py`; neither is obvious from the
proposal, and getting either wrong produces a null result for a definitional
reason rather than a scientific one.

**1. A scale-and-shift-invariant loss cannot see an affine bias as error.**
Fitting `(s,t)` to a corrupted target `a·d+b` yields `(a·s, a·t+b)`, so the
residual is exactly `a×` the clean residual: the gradient *direction* is
unchanged and only its magnitude scales. Under SSI an affine bias is **a gain
on λ, not a geometry error**; under Pearson it is a literal no-op. Only the
absolute loss registers it. So Axis C reaches geometry mainly through the
**init** path, where backprojection consumes metric depth and genuinely
displaces the cloud. `src/sweep.py` prunes the cells where no path is active.

**2. The aligned depth metric erases the injected bias too.**
`depth_metrics(align=True)` least-squares fits scale+shift before scoring, so a
depth map globally wrong by 2× scores ~0 error. Report **unaligned metric
MAE/RMSE as the primary geometry number** on Synthetic, aligned as secondary.

## Datasets (ground-truth geometry required)
Layout under `$DATA_ROOT`:
```
nerf_synthetic/<scene>/transforms_{train,val,test}.json, train/, test/
nerf_synthetic/<scene>/depth_<split>/r_<i>.npy      <- see caveat below
dtu/scan<id>/{image,mask}/, cameras.npz
dtu/Points/stl/stl<id>_total.ply                    <- Chamfer reference
```

- **NeRF-Synthetic** — hosts the controlled depth-corruption experiments.
  *Caveat:* the standard release ships **no GT depth**. It must be re-rendered
  from the original `.blend` files with Blender's depth pass into
  `depth_<split>/r_<i>.npy` (metric world units). Until that exists, Axis C
  cannot run on this dataset. This is a week-1 dependency, not a week-4 one.
- **DTU** — real scenes with structured-light GT clouds; Chamfer eval and the
  realistic operating point. Input-view ids follow the RegNeRF/PixelNeRF
  protocol (`src/data.py:DTU_SPARSE_INPUT_VIEWS`). Note that the published
  splits are **3/6/9** views — the proposal's 5-view point is ours alone and
  comparable to nothing in the literature. `cameras.npz` normalizes the scene
  into a unit sphere; the official Chamfer eval expects the original scan
  frame, so `Scene.meta["scale_mat"]` must be undone before scoring.

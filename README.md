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

**Colab is the primary and only compute host.** The Mac is for editing code,
reading results, and building figures — gsplat compiles CUDA kernels and cannot
be installed on Apple Silicon.

Open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) and run the cells in
order. Everything below is what that notebook automates.

### Storage layout, and why it matters
Only **Google Drive** is reachable from a Colab VM. A local `data/` folder and
the Kingston SSD are invisible to it, so the datasets have to live on Drive.
But Drive's FUSE mount is slow for many small files, and NeRF-Synthetic is
thousands of small PNGs — so archives live on Drive and are extracted to local
Colab disk each session:

| what | where | why |
|---|---|---|
| dataset archives | Drive | downloaded once, survive every session |
| dataset extracted | `/content` | FUSE throttles thousands of small reads |
| runs (working) | `/content` | training writes constantly |
| runs (persisted) | Drive | synced per stage so `--resume` survives a disconnect |

Re-extracting from a local archive takes seconds; re-downloading does not.

### Fetch the data from inside Colab, not from your laptop
```bash
python scripts/fetch_data.py --scenes lego chair ship \
    --archives "/content/drive/MyDrive/dl3dcv/datasets/blender" \
    --extract  /content/data
```
Colab pulls from HuggingFace at datacenter speed. Downloading the same 595 MB
on the Mac was measured at roughly 100 KB/s, and would then need uploading to
Drive on top of that.

### Local dev setup (Mac)
```bash
pip install -r requirements.txt && pytest tests/ -q
```

## Reproduce a single run
```bash
python scripts/train.py --config configs/base.yaml --out runs/demo
```

## Reproduce the full study
The sweep is **staged**: each stage fixes the winner of the previous one. The
full Cartesian product of all four axes is ~550 runs (~140–275 GPU-hours);
staged it is 145 runs (~35–70 hours). Run these from the Colab notebook.

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

## Datasets

### NeRF-Synthetic — sourced and verified
Per-scene archives from the [nerfbaselines mirror](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data),
fetched by `scripts/fetch_data.py`. Per-scene rather than the 1.27 GB monolith,
so a 3-scene sweep pulls 595 MB (lego 180, chair 130, ship 285).

Layout verified against the real archive: original NeRF format, with
`transforms_{train,val,test}.json` and `train/`, `val/`, `test/` image dirs —
what `src.data.load_blender_scene` expects.

**GT depth caveat.** The archives contain `r_*_depth_*.png`, but **only for the
test split**, and they are 8-bit normalized PNGs, not metric depth. The depth
prior is applied to the *input* views, which have no depth at all. So Axis C
still requires re-rendering metric depth from the original `.blend` files into
`depth_<split>/r_<i>.npy`. This remains the critical path.

### Depth model — Depth Anything V2
Fetched by `scripts/fetch_model.py`; wrapped by `src/depth_model.py`.

```bash
python scripts/fetch_model.py --variant large --dest <dir>   # 99 / 390 / 1341 MB
```

Needed **only for `stage4_real`**. Axis C perturbs ground-truth depth and uses
no model, so this is off the critical path.

**What it outputs matters more here than usual.** DAv2 predicts *relative
inverse depth*, defined only up to scale and shift — it is not metric. That
ambiguity is the phenomenon this project studies, so the code refuses to hide
it:

- As a **loss target**, feed the raw prediction to a scale-invariant loss.
  Passing it to the absolute loss compares disparity against metres.
- As an **init source**, backprojection needs metric depth, so the raw
  prediction backprojects to garbage. `align_to_metric()` fits it to something
  with real scale and **returns the `(scale, shift)` it solved** — log those:
  they are what places the real model on the Axis C degradation curve.

### DTU — not yet sourced
The route the sparse-view papers use (DNGaussian, FSGS) starts from DTU
**Rectified, 123 GB**, plus COLMAP preprocessing to recover poses. That does
not fit a Colab/Drive workflow, and Drive's free tier is 15 GB.

Deferring DTU matches the proposal's own §9 mitigation — *"lean on
NeRF-Synthetic (exact GT depth) for the controlled curve; use DTU as real-world
confirmation, not the primary measurement."* A smaller preprocessed DTU
(pixelNeRF/IDR style, with `cameras.npz`) is the thing to hunt for when the
realism check is needed; `src.data.load_dtu_scene` already targets that layout.

Expected layout under `$DATA_ROOT`:
```
nerf_synthetic/<scene>/transforms_{train,val,test}.json, train/, val/, test/
nerf_synthetic/<scene>/depth_<split>/r_<i>.npy      <- must be rendered
dtu/scan<id>/{image,mask}/, cameras.npz             <- not yet sourced
dtu/Points/stl/stl<id>_total.ply                    <- Chamfer reference
```

Input-view ids follow the RegNeRF/PixelNeRF protocol
(`src/data.py:DTU_SPARSE_INPUT_VIEWS`). The published splits are **3/6/9**
views — the proposal's 5-view point is ours alone and comparable to nothing in
the literature.

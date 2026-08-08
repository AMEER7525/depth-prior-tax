# Sparse-View Depth-Prior 3DGS — Study

Controlled study of when a monocular-depth prior helps vs. harms geometry in
sparse-view 3D Gaussian Splatting. Axes: how depth enters (init vs. loss),
number of views, and depth-prior quality.

## Structure
```
configs/     experiment + sweep YAMLs
src/         OUR code (the contribution) — clearly ours
scripts/     drivers: single run, sweep, evaluate
third_party/ the original 3DGS repo (NOT ours; do not edit)
notebooks/   analysis.ipynb — plots & qualitative figures for the report
```

## Code / third-party boundary
Everything under `src/` and `scripts/` is written by us. The original 3DGS
optimizer lives untouched under `third_party/3dgs/`; our changes to its
behaviour are made by importing from it in `src/`, never by editing it in place.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then install the 3DGS CUDA submodules per third_party/3dgs/README.md
```

## Reproduce a single run
```bash
python scripts/train.py --config configs/base.yaml --out runs/demo
```

## Reproduce the full study
```bash
python scripts/run_sweep.py --sweep configs/sweep.yaml     # launch the grid
python scripts/evaluate.py  --runs runs --out results.csv  # collect metrics
jupyter lab notebooks/analysis.ipynb                       # build report figures
```

## Datasets (ground-truth geometry required)
- NeRF-Synthetic (Blender) — exact rendered GT depth; hosts the controlled
  depth-corruption experiments.
- DTU — real scenes with structured-light GT point clouds; Chamfer eval and the
  realistic operating point. Sparse-view splits follow the RegNeRF/PixelNeRF protocol.

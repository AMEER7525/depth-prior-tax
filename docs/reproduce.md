# Reproducing the study

The analysis, figures and report need only a CPU and `results/results.csv`. Training
needs an NVIDIA GPU, because the rasterizer (gsplat) is CUDA-only; the study itself
was run on Google Colab. The full sweep of 257 runs took 9.5 GPU hours.

## Hardware

| GPU | Used here | Suitability |
|---|---|---|
| L4 (24 GB) | yes | Best fit: a stage of 20–48 runs finishes within one session |
| A100 (40 GB) | yes | Fastest; uses compute units faster than it saves time over an L4 |
| T4 (16 GB) | no | Only for short checks such as the `smoke` stage (estimated 3–4× slower than an A100) |
| CPU or TPU | no | Not usable: gsplat needs CUDA |

Runs are small (400×400 images, at most about 900k Gaussians), so memory is never
the limit. A training run took about 1.5–2 minutes; building the reference depth took
about 15 minutes per scene. gsplat compiles its CUDA kernels once per GPU type,
in about 4–5 minutes.

## Option 1: Google Colab

`notebooks/colab.ipynb` runs everything, and each of its cells is described in the
notebook itself. It keeps cheap state on the session disk and everything that costs
GPU time on Google Drive, so a new session continues where the last one stopped.

1. Open the notebook in Colab with a GPU runtime.
2. In cell 3, choose where the code comes from: `CODE_SOURCE = 'github'` clones this
   repository, and `'drive'` copies it from `MyDrive/dl3dcv/final_project`.
3. Place the datasets on Drive as described in [data.md](data.md), then run cells 1–7.
   Cell 5 builds the reference depth once per scene and stores it on Drive.
4. Cell 8 launches one stage. Start with `smoke`, then run the stages in order.
5. Cell 9 collects every run into `MyDrive/dl3dcv/results.csv`; cell 10 draws the
   figures into `MyDrive/dl3dcv/figures`.

Drive layout used by the notebook:

```
MyDrive/datasets/nerf_synthetic/nerf_synthetic/<scene>/   NeRF-Synthetic, extracted
MyDrive/datasets/DTU/SampleSet.zip                          DTU SampleSet
MyDrive/datasets/nerf_synthetic_depth/<scene>_depth.zip     reference depth (written)
MyDrive/dl3dcv/torch_extensions/<GPU>/                      compiled gsplat kernels (written)
MyDrive/dl3dcv/runs/<stage>/<run_id>/                       every finished run (written)
MyDrive/dl3dcv/results.csv, figures/                        collected results (written)
```

## Option 2: command line on a GPU machine

```bash
pip install -r requirements-gpu.txt
bash scripts/setup_gpu.sh                  # checks CUDA, compiles gsplat, runs the tests
export DATA_ROOT=/path/to/data             # dataset root, see data.md
export RUNS_ROOT=/path/to/runs             # where runs are written
```

Prepare the data ([data.md](data.md)), then:

```bash
# one run
python scripts/train.py --config configs/base.yaml --out runs/demo

# a stage of the study
python scripts/run_sweep.py --sweep configs/sweep.yaml --stage stage1 --resume

# collect every run into one CSV
python scripts/evaluate.py --runs "$RUNS_ROOT" --out results/results.csv
```

## Stages

| Stage | Purpose | Runs |
|---|---|---|
| `smoke` | Pipeline check (500 iterations) | 3 |
| `stage0_repro` | Vanilla 3DGS on the 8-view DietNeRF split of all eight scenes, against DNGaussian's reported 3DGS baseline (22.23 dB) | 8 |
| `stage1` | Initialization × view count, no depth loss | 27 |
| `stage2_init_loss` | Each initialization with the depth loss | 18 |
| `stage2_lambda` | Depth-loss weight × loss family, at the selected initialization | 48 |
| `stage3_noise` | Zero-mean noise on the prior | 40 |
| `stage3_affine` | Global scale error on the prior | 32 |
| `stage3_perimage` | Per-image scale error on the prior | 32 |
| `stage4_real` | Depth Anything V2 with oracle and SfM calibration | 20 |
| `stage5_dtu_base` | DTU: SfM and random initialization, no depth | 8 |
| `stage5_dtu_real` | DTU: Depth Anything V2 initialization and loss | 16 |
| `stage5_dtu_oracle` | DTU: structured-light depth initialization, with and without loss | 8 |

Each stage in `sweep.yaml` is configured with the choices of the earlier stages
(marked with `<-` comments). Useful options of `run_sweep.py`:

- `--dry-run --explain-pruning` prints the grid and why each pruned configuration was
  removed, without training.
- `--resume` skips every run that already has a finished `metrics.json`, so an
  interrupted stage continues where it stopped.
- `--shard 0/2` and `--shard 1/2` split one stage between two machines or sessions
  with no overlap.
- `--mirror <dir>` copies each run there as soon as it finishes.

Each run directory holds `config.yaml`, `metrics.json`, `train_log.json`, and rendered
RGB and depth images for three held-out views.

## Figures and report

```bash
python scripts/make_report_figures.py      # initialization, loss sweep, corruption, DTU figures
python scripts/validate_reference_depth.py --exact exact_depth --scene lego chair ship \
    --figure report/figures/reference_error   # needs the Blender renders, see data.md
cd report && latexmk -pdf report.tex
```

`notebooks/analysis.ipynb` draws the degradation curves and the trade-off map and prints
the analysis tables.

# Data

Every script reads datasets from `$DATA_ROOT` (default: `./data`):

```
$DATA_ROOT/
├── nerf_synthetic/<scene>/        NeRF-Synthetic scene: train/, val/, test/, transforms_*.json
│   ├── depth_train/r_<i>.npy      reference depth (scripts/make_gt_depth.py)
│   └── depth_test/r_<i>.npy
├── dtu/                           DTU SampleSet (scripts/prepare_dtu.py)
│   ├── scan<id>/                  image/, cameras.npz, depth/
│   ├── Points/stl/                structured-light point clouds
│   └── ObsMask/                   observability masks and ground planes
└── cache/dav2/                    cached Depth Anything V2 predictions
```

## NeRF-Synthetic

The synthetic scenes of [NeRF](https://github.com/bmild/nerf). The study uses lego,
chair and ship; the reproduction check uses all eight scenes.

```bash
python scripts/fetch_data.py --scenes lego chair ship --archives archives --extract "$DATA_ROOT"
```

`scripts/download_to_drive.py` downloads the same archives into a Google Drive for
Desktop folder instead, for use from Colab.

## Reference depth

NeRF-Synthetic ships no metric depth. `scripts/make_gt_depth.py` builds a reference for
each scene: 3DGS fitted to all 400 views at 800×800, with the alpha masks as extra
supervision, rendered as expected depth at every train and test pose. This takes about
15 minutes per scene on a Colab GPU.

```bash
python scripts/make_gt_depth.py --scene lego chair ship --persist /path/to/store
```

`--persist` zips the depth maps into a directory and restores them from it on later
runs. Each scene also gets a `depth_reference.json` with multi-view consistency and rank
correlation against the release's 8-bit depth images. Those measure self-consistency,
not accuracy.

### Validation against exact Blender depth

The modified Blender scenes the dataset was rendered from are not part of the release.
The NeRF authors shared them in [bmild/nerf#59](https://github.com/bmild/nerf/issues/59)
(that link has since expired), and a community copy with repaired texture paths,
`blend_files_fixed.zip`, is linked from
[bmild/nerf#198](https://github.com/bmild/nerf/issues/198). We used them only to measure
the reference. With Blender installed (tested with 5.2):

```bash
for s in lego chair ship; do
  blender -b blend_files_fixed/$s.blend -P scripts/render_blender_depth.py -- \
      --transforms "$DATA_ROOT/nerf_synthetic/$s/transforms_test.json" \
      --ids $(seq 0 8 192) --out exact_depth/$s/test
done
python scripts/validate_reference_depth.py --exact exact_depth --scene lego chair ship \
    --figure report/figures/reference_error
```

Rendering the input views the same way into `exact_depth/<scene>/train/` (from
`transforms_train.json`) also scores the reference on the views priors are built from.
All renders together take under a minute on a laptop CPU. Before rendering, the script
checks on a synthetic plane that Blender's depth pass is planar and upright.

The renders reproduce the released geometry: silhouettes match the released alpha masks
at IoU ≥ 0.997, and where exact and reference depth disagree by more than 3%, the
released depth images side with the exact depth on 95–99.9% of those pixels. Results
(`results/reference_validation.json`):

| Scene | AbsRel vs exact depth (eval / input views) | Within 1% | Multi-view consistency |
|---|---|---|---|
| Lego | 0.22% / 0.22% | 94.8% | 0.043% |
| Chair | 0.36% / 0.38% | 90.8% | 0.049% |
| Ship | 2.33% / 2.39% | 48.3% | 0.171% |

The corruption and Depth Anything V2 stages use only Lego and Chair, where the reference
error is 10–18 times smaller than the smallest corruption (3.9%). Recomputing every
corrupted prior against exact depth moves no prior error by more than 0.25 percentage
points. On Ship the reconstruction sits in front of the water surface (a median 5.7% on
about a quarter of its pixels); Ship enters only the initialization and view-count
stages. The multi-view consistency column shows why self-consistency is not accuracy: a
surface offset the same way in every view is perfectly consistent.

## DTU

The official SampleSet of the [DTU MVS dataset](https://roboimagedata.compute.dtu.dk/?page_id=36):
scans 1 and 6 with rectified images, calibration, structured-light point clouds and the
files of the official evaluation.

```bash
python scripts/prepare_dtu.py --zip /path/to/SampleSet.zip --out "$DATA_ROOT"
```

The script reads straight from the zip, without extracting its 6.9 GB. For each scan it
writes lighting 3 at 400×300 (the RegNeRF resolution), cameras normalized to a unit
sphere, and depth z-buffered from the structured-light cloud. As a check on scan 1,
depth backprojected from the three input views lies 0.29 mm from the scan, and SfM
points triangulated from those views lie a median 0.34 mm from it.

Scans 1 and 6 are not among the 15 scans sparse-view papers report, and they have no
object masks, so appearance is scored on full images and is not comparable to published
DTU numbers.

## Depth Anything V2

[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (large) is
downloaded from the Hugging Face Hub on first use; `scripts/fetch_model.py --dest <dir>`
pins the weights to a directory instead. Predictions are cached per view under
`$DATA_ROOT/cache/dav2`. How its relative depth is calibrated is described in
[design.md](design.md#depth-anything-v2-calibration).

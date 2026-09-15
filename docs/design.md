# Design notes

How the code is organized, what it measures, and the implementation choices the
results depend on. Each choice below is pinned by a test in `tests/`: getting one
wrong would bias a result for a definitional reason rather than a scientific one.

## Study axes

The study varies three things, plus the depth-loss weight λ:

| Axis | Settings | Code |
|---|---|---|
| A — how depth enters | SfM, random or depth initialization, each with and without a depth loss | `src/gaussians.py`, `src/sfm.py` |
| B — number of views | 3, 5, 9 on NeRF-Synthetic (nested farthest-view splits); 3, 9 on DTU (RegNeRF ordering) | `src/data.py` |
| C — prior quality | clean reference → zero-mean noise → global affine → per-image affine → Depth Anything V2 | `src/corruptions.py`, `src/depth_model.py` |
| λ and loss family | 0, 0.03, 0.1, 0.3, 1 × SSI, Pearson, absolute | `src/depth_loss.py` |

## Module map

| Module | Role |
|---|---|
| `src/data.py` | NeRF-Synthetic and DTU loaders, view splits, depth downscaling that never invents surfaces |
| `src/gaussians.py` | Gaussian parameters, random/SfM/depth initialization, opacity reset |
| `src/render.py` | The only module that calls gsplat |
| `src/sfm.py` | Known-pose triangulation: SIFT, ratio test, epipolar check, DLT |
| `src/depth_init.py` | Backprojection of depth maps into colored points |
| `src/depth_loss.py` | SSI, Pearson and absolute depth losses |
| `src/corruptions.py` | Noise, global affine and per-image affine corruption |
| `src/depth_model.py` | Depth Anything V2 inference, caching and calibration |
| `src/metrics.py` | PSNR, SSIM, LPIPS |
| `src/eval_geometry.py` | Depth error, Chamfer, official DTU Chamfer, floaters |
| `src/sweep.py` | Grid expansion with pruning of degenerate configurations |
| `src/analysis.py` | Result loading, matched baselines, divergence detection, figures |

## Code and third-party boundary

Everything under `src/`, `scripts/` and `tests/` is ours. The rasterizer is the
installed `gsplat` package and is never edited. We chose gsplat over the INRIA
reference implementation because its `render_mode="RGB+ED"` exposes expected depth
as a rendered channel, so the depth loss is ours to define, and its
`DefaultStrategy` implements vanilla 3DGS densification. The INRIA code hard-codes
its depth regularization, which would have meant patching third-party code.

## Metrics

- **Appearance** on held-out views: PSNR, SSIM and LPIPS (VGG). NeRF-Synthetic uses
  25 test views (every 8th); DTU uses the 25-view RegNeRF test split.
- **Geometry, primary:** unaligned depth MAE against the reference on NeRF-Synthetic,
  and the official DTU Chamfer distance in millimetres on DTU.
- **Geometry, secondary:** aligned depth metrics, Chamfer between clouds fused from
  rendered and reference depth, and the fraction of opaque Gaussians farther than 2%
  of the scene diagonal from any reference surface (floaters).
- **Prior quality**, for every run that uses depth: the prior's unaligned and aligned
  AbsRel against the reference on the input views. This is the shared x-axis of the
  corruption curves, which is how Depth Anything V2 is placed on them.
- **Cost:** number of Gaussians, training time, and the iteration at which a probe
  PSNR first reaches the vanilla baseline's final value.

## Choices the results depend on

1. **A scale-and-shift-invariant loss cannot see an affine bias as error.** Fitting
   `(s, t)` to `a·d + b` yields `(a·s, a·t + b)`, so the residual is exactly `a` times
   the clean one: a global affine bias only rescales λ, and a per-image bias becomes
   per-view weights. Pearson correlation ignores it entirely. Only the absolute loss
   registers it, and depth initialization always does. `src/sweep.py` prunes the
   configurations where no path can see the corruption.
2. **Alignment before evaluation erases the injected bias too**, so unaligned depth
   error is the primary geometry number.
3. **Corruption touches valid pixels only, scaled by their median.** NeRF-Synthetic
   views are mostly background (depth 0); a median over all pixels is 0, which would
   silently turn every noise level into a no-op.
4. **The rasterizer and the densification strategy must agree on packing.**
   gsplat's `rasterization()` defaults to `packed=True` and `DefaultStrategy` to
   `packed=False`; left at their defaults, densification accumulates gradients onto
   the wrong Gaussians. `src/render.py:PACKED` sets both.
5. **One pixel convention everywhere:** pixel centres at +0.5, as gsplat rasterizes.
   Backprojection, SfM keypoints and DTU calibration all use it.
6. **The schedule is scaled to the budget:** the means learning rate decays over the
   run, and densification stops at half the iterations, so the last opacity reset is
   never just before evaluation.
7. **Opacity is reset in our code, because gsplat 1.5.3 never does it.** Its reset
   test is `step % reset_every == 0 & step > 0`; `&` binds tighter than `==`, so the
   condition is always false. Without the reset, random-initialization floaters are
   never cleared. `scripts/train.py` disables gsplat's reset and applies vanilla
   3DGS's schedule instead, recording `opacity_resets` in each run's `metrics.json`.

## Depth Anything V2 calibration

The model predicts relative inverse depth, so calibration happens in inverse-depth
space, followed by inversion. The oracle setting (`depth_align: gt`) fits each image
to the inverse reference depth by least squares. The realistic setting
(`depth_align: sfm`) fits it to the inverse depth of the triangulated points that
view helped triangulate, or of all points projecting into the image when fewer than
five are available. With three or more points the fit is least squares plus one
outlier-rejection pass; with two points it is exact; with one point, or a
non-positive scale, only the scale is fitted. See
`src/depth_model.py:align_inverse_to_points`.

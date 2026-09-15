"""The only module that touches gsplat -- i.e. the only part needing CUDA.

Everything else (initialization, losses, corruption, metrics) runs anywhere, so
the untestable surface is kept to this one function. Tests substitute a fake
rasterizer to exercise the plumbing without a GPU.
"""
from __future__ import annotations

import numpy as np
import torch

# render_mode that returns RGB plus expected (opacity-normalized) depth as the
# last channel. Expected depth, not per-Gaussian z: it is what a depth loss
# should supervise, because it accounts for transmittance.
RENDER_MODE = "RGB+ED"

# Must match what the densification strategy is told in step_post_backward.
# gsplat's rasterization() defaults to packed=True while DefaultStrategy
# defaults to packed=False; leaving both at their defaults makes the strategy
# read packed [nnz] tensors as if they were [C, N], and densification silently
# accumulates gradients onto the wrong Gaussians. Pinned here, used by both.
PACKED = False


def _rasterization():
    """Imported lazily so this module can be imported on a machine without CUDA."""
    from gsplat import rasterization
    return rasterization


def viewmats_from_c2w(c2w):
    """Camera-to-world -> world-to-camera, which is what gsplat wants.

    Inverted via the rigid-transform identity rather than a general inverse:
    exact, and it cannot silently degrade if the pose picked up numerical drift.
    """
    c2w = torch.as_tensor(np.asarray(c2w), dtype=torch.float32)
    if c2w.ndim == 2:
        c2w = c2w[None]
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    Rt = R.transpose(1, 2)
    w2c = torch.zeros_like(c2w)
    w2c[:, :3, :3] = Rt
    w2c[:, :3, 3] = -torch.bmm(Rt, t[..., None])[..., 0]
    w2c[:, 3, 3] = 1.0
    return w2c


def render(gaussians, views, device, rasterize=None, background=0.0, sh_degree=None):
    """Render a batch of views.

    Colour goes through gsplat's spherical-harmonics path exactly as in
    vanilla 3DGS; `sh_degree` is the degree active at this step (3DGS raises
    it progressively), defaulting to the cloud's maximum.

    Returns (rgb, depth, alpha, info):
        rgb   (C, H, W, 3)   composited over `background`
        depth (C, H, W)      expected depth
        alpha (C, H, W, 1)
        info  the rasterizer meta dict, which the densification strategy needs
    """
    rasterize = rasterize or _rasterization()
    a = gaussians.activated
    if sh_degree is None:
        sh_degree = gaussians.sh_degree

    viewmats = torch.stack([viewmats_from_c2w(v.c2w)[0] for v in views]).to(device)
    Ks = torch.stack([torch.as_tensor(v.K, dtype=torch.float32) for v in views]).to(device)
    H, W = views[0].hw

    out, alphas, info = rasterize(
        means=a["means"], quats=a["quats"], scales=a["scales"],
        opacities=a["opacities"], colors=a["sh"],
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        render_mode=RENDER_MODE, sh_degree=sh_degree, packed=PACKED,
    )
    rgb, depth = out[..., :3], out[..., 3]
    # The rasterizer returns premultiplied colour, so add the uncovered part of
    # the background back. NeRF-Synthetic is evaluated on a fixed background.
    bg = torch.as_tensor(background, dtype=rgb.dtype, device=rgb.device)
    rgb = rgb + (1.0 - alphas) * bg
    return rgb, depth, alphas, info


def to_image(rgb):
    """(H,W,3) float tensor -> uint8 numpy, for saving renders."""
    x = rgb.detach().clamp(0, 1).cpu().numpy()
    return (x * 255).astype(np.uint8)

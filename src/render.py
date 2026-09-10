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


def render(gaussians, views, device, rasterize=None, background=0.0):
    """Render a batch of views.

    Returns (rgb, depth, alpha, info):
        rgb   (C, H, W, 3)   composited over `background`
        depth (C, H, W)      expected depth
        alpha (C, H, W, 1)
        info  the rasterizer meta dict, which the densification strategy needs
    """
    rasterize = rasterize or _rasterization()
    a = gaussians.activated

    viewmats = torch.stack([viewmats_from_c2w(v.c2w)[0] for v in views]).to(device)
    Ks = torch.stack([torch.as_tensor(v.K, dtype=torch.float32) for v in views]).to(device)
    H, W = views[0].hw

    out, alphas, info = rasterize(
        means=a["means"], quats=a["quats"], scales=a["scales"],
        opacities=a["opacities"], colors=a["colors"],
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        render_mode=RENDER_MODE, sh_degree=None,
    )
    rgb, depth = out[..., :3], out[..., 3]
    # NeRF-Synthetic is evaluated composited on a fixed background; the
    # rasterizer returns premultiplied colour, so add the uncovered part back.
    rgb = rgb + (1.0 - alphas) * background
    return rgb, depth, alphas, info


def to_image(rgb):
    """(H,W,3) float tensor -> uint8 numpy, for saving renders."""
    x = rgb.detach().clamp(0, 1).cpu().numpy()
    return (x * 255).astype(np.uint8)

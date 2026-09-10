"""Appearance metrics on held-out views: PSNR, SSIM, LPIPS.

The geometry half lives in src/eval_geometry.py. Both are computed for every
run: a run with appearance metrics only cannot contribute to the
appearance-vs-geometry map, which is the whole contribution.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None]
    win = (g @ g.T).expand(channels, 1, size, size).contiguous()
    return win


def ssim_tensor(pred, target, window_size=11, max_val=1.0):
    """Differentiable SSIM -- returns a tensor, for use inside the training loss."""
    def chw(x):
        return x.permute(2, 0, 1)[None] if x.ndim == 3 and x.shape[-1] == 3 else x[None]

    p, t = chw(pred), chw(target)
    c = p.shape[1]
    win = _gaussian_window(window_size, channels=c, device=p.device)
    pad = window_size // 2

    mu1 = F.conv2d(p, win, padding=pad, groups=c)
    mu2 = F.conv2d(t, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(p * p, win, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(t * t, win, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(p * t, win, padding=pad, groups=c) - mu1_mu2

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m.mean()


def ssim(pred, target, window_size=11, max_val=1.0):
    """SSIM as a float, for reporting."""
    with torch.no_grad():
        return float(ssim_tensor(pred, target, window_size, max_val).item())


class LPIPS:
    """Lazily-constructed LPIPS, so importing this module needs no weights."""

    def __init__(self, net="vgg", device="cpu"):
        self.net, self.device, self._model = net, device, None

    def __call__(self, pred, target):
        if self._model is None:
            import lpips as _lpips
            self._model = _lpips.LPIPS(net=self.net).to(self.device).eval()

        def to_nchw(x):
            x = x.permute(2, 0, 1)[None] if x.ndim == 3 and x.shape[-1] == 3 else x[None]
            return (x * 2 - 1).to(self.device)          # LPIPS expects [-1, 1]

        with torch.no_grad():
            return float(self._model(to_nchw(pred), to_nchw(target)).item())


def appearance_metrics(preds, targets, lpips_fn=None):
    """Mean PSNR / SSIM / LPIPS over a list of (H,W,3) tensors."""
    out = {"psnr": [], "ssim": [], "lpips": []}
    for p, t in zip(preds, targets):
        out["psnr"].append(psnr(p, t))
        out["ssim"].append(ssim(p, t))
        if lpips_fn is not None:
            out["lpips"].append(lpips_fn(p, t))
    return {k: (sum(v) / len(v) if v else None) for k, v in out.items()}

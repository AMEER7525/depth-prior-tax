"""Depth-based initialization of Gaussians (Axis A, option 'depth').

Backproject a (possibly corrupted) depth map into a world-space point cloud
from known intrinsics/extrinsics, to seed 3DGS instead of / alongside SfM.

Pixel convention: pixel (i, j) has its centre at (j + 0.5, i + 0.5), which is
what gsplat rasterizes and what every K in src/data.py assumes. Backprojecting
integer coordinates instead would shift every ray by half a pixel relative to
the renders the depth is later compared against.
"""
from __future__ import annotations
import numpy as np


def backproject_depth(depth, K, c2w, rgb=None, stride=1):
    """Backproject a depth map to a world-space point cloud.

    Args:
        depth: (H, W) metric depth (z along the camera's forward axis).
        K:     (3, 3) intrinsics.
        c2w:   (4, 4) camera-to-world, OpenCV convention.
        rgb:   optional (H, W, 3) colors.
        stride:pixel subsample for a lighter init.

    Returns:
        points (N, 3), and colors (N, 3) if rgb is given.
    """
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    z = depth[ys, xs].reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (xs + 0.5 - cx) / fx * z
    y_cam = (ys + 0.5 - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    pts_world = (np.asarray(c2w) @ pts_cam.T).T[:, :3]
    if rgb is not None:
        return pts_world, rgb[ys, xs].reshape(-1, 3)
    return pts_world

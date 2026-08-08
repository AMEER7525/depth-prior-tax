"""Dataset loading and sparse-view splits for DTU and NeRF-Synthetic.

Thin skeletons — fill in parsing for your downloaded copies. The DTU sparse
input-view ids follow the common RegNeRF / PixelNeRF protocol so results are
comparable to published numbers. VERIFY the exact ids against the paper you cite.
"""
from __future__ import annotations

# NOTE: verify these ids against the RegNeRF/PixelNeRF DTU protocol you cite.
DTU_SPARSE_INPUT_VIEWS = [25, 22, 28, 40, 44, 48, 0, 8, 13]


def dtu_sparse_split(n_views):
    """Return the input view ids for an n-view DTU experiment."""
    return DTU_SPARSE_INPUT_VIEWS[:n_views]


def load_dtu_scene(root, scan_id):
    """TODO: return images, K, c2w per view, and the GT point cloud for a DTU scan."""
    raise NotImplementedError


def load_blender_scene(root, scene):
    """TODO: return images, poses, and rendered GT depth for a NeRF-Synthetic scene."""
    raise NotImplementedError

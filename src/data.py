"""Dataset loading and sparse-view splits for DTU and NeRF-Synthetic.

Host-agnostic: the dataset root is resolved from $DATA_ROOT (set differently on
the 4080 box and on Colab), falling back to ./data. Nothing else in the codebase
should hard-code a dataset path.

Both loaders return a `Scene`: a list of `View`s plus whatever ground-truth
geometry that dataset provides. Everything downstream (init, depth loss,
evaluation) consumes `Scene` and never touches the on-disk layout.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Sparse-view protocols
# --------------------------------------------------------------------------
# DTU input-view ids from the RegNeRF / PixelNeRF sparse-view protocol, in the
# order that makes prefixes valid: [:3] and [:6] and [:9] are the published
# 3/6/9-view splits. DNGaussian and FSGS both use these ids, so a 3-view number
# produced with this split is directly comparable to their tables.
DTU_SPARSE_INPUT_VIEWS = [25, 22, 28, 40, 44, 48, 0, 8, 13]

# The published protocol defines 3, 6 and 9 views. Our proposal sweeps 3/5/9.
# 5 is OURS -- a prefix of the same ordering, but no paper reports it, so a
# 5-view number can be compared across our own arms and to nothing else.
DTU_PUBLISHED_VIEW_COUNTS = (3, 6, 9)

# The 15 DTU scans used by the sparse-view line of work (RegNeRF Tab. 1).
DTU_SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]

BLENDER_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials",
                  "mic", "ship"]


def resolve_data_root(root=None):
    """Dataset root: explicit arg > $DATA_ROOT > ./data."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("DATA_ROOT", "data"))


def dtu_sparse_split(n_views):
    """Input view ids for an n-view DTU experiment.

    Raises for n_views we cannot serve rather than silently truncating, so a
    typo in a sweep config fails at expansion time and not 40 minutes into a run.
    """
    if not 1 <= n_views <= len(DTU_SPARSE_INPUT_VIEWS):
        raise ValueError(
            f"n_views={n_views} outside 1..{len(DTU_SPARSE_INPUT_VIEWS)} "
            f"supported by the RegNeRF protocol ordering")
    return DTU_SPARSE_INPUT_VIEWS[:n_views]


def is_published_split(n_views):
    """True if this view count matches a split other papers report."""
    return n_views in DTU_PUBLISHED_VIEW_COUNTS


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class View:
    """One posed image, with optional ground-truth depth."""
    image: np.ndarray            # (H, W, 3) float32 in [0, 1]
    K: np.ndarray                # (3, 3) intrinsics
    c2w: np.ndarray              # (4, 4) camera-to-world, OpenCV convention
    depth: np.ndarray | None = None   # (H, W) metric depth, 0 = invalid
    mask: np.ndarray | None = None    # (H, W) bool, True = foreground
    name: str = ""

    @property
    def hw(self):
        return self.image.shape[:2]


@dataclass
class Scene:
    """A scene split: the views plus whatever GT geometry the dataset ships."""
    name: str
    views: list[View]
    gt_points: np.ndarray | None = None     # (M, 3) DTU structured-light cloud
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.views)

    def subset(self, indices):
        return Scene(self.name, [self.views[i] for i in indices],
                     self.gt_points, dict(self.meta))

    @property
    def has_depth(self):
        return all(v.depth is not None for v in self.views)


# --------------------------------------------------------------------------
# NeRF-Synthetic (Blender)
# --------------------------------------------------------------------------
# Camera convention: transforms_*.json stores OpenGL c2w (x right, y up, z back).
# We flip y and z once, here, so every View in the codebase is OpenCV (x right,
# y down, z forward) and depth_init.backproject_depth needs no per-dataset case.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_blender_scene(root=None, scene="lego", split="train", load_depth=True):
    """Load a NeRF-Synthetic split.

    GT depth is NOT part of the standard nerf_synthetic release. Render it with
    scripts/render_blender_depth.py (Blender depth pass) into
    <scene>/depth_<split>/r_<i>.npy in metric world units. Without it the
    controlled-degradation experiments (Axis C) cannot run on this dataset.
    """
    root = resolve_data_root(root)
    scene_dir = root / "nerf_synthetic" / scene
    tf_path = scene_dir / f"transforms_{split}.json"
    if not tf_path.exists():
        raise FileNotFoundError(
            f"{tf_path} not found. Expected layout: $DATA_ROOT/nerf_synthetic/<scene>/")

    import imageio.v2 as imageio

    with open(tf_path) as f:
        meta = json.load(f)

    frames = meta["frames"]
    depth_dir = scene_dir / f"depth_{split}"
    views = []
    for i, frame in enumerate(frames):
        img_path = scene_dir / (frame["file_path"].lstrip("./") + ".png")
        rgba = np.asarray(imageio.imread(img_path), dtype=np.float32) / 255.0
        # Blender renders RGBA on transparent background; composite on black to
        # match the standard NeRF/3DGS evaluation protocol, and keep alpha as
        # the foreground mask so geometry metrics can ignore empty space.
        if rgba.shape[-1] == 4:
            alpha = rgba[..., 3]
            image = rgba[..., :3] * alpha[..., None]
            mask = alpha > 0.5
        else:
            image, mask = rgba[..., :3], None

        H, W = image.shape[:2]
        focal = 0.5 * W / np.tan(0.5 * float(meta["camera_angle_x"]))
        K = np.array([[focal, 0, W / 2.0],
                      [0, focal, H / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64) @ _GL_TO_CV

        depth = None
        if load_depth:
            d_path = depth_dir / f"r_{i}.npy"
            if d_path.exists():
                depth = np.load(d_path).astype(np.float32)

        views.append(View(image=image, K=K, c2w=c2w, depth=depth, mask=mask,
                          name=Path(frame["file_path"]).name))

    return Scene(name=scene, views=views,
                 meta={"dataset": "blender", "split": split,
                       "camera_angle_x": meta["camera_angle_x"]})


# --------------------------------------------------------------------------
# DTU
# --------------------------------------------------------------------------
def _K_Rt_from_P(P):
    """Decompose a 3x4 projection matrix into K and a camera-to-world pose.

    This is the IDR/NeuS convention that the sparse-view DTU releases use.
    """
    import cv2
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]
    K = K / K[2, 2]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.transpose()
    c2w[:3, 3] = (t[:3] / t[3])[:, 0]
    return K.astype(np.float64), c2w


def load_dtu_scene(root=None, scan_id=8, load_masks=True, load_gt_points=False):
    """Load a DTU scan in the IDR-style layout used by the sparse-view papers.

    Expected layout under $DATA_ROOT/dtu/scan<id>/:
        image/000000.png ...      rectified images
        mask/000000.png ...       object masks (used by the DTU Chamfer eval)
        cameras.npz               world_mat_<i>, scale_mat_<i>

    `scale_mat` normalizes the scene into a unit sphere. We fold it into the
    projection so poses come back in NORMALIZED coordinates, and stash the
    matrix in meta -- the official DTU Chamfer eval expects points back in the
    original scan frame, so eval must undo it.
    """
    root = resolve_data_root(root)
    scan_dir = root / "dtu" / f"scan{scan_id}"
    cam_path = scan_dir / "cameras.npz"
    if not cam_path.exists():
        raise FileNotFoundError(
            f"{cam_path} not found. Expected layout: $DATA_ROOT/dtu/scan<id>/")

    import imageio.v2 as imageio

    cams = np.load(cam_path)
    img_paths = sorted((scan_dir / "image").glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"no images under {scan_dir / 'image'}")

    views = []
    scale_mat_0 = None
    for i, img_path in enumerate(img_paths):
        world_mat = cams[f"world_mat_{i}"]
        scale_mat = cams.get(f"scale_mat_{i}", np.eye(4))
        if scale_mat_0 is None:
            scale_mat_0 = scale_mat
        K, c2w = _K_Rt_from_P((world_mat @ scale_mat)[:3, :4])

        image = np.asarray(imageio.imread(img_path), dtype=np.float32) / 255.0
        mask = None
        if load_masks:
            m_path = scan_dir / "mask" / img_path.name
            if m_path.exists():
                m = np.asarray(imageio.imread(m_path))
                mask = (m[..., 0] if m.ndim == 3 else m) > 127

        views.append(View(image=image[..., :3], K=K, c2w=c2w, mask=mask,
                          name=img_path.stem))

    gt_points = None
    if load_gt_points:
        ply = root / "dtu" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
        if ply.exists():
            gt_points = _load_ply_points(ply)

    return Scene(name=f"scan{scan_id}", views=views, gt_points=gt_points,
                 meta={"dataset": "dtu", "scan_id": scan_id,
                       "scale_mat": scale_mat_0})


def _load_ply_points(path):
    """Read XYZ from a PLY point cloud (ascii or binary_little_endian)."""
    with open(path, "rb") as f:
        header, n_verts, props, fmt = [], 0, [], None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property") and len(header) and n_verts:
                props.append(line.split())
            elif line == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_verts)
            return np.ascontiguousarray(data[:, :3])
        dtype = np.dtype([(p[2], _PLY_TYPES[p[1]]) for p in props])
        arr = np.frombuffer(f.read(dtype.itemsize * n_verts), dtype=dtype)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
              "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}

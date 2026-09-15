"""Dataset loading and sparse-view splits for DTU and NeRF-Synthetic.

Host-agnostic: the dataset root is resolved from $DATA_ROOT, falling back to
./data. On Colab this points at local disk (/content/data), not the Drive mount
-- see the README on why extracted data must not live on FUSE. Nothing else in
the codebase should hard-code a dataset path.

Both loaders return a `Scene`: a list of `View`s plus whatever ground-truth
geometry that dataset provides. Everything downstream (init, depth loss,
evaluation) consumes `Scene` and never touches the on-disk layout.

Pixel convention everywhere: pixel (i, j) has its centre at (j + 0.5, i + 0.5),
matching gsplat. Under it, downscaling by an integer factor f is exactly
K[:2] /= f, with no half-pixel correction.
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
# Views RegNeRF drops from the test set (DNGaussian's loader uses the same list).
DTU_EXCLUDED_VIEWS = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
DTU_N_VIEWS = 49
DTU_TEST_VIEWS = [i for i in range(DTU_N_VIEWS)
                  if i not in DTU_SPARSE_INPUT_VIEWS + DTU_EXCLUDED_VIEWS]

# The published protocol defines 3, 6 and 9 views. This study sweeps 3/5/9.
# 5 is OURS -- a prefix of the same ordering, but no paper reports it, so a
# 5-view number can be compared across our own arms and to nothing else.
DTU_PUBLISHED_VIEW_COUNTS = (3, 6, 9)

# The 15 DTU scans used by the sparse-view line of work (RegNeRF Tab. 1).
DTU_SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
# The two scans in the official DTU SampleSet (what scripts/prepare_dtu.py
# builds). Not in the benchmark above, so they are a realism check with the
# official geometry metric, not a reproduction of published numbers.
DTU_SAMPLESET_SCANS = [1, 6]

BLENDER_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials",
                  "mic", "ship"]
# The 8-view NeRF-Synthetic split of DietNeRF, as used by DNGaussian's loader.
# The one Blender sparse-view protocol with published 3DGS baselines.
BLENDER_DIETNERF_8 = [2, 16, 26, 55, 73, 76, 86, 93]


def farthest_view_sampling(c2ws, n, start=0):
    """Greedy farthest-point sampling on camera viewing directions.

    NeRF-Synthetic's 100 training poses are random on the upper hemisphere,
    so evenly spaced *indices* are just a random subset. This picks views that
    cover the object as evenly as possible, deterministically, and every
    prefix is itself a valid split: the 3 views are inside the 5, which are
    inside the 9, so a view-count sweep changes one thing at a time.
    """
    centres = np.stack([np.asarray(c)[:3, 3] for c in c2ws])
    dirs = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
    chosen = [int(start)]
    dist = np.full(len(dirs), np.inf)
    while len(chosen) < n:
        dist = np.minimum(dist, np.linalg.norm(dirs - dirs[chosen[-1]], axis=1))
        dist[chosen] = -1.0
        chosen.append(int(np.argmax(dist)))
    return chosen


def blender_sparse_split(n_views, c2ws=None, protocol="fps", n_total=None):
    """Input view ids for an n-view NeRF-Synthetic experiment.

    protocol:
      "fps"      farthest-view sampling (OURS; nested prefixes). Needs c2ws.
      "dietnerf" the published 8-view split; n_views must be 8.
      "even"     evenly spaced indices (the old behaviour).
    n_views <= 0 means every training view.
    """
    total = n_total if n_total is not None else (len(c2ws) if c2ws is not None else 100)
    if n_views <= 0:
        return list(range(total))
    if not 1 <= n_views <= total:
        raise ValueError(f"n_views={n_views} outside 1..{total}")
    if protocol == "dietnerf":
        if n_views != len(BLENDER_DIETNERF_8):
            raise ValueError(f"the DietNeRF split has 8 views, not {n_views}")
        return list(BLENDER_DIETNERF_8)
    if protocol == "fps":
        if c2ws is None:
            raise ValueError("protocol 'fps' needs the training poses")
        return farthest_view_sampling(c2ws, n_views)
    if protocol == "even":
        return [int(round(i)) for i in np.linspace(0, total, n_views, endpoint=False)]
    raise ValueError(f"unknown split protocol {protocol!r}")


def evenly_spaced(n_pick, n_total):
    """Deterministic subsample of held-out views, to keep evaluation affordable."""
    n_pick = min(n_pick, n_total)
    return [int(round(i)) for i in np.linspace(0, n_total, n_pick, endpoint=False)]


def resolve_data_root(root=None):
    """Dataset root: explicit arg > $DATA_ROOT > ./data."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("DATA_ROOT", "data"))


def dtu_sparse_split(n_views):
    """Input view ids for an n-view DTU experiment (n <= 0: every non-test view).

    Raises for n_views we cannot serve rather than silently truncating, so a
    typo in a sweep config fails at expansion time and not 40 minutes into a run.
    """
    if n_views <= 0:
        return [i for i in range(DTU_N_VIEWS) if i not in DTU_TEST_VIEWS]
    if not 1 <= n_views <= len(DTU_SPARSE_INPUT_VIEWS):
        raise ValueError(
            f"n_views={n_views} outside 1..{len(DTU_SPARSE_INPUT_VIEWS)} "
            f"supported by the RegNeRF protocol ordering")
    return DTU_SPARSE_INPUT_VIEWS[:n_views]


def is_published_split(n_views):
    """True if this view count matches a split other papers report."""
    return n_views in DTU_PUBLISHED_VIEW_COUNTS


def parse_scan_id(scene):
    """'scan6', 'scan006', 6 or '6' -> 6."""
    s = str(scene)
    return int(s[4:] if s.startswith("scan") else s)


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
# Resampling
# --------------------------------------------------------------------------
def _block_mean(x, f):
    """Average f x f pixel blocks: exact area downsampling for integer f."""
    if f == 1:
        return x
    H, W = x.shape[:2]
    H2, W2 = H // f, W // f
    x = x[:H2 * f, :W2 * f]
    return x.reshape(H2, f, W2, f, *x.shape[2:]).mean(axis=(1, 3))


def downscale_depth(depth, f, rel_tol=0.01):
    """Area-downsample depth WITHOUT inventing surfaces.

    A block is kept only if all its pixels are valid and agree to within
    rel_tol; silhouette and occlusion-edge blocks become invalid (0) instead of
    averaging a foreground and a background depth into a point on neither.
    """
    if f == 1:
        return np.asarray(depth, dtype=np.float32)
    d = np.asarray(depth, dtype=np.float64)
    H, W = d.shape
    H2, W2 = H // f, W // f
    b = d[:H2 * f, :W2 * f].reshape(H2, f, W2, f)
    valid = (np.isfinite(b) & (b > 0)).all(axis=(1, 3))
    bz = np.where(np.isfinite(b), b, 0.0)
    mean = bz.mean(axis=(1, 3))
    spread = bz.max(axis=(1, 3)) - bz.min(axis=(1, 3))
    ok = valid & (spread <= rel_tol * np.maximum(mean, 1e-12))
    return np.where(ok, mean, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# NeRF-Synthetic (Blender)
# --------------------------------------------------------------------------
# Camera convention: transforms_*.json stores OpenGL c2w (x right, y up, z back).
# We flip y and z once, here, so every View in the codebase is OpenCV (x right,
# y down, z forward) and depth_init.backproject_depth needs no per-dataset case.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _blender_meta(root, scene, split):
    scene_dir = resolve_data_root(root) / "nerf_synthetic" / scene
    tf_path = scene_dir / f"transforms_{split}.json"
    if not tf_path.exists():
        raise FileNotFoundError(
            f"{tf_path} not found. Expected layout: $DATA_ROOT/nerf_synthetic/<scene>/")
    with open(tf_path) as f:
        return scene_dir, json.load(f)


def blender_poses(root=None, scene="lego", split="train"):
    """OpenCV c2w for every frame of a split, without decoding any image."""
    _, meta = _blender_meta(root, scene, split)
    return [np.asarray(fr["transform_matrix"], dtype=np.float64) @ _GL_TO_CV
            for fr in meta["frames"]]


def load_blender_scene(root=None, scene="lego", split="train", load_depth=True,
                       downscale=1, background=0.0, indices=None):
    """Load a NeRF-Synthetic split (optionally only the frames in `indices`).

    Images are composited over `background` (1.0 = white, the convention of
    every published NeRF-Synthetic number) after area-downsampling the
    premultiplied RGBA, and alpha becomes the foreground mask.

    GT depth is NOT part of the nerf_synthetic release. scripts/make_gt_depth.py
    writes it to <scene>/depth_<split>/r_<i>.npy (full resolution, depth along
    the camera's forward axis, 0 = background).
    """
    import imageio.v2 as imageio

    scene_dir, meta = _blender_meta(root, scene, split)
    frames = meta["frames"]
    indices = list(range(len(frames))) if indices is None else list(indices)
    depth_dir = scene_dir / f"depth_{split}"
    bg = np.asarray(background, dtype=np.float32)

    views = []
    for i in indices:
        frame = frames[i]
        stem = Path(frame["file_path"]).name
        img_path = scene_dir / (frame["file_path"].lstrip("./") + ".png")
        rgba = np.asarray(imageio.imread(img_path), dtype=np.float32) / 255.0
        if rgba.shape[-1] == 4:
            alpha = rgba[..., 3:4]
            premult = _block_mean(rgba[..., :3] * alpha, downscale)
            alpha = _block_mean(alpha, downscale)
            image = premult + (1.0 - alpha) * bg
            mask = alpha[..., 0] > 0.5
        else:
            image, mask = _block_mean(rgba[..., :3], downscale), None

        H, W = image.shape[:2]
        focal = 0.5 * W / np.tan(0.5 * float(meta["camera_angle_x"]))
        K = np.array([[focal, 0, W / 2.0],
                      [0, focal, H / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64) @ _GL_TO_CV

        depth = None
        if load_depth:
            d_path = depth_dir / f"{stem}.npy"
            if d_path.exists():
                depth = downscale_depth(np.load(d_path), downscale)

        views.append(View(image=image.astype(np.float32), K=K, c2w=c2w,
                          depth=depth, mask=mask, name=stem))

    return Scene(name=scene, views=views,
                 meta={"dataset": "blender", "split": split, "indices": indices,
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


def load_dtu_scene(root=None, scan_id=8, load_masks=True, load_gt_points=False,
                   load_depth=True, downscale=1):
    """Load a DTU scan in the IDR-style layout used by the sparse-view papers.

    Expected layout under $DATA_ROOT/dtu/scan<id>/:
        image/000000.png ...      rectified images
        mask/000000.png ...       object masks (optional)
        depth/000000.npy ...      GT depth, normalized units (optional; written
                                  by scripts/prepare_dtu.py from the STL cloud)
        cameras.npz               world_mat_<i>, scale_mat_<i>

    `scale_mat` normalizes the scene into a unit sphere. We fold it into the
    projection so poses come back in NORMALIZED coordinates, and stash the
    matrix in meta -- the official DTU Chamfer eval expects points back in the
    original scan frame, so eval must undo it.
    """
    import imageio.v2 as imageio

    scan_id = parse_scan_id(scan_id)
    root = resolve_data_root(root)
    scan_dir = root / "dtu" / f"scan{scan_id}"
    cam_path = scan_dir / "cameras.npz"
    if not cam_path.exists():
        raise FileNotFoundError(
            f"{cam_path} not found. Expected layout: $DATA_ROOT/dtu/scan<id>/ "
            "(build it with scripts/prepare_dtu.py)")

    cams = np.load(cam_path)
    img_paths = sorted((scan_dir / "image").glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"no images under {scan_dir / 'image'}")

    views = []
    scale_mat_0 = None
    for i, img_path in enumerate(img_paths):
        world_mat = cams[f"world_mat_{i}"]
        key = f"scale_mat_{i}"
        scale_mat = cams[key] if key in cams.files else np.eye(4)
        if scale_mat_0 is None:
            scale_mat_0 = scale_mat
        K, c2w = _K_Rt_from_P((world_mat @ scale_mat)[:3, :4])

        image = np.asarray(imageio.imread(img_path), dtype=np.float32) / 255.0
        image = _block_mean(image[..., :3], downscale)
        K = K.copy()
        K[:2] /= downscale

        mask = None
        if load_masks:
            m_path = scan_dir / "mask" / img_path.name
            if m_path.exists():
                m = np.asarray(imageio.imread(m_path), dtype=np.float32)
                m = (m[..., 0] if m.ndim == 3 else m) / 255.0
                mask = _block_mean(m, downscale) > 0.5

        depth = None
        if load_depth:
            d_path = scan_dir / "depth" / f"{img_path.stem}.npy"
            if d_path.exists():
                depth = downscale_depth(np.load(d_path), downscale, rel_tol=0.02)

        views.append(View(image=image.astype(np.float32), K=K, c2w=c2w,
                          depth=depth, mask=mask, name=img_path.stem))

    gt_points = None
    if load_gt_points:
        ply = root / "dtu" / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
        if ply.exists():
            gt_points = _load_ply_points(ply)

    return Scene(name=f"scan{scan_id}", views=views, gt_points=gt_points,
                 meta={"dataset": "dtu", "scan_id": scan_id,
                       "scale_mat": scale_mat_0})


def load_dtu_eval_assets(root=None, scan_id=1):
    """What the official DTU geometry eval needs, or None if any piece is missing.

    Returns {"stl": (M,3) mm, "obs_mask": bool volume, "bb": (2,3), "res": float,
    "plane": (4,)}, read from $DATA_ROOT/dtu/{Points/stl, ObsMask}/.
    """
    import scipy.io as sio

    scan_id = parse_scan_id(scan_id)
    d = resolve_data_root(root) / "dtu"
    stl = d / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    obs = d / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    plane = d / "ObsMask" / f"Plane{scan_id}.mat"
    if not (stl.exists() and obs.exists() and plane.exists()):
        return None
    m = sio.loadmat(obs)
    return {"stl": _load_ply_points(stl),
            "obs_mask": m["ObsMask"].astype(bool),
            "bb": m["BB"].astype(np.float64),
            "res": float(np.asarray(m["Res"]).reshape(-1)[0]),
            "plane": sio.loadmat(plane)["P"].astype(np.float64).reshape(4)}


def _load_ply_points(path):
    """Read XYZ from a PLY point cloud (ascii or binary_little_endian).

    Only the vertex element's properties are read, so files that also carry
    faces (or put other elements after the vertices) parse correctly.
    """
    with open(path, "rb") as f:
        fmt, n_verts, props, current = None, 0, [], None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: PLY header has no end_header")
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                parts = line.split()
                current = parts[1]
                if current == "vertex":
                    n_verts = int(parts[2])
                elif n_verts == 0:
                    raise ValueError(f"{path}: vertex element must come first")
            elif line.startswith("property") and current == "vertex":
                props.append(line.split())
            elif line == "end_header":
                break
        names = [p[-1] for p in props]
        xyz = [names.index(c) for c in ("x", "y", "z")]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_verts, ndmin=2)
            return np.ascontiguousarray(data[:, xyz], dtype=np.float64)
        dtype = np.dtype([(p[-1], _PLY_TYPES[p[1]]) for p in props])
        arr = np.frombuffer(f.read(dtype.itemsize * n_verts), dtype=dtype)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
              "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}

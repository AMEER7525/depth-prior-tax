"""Known-pose triangulation: the 'sfm' initialization (Axis A, option 1).

Vanilla 3DGS initializes from a COLMAP SfM cloud. In the sparse-view setting
the poses are given (NeRF-Synthetic transforms, DTU calibration), so the part
of SfM that matters for initialization is what COLMAP's point_triangulator
does with known poses: detect features, match them, keep matches consistent
with the known epipolar geometry, triangulate, and filter. This module does
exactly that -- OpenCV SIFT for features, numpy for the geometry -- so it needs
no COLMAP install and behaves identically on the Mac and on Colab.

It is given nothing a real SfM run would not have: no depth, and no mask
beyond the foreground alpha that NeRF-Synthetic ships inside its RGBA images.
With 3 views it returns few points. That is the finding, not a bug -- few
reliable correspondences are exactly why sparse-view initialization is hard,
and the point count is logged per run.

Pixel convention: gsplat puts pixel (i, j)'s centre at (j + 0.5, i + 0.5), and
so does every K in src/data.py. OpenCV keypoints put it at (j, i). Keypoints
are shifted by +0.5 once, in detect(), and nowhere else.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Two-view geometry from known poses (pure numpy, testable anywhere)
# --------------------------------------------------------------------------
def projection_matrix(K, c2w):
    """3x4 P = K [R | t] with [R | t] the world-to-camera transform."""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    return np.asarray(K, dtype=np.float64) @ w2c[:3, :4]


def _skew(t):
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def fundamental_from_poses(K1, c2w1, K2, c2w2):
    """F with x2^T F x1 = 0 for pixel x1 in camera 1 and x2 in camera 2."""
    w2c1 = np.linalg.inv(np.asarray(c2w1, dtype=np.float64))
    w2c2 = np.linalg.inv(np.asarray(c2w2, dtype=np.float64))
    R = w2c2[:3, :3] @ w2c1[:3, :3].T
    t = w2c2[:3, 3] - R @ w2c1[:3, 3]
    F = np.linalg.inv(K2).T @ _skew(t) @ R @ np.linalg.inv(K1)
    n = np.linalg.norm(F)
    return F / n if n > 0 else F


def sampson_distance(F, x1, x2, eps=1e-12):
    """First-order geometric epipolar error, in squared pixels. x1, x2: (N, 2)."""
    h1 = np.concatenate([x1, np.ones((len(x1), 1))], axis=1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1))], axis=1)
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.sum(h2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, eps)


def triangulate_dlt(P1, P2, x1, x2):
    """Linear (DLT) triangulation of N correspondences. Returns (N, 3)."""
    A = np.stack([
        x1[:, 0:1] * P1[2] - P1[0],
        x1[:, 1:2] * P1[2] - P1[1],
        x2[:, 0:1] * P2[2] - P2[0],
        x2[:, 1:2] * P2[2] - P2[1],
    ], axis=1)                                  # (N, 4, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1]
    w = X[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return X[:, :3] / w


def project(P, X):
    """Project (N, 3) world points; returns pixels (N, 2) and depth-like w (N,)."""
    h = np.concatenate([X, np.ones((len(X), 1))], axis=1) @ P.T
    w = h[:, 2]
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return h[:, :2] / safe[:, None], w


def triangulation_angle_deg(c1, c2, X):
    """Angle at X between the rays to the two camera centres."""
    r1 = X - c1
    r2 = X - c2
    cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1)
                                     * np.linalg.norm(r2, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def triangulate_pair(K1, c2w1, K2, c2w2, x1, x2, max_epipolar_px=2.0,
                     max_reproj_px=2.0, min_angle_deg=1.0):
    """Epipolar-filter, triangulate and quality-filter one image pair.

    Returns (points (M, 3), keep_index (M,)) where keep_index indexes x1/x2.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    if len(x1) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)

    F = fundamental_from_poses(K1, c2w1, K2, c2w2)
    idx = np.nonzero(sampson_distance(F, x1, x2) < max_epipolar_px ** 2)[0]
    if len(idx) == 0:
        return np.zeros((0, 3)), idx

    P1, P2 = projection_matrix(K1, c2w1), projection_matrix(K2, c2w2)
    X = triangulate_dlt(P1, P2, x1[idx], x2[idx])

    u1, w1 = project(P1, X)
    u2, w2 = project(P2, X)
    err = np.maximum(np.linalg.norm(u1 - x1[idx], axis=1),
                     np.linalg.norm(u2 - x2[idx], axis=1))
    ang = triangulation_angle_deg(np.asarray(c2w1)[:3, 3], np.asarray(c2w2)[:3, 3], X)
    ok = (w1 > 0) & (w2 > 0) & (err < max_reproj_px) & (ang > min_angle_deg)
    return X[ok], idx[ok]


# --------------------------------------------------------------------------
# Features (OpenCV)
# --------------------------------------------------------------------------
def detect(view, max_features=8000):
    """SIFT keypoints (in gsplat's +0.5 pixel convention) and descriptors."""
    import cv2

    img = (np.clip(view.image, 0.0, 1.0) * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = None
    if view.mask is not None:
        mask = (np.asarray(view.mask, dtype=bool).astype(np.uint8) * 255)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, desc = sift.detectAndCompute(gray, mask)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float64) + 0.5
    return pts, desc.astype(np.float32)


def match(desc1, desc2, ratio=0.9):
    """Lowe ratio test plus mutual-nearest check. Returns (i1, i2) index arrays."""
    import cv2

    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(desc1, desc2, k=2)
    bwd = {m[0].queryIdx: m[0].trainIdx
           for m in bf.knnMatch(desc2, desc1, k=2) if len(m) >= 1}
    i1, i2 = [], []
    for m in fwd:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance and bwd.get(a.trainIdx) == a.queryIdx:
            i1.append(a.queryIdx)
            i2.append(a.trainIdx)
    return np.array(i1, dtype=int), np.array(i2, dtype=int)


def _sample_colors(image, px):
    """Nearest-pixel colours at +0.5-convention pixel coordinates."""
    H, W = image.shape[:2]
    j = np.clip(np.floor(px[:, 0]).astype(int), 0, W - 1)
    i = np.clip(np.floor(px[:, 1]).astype(int), 0, H - 1)
    return image[i, j]


def triangulate_views(views, max_features=20000, ratio=0.9, max_epipolar_px=2.0,
                      max_reproj_px=2.0, min_angle_deg=1.0):
    """Sparse point cloud from posed views, COLMAP point_triangulator style.

    Returns (points (M, 3), colors (M, 3), stats). Every pair of views is
    matched; a point seen in several pairs appears several times, which is
    harmless for initialization (densification prunes duplicates anyway).

    The ratio test (0.9) is looser than COLMAP's 0.8 because every surviving
    match must ALSO lie within max_epipolar_px of its epipolar line under the
    known poses -- a stronger geometric check than the RANSAC COLMAP can run
    without poses. stats["_obs"] records the pair each point came from, so a
    caller can use only the points a given view actually observed.
    """
    feats = [detect(v, max_features) for v in views]
    all_pts, all_cols, all_obs = [], [], []
    stats = {"n_keypoints": [len(f[0]) for f in feats], "pairs": []}
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            (pa, da), (pb, db) = feats[a], feats[b]
            ia, ib = match(da, db, ratio)
            X, keep = triangulate_pair(
                views[a].K, views[a].c2w, views[b].K, views[b].c2w,
                pa[ia], pb[ib], max_epipolar_px, max_reproj_px, min_angle_deg)
            stats["pairs"].append({"pair": [a, b], "matches": int(len(ia)),
                                   "points": int(len(X))})
            if len(X):
                all_pts.append(X)
                all_cols.append(_sample_colors(views[a].image, pa[ia][keep]))
                all_obs.append(np.tile([a, b], (len(X), 1)))
    if not all_pts:
        stats["n_points"] = 0
        stats["_obs"] = np.zeros((0, 2), dtype=int)
        return np.zeros((0, 3)), np.zeros((0, 3)), stats
    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    stats["n_points"] = int(len(pts))
    stats["_obs"] = np.concatenate(all_obs)
    return pts, cols, stats


def sparse_depths(points, view):
    """Project a sparse cloud into a view: pixel coords (N, 2) and depth (N,).

    Used to align a relative monocular depth map to metric scale the way a
    practitioner would -- against SfM points -- rather than against GT.
    """
    P = projection_matrix(view.K, view.c2w)
    px, w = project(P, np.asarray(points, dtype=np.float64))
    H, W = view.hw
    ok = (w > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return px[ok], w[ok]

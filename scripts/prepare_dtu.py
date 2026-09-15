"""Build $DATA_ROOT/dtu from the official DTU SampleSet.zip (scans 1 and 6).

    python scripts/prepare_dtu.py \
        --zip "/content/drive/MyDrive/datasets/DTU/SampleSet.zip" --out "$DATA_ROOT"

Members are read straight out of the zip, so the 6.9 GB archive is never
extracted -- about 250 MB per scan is actually read. Per scan it writes the
IDR-style layout src.data.load_dtu_scene reads:

    dtu/scan<id>/image/000000.png ...  rectified, lighting 3 (the most diffuse,
                                       what RegNeRF / pixelNeRF / IDR use),
                                       area-downsampled 4x to 400x300 -- the
                                       RegNeRF sparse-view resolution
    dtu/scan<id>/cameras.npz           world_mat_i: the calibration's 3x4 P in
                                       the 400x300 pixel frame (+0.5 pixel-centre
                                       convention); scale_mat_i: normalization
                                       from the ObsMask bounding box to a unit sphere
    dtu/scan<id>/depth/000000.npy      GT depth in NORMALIZED units, z-buffered
                                       from the structured-light STL cloud
    dtu/Points/stl/stl<id:03d>_total.ply, dtu/ObsMask/{ObsMask<id>_10,Plane<id>}.mat
                                       what the official Chamfer evaluation needs

WHAT THESE TWO SCANS ARE FOR. The SampleSet holds scans 1 and 6, which are not
among the 15 scans sparse-view papers report, so they cannot reproduce a
published number. They serve as the real-world confirmation: real images,
real calibration, and the official structured-light geometry metric.
No object masks exist for them, so appearance is scored on full images.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import DTU_N_VIEWS, _load_ply_points  # noqa: E402

LIGHT = 3          # most diffuse lighting
FACTOR = 4         # 1600x1200 -> 400x300


def find_prefix(z):
    """Path inside the zip up to and including 'MVS Data/'."""
    tail = "Calibration/cal18/pos_001.txt"
    for n in z.namelist():
        if n.endswith(tail):
            return n[: -len(tail)]
    raise SystemExit("this zip has no Calibration/cal18/pos_001.txt -- is it SampleSet.zip?")


def downscale_P(P, f):
    """Projection in DTU's integer-centre pixels -> +0.5-centre pixels, downscaled by f."""
    to_centre = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])
    return np.diag([1.0 / f, 1.0 / f, 1.0]) @ to_centre @ P


def normalization(bb):
    """scale_mat mapping the unit sphere onto the ObsMask bounding box's sphere."""
    bb = np.asarray(bb, dtype=np.float64)
    centre = bb.mean(axis=0)
    radius = 0.5 * float(np.linalg.norm(bb[1] - bb[0]))
    S = np.eye(4)
    S[:3, :3] *= radius
    S[:3, 3] = centre
    return S, radius


def zbuffer_depth(P, pts, H, W, rel_tol=0.02, win=5):
    """Nearest-point depth per pixel from a dense cloud, with leak rejection.

    A pixel whose depth sits more than rel_tol behind the nearest surface in
    its win x win neighbourhood is a back-surface point showing through a gap
    in the front surface, and is dropped rather than kept as wrong GT.
    """
    from scipy.ndimage import minimum_filter

    h = pts @ P[:, :3].T + P[:, 3]
    z = h[:, 2] / np.linalg.norm(P[2, :3])       # metric depth along the axis
    ok = z > 0
    u, v, z = h[ok, 0] / h[ok, 2], h[ok, 1] / h[ok, 2], z[ok]
    j, i = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inb = (j >= 0) & (j < W) & (i >= 0) & (i < H)
    idx, z = i[inb] * W + j[inb], z[inb]
    order = np.lexsort((z, idx))
    idx, z = idx[order], z[order]
    first = np.ones(len(idx), dtype=bool)
    first[1:] = idx[1:] != idx[:-1]
    depth = np.zeros(H * W)
    depth[idx[first]] = z[first]
    depth = depth.reshape(H, W)
    local = minimum_filter(np.where(depth > 0, depth, np.inf), size=win, mode="nearest")
    depth[(depth > 0) & (depth > local * (1 + rel_tol))] = 0.0
    return depth


def extract(z, member, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with z.open(member) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


def prepare_scan(z, prefix, scan, out):
    import cv2
    import scipy.io as sio

    dtu = out / "dtu"
    sdir = dtu / f"scan{scan}"
    (sdir / "image").mkdir(parents=True, exist_ok=True)
    (sdir / "depth").mkdir(parents=True, exist_ok=True)

    stl_path = dtu / "Points" / "stl" / f"stl{scan:03d}_total.ply"
    obs_path = dtu / "ObsMask" / f"ObsMask{scan}_10.mat"
    plane_path = dtu / "ObsMask" / f"Plane{scan}.mat"
    extract(z, f"{prefix}Points/stl/stl{scan:03d}_total.ply", stl_path)
    extract(z, f"{prefix}ObsMask/ObsMask{scan}_10.mat", obs_path)
    extract(z, f"{prefix}ObsMask/Plane{scan}.mat", plane_path)
    stl = _load_ply_points(stl_path)
    S, radius = normalization(sio.loadmat(obs_path)["BB"])

    cams, coverage = {}, []
    for k in range(DTU_N_VIEWS):
        pos = k + 1
        P = np.loadtxt(io.StringIO(
            z.read(f"{prefix}Calibration/cal18/pos_{pos:03d}.txt").decode()))
        buf = np.frombuffer(
            z.read(f"{prefix}Rectified/scan{scan}/rect_{pos:03d}_{LIGHT}_r5000.png"),
            np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        H, W = img.shape[0] // FACTOR, img.shape[1] // FACTOR
        cv2.imwrite(str(sdir / "image" / f"{k:06d}.png"),
                    cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))

        Pd = downscale_P(P, FACTOR)
        cams[f"world_mat_{k}"] = np.vstack([Pd, [0.0, 0.0, 0.0, 1.0]])
        cams[f"scale_mat_{k}"] = S
        depth = zbuffer_depth(Pd, stl, H, W)
        np.save(sdir / "depth" / f"{k:06d}.npy", (depth / radius).astype(np.float32))
        coverage.append(float((depth > 0).mean()))
    np.savez(sdir / "cameras.npz", **cams)

    meta = {"scan": scan, "source": "DTU SampleSet (official)", "lighting": LIGHT,
            "downscale": FACTOR, "image_hw": [H, W], "radius_mm": radius,
            "centre_mm": S[:3, 3].tolist(), "stl_points": int(len(stl)),
            "depth_coverage_mean": float(np.mean(coverage)),
            "depth_coverage_min": float(np.min(coverage))}
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  scan{scan}: {DTU_N_VIEWS} views {W}x{H}, GT depth covers "
          f"{100 * meta['depth_coverage_mean']:.0f}% of pixels on average "
          f"(min {100 * meta['depth_coverage_min']:.0f}%)")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="path to SampleSet.zip")
    ap.add_argument("--out", required=True, help="DATA_ROOT to write dtu/ into")
    ap.add_argument("--scans", nargs="+", type=int, default=[1, 6])
    args = ap.parse_args()

    out = Path(args.out)
    with zipfile.ZipFile(args.zip) as z:
        prefix = find_prefix(z)
        for scan in args.scans:
            if (out / "dtu" / f"scan{scan}" / "cameras.npz").exists():
                print(f"  scan{scan}: already prepared")
                continue
            prepare_scan(z, prefix, scan, out)


if __name__ == "__main__":
    main()

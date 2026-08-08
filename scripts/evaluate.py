"""Collect metrics from finished runs into one CSV for the analysis notebook."""
from __future__ import annotations
import argparse
import csv
import glob
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", help="directory of run outputs")
    p.add_argument("--out", default="results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if not os.path.isdir(run_dir):
            continue
        # TODO: load held-out renders + reconstructed point cloud for this run, then
        #   appearance: PSNR / SSIM / LPIPS on held-out views
        #   geometry:   src.eval_geometry.chamfer_distance / depth_metrics
        # rows.append({"run": os.path.basename(run_dir), **metrics})
        pass
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

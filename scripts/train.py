"""Train a single depth-prior 3DGS run from a config.

Driver that wires our src/ modules to the original 3DGS optimizer in
third_party/3dgs. Integration points are marked TODO; they depend on the exact
3DGS repo you clone there.
"""
from __future__ import annotations
import argparse
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment yaml")
    p.add_argument("--out", default="runs/demo", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load data + sparse split                 -> src.data
    # 2. Estimate/load depth, then degrade it      -> depth model + src.corruptions.degrade
    # 3. Initialize Gaussians:
    #      "sfm"    -> COLMAP points from the 3DGS repo
    #      "random" -> random points in the scene volume
    #      "depth"  -> src.depth_init.backproject_depth(...)
    # 4. Train with  L_rgb + lambda * depth_loss   -> src.depth_loss.depth_loss(kind=...)
    #      (render expected depth from the Gaussians for the depth term)
    # 5. Save renders, point cloud, and logs to args.out
    raise NotImplementedError("wire to third_party/3dgs — see README")


if __name__ == "__main__":
    main()

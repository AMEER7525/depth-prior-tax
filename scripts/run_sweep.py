"""Expand a sweep config into individual runs (init x views x depth-quality x lambda)."""
from __future__ import annotations
import argparse
import itertools
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, help="path to sweep yaml")
    p.add_argument("--dry-run", action="store_true", help="print the grid, launch nothing")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.sweep) as f:
        grid = yaml.safe_load(f)

    axes = grid["axes"]              # name -> list of values
    keys = list(axes)
    combos = list(itertools.product(*(axes[k] for k in keys)))
    print(f"{len(combos)} settings x {len(grid.get('scenes', [None]))} scenes")
    for combo in combos:
        setting = dict(zip(keys, combo))
        for scene in grid.get("scenes", [None]):
            setting_full = {**setting, "scene": scene}
            print("RUN", setting_full)
            if args.dry_run:
                continue
            # TODO: write a per-run config from setting_full, then
            #   subprocess.run(["python", "scripts/train.py", "--config", cfg, "--out", out])


if __name__ == "__main__":
    main()

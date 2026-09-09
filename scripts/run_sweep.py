"""Expand a staged sweep into runs and launch them, with resume.

Designed for the 4080-primary / Colab-overflow split: runs are addressed by a
stable id, so both hosts can write into the same runs/ tree (synced or mounted)
and --resume makes either host skip what the other already finished. Use
--shard i/n to split the grid across the two.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.sweep import expand, run_id  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, help="path to sweep yaml")
    p.add_argument("--stage", default="stage1", help="which stage block to run")
    p.add_argument("--base", default=None,
                   help="config to override (default: <repo>/configs/base.yaml)")
    p.add_argument("--runs", default=None,
                   help="output root (default $RUNS_ROOT, else runs/)")
    p.add_argument("--dry-run", action="store_true", help="print the grid, launch nothing")
    p.add_argument("--explain-pruning", action="store_true",
                   help="list the cells dropped as degenerate and why")
    p.add_argument("--keep-degenerate", action="store_true",
                   help="do not prune provably-null cells (for auditing)")
    p.add_argument("--resume", action="store_true",
                   help="skip runs whose output dir already holds metrics.json")
    p.add_argument("--shard", default=None, metavar="i/n",
                   help="run only shard i of n (e.g. 0/2 on the 4080, 1/2 on Colab)")
    return p.parse_args()


def runs_root(arg):
    return Path(arg or os.environ.get("RUNS_ROOT", "runs"))


def is_finished(run_dir):
    return (run_dir / "metrics.json").exists()


def main():
    args = parse_args()
    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    if args.stage not in sweep:
        sys.exit(f"stage '{args.stage}' not in {args.sweep}; "
                 f"have {[k for k in sweep if k.startswith('stage')]}")
    base_path = Path(args.base) if args.base else REPO / "configs" / "base.yaml"
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    cells, pruned = expand(sweep[args.stage], keep_degenerate=args.keep_degenerate)

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [c for j, c in enumerate(cells) if j % n == i]

    root = runs_root(args.runs)
    total = len(cells) + len(pruned)
    print(f"[{args.stage}] {total} cells -> {len(cells)} to run, "
          f"{len(pruned)} pruned ({100 * len(pruned) / max(total, 1):.0f}% saved)")

    if args.explain_pruning:
        by_reason = {}
        for c in pruned:
            by_reason.setdefault(c["_reason"].split(":")[0], []).append(c)
        for reason, group in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(group):4d}  {reason}")

    pending = [c for c in cells
               if not (args.resume and is_finished(root / args.stage / run_id(c)))]
    skipped = len(cells) - len(pending)
    if skipped:
        print(f"resume: skipping {skipped} finished run(s)")

    launched = 0
    for cell in pending:
        rid = run_id(cell)
        out = root / args.stage / rid

        cfg = _apply_cell(base_cfg, cell)
        _check_corruption_reachable(cfg, cell, rid)
        if args.dry_run:
            print("RUN", rid)
            continue

        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(out / "cell.json", "w") as f:
            json.dump(cell, f, indent=2)

        print(f"[{launched + 1}/{len(pending)}] {rid}")
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(cfg_path),
             "--out", str(out)])
        if result.returncode != 0:
            # One bad scene must not abandon a 200-run grid; record and continue.
            (out / "FAILED").write_text(f"exit {result.returncode}\n")
            print(f"  !! failed (exit {result.returncode}), continuing")
        launched += 1



def _check_corruption_reachable(cfg, cell, rid):
    """Refuse to run an Axis C cell whose corruption cannot take effect.

    The degradation knobs perturb GROUND-TRUTH depth. If depth.model names a
    real predictor there is no GT to perturb, so sigma_rel/scale/shift are
    ignored -- and the whole degradation curve comes out flat for a reason that
    has nothing to do with the hypothesis. Fail loudly instead.
    """
    corrupting = (cell.get("sigma_rel", 0.0) != 0.0
                  or cell.get("scale", 1.0) != 1.0
                  or cell.get("shift", 0.0) != 0.0)
    model = cfg.get("depth", {}).get("model")
    if corrupting and model != "gt":
        sys.exit(
            f"\n{rid}\n"
            f"  sweeps a depth corruption but depth.model is '{model}'.\n"
            f"  Corruption applies to GT depth only; these runs would silently\n"
            f"  ignore it and produce a flat curve. Set depth.model: gt in the\n"
            f"  base config, or add depth_model: [gt] to this stage's axes.")


def _apply_cell(base_cfg, cell):
    """Overlay a flat sweep cell onto the nested base config."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    routes = {
        "init": ("init",),
        "scene": ("scene",),
        "n_views": ("n_views",),
        "depth_kind": ("loss", "depth_kind"),
        "depth_lambda": ("loss", "depth_lambda"),
        "sigma_rel": ("depth", "sigma_rel"),
        "scale": ("depth", "scale"),
        "shift": ("depth", "shift"),
        "depth_model": ("depth", "model"),
    }
    for key, value in cell.items():
        if key.startswith("_") or value is None:
            continue
        path = routes.get(key)
        if path is None:
            raise KeyError(f"sweep axis '{key}' has no route into base.yaml")
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return cfg


if __name__ == "__main__":
    main()

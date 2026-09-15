"""Collect every finished run into one CSV for the analysis notebook.

    python scripts/evaluate.py --runs $RUNS_ROOT --out results.csv

run_sweep.py lays runs out as <runs>/<stage>/<run_id>/metrics.json. Each row is
one run: its stage, its config flattened to the sweep's axis names (so the
notebook can group by init / n_views / depth_lambda / ... directly), and every
scalar it reported. Failed runs go to a second CSV with their error, because
"SfM could not initialize at 3 views" is a result too.

Two kinds of run are left out by default -- and counted, so nothing vanishes
silently:
  * pre-fix runs: trained before train.py took over the opacity reset that
    gsplat 1.5.3 never performs. They carry no `opacity_resets` record.
  * orphans: runs of a stage in configs/sweep.yaml whose settings are no longer
    in that stage (a changed winner, a changed schedule).
--all keeps both.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# column -> path into the nested run config, and the value when absent.
FLAT = {
    "dataset":       (("dataset",), "blender"),
    "scene":         (("scene",), None),
    "n_views":       (("n_views",), None),
    "init":          (("init",), None),
    "split":         (("split",), "fps"),
    "downscale":     (("downscale",), 1),
    "background":    (("background",), "black"),
    "depth_model":   (("depth", "model"), "gt"),
    "depth_align":   (("depth", "align"), "gt"),
    "sigma_rel":     (("depth", "sigma_rel"), 0.0),
    "noise_corr_px": (("depth", "noise_corr_px"), 0.0),
    "scale":         (("depth", "scale"), 1.0),
    "shift":         (("depth", "shift"), 0.0),
    "scale_jitter":  (("depth", "scale_jitter"), 0.0),
    "shift_jitter":  (("depth", "shift_jitter"), 0.0),
    "depth_kind":    (("loss", "depth_kind"), "ssi"),
    "depth_lambda":  (("loss", "depth_lambda"), 0.0),
    "iterations":    (("train", "iterations"), None),
    "sh_degree":     (("train", "sh_degree"), 0),
    "seed":          (("train", "seed"), 0),
}
# Nested record entries that are not per-run scalars worth a column.
_SKIP = {"config", "log"}
# Reproduction target: DNGaussian (CVPR'24), 3DGS row, NeRF-Synthetic 8 views.
REPRO_TARGET = {"psnr": 22.23, "ssim": 0.858, "lpips": 0.114}


def _get(d, path, default):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _scalar(v):
    return v is None or isinstance(v, (bool, int, float, str))


def _flatten(prefix, obj, row):
    """Scalars at any depth become columns; lists become JSON strings."""
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if _scalar(v):
            row[key] = v
        elif isinstance(v, dict):
            _flatten(f"{key}_", v, row)
        elif isinstance(v, list) and k in ("psnr_curve", "train_ids", "eval_ids"):
            row[key] = json.dumps(v)


def _stage(run_dir, root):
    rel = Path(run_dir).relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def is_current(rec):
    """Trained with the opacity-reset fix (train.py records the reset steps)."""
    return "opacity_resets" in rec


def load_sweep(path):
    """{stage: stage config} from sweep.yaml, or {} when disabled/absent."""
    if not path or not Path(path).exists():
        return {}
    import yaml
    return yaml.safe_load(open(path)) or {}


def current_run_ids(sweep):
    """{stage: set of run ids} that the sweep config would run today."""
    sys.path.insert(0, str(REPO))
    from src.sweep import expand, run_id
    return {name: {run_id(c) for c in expand(stage)[0]} for name, stage in sweep.items()}


def row_for(metrics_path, root, rec=None):
    rec = rec if rec is not None else json.loads(Path(metrics_path).read_text())
    run_dir = Path(metrics_path).parent
    row = {"stage": _stage(run_dir, root), "run_id": run_dir.name}
    cfg = rec.get("config", {})
    for col, (path, default) in FLAT.items():
        row[col] = _get(cfg, path, default)
    body = {k: v for k, v in rec.items() if k not in _SKIP}
    # "depth" and "init" hold run bookkeeping whose leaf names would collide
    # with config columns ("model", "init", ...), so they keep a prefix.
    for k in ("depth", "init"):
        if isinstance(body.get(k), dict):
            _flatten(f"{k}_", body.pop(k), row)
    _flatten("", body, row)
    row["n_opacity_resets"] = len(rec.get("opacity_resets") or [])
    return row


def failure_for(run_dir, root):
    err = Path(run_dir) / "error.txt"
    cell = Path(run_dir) / "cell.json"
    row = {"stage": _stage(run_dir, root), "run_id": Path(run_dir).name,
           "error": err.read_text().strip().splitlines()[-1] if err.exists() else ""}
    if cell.exists():
        row.update({k: v for k, v in json.loads(cell.read_text()).items()
                    if not k.startswith("_")})
    return row


def write_csv(rows, path, lead=()):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(lead) + [c for c in FLAT if c not in lead]
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def repro_summary(rows, expected_scenes):
    """The reproduction check per scene, naming any scene that has not run yet."""
    by_scene = {r["scene"]: r for r in rows if r["stage"] == "stage0_repro"}
    if not by_scene:
        return
    print("\nstage0_repro -- vanilla 3DGS, NeRF-Synthetic 8 views:")
    for s in sorted(by_scene):
        r = by_scene[s]
        print(f"  {s:10s} PSNR {float(r['psnr']):6.2f}  SSIM {float(r['ssim']):.3f}  "
              f"LPIPS {float(r['lpips']):.3f}")
    mean = {k: sum(float(r[k]) for r in by_scene.values()) / len(by_scene)
            for k in REPRO_TARGET}
    print(f"  mean over {len(by_scene)} scene(s): PSNR {mean['psnr']:.2f} / SSIM "
          f"{mean['ssim']:.3f} / LPIPS {mean['lpips']:.3f}   published: "
          f"{REPRO_TARGET['psnr']} / {REPRO_TARGET['ssim']} / {REPRO_TARGET['lpips']}")
    missing = sorted(set(expected_scenes) - set(by_scene))
    if missing:
        print(f"  NOT YET RUN: {', '.join(missing)} -- the reproduction check is the mean over all "
              f"{len(expected_scenes)} scenes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", help="root of run outputs")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--sweep", default=str(REPO / "configs" / "sweep.yaml"),
                   help="sweep config that defines the current runs ('' disables "
                        "the orphan filter)")
    p.add_argument("--all", action="store_true",
                   help="keep runs trained before the opacity-reset fix, and orphans")
    args = p.parse_args()

    root = Path(args.runs)
    sweep = load_sweep(args.sweep)
    current = current_run_ids(sweep) if sweep else {}

    def orphan(run_dir):
        st = _stage(run_dir, root)
        return st in current and Path(run_dir).name not in current[st]

    rows, bad, pre_fix, orphans = [], [], 0, 0
    for m in sorted(root.rglob("metrics.json")):
        try:
            rec = json.loads(m.read_text())
        except (json.JSONDecodeError, OSError) as e:
            bad.append(f"{m}: {e}")
            continue
        if not args.all and orphan(m.parent):
            orphans += 1
            continue
        if not args.all and not is_current(rec):
            pre_fix += 1
            continue
        rows.append(row_for(m, root, rec))
    failures = [failure_for(f.parent, root) for f in sorted(root.rglob("FAILED"))
                if not (f.parent / "metrics.json").exists()
                and (args.all or not orphan(f.parent))]

    write_csv(rows, args.out, lead=("stage", "run_id"))
    if rows:
        print(f"wrote {len(rows)} run(s) to {args.out}")
    else:
        print(f"no finished runs (metrics.json) under {root}; nothing written")
    if pre_fix:
        print(f"left out {pre_fix} run(s) trained before the opacity-reset fix "
              "(run_sweep --resume re-runs them)")
    if orphans:
        print(f"left out {orphans} run(s) whose settings are no longer in "
              f"{Path(args.sweep).name} (--all keeps them)")
    if failures:
        fail_path = Path(args.out).with_name(Path(args.out).stem + "_failures.csv")
        write_csv(failures, fail_path, lead=("stage", "run_id", "error"))
        print(f"{len(failures)} failed run(s) -> {fail_path}")
    for b in bad:
        print("  unreadable:", b)
    repro_summary(rows, (sweep.get("stage0_repro") or {}).get("scenes", []))


if __name__ == "__main__":
    main()

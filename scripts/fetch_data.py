"""Fetch datasets in the shape Colab actually wants to read them.

The Colab access pattern matters more than the download. Reading thousands of
small PNGs over the Drive FUSE mount is slow and rate-limited; reading one
large archive is not. So the split is:

    archives  -> Google Drive   (persistent, downloaded once, survives sessions)
    extracted -> /content       (local Colab disk, fast, rebuilt each session)

Re-extracting from a local archive takes seconds. Re-downloading does not, and
neither does training off a FUSE mount.

    python scripts/fetch_data.py --scenes lego chair ship \
        --archives "/content/drive/MyDrive/datasets/blender" \
        --extract  /content/data
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Per-scene archives from the nerfbaselines mirror. Per-scene rather than the
# 1.27 GB monolith so a 3-scene sweep pulls 595 MB instead of everything.
BLENDER_BASE = ("https://huggingface.co/datasets/nerfbaselines/"
                "nerfbaselines-data/resolve/main/blender")
BLENDER_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials",
                  "mic", "ship"]
# Approximate sizes (MB) so the script can report the cost before spending it.
BLENDER_MB = {"chair": 130, "drums": 152, "ficus": 134, "hotdog": 158,
              "lego": 180, "materials": 144, "mic": 100, "ship": 285}


def human(n):
    return f"{n / 1e6:.0f} MB"


def download(url, dest, quiet=False):
    """Download with resume. Returns False if the file was already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
    if total and have == total:
        if not quiet:
            print(f"  have  {dest.name} ({human(total)})")
        return False

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have and not quiet:
        print(f"  resume {dest.name} from {human(have)}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(dest, "ab" if have else "wb") as f:
        got = have
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if not quiet and total:
                pct = 100 * got / total
                print(f"\r  {dest.name}  {pct:5.1f}%  {human(got)}/{human(total)}",
                      end="", flush=True)
    if not quiet:
        print()
    return True


def extract_scene(archive, out_root, scene):
    """Extract into <out_root>/nerf_synthetic/<scene>/, normalizing layout.

    Mirrors disagree on whether the zip has a top-level scene directory, so
    locate transforms_train.json after extracting and reparent to match what
    src.data.load_blender_scene expects.
    """
    target = out_root / "nerf_synthetic" / scene
    if (target / "transforms_train.json").exists():
        print(f"  have  {scene}/ extracted")
        return target

    staging = out_root / ".staging" / scene
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(staging)

    found = next(staging.rglob("transforms_train.json"), None)
    if found is None:
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(f"ERROR: {archive.name} contains no transforms_train.json.\n"
                 f"       src/data.py cannot read this layout.")

    src = found.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    shutil.rmtree(out_root / ".staging", ignore_errors=True)
    print(f"  ok    {scene}/ -> {target}")
    return target


def verify(scene_dir):
    """Confirm the extracted scene is actually loadable, not just present."""
    import json
    n_img = len(list(scene_dir.rglob("*.png")))
    splits = {}
    for split in ("train", "val", "test"):
        p = scene_dir / f"transforms_{split}.json"
        splits[split] = len(json.load(open(p))["frames"]) if p.exists() else 0
    ok = splits["train"] > 0 and n_img > 0
    print(f"        {n_img} png | frames train={splits['train']} "
          f"val={splits['val']} test={splits['test']} | "
          f"{'OK' if ok else 'INCOMPLETE'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=["lego", "chair", "ship"],
                   help="scenes to fetch (default: the three the sweep uses)")
    p.add_argument("--all", action="store_true", help="fetch all 8 scenes")
    p.add_argument("--archives", required=True,
                   help="where zips live (put this on Drive in Colab)")
    p.add_argument("--extract", required=True,
                   help="where to extract (put this on local disk in Colab)")
    p.add_argument("--archives-only", action="store_true",
                   help="download but do not extract")
    args = p.parse_args()

    scenes = BLENDER_SCENES if args.all else args.scenes
    unknown = set(scenes) - set(BLENDER_SCENES)
    if unknown:
        sys.exit(f"unknown scene(s): {sorted(unknown)}\n"
                 f"available: {BLENDER_SCENES}")

    arch_dir, out_root = Path(args.archives), Path(args.extract)
    total_mb = sum(BLENDER_MB.get(s, 150) for s in scenes)
    print(f"NeRF-Synthetic: {len(scenes)} scene(s), ~{total_mb} MB")
    print(f"  archives -> {arch_dir}")
    print(f"  extract  -> {out_root}\n")

    ok = True
    for scene in scenes:
        print(f"[{scene}]")
        archive = arch_dir / f"{scene}.zip"
        download(f"{BLENDER_BASE}/{scene}.zip", archive)
        if args.archives_only:
            continue
        ok &= verify(extract_scene(archive, out_root, scene))

    if not args.archives_only:
        print(f"\nDATA_ROOT={out_root}")
        print("NOTE: NeRF-Synthetic ships NO ground-truth depth. Axis C needs it "
              "rendered\n      from the original .blend files before it can run.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

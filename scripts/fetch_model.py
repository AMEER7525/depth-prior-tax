"""Download Depth Anything V2 weights to a directory (Drive, SSD, or local).

Needed only for stage4_real -- the proposal's "one real monocular operating
point". Axis C runs on perturbed GROUND-TRUTH depth and needs no model at all,
so this is not on the critical path.

    python scripts/fetch_model.py --variant large \
        --dest "/content/drive/MyDrive/models/depth_anything_v2_large"

On Colab the Hub download takes seconds and transformers caches it, so keeping
weights on Drive is only worth the quota if you want them offline or pinned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.depth_model import DA_V2_REPOS, DEFAULT_VARIANT  # noqa: E402

SIZES_MB = {"small": 99, "base": 390, "large": 1341}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(DA_V2_REPOS))
    ap.add_argument("--dest", required=True, help="directory to download into")
    ap.add_argument("--check", action="store_true",
                    help="report what is already there, download nothing")
    args = ap.parse_args()

    dest = Path(args.dest)
    repo = DA_V2_REPOS[args.variant]

    weights = dest / "model.safetensors"
    if weights.exists():
        mb = weights.stat().st_size / 1e6
        print(f"already present: {dest} ({mb:.0f} MB)")
        if args.check or mb > SIZES_MB[args.variant] * 0.95:
            return
        print("  size looks short; re-downloading")
    elif args.check:
        print(f"absent: {dest} (~{SIZES_MB[args.variant]} MB to fetch)")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")

    print(f"{repo}  (~{SIZES_MB[args.variant]} MB)  ->  {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    # local_dir gives a plain directory rather than the blob/symlink cache
    # layout, which Drive does not handle well.
    snapshot_download(repo_id=repo, local_dir=str(dest),
                      allow_patterns=["*.json", "*.safetensors"])

    got = sorted(p.name for p in dest.iterdir() if p.is_file())
    print(f"\nfiles: {got}")
    if "model.safetensors" not in got:
        sys.exit("ERROR: model.safetensors missing after download")
    print(f"size : {weights.stat().st_size / 1e6:.0f} MB")
    print(f"\nUse it with:  load_depth_model(local_dir='{dest}')")


if __name__ == "__main__":
    main()

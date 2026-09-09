#!/usr/bin/env python3
"""Download the NeRF-Synthetic archives into Google Drive, from this Mac.

No Colab needed. Archives land in your Google Drive for Desktop folder, which
syncs them to the cloud, so a later Colab session can mount Drive and read them
without downloading anything again.

Why archives and not extracted scenes: Colab extracts to its own local disk each
session, because reading thousands of small PNGs over the Drive FUSE mount is
slow and rate-limited. Drive only ever needs to hold the zips. Extracting here
as well would upload ~4x the bytes for no benefit.

Downloads are staged locally and only moved into Drive once verified, so Drive
never sees a truncated archive and never tries to sync a file that is still
being written. Interrupted downloads resume: re-run the same command.

    python download_datasets.py --account 75@      # lego, chair, ship (~595 MB)
    python download_datasets.py --all              # all 8 scenes (~1.3 GB)
    python download_datasets.py --check            # report what Drive already has
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from fetch_data import (  # noqa: E402
    BLENDER_BASE, BLENDER_MB, BLENDER_SCENES, download, human)

DEFAULT_SCENES = ["lego", "chair", "ship"]      # the three the sweep uses


def drive_accounts():
    """Every Google Drive for Desktop mount, as {account email: My Drive path}."""
    out = {}
    for p in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive")):
        out[p.parent.name.replace("GoogleDrive-", "")] = p
    return out


def pick_drive(account=None):
    """Resolve which Drive account to write into.

    Never guesses between several. The archives are only useful if Colab mounts
    the SAME account, and a wrong choice here wastes both the upload and the
    quota -- with the failure surfacing much later, in Colab, as missing data.
    """
    accounts = drive_accounts()
    if not accounts:
        sys.exit("Google Drive for Desktop not found under ~/Library/CloudStorage.\n"
                 "Start Drive for Desktop, or pass --dest explicitly.")
    if account:
        matches = [a for a in accounts if account in a]
        if len(matches) == 1:
            return accounts[matches[0]]
        sys.exit(f"--account {account!r} matched {len(matches)} of "
                 f"{list(accounts)}; be more specific.")
    if len(accounts) == 1:
        return next(iter(accounts.values()))
    sys.exit("Several Google accounts are signed in:\n"
             + "\n".join(f"    {a}" for a in accounts)
             + "\n\nPick one with --account, e.g.\n"
               f"    python download_datasets.py --account {next(iter(accounts))}\n\n"
               "Use the SAME account you will open Colab with -- Colab mounts one\n"
               "account's Drive, and archives under the other are invisible to it.")


def verify_zip(path, scene):
    """Confirm the archive is complete and holds the layout src/data.py needs.

    A truncated download is the likely failure on a slow link, and a broken zip
    sitting in Drive is worse than no zip: Colab would fail hours later with a
    confusing error. Read the central directory and require transforms_train.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            bad = z.testzip()
    except (zipfile.BadZipFile, OSError) as e:
        return False, f"unreadable ({type(e).__name__}: {e})"
    if bad:
        return False, f"CRC failed on {bad}"
    if not any(n.endswith("transforms_train.json") for n in names):
        return False, "no transforms_train.json inside"
    n_png = sum(1 for n in names if n.endswith(".png"))
    return True, f"{len(names)} entries, {n_png} png"


def report(dest, scenes):
    print(f"Drive folder: {dest}\n")
    total = missing = 0
    for scene in scenes:
        p = dest / f"{scene}.zip"
        if not p.exists():
            print(f"  {scene:10s} -- absent (~{BLENDER_MB.get(scene, 150)} MB to fetch)")
            missing += 1
            continue
        ok, detail = verify_zip(p, scene)
        size = p.stat().st_size
        total += size
        print(f"  {scene:10s} {'OK  ' if ok else 'BAD '} {human(size):>9s}  {detail}")
    print(f"\n  {human(total)} present, {missing} scene(s) missing")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES,
                    help=f"default: {' '.join(DEFAULT_SCENES)}")
    ap.add_argument("--all", action="store_true", help="all 8 scenes")
    ap.add_argument("--dest", default=None,
                    help="target dir (default: <Google Drive>/datasets/blender)")
    ap.add_argument("--account", default=None,
                    help="which Google account's Drive to use (substring of the "
                         "email); required when several are signed in")
    ap.add_argument("--staging", default=None,
                    help="where to download before moving into Drive "
                         "(default: <dest>/../.staging)")
    ap.add_argument("--check", action="store_true",
                    help="report what is already there, download nothing")
    args = ap.parse_args()

    scenes = BLENDER_SCENES if args.all else args.scenes
    unknown = set(scenes) - set(BLENDER_SCENES)
    if unknown:
        sys.exit(f"unknown scene(s): {sorted(unknown)}\navailable: {BLENDER_SCENES}")

    dest = (Path(args.dest) if args.dest
            else pick_drive(args.account) / "datasets" / "blender")
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        report(dest, scenes)
        return

    # Stage outside Drive so a partial file is never visible to the sync client.
    staging = Path(args.staging) if args.staging else dest.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

    todo = [s for s in scenes if not (dest / f"{s}.zip").exists()]
    have = len(scenes) - len(todo)
    mb = sum(BLENDER_MB.get(s, 150) for s in todo)
    print(f"NeRF-Synthetic -> {dest}")
    if have:
        print(f"  {have} scene(s) already present, skipping")
    if not todo:
        print("  nothing to do")
        return
    print(f"  {len(todo)} to fetch, ~{mb} MB")
    print(f"  staging in {staging}\n")

    failed = []
    for scene in todo:
        print(f"[{scene}]")
        tmp = staging / f"{scene}.zip"
        try:
            download(f"{BLENDER_BASE}/{scene}.zip", tmp)
        except KeyboardInterrupt:
            print(f"\n  interrupted -- {human(tmp.stat().st_size if tmp.exists() else 0)} "
                  f"kept in staging; re-run to resume")
            raise SystemExit(130)
        except Exception as e:
            print(f"  download failed: {e}")
            failed.append(scene)
            continue

        ok, detail = verify_zip(tmp, scene)
        if not ok:
            print(f"  REJECTED: {detail}")
            print(f"  leaving {tmp} in staging; re-run to resume the download")
            failed.append(scene)
            continue
        print(f"  verified: {detail}")
        shutil.move(str(tmp), str(dest / f"{scene}.zip"))
        print(f"  -> {dest / f'{scene}.zip'}")

    try:
        staging.rmdir()
    except OSError:
        pass        # still holds a partial download; that is intentional

    print()
    report(dest, scenes)
    if failed:
        print(f"\n{len(failed)} scene(s) incomplete: {failed}. Re-run to resume.")
        sys.exit(1)
    print("\nDrive for Desktop now syncs these to the cloud. Wait for the sync to")
    print("finish before running the Colab notebook, and skip its fetch cell's")
    print("download -- it will find the archives already present.")


if __name__ == "__main__":
    main()

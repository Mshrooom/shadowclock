#!/usr/bin/env python3
"""
verify_dataset.py - check that every image / mask / height map listed in metadata.jsonl exists and is readable.

    python verify_dataset.py --data data            # report only
    python verify_dataset.py --data data --prune    # also rewrite metadata.jsonl without the broken rows
                                                    # (original saved as metadata.jsonl.bak)

Paths are normalised to forward slashes, so a dataset rendered on Windows can be used on Linux.
Exit status is 1 if anything is missing/unreadable (and not pruned), else 0.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()

    meta_path = os.path.join(args.data, "metadata.jsonl")
    rows = [json.loads(l) for l in open(meta_path)]
    bad, good = [], []
    for r in rows:
        r["image"] = r["image"].replace("\\", "/")
        if r.get("shadow_gt"):
            r["shadow_gt"] = r["shadow_gt"].replace("\\", "/")
        problem = None
        img = os.path.join(args.data, r["image"])
        if not os.path.exists(img):
            problem = "image missing"
        elif os.path.getsize(img) == 0:
            problem = "image empty"
        else:
            try:
                with Image.open(img) as im:
                    im.load()
                    if im.size != (r["size_px"], r["size_px"]):
                        problem = f"image size {im.size} != {r['size_px']}"
            except Exception as exc:  # noqa: BLE001
                problem = f"image unreadable ({type(exc).__name__})"
        if problem is None and r.get("shadow_gt"):
            sh = os.path.join(args.data, r["shadow_gt"])
            if not os.path.exists(sh):
                problem = "shadow mask missing"
        if problem is None:
            hp = os.path.join(args.data, "scenes", f"scene_{r['scene']:03d}_height.npy")
            if not os.path.exists(hp):
                problem = "height map missing"
        (bad if problem else good).append((r, problem))

    print(f"rows in metadata : {len(rows)}")
    print(f"OK               : {len(good)}")
    print(f"broken           : {len(bad)}")
    if bad:
        print("\nbroken rows by problem :", dict(Counter(p for _, p in bad)))
        print("broken rows by scene   :", dict(sorted(Counter(r['scene'] for r, _ in bad).items())))
        print("broken rows by lat     :", dict(sorted(Counter(r['lat'] for r, _ in bad).items())))
        print("broken rows by hour UTC:", dict(sorted(Counter(r['utc_minute'] // 60 for r, _ in bad).items())))
        print("first 10:", [r["image"] for r, _ in bad[:10]])
    if args.prune and bad:
        shutil.copyfile(meta_path, meta_path + ".bak")
        with open(meta_path, "w") as f:
            for r, _ in good:
                f.write(json.dumps(r) + "\n")
        print(f"\npruned: metadata.jsonl now lists {len(good)} rows (backup: metadata.jsonl.bak)")
        return 0
    if bad:
        print("\nFix: re-run the SAME generate_dataset.py command (it re-renders only the missing images), "
              "or run this script with --prune to continue with what exists.")
        return 1
    print("dataset is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())

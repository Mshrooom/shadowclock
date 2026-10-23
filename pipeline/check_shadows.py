#!/usr/bin/env python3
"""
check_shadows.py - verify that rendered shadows match the analytic ground truth.

For every (sampled) image it thresholds the luminance of *ground* pixels (height == 0)
with Otsu and reports the IoU against the analytic shadow mask. A healthy dataset gives
IoU close to 1; a flipped north/east convention or a wrong sun azimuth gives a much lower
IoU. Use this once after rendering (and again if you change the generator).

    python check_shadows.py --data data --max-images 40
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    return centers[int(np.argmax(between))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    rows = [r for r in rows if r.get("shadow_gt")]
    if not rows:
        raise SystemExit("no rows with shadow_gt; render without --no-shadow-gt")
    for r in rows:  # metadata written on Windows may contain backslashes
        r["image"], r["shadow_gt"] = r["image"].replace("\\", "/"), r["shadow_gt"].replace("\\", "/")
    present = [r for r in rows if os.path.exists(os.path.join(args.data, r["image"]))
               and os.path.exists(os.path.join(args.data, r["shadow_gt"]))]
    if len(present) < len(rows):
        print(f"WARNING: {len(rows) - len(present)} of {len(rows)} listed images/masks are missing on disk. "
              f"Run verify_dataset.py --data {args.data} (add --prune to drop them from metadata).")
    rows = present
    if not rows:
        raise SystemExit("none of the listed images exist")
    rng = np.random.default_rng(args.seed)
    if len(rows) > args.max_images:
        rows = [rows[i] for i in rng.choice(len(rows), args.max_images, replace=False)]

    ious, flip_ious, details = [], [], []
    for r in rows:
        img = np.asarray(Image.open(os.path.join(args.data, r["image"])).convert("RGB"), dtype=float) / 255.0
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        H = np.load(os.path.join(args.data, "scenes", f"scene_{r['scene']:03d}_height.npy"))
        if H.shape != lum.shape:
            raise SystemExit(f"{r['image']}: image {lum.shape} != height map {H.shape}")
        gt = np.load(os.path.join(args.data, r["shadow_gt"]))["shadow"]
        ground = H == 0
        thr = otsu(lum[ground])
        pred = (lum < thr) & ground
        gt_g = gt & ground
        inter, union = (pred & gt_g).sum(), (pred | gt_g).sum()
        ious.append(inter / max(union, 1))
        details.append((ious[-1], r["image"], r["sun_el_deg"], r["sun_az_deg"], float(gt_g.sum()) / max(int(ground.sum()), 1)))
        # negative control: mirror the analytic mask left-right (what a wrong east/west convention would give)
        gm = gt_g[:, ::-1]
        flip_ious.append((pred & gm).sum() / max((pred | gm).sum(), 1))
    print(f"images checked          : {len(ious)}")
    print(f"mean IoU (render vs GT) : {np.mean(ious):.3f}   min {np.min(ious):.3f}")
    print(f"mean IoU (mirrored GT)  : {np.mean(flip_ious):.3f}   <- should be clearly lower")
    print("worst 5 images (IoU, sun elevation, sun azimuth, shadowed fraction of ground):")
    for iou, name, el, az, frac in sorted(details)[:5]:
        print(f"    {iou:.3f}  el={el:5.1f}  az={az:5.1f}  shadow={100 * frac:4.1f}%  {name}")
    if np.mean(ious) < 0.9:
        print("WARNING: low agreement; check conventions (north up / east right) before trusting the dataset")


if __name__ == "__main__":
    main()

"""Blender-free end-to-end smoke test: toy overhead renders (numpy) -> make_pairs.py -> baseline.py -> score.parse.

The toy images follow the benchmark conventions (north up, east right, green ground, grey roofs, shadows = darker ground
from the exact analytic shadow mask), so the pixel baseline must order them correctly given the hemisphere (R1).
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

from geom import analytic_shadow_mask, rasterize_height_polys
from score import parse
from solar import solar_position

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "pipeline")


def rect(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def make_toy_dataset(root, lat):
    size, gsd = 512, 0.5
    blds = [
        {"rings": [rect(-60, -50, 22, 16)], "height": 24.0},
        {"rings": [rect(55, -45, 18, 24)], "height": 30.0},
        {"rings": [rect(-55, 55, 26, 18)], "height": 18.0},
        {"rings": [rect(50, 55, 20, 20)], "height": 27.0},
    ]
    H = rasterize_height_polys(blds, size, gsd)
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "scenes"), exist_ok=True)
    np.save(os.path.join(root, "scenes", "scene_000_height.npy"), H)
    rows = []
    for hour in range(8, 17):
        az, el = (float(x) for x in solar_position(lat, 0.0, 2027, 3, 20, float(hour)))
        if not 20 <= el <= 70:
            continue
        img = np.zeros((size, size, 3), dtype=float)
        img[:] = (70, 160, 70)
        sh = analytic_shadow_mask(H, gsd, az, el) & (H == 0)
        img[sh] = np.array((70, 160, 70)) * 0.4
        img[H > 0] = (170, 150, 135)
        name = f"images/s000_lat{int(lat):+03d}_2027-03-20_{hour:02d}00.png"
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(root, name))
        rows.append(dict(id=name, scene=0, image=name, shadow_gt=None, lat=lat, lon=0.0, date="2027-03-20",
                         utc_minute=hour * 60, utc_hour=float(hour), sun_az_deg=az, sun_el_deg=el, gsd_m=gsd, size_px=size))
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def run(args):
    return subprocess.run([sys.executable] + args, capture_output=True, text=True, check=True, cwd=PIPE)


def accuracy(pairs, preds, regime):
    ok = tot = 0
    for p in pairs:
        got = preds[(p["id"], regime)]
        tot += 1
        ok += int(got == ("A" if p["a_earlier"] else "B"))
    return ok / tot


def test_pairs_and_pixel_baseline_in_both_hemispheres(tmp_path):
    for lat in (40.0, -40.0):
        d = tmp_path / f"d{int(lat)}"
        assert make_toy_dataset(str(d), lat) >= 5
        pairs_f, preds_f = str(d / "pairs.jsonl"), str(d / "preds.jsonl")
        run(["make_pairs.py", "--data", str(d), "--out", pairs_f, "--dt", "1,2", "--max-el", "70", "--add-swapped"])
        run(["baseline.py", "--data", str(d), "--pairs", pairs_f, "--out", preds_f, "--method", "components"])
        pairs = [json.loads(l) for l in open(pairs_f)]
        assert pairs and all(p["hemisphere"] == ("northern" if lat > 0 else "southern") for p in pairs)
        preds = {}
        for l in open(preds_f):
            r = json.loads(l)
            preds[(r["id"], r["regime"])] = parse(r["response"])[0]
        assert accuracy(pairs, preds, "R1") >= 0.95, f"lat {lat}: R1 accuracy too low"
        assert accuracy(pairs, preds, "R2") >= 0.95, f"lat {lat}: R2 accuracy too low"

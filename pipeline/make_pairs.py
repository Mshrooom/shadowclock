#!/usr/bin/env python3
"""
make_pairs.py - build the evaluation pairs (and VLM prompts) from metadata.jsonl.

Each pair = two renders of the SAME scene, SAME latitude and SAME date, at different times
of day. Ground truth: which image is earlier and the elapsed time in hours.

Three information regimes (prompt text is stored per pair under "prompts"):
    R0  nothing but the two images (north-up stated)         -> hemisphere is NOT given
    R1  + hemisphere (northern / southern)
    R2  + latitude, longitude, date and the clock convention

Output: pairs.jsonl, one JSON object per line:
    id, scene, group, image_a, image_b, a_earlier, elapsed_hours, dt_bucket, lat, hemisphere,
    date, prompts {R0,R1,R2}

Typical use:
    python make_pairs.py --data data --out pairs.jsonl --dt 0.5,1,2,4 --max-per-dt 4 --add-swapped
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

BASE = ("You are given two overhead (satellite-style) images, A and B, of the same place on the same "
        "day. North is up and east is to the right in both images. The only light source is the sun. "
        "Using the shadows, decide which image was captured EARLIER in the day and estimate how many "
        "hours apart the two images were captured. ")
# v1 embedded a concrete example answer ("A", 1.5). A pilot showed a 3B model copying it, so v2 states the format
# WITHOUT any example values (nothing to copy). v1 is kept only for the example-anchoring ablation.
FORMAT_V1 = ("Respond with JSON only, for example {\"earlier\": \"A\", \"elapsed_hours\": 1.5}. "
             "\"earlier\" must be \"A\" or \"B\".")
FORMAT_V2 = ("Respond with JSON only. The JSON object must have exactly two keys: \"earlier\", whose value is the "
             "single letter A or B (the image captured earlier in the day), and \"elapsed_hours\", whose value is "
             "a number (the hours between the two images).")


def hemisphere(lat):
    return "northern" if lat >= 0 else "southern"


FORMAT_COT = ("First think step by step in a few sentences: how do the shadows differ between the two images, and how does "
              "the sun move over a day at this location? Then finish with one final line that contains ONLY a JSON object "
              "with exactly two keys: \"earlier\" (the single letter A or B, the image captured earlier in the day) and "
              "\"elapsed_hours\" (a number, the hours between the two images).")


def build_prompts(lat, lon, date, version="v2", az=None, swap_hemisphere=False):
    FORMAT = FORMAT_V1 if version == "v1" else (FORMAT_COT if version == "v2cot" else FORMAT_V2)
    r0 = BASE + FORMAT
    r1 = BASE + f"The location is in the {hemisphere(lat)} hemisphere. " + FORMAT
    r2 = (BASE + f"The location is at latitude {lat:.1f} degrees ({'N' if lat >= 0 else 'S'}), longitude "
          f"{lon:.1f} degrees E, and the date is {date}. Times are local mean solar time. " + FORMAT)
    out = {"R0": r0, "R1": r1, "R2": r2}
    if az is not None:  # R3 (exploratory): perception is handed over, so only the reasoning remains
        stated = hemisphere(-lat if swap_hemisphere else lat)  # swap: deliberately state the opposite hemisphere
        out["R3"] = (BASE + f"The location is in the {stated} hemisphere. A separate measurement tool reports the "
                     "direction in which the shadows point, as a compass bearing in degrees clockwise from north "
                     f"(0 = north, 90 = east, 180 = south, 270 = west): Image A: {az[0]:.0f} degrees, Image B: "
                     f"{az[1]:.0f} degrees. " + FORMAT)
        out["R4"] = (BASE + f"The location is in the {hemisphere(lat)} hemisphere. A separate measurement tool reports the "
                     "direction in which the shadows point, as a compass bearing in degrees clockwise from north "
                     f"(0 = north, 90 = east, 180 = south, 270 = west): Image A: {az[0]:.0f} degrees, Image B: "
                     f"{az[1]:.0f} degrees. Rule: seen from above, the sun moves clockwise over the day in the northern "
                     "hemisphere, so shadow bearings INCREASE with time (a bearing wraps from 359 to 0); in the southern "
                     "hemisphere the sun moves counter-clockwise, so shadow bearings DECREASE with time (wrapping from 0 to "
                     "359). The two images are less than 6 hours apart, so compare the two bearings the short way around "
                     "the circle. " + FORMAT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--dt", default="0.5,1,2,4", help="allowed time gaps in hours, comma separated")
    ap.add_argument("--max-per-dt", type=int, default=4, help="max pairs per (scene, lat, date, dt)")
    ap.add_argument("--add-swapped", action="store_true",
                    help="also emit the A/B-swapped twin of every pair (enables the consistency metric)")
    ap.add_argument("--max-el", type=float, default=90.0,
                    help="drop images whose sun elevation exceeds this (deg). Near-zenith sun leaves almost no shadow "
                         "to read; 70 is a sensible choice (shadow length = height / tan(el))")
    ap.add_argument("--prompt-version", default="v2", choices=["v1", "v2", "v2cot"],
                    help="v2 (default): format stated without example values; v1: old prompt with an example answer; "
                         "v2cot: v2 plus step-by-step reasoning before the final JSON line (use --max-new-tokens 400+)")
    ap.add_argument("--swap-hemisphere", action="store_true",
                    help="with --with-azimuth: the R3 prompt states the OPPOSITE hemisphere (control: does the model use the stated hemisphere?)")
    ap.add_argument("--with-azimuth", action="store_true",
                    help="also write an R3 prompt that states the true shadow bearings (use a SEPARATE output file)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dts = [float(x) for x in args.dt.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    for r in rows:  # metadata written on Windows may contain backslashes
        r["image"] = r["image"].replace("\\", "/")
    n_before = len(rows)
    rows = [r for r in rows if r["sun_el_deg"] <= args.max_el]
    print(f"--max-el {args.max_el:g}: kept {len(rows)} of {n_before} images")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["lat"], r["date"])].append(r)

    rng = np.random.default_rng(args.seed)
    pairs = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda r: r["utc_minute"])
        for dt in dts:
            cands = []
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    if abs((b["utc_minute"] - a["utc_minute"]) / 60.0 - dt) < 1e-6:
                        cands.append((a, b))
            if len(cands) > args.max_per_dt:
                idx = rng.choice(len(cands), args.max_per_dt, replace=False)
                cands = [cands[k] for k in sorted(idx)]
            for early, late in cands:
                variants = [bool(rng.integers(2))]
                if args.add_swapped:
                    variants.append(not variants[0])
                for k, swap in enumerate(variants):
                    first, second = (late, early) if swap else (early, late)
                    pid = f"{len(pairs):06d}"
                    pairs.append(dict(
                        id=pid, twin_of=None if k == 0 else pairs[-1]["id"],
                        scene=early["scene"], group=f"s{early['scene']:03d}_lat{int(early['lat']):+03d}_{early['date']}",
                        image_a=first["image"], image_b=second["image"],
                        a_earlier=bool(not swap), elapsed_hours=float(dt), dt_bucket=float(dt),
                        lat=early["lat"], lon=early["lon"], hemisphere=hemisphere(early["lat"]),
                        date=early["date"], max_el_deg=float(max(early["sun_el_deg"], late["sun_el_deg"])),
                        prompt_version=args.prompt_version,
                        prompts=build_prompts(early["lat"], early["lon"], early["date"], args.prompt_version,
                                              az=((first["sun_az_deg"] + 180.0) % 360.0, (second["sun_az_deg"] + 180.0) % 360.0)
                                              if args.with_azimuth else None, swap_hemisphere=args.swap_hemisphere)))
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    n_s = sum(p["hemisphere"] == "southern" for p in pairs)
    print(f"wrote {len(pairs)} pairs to {args.out}  (southern: {n_s}, northern: {len(pairs) - n_s})")


if __name__ == "__main__":
    main()

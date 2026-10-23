#!/usr/bin/env python3
"""
make_probes.py - single-image "perception" probes.

Question: in which of 8 compass directions do the shadows point? (north-up image)
Ground truth = (sun azimuth + 180) mod 360, binned into 45-degree sectors centred on N, NE, E, ...
Purpose: a model that fails the ordering task but passes this probe has a REASONING failure; a model that
fails the probe cannot read shadow direction at all (PERCEPTION failure). Chance: 12.5% exact,
37.5% within one sector.

    python make_probes.py --data data --out probes.jsonl --n 200 --max-el 70 --seed 0
"""
import argparse
import json
import os
import random

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PROMPT = ("This is an overhead (satellite-style) image, north is up and east is to the right. The only light "
          "source is the sun, and the buildings cast shadows on the ground. In which compass direction do "
          "the shadows point (the direction from a building toward the end of its shadow)? Choose one of "
          "N, NE, E, SE, S, SW, W, NW. Respond with JSON only: one key, \"shadow_direction\", whose value is "
          "your chosen direction.")
PROMPT_V1 = PROMPT.replace("Respond with JSON only: one key, \"shadow_direction\", whose value is your chosen "
                           "direction.", "Respond with JSON only, for example {\"shadow_direction\": \"NE\"}.")


def sector_of(az_deg):
    return SECTORS[int(((az_deg % 360.0) + 22.5) // 45.0) % 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="probes.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-el", type=float, default=70.0)
    ap.add_argument("--prompt-version", default="v2", choices=["v1", "v2"],
                    help="v1 = old prompt with an example answer (ablation only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    rows = [r for r in rows if r["sun_el_deg"] <= args.max_el]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            shadow_az = (r["sun_az_deg"] + 180.0) % 360.0
            f.write(json.dumps(dict(
                id=f"probe{i:05d}", image=r["image"].replace("\\", "/"), prompt=PROMPT if args.prompt_version == "v2" else PROMPT_V1,
                prompt_version=args.prompt_version, scene=r["scene"],
                lat=r["lat"], sun_el_deg=r["sun_el_deg"], shadow_az_deg=shadow_az,
                truth=sector_of(shadow_az))) + "\n")
    print(f"wrote {len(rows)} probes to {args.out}")


if __name__ == "__main__":
    main()

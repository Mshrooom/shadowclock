#!/usr/bin/env python3
"""
make_samples.py - small JPEG thumbnails of the rendered OSM-derived scenes (for the README and the map popups).

For every city: the equinox date at the hours closest to 08:00, 12:00 and 16:00 local mean solar time, 256 px.

    python scripts/make_samples.py --data data_real --out data/samples
"""
import argparse
import json
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--date", default="2027-03-20")
    ap.add_argument("--hours", default="8,12,16")
    ap.add_argument("--px", type=int, default=256)
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    os.makedirs(args.out, exist_ok=True)
    index = []
    for city in sorted({m["city"] for m in meta}):
        ms = [m for m in meta if m["city"] == city and m["date"] == args.date]
        for h in (float(x) for x in args.hours.split(",")):
            if not ms:
                continue
            m = min(ms, key=lambda m: abs(m["utc_hour"] - h))  # 'utc_hour' holds the LOCAL MEAN solar hour (see README)
            name = f"{city}_{m['date']}_{int(m['utc_hour']):02d}00.jpg"
            Image.open(os.path.join(args.data, m["image"])).convert("RGB").resize((args.px, args.px), Image.LANCZOS).save(
                os.path.join(args.out, name), quality=82, optimize=True)
            index.append({"city": city, "file": name, "date": m["date"], "hour": m["utc_hour"],
                          "sun_az_deg": round(m["sun_az_deg"], 1), "sun_el_deg": round(m["sun_el_deg"], 1)})
    json.dump(index, open(os.path.join(args.out, "index.json"), "w"), indent=1)
    print(len(index), "thumbnails ->", args.out)


if __name__ == "__main__":
    main()

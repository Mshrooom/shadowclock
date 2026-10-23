#!/usr/bin/env python3
"""
build_city_results.py - per-city summary of the OSM-derived scenes and of a prediction file on them.

Reads the rendered dataset (metadata.jsonl + scenes/scene_NNN.json), the pairs file and any predictions file
(same format as score.py), and writes results/real_per_city.csv and results/real_per_city.json.

Accuracy is computed on DISTINCT pairs only (the A/B-swapped twin of a pair carries no new information),
with a Wilson 95% interval. n is small per city (roughly 40-80 distinct pairs sharing one scene), so read the
intervals, not just the point estimates.

    python scripts/build_city_results.py --data data_real --pairs results/pairs_real.jsonl \
        --preds results/preds_baseline_auto_real.jsonl --model analytic_baseline --out results
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
from score import parse, wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder with metadata.jsonl and scenes/")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--model", default="analytic_baseline")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl"))]
    scenes = {}
    for m in meta:
        if m["scene"] not in scenes:
            doc = json.load(open(os.path.join(args.data, "scenes", f"scene_{m['scene']}.json")))
            scenes[m["scene"]] = doc
    n_images = defaultdict(int)
    for m in meta:
        n_images[m["scene"]] += 1

    pairs = {}
    for l in open(args.pairs):
        r = json.loads(l)
        pairs[r["id"]] = r
    pred = defaultdict(dict)  # regime -> id -> (earlier, elapsed)
    for l in open(args.preds):
        r = json.loads(l)
        if r.get("model", args.model) != args.model:
            continue
        pred[r["regime"]][r["id"]] = parse(r["response"])

    rows = []
    for sc, doc in sorted(scenes.items()):
        row = {
            "scene": sc, "city": doc["city"], "lat": doc["lat"], "lon": doc["lon"],
            "n_images": n_images[sc], "n_buildings": doc["n_buildings"],
            "built_fraction": round(doc["built_fraction_of_field"], 3),
            "explicit_height_fraction": round(doc["explicit_height_fraction"], 3),
            "max_height_m": round(doc["max_height_m"], 1),
        }
        ids = [i for i, p in pairs.items() if p["scene"] == sc and p["twin_of"] is None]
        row["n_distinct_pairs"] = len(ids)
        for reg in ("R1", "R2"):
            k = n = 0
            err = []
            for i in ids:
                got = pred[reg].get(i)
                if got is None or got[0] not in ("A", "B"):
                    continue
                n += 1
                k += int(got[0] == ("A" if pairs[i]["a_earlier"] else "B"))
                if got[1] is not None:
                    err.append(abs(float(got[1]) - pairs[i]["elapsed_hours"]))
            lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
            row[f"{reg}_acc"] = round(100.0 * k / n, 1) if n else None
            row[f"{reg}_ci_lo"] = round(100.0 * lo, 1) if n else None
            row[f"{reg}_ci_hi"] = round(100.0 * hi, 1) if n else None
            row[f"{reg}_n"] = n
            row[f"{reg}_mae_h"] = round(sum(err) / len(err), 2) if err else None
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "real_per_city.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"model": args.model, "cities": rows}, open(os.path.join(args.out, "real_per_city.json"), "w"), indent=1)
    print(f"{'city':<13}{'lat':>7}{'bldg':>6}{'expl%':>7}{'R1 acc [95% CI]':>24}{'R2 acc':>9}{'n':>5}")
    for r in rows:
        print(f"{r['city']:<13}{r['lat']:>7.1f}{r['n_buildings']:>6}{100 * r['explicit_height_fraction']:>7.0f}"
              f"{r['R1_acc']:>9.1f} [{r['R1_ci_lo']:>5.1f},{r['R1_ci_hi']:>5.1f}]{r['R2_acc']:>9.1f}{r['R1_n']:>5}")


if __name__ == "__main__":
    main()

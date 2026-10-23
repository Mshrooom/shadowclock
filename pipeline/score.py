#!/usr/bin/env python3
"""
score.py - score predictions of any model (VLM or the analytic baseline).

Predictions file: JSONL, one line per (pair, regime[, model]):
    {"id": "000123", "regime": "R1", "model": "qwen2.5-vl-7b", "response": "<raw model text>"}
The response must contain a JSON object with "earlier" ("A"/"B") and optionally "elapsed_hours".
Unparseable answers count as wrong (and are reported as the invalid rate); they are never dropped.

    python score.py --pairs pairs.jsonl --preds preds_baseline.jsonl [--preds more.jsonl ...] [--json out.json]

Reported per (model, regime): ordering accuracy overall, by hemisphere and by time gap with 95% Wilson
intervals; MAE of the elapsed-time estimate (hours); fraction of elapsed estimates within 0.5 h;
invalid rate; and, if the pairs were built with --add-swapped, swap consistency (a model that
really reads the shadows must flip its answer when A and B are swapped).
Chance level for ordering is 50%.
"""
import argparse
import json
import math
import re
from collections import defaultdict

JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def parse(response):
    """Return (earlier 'A'|'B'|None, elapsed float|None)."""
    for m in reversed(JSON_RE.findall(response or "")):  # last valid object wins (matters for step-by-step answers)
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        e = str(obj.get("earlier", "")).strip().upper()[:1]
        earlier = e if e in ("A", "B") else None
        try:
            elapsed = float(obj.get("elapsed_hours"))
        except (TypeError, ValueError):
            elapsed = None
        if earlier is not None or elapsed is not None:
            return earlier, elapsed
    return None, None


def fmt(k, n):
    if n == 0:
        return "   n/a        "
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--preds", required=True, action="append")
    ap.add_argument("--json", default=None, help="optional path for a machine-readable summary")
    ap.add_argument("--only-predicted", action="store_true",
                    help="score only pairs that have a prediction (use for pilots on a subset); default counts missing pairs as wrong")
    args = ap.parse_args()

    pairs = {}
    for l in open(args.pairs):
        p = json.loads(l)
        pairs[p["id"]] = p
    preds = defaultdict(dict)  # (model, regime) -> id -> (earlier, elapsed)
    for path in args.preds:
        for l in open(path):
            r = json.loads(l)
            if r["id"] not in pairs:
                continue
            preds[(r.get("model", path), r["regime"])][r["id"]] = parse(r["response"])

    summary = {}
    for (model, regime), table in sorted(preds.items()):
        use = {k: v for k, v in pairs.items() if k in table} if args.only_predicted else pairs
        missing = len(use) - len(table)
        buckets = defaultdict(lambda: [0, 0])  # name -> [correct, total]
        maes, within = [], []
        invalid = 0
        for pid, p in use.items():
            earlier, elapsed = table.get(pid, (None, None))
            truth = "A" if p["a_earlier"] else "B"
            ok = earlier == truth
            invalid += earlier is None
            el = p.get("max_el_deg")
            el_name = None if el is None else ("sun<=45" if el <= 45 else "sun45-70" if el <= 70 else "sun>70")
            for name in [n_ for n_ in ("all", f"hemi={p['hemisphere']}", f"dt={p['dt_bucket']:g}h", el_name) if n_]:
                buckets[name][0] += ok
                buckets[name][1] += 1
            if elapsed is not None:
                err = abs(elapsed - p["elapsed_hours"])
                maes.append(err)
                within.append(err <= 0.5)
        # swap consistency: twin pairs must receive opposite answers
        cons_k = cons_n = 0
        for pid, p in use.items():
            if p.get("twin_of"):
                a, b = table.get(pid, (None, None))[0], table.get(p["twin_of"], (None, None))[0]
                if a is not None and b is not None:
                    cons_n += 1
                    cons_k += a != b
        n = len(use)
        print(f"\n=== model: {model}   regime: {regime}   pairs: {n}   missing: {missing}   "
              f"invalid: {100 * invalid / n:.1f}% ===")
        print("  ordering accuracy % [95% CI]  (chance = 50)")
        for name in sorted(buckets, key=lambda s: (s != "all", s)):
            k, t = buckets[name]
            print(f"    {name:<14s} {fmt(k, t)}   n={t}")
        if maes:
            print(f"  elapsed-time MAE: {sum(maes) / len(maes):.2f} h   within 0.5 h: {100 * sum(within) / len(within):.1f}%")
        if cons_n:
            print(f"  swap consistency: {fmt(cons_k, cons_n)}   n={cons_n}")
        summary[f"{model}|{regime}"] = dict(
            n=n, invalid_rate=invalid / n,
            accuracy={k: v[0] / v[1] for k, v in buckets.items()},
            elapsed_mae_h=(sum(maes) / len(maes)) if maes else None,
            swap_consistency=(cons_k / cons_n) if cons_n else None)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"\nsummary written to {args.json}")


if __name__ == "__main__":
    main()

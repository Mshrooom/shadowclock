#!/usr/bin/env python3
"""
hemisphere_flip.py - does the model USE the hemisphere stated in the R3 prompt?

Inputs: the R3 prediction file from the normal prompt and from the prompt that states the OPPOSITE hemisphere
(make_pairs.py --with-azimuth --swap-hemisphere), same pair ids.
Under a false hemisphere a correct reasoner must give the opposite ordering for every pair whose azimuth change is
well below 180 degrees (all 1, 2, 4 h pairs). So:
  flip rate            = fraction of pairs whose answer changes between the two prompts (correct reasoner: ~100%)
  accuracy (true)      = answers vs the real ordering, normal prompt
  agreement with false premise = swapped-prompt answers vs the ordering implied by the FALSE hemisphere
A model that ignores the hemisphere has a low flip rate and low agreement; a model that applies one fixed rotation
direction (e.g. always the northern rule) has a low flip rate too.

    python hemisphere_flip.py --pairs pairs_r3.jsonl --normal r3_normal.jsonl --swapped r3_swapped.jsonl
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import parse, wilson  # noqa: E402


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["regime"] == "R3":
            out[r["id"]] = parse(r["response"])[0]
    return out


def line(label, k, n):
    if n == 0:
        return f"  {label:<46s} n/a"
    lo, hi = wilson(k / 2, n / 2)  # twins are not independent: interval on n/2 distinct pairs
    return f"  {label:<46s} {100 * k / n:5.1f}% [{100 * lo:4.1f},{100 * hi:5.1f}]  n={n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--swapped", required=True)
    args = ap.parse_args()
    pairs = {json.loads(l)["id"]: json.loads(l) for l in open(args.pairs)}
    a, b = load(args.normal), load(args.swapped)
    ids = [i for i in pairs if i in a and i in b and a[i] and b[i]]
    flips = sum(a[i] != b[i] for i in ids)
    acc_true = sum(a[i] == ("A" if pairs[i]["a_earlier"] else "B") for i in ids)
    agree_false = sum(b[i] == ("B" if pairs[i]["a_earlier"] else "A") for i in ids)  # opposite of the true ordering
    print(f"pairs answered in both files: {len(ids)}  (invalid/missing excluded)")
    print(line("flip rate (correct reasoner: ~100%)", flips, len(ids)))
    print(line("accuracy vs TRUE ordering, normal prompt", acc_true, len(ids)))
    print(line("agreement with the FALSE premise, swapped prompt", agree_false, len(ids)))
    for h in ("northern", "southern"):
        sub = [i for i in ids if pairs[i]["hemisphere"] == h]
        print(f"  [{h} pairs]" + line("flip rate", sum(a[i] != b[i] for i in sub), len(sub))[1:])
    print("\nreading: a model that uses the stated hemisphere flips (rate near 100%) and follows the false premise; "
          "a low flip rate means it ignores the stated hemisphere or applies a fixed rotation direction.")


if __name__ == "__main__":
    main()

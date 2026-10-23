#!/usr/bin/env python3
"""
evaluate_hypotheses.py - apply the frozen thresholds in HYPOTHESES.md mechanically.

    python evaluate_hypotheses.py --pairs pairs.jsonl --probes probes.jsonl \
        --preds preds_baseline.jsonl --preds qwen.jsonl --preds smol.jsonl --json verdicts.json
    # pilots on a subset: add --allow-partial (scores only the pairs a model answered)

Rules (identical to HYPOTHESES.md; change them there AND here BEFORE running any model):
  gate : a (model, regime) with > 10% unparseable answers is NOT SCORED
  H1   : R0.  SUPPORTED if north >= 60 and north - south >= 20; FALSIFIED if south >= 60 or |north - south| < 10; else INCONCLUSIVE
  H2   : R1 accuracy on 1 h pairs.  SOLVED >= 90, CHANCE <= 55, else PARTIAL.
         roll-up: SUPPORTED if >= half of scored models are PARTIAL; FALSIFIED if all are SOLVED; else INCONCLUSIVE
  H3   : elapsed time.  MAE_ns = MAE of the best constant guess.  SUPPORTED if MAE_R1 >= 0.75*MAE_ns and MAE_R2 >= 0.5 h;
         FALSIFIED if MAE_R2 < 0.5 h; else INCONCLUSIVE.  roll-up as H2 (>= half supported / all falsified)
  H4   : R1 swap consistency.  SUPPORTED if < 90%; FALSIFIED if >= 95%; else INCONCLUSIVE.  roll-up as H3
  probe: perception OK if within-one-sector accuracy >= 70%
The analytic baseline is printed as a reference and excluded from roll-ups.
Accuracy intervals: Wilson interval on the number of DISTINCT pairs (twins halve n) and a bootstrap over
scenes (with only ~8 scenes this is crude; treat both as optimistic).
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import parse, wilson  # noqa: E402

REFERENCE_MODELS = {"analytic_baseline"}
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
JSON_RE = re.compile(r"\{[^{}]*\}", re.S)
MAX_INVALID = 0.10


def parse_probe(response):
    for m in JSON_RE.findall(response or ""):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        d = str(obj.get("shadow_direction", "")).strip().upper()
        if d in SECTORS:
            return d
    return None


def fmt_pct(x):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:5.1f}"


def acc_ci(recs, twins):
    n = len(recs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    k = sum(r["correct"] for r in recs)
    p = k / n
    n_eff = n / 2 if twins else n
    lo, hi = wilson(p * n_eff, n_eff)
    by = defaultdict(lambda: [0, 0])
    for r in recs:
        by[r["scene"]][0] += r["correct"]
        by[r["scene"]][1] += 1
    arr = np.array(list(by.values()), dtype=float)
    if len(arr) >= 2:
        idx = np.random.default_rng(0).integers(0, len(arr), (1000, len(arr)))
        s = arr[idx, 0].sum(1) / arr[idx, 1].sum(1)
        blo, bhi = np.percentile(s, [2.5, 97.5])
    else:
        blo = bhi = float("nan")
    return p, lo, hi, blo, bhi


def line_acc(label, recs, twins):
    p, lo, hi, blo, bhi = acc_ci(recs, twins)
    return f"{label} {fmt_pct(p)}% [Wilson {fmt_pct(lo)},{fmt_pct(hi)}] [scene-boot {fmt_pct(blo)},{fmt_pct(bhi)}] n={len(recs)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--preds", required=True, action="append")
    ap.add_argument("--allow-partial", action="store_true", help="score only pairs a model answered (pilots)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pairs = {}
    for l in open(args.pairs):
        p = json.loads(l)
        pairs[p["id"]] = p
    twins = any(p.get("twin_of") for p in pairs.values())
    probes = {}
    if args.probes:
        for l in open(args.probes):
            q = json.loads(l)
            probes[q["id"]] = q

    raw = defaultdict(lambda: defaultdict(dict))  # model -> regime -> id -> response
    for path in args.preds:
        for l in open(path):
            r = json.loads(l)
            raw[r.get("model", path)][r["regime"]][r["id"]] = r["response"]

    elapsed_all = sorted(p["elapsed_hours"] for p in pairs.values())
    const = elapsed_all[len(elapsed_all) // 2]
    mae_ns = float(np.mean([abs(e - const) for e in elapsed_all]))
    print(f"no-skill elapsed-time reference: constant guess {const:g} h -> MAE {mae_ns:.3f} h")

    results, verdicts = {}, defaultdict(dict)
    for model in sorted(raw):
        reg = raw[model]
        out = dict(reference=model in REFERENCE_MODELS)
        print(f"\n================ {model}{'  (reference, excluded from roll-ups)' if out['reference'] else ''}")

        recs_by_regime, gate = {}, {}
        for regime in ("R0", "R1", "R2"):
            if regime not in reg:
                gate[regime] = "NOT RUN"
                continue
            table = reg[regime]
            ids = [i for i in pairs if i in table] if args.allow_partial else list(pairs)
            missing = sum(i not in table for i in ids)
            recs = []
            for i in ids:
                p = pairs[i]
                earlier, elapsed = parse(table.get(i)) if i in table else (None, None)
                recs.append(dict(
                    id=i, scene=p["scene"], hemi=p["hemisphere"], dt=p["dt_bucket"], earlier=earlier,
                    correct=earlier == ("A" if p["a_earlier"] else "B"), valid=earlier is not None,
                    err=None if elapsed is None else abs(elapsed - p["elapsed_hours"]), twin_of=p.get("twin_of")))
            inv = sum(not r["valid"] for r in recs) / max(len(recs), 1)
            recs_by_regime[regime] = recs
            gate[regime] = "OK" if (inv <= MAX_INVALID and missing == 0) else (
                f"NOT SCORED (invalid {100 * inv:.0f}%, missing {missing})")
            print(f"  {regime}: {len(recs)} pairs, invalid {100 * inv:.1f}%, missing {missing}  -> {gate[regime]}")

        # ---- H1
        v1 = "NOT SCORED"
        if gate.get("R0") == "OK":
            R = recs_by_regime["R0"]
            north = [r for r in R if r["hemi"] == "northern"]
            south = [r for r in R if r["hemi"] == "southern"]
            pn, ps = acc_ci(north, twins)[0], acc_ci(south, twins)[0]
            if ps >= 0.60 or abs(pn - ps) < 0.10:
                v1 = "FALSIFIED"
            elif pn >= 0.60 and pn - ps >= 0.20:
                v1 = "SUPPORTED"
            else:
                v1 = "INCONCLUSIVE"
            print("  H1 (R0): " + line_acc("north", north, twins) + "\n           " + line_acc("south", south, twins) + f"\n           -> {v1}")
        verdicts["H1"][model] = v1

        # ---- H2
        v2 = "NOT SCORED"
        if gate.get("R1") == "OK":
            R = recs_by_regime["R1"]
            one = [r for r in R if r["dt"] == 1.0]
            p1 = acc_ci(one, twins)[0] if one else acc_ci(R, twins)[0]
            v2 = "SOLVED" if p1 >= 0.90 else ("CHANCE" if p1 <= 0.55 else "PARTIAL")
            by_dt = {dt: acc_ci([r for r in R if r["dt"] == dt], twins)[0] for dt in sorted({r["dt"] for r in R})}
            print("  H2 (R1): " + line_acc("1h-pairs" if one else "all", one or R, twins)
                  + "\n           " + line_acc("overall (descriptive)", R, twins)
                  + "\n           by gap: " + ", ".join(f"{dt:g}h {fmt_pct(a)}%" for dt, a in by_dt.items())
                  + f"\n           -> {v2}")
        verdicts["H2"][model] = v2

        # ---- H3
        v3 = "NOT SCORED"
        if gate.get("R1") == "OK" and gate.get("R2") == "OK":
            def mae(regime):
                e = [r["err"] for r in recs_by_regime[regime] if r["err"] is not None]
                return (float(np.mean(e)) if e else float("nan")), len(e)
            m1, n1 = mae("R1")
            m2, n2 = mae("R2")
            if math.isnan(m2) or m2 < 0.5:
                v3 = "FALSIFIED" if not math.isnan(m2) else "NOT SCORED"
            elif not math.isnan(m1) and m1 >= 0.75 * mae_ns:
                v3 = "SUPPORTED"
            else:
                v3 = "INCONCLUSIVE"
            print(f"  H3: MAE R1 {m1:.2f} h (n={n1}), MAE R2 {m2:.2f} h (n={n2}), no-skill {mae_ns:.2f} h -> {v3}")
        verdicts["H3"][model] = v3

        if gate.get("R2") == "OK":  # descriptive only: ordering accuracy when latitude/longitude/date are given
            print("  R2 ordering (descriptive): " + line_acc("overall", recs_by_regime["R2"], twins))

        # ---- H4
        v4 = "NOT SCORED"
        if gate.get("R1") == "OK":
            R = {r["id"]: r for r in recs_by_regime["R1"]}
            both = [(r, R[r["twin_of"]]) for r in R.values() if r["twin_of"] and r["twin_of"] in R]
            both = [(a, b) for a, b in both if a["valid"] and b["valid"]]
            valid_ans = [r["earlier"] for r in R.values() if r["valid"]]
            a_rate = valid_ans.count("A") / max(len(valid_ans), 1)
            if both:
                cons = sum(a["earlier"] != b["earlier"] for a, b in both) / len(both)
                v4 = "SUPPORTED" if cons < 0.90 else ("FALSIFIED" if cons >= 0.95 else "INCONCLUSIVE")
                nv = len(valid_ans)
                alo, ahi = wilson(a_rate * nv, nv)
                print(f"  H4 (R1): swap consistency {100 * cons:.1f}% (n={len(both)} twin pairs), answers 'A' {100 * a_rate:.0f}% "
                      f"[Wilson {100 * alo:.0f},{100 * ahi:.0f}] -> {v4}\n"
                      "           note: a model guessing at random also has consistency ~50%, so < 90% does not by itself "
                      "show position bias; use the A-rate interval for that")
            else:
                print("  H4: no twin pairs (build pairs with --add-swapped) -> NOT SCORED")
        verdicts["H4"][model] = v4

        # ---- R3 / R4 (exploratory, descriptive only): shadow bearings (and for R4 also the sun-motion rule) are given in the prompt
        for xr in ("R3", "R4"):
          if xr in reg:
              ids = [i for i in pairs if i in reg[xr]] if args.allow_partial else list(pairs)
              R3 = []
              for i in ids:
                  p = pairs[i]
                  earlier, _ = parse(reg[xr].get(i)) if i in reg[xr] else (None, None)
                  R3.append(dict(scene=p["scene"], hemi=p["hemisphere"], correct=earlier == ("A" if p["a_earlier"] else "B"),
                                 valid=earlier is not None))
              inv3 = sum(not r["valid"] for r in R3) / max(len(R3), 1)
              print(f"  {xr} (exploratory, true shadow bearings given in text{' + the sun-motion rule' if xr == 'R4' else ''}; invalid {100 * inv3:.1f}%):")
              print("     " + line_acc("overall", R3, twins))
              for h in ("northern", "southern"):
                  print("     " + line_acc(h, [r for r in R3 if r["hemi"] == h], twins))
              if xr == "R3":
                  print("     reading: well above 50% here but at chance in R1 => the failure is perception; at chance here too => reasoning fails as well")
              else:
                  print("     reading: well above 50% here => the model lacked the sun-motion rule but can apply it; still near/below 50% => it cannot apply an explicit rule")
              p3, lo3, hi3 = acc_ci(R3, twins)[:3]
              if hi3 < 0.5:
                  print("     NOTE: the interval lies entirely BELOW 50% -> answers are systematically wrong, i.e. the model applies a"
                        " consistent but incorrect rule (see the agreement check in the notebook), not random guessing")

        # ---- probe
        perception = None
        if probes and "PROBE" in reg:
            ids = [i for i in probes if i in reg["PROBE"]]
            exact = within = invalid = 0
            for i in ids:
                d = parse_probe(reg["PROBE"][i])
                if d is None:
                    invalid += 1
                    continue
                dist = abs(SECTORS.index(d) - SECTORS.index(probes[i]["truth"]))
                dist = min(dist, 8 - dist)
                exact += dist == 0
                within += dist <= 1
            n = max(len(ids), 1)
            perception = within / n >= 0.70
            # honest reference: the sectors are NOT equally likely (E/W shadows are the most common), so the right
            # "no skill" level is the best CONSTANT answer on these probes, not the uniform 1/8 and 3/8
            truth_cnt = Counter(probes[i]["truth"] for i in ids)
            const_exact = max(truth_cnt.values()) / n
            const_within = max(sum(truth_cnt[SECTORS[(k + o) % 8]] for o in (-1, 0, 1)) for k in range(8)) / n
            ans_cnt = Counter(d for d in (parse_probe(reg["PROBE"][i]) for i in ids) if d)
            top = ans_cnt.most_common(1)[0] if ans_cnt else ("-", 0)
            print(f"  probe (shadow direction, n={len(ids)}): exact {100 * exact / n:.1f}% (uniform chance 12.5; best constant "
                  f"answer on these probes {100 * const_exact:.1f}%), within one sector {100 * within / n:.1f}% (uniform 37.5; "
                  f"best constant {100 * const_within:.1f}%), invalid {100 * invalid / n:.0f}% "
                  f"-> perception {'OK' if perception else 'FAILS'}")
            print(f"           most frequent answer: '{top[0]}' = {100 * top[1] / n:.0f}% of the probes "
                  f"(a model reading shadows should be well above the best-constant levels)")
        if v2 in ("PARTIAL", "CHANCE") and perception is not None:
            print("  diagnosis: " + ("reasoning failure (reads shadow direction, cannot turn it into time order)"
                                      if perception else "perception failure (cannot read shadow direction)"))
        out.update(gate=gate, H1=v1, H2=v2, H3=v3, H4=v4, perception_ok=perception)
        results[model] = out

    print("\n================ ROLL-UP (reference models excluded)")
    roll = {}
    models = [m for m in results if not results[m]["reference"]]
    for h in ("H1", "H2", "H3", "H4"):
        vs = [results[m][h] for m in models if results[m][h] != "NOT SCORED"]
        n = len(vs)
        if n == 0:
            roll[h] = "NOT SCORED (no scored model)"
        elif h == "H2":
            roll[h] = ("SUPPORTED" if sum(v == "PARTIAL" for v in vs) >= n / 2
                       else "FALSIFIED" if all(v == "SOLVED" for v in vs) else "INCONCLUSIVE")
            roll[h] += f"  [{sum(v == 'PARTIAL' for v in vs)} partial, {sum(v == 'CHANCE' for v in vs)} at chance, {sum(v == 'SOLVED' for v in vs)} solved of {n}]"
        else:
            roll[h] = ("SUPPORTED" if sum(v == "SUPPORTED" for v in vs) >= n / 2
                       else "FALSIFIED" if all(v == "FALSIFIED" for v in vs) else "INCONCLUSIVE")
            roll[h] += f"  [{sum(v == 'SUPPORTED' for v in vs)} supported, {sum(v == 'FALSIFIED' for v in vs)} falsified of {n}]"
        print(f"  {h}: {roll[h]}")
    print("  H5 (remote-sensing VLMs vs general VLMs): not testable with the models run here; drop it unless you add "
          "such models.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(no_skill_mae_h=mae_ns, models=results, rollup=roll), f, indent=1)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_figures.py - README / paper figures from results/*.json and the rendered images.

    python scripts/make_figures.py --results results --data data_real --out results/figures

Writes fig_overview.png, fig_per_city.png and fig_samples.png. Colours: blue = the analytic (pixel) baseline,
orange = open VLMs, grey = reference lines. Every bar and point is labelled directly; colour is never the only cue.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


NAMES = {"newyork": "New York", "buenosaires": "Buenos Aires", "capetown": "Cape Town"}


def nice(city):
    return NAMES.get(city, city.capitalize())


def fig_overview(res, out):
    base = res["analytic_baseline"]
    real = res["real_footprint_dataset_baseline"]
    rows = [("Analytic baseline (pixels), synthetic", base["R1_acc"], None, BLUE, "")]
    rows.append(("Analytic baseline (pixels), OSM-derived", real["R1_acc"], None, BLUE, ""))
    for m in res["models_R1"]:
        note = f"answers 'A' {m['a_rate']:.0f}%"
        if m.get("ci_lo") is not None:
            rows.append((m["label"], m["acc"], (m["ci_lo"], m["ci_hi"]), ORANGE, note))
        else:
            rows.append((m["label"], m["acc"], None, ORANGE, note + ", CI not recorded"))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, col, note) in zip(y, rows):
        ax.barh(yi, v, color=col, height=0.55)
        if ci:
            ax.plot(ci, [yi, yi], color=INK, lw=1.4)
            ax.plot([ci[0]] * 2, [yi - 0.12, yi + 0.12], color=INK, lw=1.4)
            ax.plot([ci[1]] * 2, [yi - 0.12, yi + 0.12], color=INK, lw=1.4)
        txt = f"{v:.1f}%" + (f"   ({note})" if note else "")
        ax.text(max(v, ci[1] if ci else v) + 1.5, yi, txt, va="center", fontsize=9, color=INK)
    ax.axvline(50, color=GREY, lw=1, ls=(0, (4, 3)))
    ax.text(50.8, -0.62, "chance (50%)", color=MUTED, fontsize=8.5, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("which image is earlier? accuracy in R1 (hemisphere given), %")
    ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.set_title("Pixel baseline solves it; the open VLMs tested sit at chance", loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_per_city(csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("R1_acc", "R1_ci_lo", "R1_ci_hi", "built_fraction", "n_buildings", "lat", "R1_n"):
            r[k] = float(r[k])
    order = sorted(rows, key=lambda r: r["R1_acc"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(order))
    for yi, r in zip(y, order):
        col = BLUE if r["lat"] >= 0 else ORANGE
        a1.plot([r["R1_ci_lo"], r["R1_ci_hi"]], [yi, yi], color=col, lw=2, alpha=0.55)
        a1.plot(r["R1_acc"], yi, "o", color=col, ms=7, mec="white", mew=1.2)
        a1.text(103, yi, f"{r['R1_acc']:.0f}%", va="center", fontsize=8.5, color=INK)
    a1.axvline(50, color=GREY, lw=1, ls=(0, (4, 3)))
    a1.set_yticks(y)
    a1.set_yticklabels([nice(r["city"]) for r in order])
    a1.set_xlim(30, 112)
    a1.set_xlabel("baseline ordering accuracy, R1, % (95% Wilson CI)")
    a1.set_title("By city (about 30 distinct pairs each)", loc="left", fontsize=11)
    a1.text(111, len(order) - 0.2, "blue = northern hemisphere, orange = southern", fontsize=8.5, color=MUTED, ha="right")
    a1.set_ylim(-0.7, len(order) + 0.2)
    for r in rows:
        col = BLUE if r["lat"] >= 0 else ORANGE
        a2.plot(100 * r["built_fraction"], r["R1_acc"], "o", color=col, ms=7, mec="white", mew=1.2)
    offs = {"santiago": (-4, 7, "right"), "delhi": (4, -13, "left"), "buenosaires": (0, 12, "center"), "capetown": (6, 5, "left")}
    for r in rows:
        dx, dy, ha = offs.get(r["city"], (5, 5, "left"))
        a2.annotate(nice(r["city"]), (100 * r["built_fraction"], r["R1_acc"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=MUTED, ha=ha, va="bottom")
    a2.axhline(50, color=GREY, lw=1, ls=(0, (4, 3)))
    a2.set_xlabel("built-up share of the 256 m x 256 m field, %")
    a2.set_ylabel("baseline accuracy, R1, %")
    a2.set_ylim(40, 105)
    a2.set_title("Denser scenes score lower (12 scenes, descriptive)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_samples(data, out):
    meta = [json.loads(l) for l in open(os.path.join(data, "metadata.jsonl"))]
    picks = [("delhi", "2027-03-20"), ("chicago", "2027-06-21"), ("santiago", "2027-12-21"), ("melbourne", "2027-12-21")]
    hours = (8.0, 12.0, 16.0)
    fig, axes = plt.subplots(len(picks), len(hours), figsize=(8.4, 11.0))
    for i, (city, date) in enumerate(picks):
        ms = [m for m in meta if m["city"] == city and m["date"] == date]
        for j, h in enumerate(hours):
            m = min(ms, key=lambda m: abs(m["utc_hour"] - h))  # 'utc_hour' stores the local mean solar hour
            ax = axes[i, j]
            ax.imshow(Image.open(os.path.join(data, m["image"])).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{nice(city)} {date[5:]}  {int(m['utc_hour']):02d}:00 local mean solar\n"
                         f"sun az {m['sun_az_deg']:.0f} deg, el {m['sun_el_deg']:.0f} deg", fontsize=8.5, color=MUTED)
    fig.suptitle("Same place, same date, different time: north-up nadir renders of OSM footprints", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default=None, help="rendered OSM-derived dataset (for the sample montage); skipped if absent")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = json.load(open(os.path.join(args.results, "synthetic_results.json")))
    fig_overview(res, os.path.join(args.out, "fig_overview.png"))
    fig_per_city(os.path.join(args.results, "real_per_city.csv"), os.path.join(args.out, "fig_per_city.png"))
    if args.data:
        fig_samples(args.data, os.path.join(args.out, "fig_samples.png"))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()

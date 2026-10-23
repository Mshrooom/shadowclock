#!/usr/bin/env python3
"""
baseline.py - analytic "sundial" baseline (no learning, uses only the pixels + the same
metadata a VLM is given in each regime).

Pipeline
  1. Ground = green pixels. Shadow = ground pixels darker than the Otsu threshold. Roof = non-ground.
  2. For every roof R whose cast shadow S is connected to it and touches no other roof, the union
     U = R + S is the roof footprint swept along the shadow direction. The "excess width" of U over R
     is L*|cos(phi - theta)| in projection direction phi, so theta (shadow axis) is found with a
     matched filter and its sign from the centroid offset of S relative to R.
  3. Sun azimuth = shadow azimuth + 180 deg; buildings are combined with a robust circular mean.
  4. Regime logic (image up = north, so shadow azimuth is directly a compass bearing):
       R0  hemisphere NOT given -> assumes northern hemisphere (the prior a model without hemisphere
           information has to fall back on). Elapsed time = |delta azimuth| / 15 deg per hour (crude).
       R1  hemisphere given     -> ordering from the rotation direction of the sun azimuth
           (increasing azimuth = later in the north, decreasing = later in the south).
       R2  latitude + date given -> invert the solar ephemeris (solar.py) to a clock time for each
           image; ordering and elapsed time come from the two clock times.

Output: predictions.jsonl lines {"id","regime","model","response"} where response is the JSON string
{"earlier": "A"|"B", "elapsed_hours": x}, i.e. exactly what score.py expects from a VLM.

    python baseline.py --data data --pairs pairs.jsonl --out preds_baseline.jsonl
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solar import solar_position, wrap180  # noqa: E402


# ----------------------------------------------------------------------------- image helpers
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0


def rgb_to_hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rc = m & (mx == r)
    gc = m & ~rc & (mx == g)
    bc = m & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6.0
    h[gc] = (b - r)[gc] / d[gc] + 2.0
    h[bc] = (r - g)[bc] / d[bc] + 4.0
    h /= 6.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    return centers[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


# ----------------------------------------------------------------------------- azimuth estimation
PHIS = np.deg2rad(np.arange(0, 180, 2.0))
THETAS = np.deg2rad(np.arange(0, 180, 1.0))
COS_TABLE = np.abs(np.cos(PHIS[None, :] - THETAS[:, None]))  # (n_theta, n_phi)


def widths(pts):
    proj = pts[:, 0:1] * np.cos(PHIS)[None, :] + pts[:, 1:2] * np.sin(PHIS)[None, :]
    return proj.max(0) - proj.min(0)


def shadow_axis(roof_pts, shadow_pts):
    """Return (theta_rad, length_px): shadow direction, angle from +x (east) counter-clockwise."""
    excess = widths(np.vstack([roof_pts, shadow_pts])) - widths(roof_pts)
    scores = COS_TABLE @ excess
    k = int(np.argmax(scores))
    theta = THETAS[k]
    u = np.array([math.cos(theta), math.sin(theta)])
    if (shadow_pts.mean(0) - roof_pts.mean(0)) @ u < 0:
        theta += math.pi
    length = float(excess[int(np.argmin(np.abs(PHIS - (theta % math.pi))))])
    return theta, length


def estimate_sun_azimuth(rgb, min_roof_px=100, min_shadow_px=60):
    """Return dict(az, n, coherence) or None. az = sun azimuth in deg clockwise from north."""
    h, s, v = rgb_to_hsv(rgb)
    ground = (h > 0.2) & (h < 0.5) & (s > 0.2)
    if ground.sum() < 200:
        return None
    thr = otsu(v[ground])
    lo, hi = v[ground & (v < thr)], v[ground & (v >= thr)]
    if lo.size < 30 or hi.size < 30 or hi.mean() - lo.mean() < 0.08:
        return None  # no clear lit/shadow bimodality
    shadow = ground & (v < thr)
    roof = ~ground
    st = np.ones((3, 3), dtype=int)
    lab_r, n_r = ndimage.label(roof, structure=st)
    lab_s, n_s = ndimage.label(shadow, structure=st)
    area_r = np.bincount(lab_r.ravel(), minlength=n_r + 1)

    touching = {}  # shadow component -> set of roof components it touches
    for sid in range(1, n_s + 1):
        comp = lab_s == sid
        if comp.sum() < min_shadow_px:
            continue
        ring = ndimage.binary_dilation(comp, structure=st, iterations=2) & ~comp
        counts = np.bincount(lab_r[ring], minlength=n_r + 1)
        counts[0] = 0
        touching[sid] = {int(r) for r in np.nonzero(counts >= 6)[0] if area_r[r] >= min_roof_px}

    H, W = roof.shape
    estimates = []  # (unit vector of sun direction, weight)
    for rid in range(1, n_r + 1):
        if area_r[rid] < min_roof_px:
            continue
        sids = [sid for sid, rs in touching.items() if rs == {rid}]
        if not sids:
            continue
        r_mask = lab_r == rid
        s_mask = np.isin(lab_s, sids)
        u_mask = r_mask | s_mask
        if u_mask[0].any() or u_mask[-1].any() or u_mask[:, 0].any() or u_mask[:, -1].any():
            continue  # clipped by the image border
        rr, rc = np.nonzero(r_mask)
        sr, sc = np.nonzero(s_mask)
        roof_pts = np.stack([rc, -rr], 1).astype(float)
        shadow_pts = np.stack([sc, -sr], 1).astype(float)
        theta, length = shadow_axis(roof_pts, shadow_pts)
        if length < 3.0:
            continue
        shadow_az = (90.0 - math.degrees(theta)) % 360.0  # compass bearing of the shadow
        sun_az = math.radians((shadow_az + 180.0) % 360.0)
        estimates.append((np.array([math.sin(sun_az), math.cos(sun_az)]), length))
    if not estimates:
        return None

    vecs = np.array([e[0] for e in estimates])
    w = np.array([e[1] for e in estimates])
    mean = (vecs * w[:, None]).sum(0)
    ang = np.arctan2(vecs[:, 0], vecs[:, 1])
    ref = math.atan2(mean[0], mean[1])
    keep = np.abs((ang - ref + math.pi) % (2 * math.pi) - math.pi) < math.radians(40)  # drop outliers
    if keep.sum() == 0:
        keep[:] = True
    mean = (vecs[keep] * w[keep, None]).sum(0)
    az = math.degrees(math.atan2(mean[0], mean[1])) % 360.0
    coherence = float(np.linalg.norm(mean) / w[keep].sum())
    return dict(az=az, n=int(keep.sum()), coherence=coherence)


def estimate_sun_azimuth_boundary(rgb, sigma=1.5, step_px=2.5, min_grad=0.05):
    """Robust variant for real, dense, non-rectangular footprints.
    Roof boundary pixels whose outward normal points toward a shadow vote with that normal; for a convex footprint the
    length-weighted sum of the normals of the edges facing the shadow equals (width perpendicular to the shadow) x
    (shadow direction), so the vector sum points along the shadow. No per-building association is needed."""
    h, s, v = rgb_to_hsv(rgb)
    ground = (h > 0.2) & (h < 0.5) & (s > 0.2)
    if ground.sum() < 200:
        return None
    thr = otsu(v[ground])
    lo, hi = v[ground & (v < thr)], v[ground & (v >= thr)]
    if lo.size < 30 or hi.size < 30 or hi.mean() - lo.mean() < 0.08:
        return None
    shadow = ground & (v < thr)
    sm = ndimage.gaussian_filter((~ground).astype(float), sigma)
    gy, gx = np.gradient(sm)                      # derivatives along rows (south) and columns (east)
    mag = np.hypot(gx, gy)
    rows, cols = np.nonzero(mag > min_grad)
    if rows.size < 50:
        return None
    nx, ny = -gx[rows, cols] / mag[rows, cols], gy[rows, cols] / mag[rows, cols]   # outward normal (east, north)
    H, W = shadow.shape
    qr = np.rint(rows - ny * step_px).astype(int)  # one step outward; north = decreasing row
    qc = np.rint(cols + nx * step_px).astype(int)
    ok = (qr >= 0) & (qr < H) & (qc >= 0) & (qc < W)
    in_sh = np.zeros(rows.size, dtype=bool)
    in_sh[ok] = shadow[qr[ok], qc[ok]]
    if in_sh.sum() < 20:
        return None
    vx, vy = nx[in_sh].sum(), ny[in_sh].sum()
    shadow_az = math.degrees(math.atan2(vx, vy)) % 360.0
    return dict(az=(shadow_az + 180.0) % 360.0, n=int(in_sh.sum()), coherence=float(math.hypot(vx, vy) / in_sh.sum()))


# ----------------------------------------------------------------------------- regime logic
def invert_clock(az_est, lat, lon, date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    minutes = np.arange(0, 1440, 1)
    az, el = solar_position(lat, lon, y, m, d, minutes / 60.0)
    err = np.abs(wrap180(az - az_est))
    err[el <= 0.0] = np.inf
    return float(minutes[int(np.argmin(err))]) / 60.0


def predict(regime, pair, az_a, az_b):
    """Return (earlier 'A'|'B', elapsed_hours) for one pair."""
    if regime == "R2":
        ta = invert_clock(az_a, pair["lat"], pair["lon"], pair["date"])
        tb = invert_clock(az_b, pair["lat"], pair["lon"], pair["date"])
        return ("A" if ta <= tb else "B"), abs(tb - ta)
    hemi = "northern" if regime == "R0" else pair["hemisphere"]  # R0: northern-hemisphere prior
    d = float(wrap180(az_b - az_a))
    b_later = d > 0 if hemi == "northern" else d < 0
    return ("A" if b_later else "B"), abs(d) / 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="preds_baseline.jsonl")
    ap.add_argument("--method", default="components", choices=["components", "boundary", "auto"],
                    help="components: per-building shadow axis (simple boxes); boundary: roof-edge normal vote; "
                         "auto: components, falling back to boundary where components cannot estimate")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs)]
    for p in pairs:
        p["image_a"], p["image_b"] = p["image_a"].replace("\\", "/"), p["image_b"].replace("\\", "/")
    meta = {r["image"].replace("\\", "/"): r
            for r in (json.loads(l) for l in open(os.path.join(args.data, "metadata.jsonl")))}
    cache = {}

    def az_of(img_rel):
        if img_rel not in cache:
            rgb = load_rgb(os.path.join(args.data, img_rel))
            if args.method == "boundary":
                cache[img_rel] = estimate_sun_azimuth_boundary(rgb)
            elif args.method == "auto":
                cache[img_rel] = estimate_sun_azimuth(rgb) or estimate_sun_azimuth_boundary(rgb)
            else:
                cache[img_rel] = estimate_sun_azimuth(rgb)
        return cache[img_rel]

    n_fallback = 0
    with open(args.out, "w") as f:
        for p in pairs:
            ea, eb = az_of(p["image_a"]), az_of(p["image_b"])
            for regime in ("R0", "R1", "R2"):
                if ea is None or eb is None:
                    earlier, elapsed = "A", 1.0  # abstain -> fixed guess, counted below
                    n_fallback += regime == "R0"
                else:
                    earlier, elapsed = predict(regime, p, ea["az"], eb["az"])
                resp = json.dumps({"earlier": earlier, "elapsed_hours": round(elapsed, 3)})
                f.write(json.dumps(dict(id=p["id"], regime=regime, model="analytic_baseline", response=resp)) + "\n")

    errs = [abs(float(wrap180(e["az"] - meta[k]["sun_az_deg"]))) for k, e in cache.items() if e is not None]
    n_none = sum(e is None for e in cache.values())
    print(f"images processed: {len(cache)}   no-estimate: {n_none}   pairs with fallback guess: {n_fallback}")
    if errs:
        print(f"sun-azimuth error vs ground truth (deg): median {np.median(errs):.2f}  "
              f"mean {np.mean(errs):.2f}  p90 {np.percentile(errs, 90):.2f}  max {np.max(errs):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

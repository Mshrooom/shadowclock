"""
geom.py - pure-numpy geometry helpers shared by the generator and the map-based baseline (no bpy needed).

Conventions (same as everywhere else in this project): image up = north (+Y), image right = east (+X); row 0 is
the north edge; pixel (r, c) has its centre at x = (c + 0.5) * gsd - half, y = half - (r + 0.5) * gsd with
half = size_px * gsd / 2.  A "building" is a dict with `rings` (list of rings; ring 0 = outer boundary, later rings =
courtyard holes; each ring a list of (x, y) in metres) and `height` (metres, flat roof).
"""
import math

import numpy as np


def rasterize_height_polys(buildings, size_px, gsd):
    """Height map (metres, float32) of extruded polygons with holes (even-odd rule); the tallest building wins."""
    H = np.zeros((size_px, size_px), dtype=np.float32)
    half = size_px * gsd / 2.0
    for b in buildings:
        pts = np.array([p for ring in b["rings"] for p in ring], dtype=float)
        if pts.size == 0:
            continue
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        c0, c1 = max(int(math.floor((x0 + half) / gsd)), 0), min(int(math.ceil((x1 + half) / gsd)), size_px)
        r0, r1 = max(int(math.floor((half - y1) / gsd)), 0), min(int(math.ceil((half - y0) / gsd)), size_px)
        if c1 <= c0 or r1 <= r0:
            continue
        xs = (np.arange(c0, c1) + 0.5) * gsd - half
        ys = half - (np.arange(r0, r1) + 0.5) * gsd
        X, Y = np.meshgrid(xs, ys)
        inside = np.zeros(X.shape, dtype=bool)
        for ring in b["rings"]:
            rx = np.array([p[0] for p in ring], dtype=float)
            ry = np.array([p[1] for p in ring], dtype=float)
            n = len(rx)
            for i in range(n):
                xa, ya, xb, yb = rx[i], ry[i], rx[(i + 1) % n], ry[(i + 1) % n]
                if ya == yb:
                    continue
                crosses = (ya > Y) != (yb > Y)
                x_int = xa + (Y - ya) * (xb - xa) / (yb - ya)
                inside ^= crosses & (X < x_int)
        patch = H[r0:r1, c0:c1]
        patch[inside] = np.maximum(patch[inside], float(b["height"]))
    return H


def analytic_shadow_mask(H, gsd, az_deg, el_deg):
    """Exact cast-shadow mask of a height field for a directional sun (ray marching toward the sun)."""
    n = H.shape[0]
    az, el = math.radians(az_deg), math.radians(el_deg)
    sx, sy, t = math.sin(az), math.cos(az), math.tan(el)  # horizontal unit vector toward the sun (E, N)
    rows, cols = np.mgrid[0:n, 0:n]
    shadow = np.zeros((n, n), dtype=bool)
    d_max = float(H.max()) / t + gsd
    d = gsd
    while d <= d_max:
        rr = np.rint(rows - sy * d / gsd).astype(np.int64)  # moving north = decreasing row
        cc = np.rint(cols + sx * d / gsd).astype(np.int64)
        ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
        hh = np.zeros_like(H)
        hh[ok] = H[rr[ok], cc[ok]]
        shadow |= hh > (H + d * t + 1e-3)
        d += gsd
    return shadow

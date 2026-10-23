import math

import numpy as np

from geom import analytic_shadow_mask, rasterize_height_polys


def square(cx, cy, w):
    h = w / 2
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def test_rasterize_square_height_and_area():
    H = rasterize_height_polys([{"rings": [square(0, 0, 20)], "height": 30.0}], 128, 0.5)  # 64 m field, 20 m square
    assert H.max() == 30.0
    assert abs((H > 0).sum() * 0.25 - 400.0) < 12.0


def test_courtyard_hole_is_empty():
    b = {"rings": [square(0, 0, 40), square(0, 0, 10)], "height": 20.0}
    H = rasterize_height_polys([b], 160, 0.5)
    assert H[80, 80] == 0 and H[80, 100] == 20.0


def test_shadow_length_of_a_tower_is_height_over_tan_elevation():
    n, gsd = 256, 0.5
    H = rasterize_height_polys([{"rings": [square(0, 0, 8)], "height": 20.0}], n, gsd)
    az, el = 90.0, 45.0  # sun in the east -> shadow points west, length = 20 m
    sh = analytic_shadow_mask(H, gsd, az, el) & (H == 0)
    row = n // 2
    cols = np.where(sh[row])[0]
    west_edge_col = np.where(H[row] > 0)[0].min()
    length_m = (west_edge_col - cols.min()) * gsd
    assert abs(length_m - 20.0) < 1.5
    assert cols.max() < west_edge_col  # nothing on the sun side

import math
import shutil
import subprocess
import json
import os
import random
import datetime

import numpy as np
import pytest

from solar import solar_position, sun_vector_enu, wrap180

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_equinox_noon_equator_sun_is_overhead():
    az, el = solar_position(0.0, 0.0, 2027, 3, 20, 12.0)
    assert el > 85  # equation of time and declination keep it within a few degrees of the zenith


def test_east_in_the_morning_west_in_the_evening():
    az_am, _ = solar_position(40.0, 0.0, 2027, 3, 20, 8.0)
    az_pm, _ = solar_position(40.0, 0.0, 2027, 3, 20, 16.0)
    assert 60 < az_am < 120 and 240 < az_pm < 300


def test_sun_vector_is_unit_and_points_up_when_sun_is_up():
    v = sun_vector_enu(120.0, 35.0)
    assert math.isclose(sum(c * c for c in v), 1.0, rel_tol=1e-12)
    assert v[2] > 0


def test_wrap180():
    assert wrap180(190.0) == -170.0 and wrap180(-190.0) == 170.0


def test_matches_pvlib():
    pvlib = pytest.importorskip("pvlib")
    import pandas as pd
    random.seed(3)
    worst_az = worst_el = 0.0
    for _ in range(60):
        lat, lon = random.uniform(-60, 60), random.uniform(-180, 180)
        d = datetime.date(2027, 1, 1) + datetime.timedelta(days=random.randrange(365))
        h = random.uniform(0, 24)
        t = pd.Timestamp(d.year, d.month, d.day, tz="UTC") + pd.Timedelta(hours=h)
        ref = pvlib.solarposition.get_solarposition(t, lat, lon, method="nrel_numpy")
        az, el = solar_position(lat, lon, d.year, d.month, d.day, h)
        if ref["elevation"].iloc[0] > 10:  # keep clear of the horizon, where pvlib's refraction model and ours differ by design
            worst_az = max(worst_az, abs(((float(az) - ref["azimuth"].iloc[0]) + 180) % 360 - 180))
            worst_el = max(worst_el, abs(float(el) - ref["elevation"].iloc[0]))
    assert worst_az < 0.1 and worst_el < 0.1, (worst_az, worst_el)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_port_matches_python(tmp_path):
    random.seed(1)
    cases = []
    for _ in range(300):
        lat, lon = random.uniform(-66, 66), random.uniform(-180, 180)
        d = datetime.date(2027, 1, 1) + datetime.timedelta(days=random.randrange(365))
        h = random.uniform(0, 24)
        az, el = solar_position(lat, lon, d.year, d.month, d.day, h)
        cases.append(dict(lat=lat, lon=lon, y=d.year, m=d.month, d=d.day, h=h, az=float(az), el=float(el)))
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(cases))
    out = subprocess.run(["node", os.path.join(ROOT, "tests", "solar_parity.mjs"), str(f)], capture_output=True, text=True, check=True)
    res = json.loads(out.stdout)
    assert res["worstAz"] < 1e-6 and res["worstEl"] < 1e-6, res

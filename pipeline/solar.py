"""
solar.py - NOAA-style solar position (azimuth / elevation), numpy-vectorised.

Convention used everywhere in this project
------------------------------------------
* azimuth   : degrees clockwise from true north (N=0, E=90, S=180, W=270)
* elevation : degrees above the horizon
* time      : UTC hours since 00:00 of the given calendar date (may be an array)
* longitude : degrees, east positive.  With lon=0 the UTC clock equals local
              mean time, which is what the dataset uses.
No atmospheric refraction is applied (irrelevant for shadow geometry here).
Accuracy is on the order of 0.01 deg for 1900-2100 (NOAA spreadsheet algorithm).
"""
import math
import numpy as np
from datetime import date as _date


def julian_day_0h(year, month, day):
    """Julian Day at 00:00 UTC of a Gregorian calendar date."""
    return _date(int(year), int(month), int(day)).toordinal() + 1721424.5


def solar_position(lat_deg, lon_deg, year, month, day, utc_hours):
    """Return (azimuth_deg, elevation_deg) as numpy arrays (or 0-d arrays)."""
    h = np.asarray(utc_hours, dtype=float)
    jd = julian_day_0h(year, month, day) + h / 24.0
    jc = (jd - 2451545.0) / 36525.0

    L0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = np.deg2rad(M)
    C = (np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * np.cos(np.deg2rad(omega))
    obliq_r = np.deg2rad(obliq)
    decl = np.arcsin(np.sin(obliq_r) * np.sin(np.deg2rad(app_long)))   # radians

    y = np.tan(obliq_r / 2.0) ** 2
    L0r = np.deg2rad(L0)
    eq_time = 4.0 * np.rad2deg(
        y * np.sin(2 * L0r) - 2 * e * np.sin(Mr)
        + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
        - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))   # minutes

    tst = np.mod(h * 60.0 + eq_time + 4.0 * lon_deg, 1440.0)               # true solar time, minutes
    ha = np.deg2rad(tst / 4.0 - 180.0)                                      # hour angle, radians

    lat = math.radians(lat_deg)
    cos_zen = np.clip(math.sin(lat) * np.sin(decl) + math.cos(lat) * np.cos(decl) * np.cos(ha), -1.0, 1.0)
    zen = np.arccos(cos_zen)
    sin_zen = np.sin(zen)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = (math.sin(lat) * cos_zen - np.sin(decl)) / (math.cos(lat) * sin_zen)
    cos_az = np.clip(np.nan_to_num(cos_az, nan=1.0), -1.0, 1.0)
    acos_az = np.rad2deg(np.arccos(cos_az))
    az = np.where(ha > 0, np.mod(acos_az + 180.0, 360.0), np.mod(540.0 - acos_az, 360.0))
    el = 90.0 - np.rad2deg(zen)
    return az, el


def sun_vector_enu(az_deg, el_deg):
    """Unit vector pointing FROM the ground TOWARD the sun in (east, north, up)."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))


def wrap180(x):
    """Wrap an angle (deg) to [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    for lat in (50, 0, -35):
        az, el = solar_position(lat, 0.0, 2027, 3, 20, np.array([6, 9, 12, 15, 18]))
        print(lat, np.round(az, 1), np.round(el, 1))

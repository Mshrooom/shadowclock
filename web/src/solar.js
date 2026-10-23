// NOAA solar position - a line-by-line port of pipeline/solar.py (same constants, same conventions).
// azimuth: degrees clockwise from true north; elevation: degrees above the horizon.
// utcHours: hours since 00:00 UTC of the calendar date. For the benchmark's "local mean solar time" clock use
// utcHours = localMeanHours - lonDeg / 15  (that is how the datasets were rendered; see README).

const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
const mod = (a, n) => ((a % n) + n) % n;

export function julianDay0h(year, month, day) {
  return Date.UTC(year, month - 1, day) / 86400000 + 2440587.5;
}

export function solarPosition(latDeg, lonDeg, year, month, day, utcHours) {
  const jd = julianDay0h(year, month, day) + utcHours / 24;
  const jc = (jd - 2451545.0) / 36525.0;

  const L0 = mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360);
  const M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
  const e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
  const Mr = rad(M);
  const C =
    Math.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
    Math.sin(2 * Mr) * (0.019993 - 0.000101 * jc) +
    Math.sin(3 * Mr) * 0.000289;
  const trueLong = L0 + C;
  const omega = 125.04 - 1934.136 * jc;
  const appLong = trueLong - 0.00569 - 0.00478 * Math.sin(rad(omega));
  const meanObliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
  const obliq = meanObliq + 0.00256 * Math.cos(rad(omega));
  const obliqR = rad(obliq);
  const decl = Math.asin(Math.sin(obliqR) * Math.sin(rad(appLong)));

  const y = Math.tan(obliqR / 2) ** 2;
  const L0r = rad(L0);
  const eqTime =
    4.0 *
    deg(
      y * Math.sin(2 * L0r) -
        2 * e * Math.sin(Mr) +
        4 * e * y * Math.sin(Mr) * Math.cos(2 * L0r) -
        0.5 * y * y * Math.sin(4 * L0r) -
        1.25 * e * e * Math.sin(2 * Mr)
    );

  const tst = mod(utcHours * 60.0 + eqTime + 4.0 * lonDeg, 1440.0);
  const ha = rad(tst / 4.0 - 180.0);

  const lat = rad(latDeg);
  const cosZen = Math.min(1, Math.max(-1, Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.cos(ha)));
  const zen = Math.acos(cosZen);
  const sinZen = Math.sin(zen);
  let cosAz = (Math.sin(lat) * cosZen - Math.sin(decl)) / (Math.cos(lat) * sinZen);
  if (!Number.isFinite(cosAz)) cosAz = 1;
  cosAz = Math.min(1, Math.max(-1, cosAz));
  const acosAz = deg(Math.acos(cosAz));
  const az = ha > 0 ? mod(acosAz + 180, 360) : mod(540 - acosAz, 360);
  return { az, el: 90 - deg(zen) };
}

// local mean solar time (hours) -> sun position for a city
export function sunAtLocalMean(latDeg, lonDeg, dateStr, localMeanHours) {
  const [y, m, d] = dateStr.split("-").map(Number);
  return solarPosition(latDeg, lonDeg, y, m, d, localMeanHours - lonDeg / 15);
}

// unit vector from the ground toward the sun in three.js axes: x = east, y = up, z = SOUTH (so north is -z)
export function sunDirection(az, el) {
  const a = rad(az);
  const e = rad(el);
  return [Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a)];
}

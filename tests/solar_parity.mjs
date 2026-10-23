// usage: node tests/solar_parity.mjs cases.json   -> prints max abs error vs the python reference
import { readFileSync } from "node:fs";
import { solarPosition } from "../web/src/solar.js";
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
let worstAz = 0, worstEl = 0;
for (const c of cases) {
  const r = solarPosition(c.lat, c.lon, c.y, c.m, c.d, c.h);
  worstAz = Math.max(worstAz, Math.abs(((r.az - c.az + 540) % 360) - 180));
  worstEl = Math.max(worstEl, Math.abs(r.el - c.el));
}
console.log(JSON.stringify({ n: cases.length, worstAz, worstEl }));

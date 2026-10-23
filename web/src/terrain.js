import * as THREE from "three";

// zlib-compressed little-endian uint16 decimetres -> Uint16Array
export async function decodeHeights(b64) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint16Array(buf);
}

// roof colour ramp by building height (metres): sand -> terracotta -> light grey -> slate -> deep blue-grey
const RAMP = [
  [0, "#d3bb96"], [6, "#d69f78"], [14, "#cf9070"], [25, "#c4c9d2"], [50, "#a7b4c9"], [110, "#8798b8"], [300, "#7487ad"],
].map(([h, c]) => [h, new THREE.Color(c)]);

export function roofColor(h, out = new THREE.Color()) {
  if (h <= RAMP[0][0]) return out.copy(RAMP[0][1]);
  for (let i = 1; i < RAMP.length; i++) {
    if (h <= RAMP[i][0]) {
      const t = (h - RAMP[i - 1][0]) / (RAMP[i][0] - RAMP[i - 1][0]);
      return out.copy(RAMP[i - 1][1]).lerp(RAMP[i][1], t);
    }
  }
  return out.copy(RAMP[RAMP.length - 1][1]);
}

// Flat-roof prisms from a height field. Axes: x = east, y = up, z = south (north is -z); row 0 of the grid is north.
// Roofs are merged along rows where the height is equal; north/south walls too; east/west walls are per cell.
function blurredMask(hdm, size, passes = 3, rad = 2) {
  let a = new Float32Array(size * size);
  for (let i = 0; i < a.length; i++) a[i] = hdm[i] > 0 ? 1 : 0;
  let b = new Float32Array(size * size);
  const w = 2 * rad + 1;
  for (let p = 0; p < passes; p++) {
    for (let r = 0; r < size; r++) {            // horizontal box blur
      for (let c = 0; c < size; c++) {
        let t = 0;
        for (let k = -rad; k <= rad; k++) t += a[r * size + Math.min(size - 1, Math.max(0, c + k))];
        b[r * size + c] = t / w;
      }
    }
    for (let r = 0; r < size; r++) {            // vertical box blur
      for (let c = 0; c < size; c++) {
        let t = 0;
        for (let k = -rad; k <= rad; k++) t += b[Math.min(size - 1, Math.max(0, r + k)) * size + c];
        a[r * size + c] = t / w;
      }
    }
  }
  return a;
}

export function buildBuildingGeometry(hdm, size, gsd) {
  const half = (size * gsd) / 2;
  // The 0.5 m raster turns oblique walls into staircases whose alternating N/S and E/W faces would shade like stripes.
  // For lighting only, walls take the outward normal of a blurred footprint mask (shadows still use the true geometry).
  const B = blurredMask(hdm, size);
  const bAt = (r, c) => B[Math.min(size - 1, Math.max(0, r)) * size + Math.min(size - 1, Math.max(0, c))];
  const wallNormal = (r, c, axis) => {
    const gx = bAt(r, c + 1) - bAt(r, c - 1);
    const gz = bAt(r + 1, c) - bAt(r - 1, c);
    let nx = -gx, nz = -gz;
    const len = Math.hypot(nx, nz);
    if (len < 1e-4) return axis;
    nx /= len; nz /= len;
    return nx * axis[0] + nz * axis[2] > 0.3 ? [nx, 0, nz] : axis;
  };
  const P = [];
  const N = [];
  const C = [];
  const col = new THREE.Color();
  const wallCol = new THREE.Color();
  const at = (r, c) => (r < 0 || r >= size || c < 0 || c >= size ? 0 : hdm[r * size + c]);

  const push = (a, b, c, n, rgb) => {
    for (const v of [a, b, c]) {
      P.push(v[0], v[1], v[2]);
      N.push(n[0], n[1], n[2]);
      C.push(rgb.r, rgb.g, rgb.b);
    }
  };
  const quadTop = (x0, x1, z0, z1, y, rgb) => {
    push([x0, y, z1], [x1, y, z1], [x1, y, z0], [0, 1, 0], rgb);
    push([x0, y, z1], [x1, y, z0], [x0, y, z0], [0, 1, 0], rgb);
  };

  for (let r = 0; r < size; r++) {
    let c = 0;
    while (c < size) {
      const h = at(r, c);
      if (h === 0) { c++; continue; }
      let c1 = c + 1;
      while (c1 < size && at(r, c1) === h) c1++;
      const hm = h / 10;
      roofColor(hm, col);
      wallCol.copy(col).multiplyScalar(0.8);
      const x0 = c * gsd - half;
      const x1 = c1 * gsd - half;
      const zN = r * gsd - half;
      const zS = (r + 1) * gsd - half;
      quadTop(x0, x1, zN, zS, hm, col);
      // north / south walls over the run, split where the neighbour height changes
      for (const side of [-1, 1]) {
        let k = c;
        while (k < c1) {
          const nh = at(r + side, k);
          if (nh >= h) { k++; continue; }
          let k1 = k + 1;
          while (k1 < c1 && at(r + side, k1) === nh) k1++;
          const xa = k * gsd - half;
          const xb = k1 * gsd - half;
          const yb = nh / 10;
          const mid = (k + k1) >> 1;
          if (side === -1) {
            const nn = wallNormal(r, mid, [0, 0, -1]);
            push([xb, yb, zN], [xa, yb, zN], [xa, hm, zN], nn, wallCol);
            push([xb, yb, zN], [xa, hm, zN], [xb, hm, zN], nn, wallCol);
          } else {
            const nn = wallNormal(r, mid, [0, 0, 1]);
            push([xa, yb, zS], [xb, yb, zS], [xb, hm, zS], nn, wallCol);
            push([xa, yb, zS], [xb, hm, zS], [xa, hm, zS], nn, wallCol);
          }
          k = k1;
        }
      }
      // east / west walls per cell
      for (let k = c; k < c1; k++) {
        const z0 = zN;
        const z1 = zS;
        const nhE = at(r, k + 1);
        if (nhE < h) {
          const xe = (k + 1) * gsd - half;
          const yb = nhE / 10;
          const ne = wallNormal(r, k, [1, 0, 0]);
          push([xe, yb, z1], [xe, yb, z0], [xe, hm, z0], ne, wallCol);
          push([xe, yb, z1], [xe, hm, z0], [xe, hm, z1], ne, wallCol);
        }
        const nhW = at(r, k - 1);
        if (nhW < h) {
          const xw = k * gsd - half;
          const yb = nhW / 10;
          const nw = wallNormal(r, k, [-1, 0, 0]);
          push([xw, yb, z0], [xw, yb, z1], [xw, hm, z1], nw, wallCol);
          push([xw, yb, z0], [xw, hm, z1], [xw, hm, z0], nw, wallCol);
        }
      }
      c = c1;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(C, 3));
  g.computeBoundingSphere();
  return g;
}

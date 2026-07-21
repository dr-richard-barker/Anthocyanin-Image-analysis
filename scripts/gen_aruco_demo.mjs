// Generate a demo image containing four real ARUCO_MIP_36h12 fiducials, used to
// exercise the ArUco-decoder fallback in colorcalib.ts (the path taken when the
// geometric corner-square detector can't lock onto a clean quad).
//
// The four markers are laid out on a flat plane, then that plane is warped into a
// PERSPECTIVE-TILTED quad — simulating a phone photo taken at an angle. The tilt
// makes the near markers larger than the far ones, so the geometric detector's
// "four similar-sized corner squares" heuristic fails, while the ArUco decoder
// (perspective-invariant by design) still reads all four. That routes detection
// through the fallback we want to test.
//
// Marker encoding (matches js-aruco2 AR.Detector.getMarker):
//   - 8x8 cell grid; markSize = sqrt(36)+2 = 8. Outer ring BLACK.
//   - Interior 6x6: bit = codeBits[(r-1)*6 + (c-1)]; '1' -> WHITE, '0' -> BLACK.
//   - Row-major interior == the dictionary code (decoder tries 4 rotations).
//
// Output: public/demos/aruco_markers_demo.png (8-bit grayscale, zlib-deflated).
// Run: node scripts/gen_aruco_demo.mjs

import { deflateSync } from 'node:zlib';
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// First four ARUCO_MIP_36h12 codes (ids 0..3).
const CODES = [0xd2b63a09d, 0x6001134e5, 0x1206fbe72, 0xff8ad6cb4];
const bin36 = (h) => h.toString(2).padStart(36, '0');

// --- 1. render the flat marker plane (large, for clean supersampling) ---
const SW = 1200, SH = 840, CELL = 26, MS = 8 * CELL, PAD = 90;
const src = new Uint8Array(SW * SH).fill(255);
function drawMarker(ox, oy, code) {
  const b = bin36(code);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const border = r === 0 || r === 7 || c === 0 || c === 7;
      const black = border ? true : b[(r - 1) * 6 + (c - 1)] === '0';
      if (!black) continue; // white cells already match the background
      for (let dy = 0; dy < CELL; dy++)
        for (let dx = 0; dx < CELL; dx++) {
          const x = ox + c * CELL + dx, y = oy + r * CELL + dy;
          if (x >= 0 && x < SW && y >= 0 && y < SH) src[y * SW + x] = 0;
        }
    }
  }
}
drawMarker(PAD, PAD, CODES[0]);                    // TL
drawMarker(SW - PAD - MS, PAD, CODES[1]);          // TR
drawMarker(SW - PAD - MS, SH - PAD - MS, CODES[2]);// BR
drawMarker(PAD, SH - PAD - MS, CODES[3]);          // BL

// --- 2. homography output(x,y) -> source(u,v) from 4 corner correspondences ---
// Solve 8 unknowns [a..h]: u=(ax+by+c)/(gx+hy+1), v=(dx+ey+f)/(gx+hy+1).
function solve8(A, y) {
  const n = 8;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
    [A[i], A[p]] = [A[p], A[i]]; [y[i], y[p]] = [y[p], y[i]];
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
      y[k] -= f * y[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}
function homography(dst, srcPts) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = dst[i], { x: u, y: v } = srcPts[i];
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v]); b.push(v);
  }
  const h = solve8(A, b);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

// --- 3. warp into a tilted output quad and bilinear-sample the source ---
const OW = 1120, OH = 760;
// Output quad for the marker plane: top edge shorter/higher (farther away).
const dstQuad = [
  { x: 300, y: 120 },  // TL (far)
  { x: 760, y: 150 },  // TR (far)
  { x: 970, y: 650 },  // BR (near)
  { x: 110, y: 610 },  // BL (near)
];
const srcQuad = [
  { x: 0, y: 0 }, { x: SW - 1, y: 0 }, { x: SW - 1, y: SH - 1 }, { x: 0, y: SH - 1 },
];
const H = homography(dstQuad, srcQuad);
const sample = (u, v) => {
  if (u < 0 || v < 0 || u >= SW - 1 || v >= SH - 1) return 255;
  const x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0;
  const a = src[y0 * SW + x0], b = src[y0 * SW + x0 + 1];
  const c = src[(y0 + 1) * SW + x0], d = src[(y0 + 1) * SW + x0 + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
};
const img = new Uint8Array(OW * OH).fill(245); // faint off-white "paper"
for (let y = 0; y < OH; y++) {
  for (let x = 0; x < OW; x++) {
    const w = H[6] * x + H[7] * y + 1;
    const u = (H[0] * x + H[1] * y + H[2]) / w;
    const v = (H[3] * x + H[4] * y + H[5]) / w;
    if (u >= 0 && v >= 0 && u < SW && v < SH) img[y * OW + x] = Math.round(sample(u, v));
  }
}

// --- 4. minimal 8-bit grayscale PNG encoder ---
function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}
function chunk(type, data) {
  const t = Buffer.from(type, 'ascii');
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([t, data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(OW, 0); ihdr.writeUInt32BE(OH, 4);
ihdr[8] = 8; ihdr[9] = 0; // 8-bit grayscale
const raw = Buffer.alloc(OH * (OW + 1));
for (let y = 0; y < OH; y++) {
  raw[y * (OW + 1)] = 0; // filter: none
  for (let x = 0; x < OW; x++) raw[y * (OW + 1) + 1 + x] = img[y * OW + x];
}
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', deflateSync(raw, { level: 9 })),
  chunk('IEND', Buffer.alloc(0)),
]);

const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'demos');
mkdirSync(outDir, { recursive: true });
const out = join(outDir, 'aruco_markers_demo.png');
writeFileSync(out, png);
console.log('wrote', out, png.length, 'bytes', `(${OW}x${OH}, tilted, markers ids 0-3)`);

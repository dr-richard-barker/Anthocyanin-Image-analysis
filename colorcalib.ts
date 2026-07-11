// Astrocalibration marker support — fully client-side, no network / AI.
//
// Implements the essentials of the PlantCV Astrobotany colour-correction
// workflow (astro_color_matrix + affine_color_correction) for the browser:
//   1. detect the 4 corner ArUco fiducials (js-aruco2, ARUCO_MIP_36h12)
//   2. build a sampling quad from the corners (user-adjustable in the UI)
//   3. sample the 15 reference chips through that quad
//   4. fit a 3x4 affine colour transform source -> Astro standard
//   5. apply it per-pixel in the analysis pipeline
//
// See docs/astrocalibration.md for the marker anatomy and rationale.

export interface Pt { x: number; y: number; }

// The 15-chip Astrobotany standard (pcv.transform.astro_color_matrix()), RGB in
// 0-1. u,v are the chip centre in normalised quad coords (0..1 across the 4
// marker-centre corners TL->TR = u, TL->BL = v), measured from a reference
// marker image. They are only a starting guess — the UI lets the user nudge the
// quad corners to fit their own photo.
export interface AstroChip { id: number; u: number; v: number; std: [number, number, number]; name: string; }

export const ASTRO_CHIPS: AstroChip[] = [
  // solid colour column
  { id: 10, u: 0.55, v: 0.06, std: [0.18, 0.23, 0.50], name: 'blue' },
  { id: 20, u: 0.55, v: 0.27, std: [0.34, 0.62, 0.25], name: 'green' },
  { id: 30, u: 0.54, v: 0.52, std: [0.71, 0.25, 0.21], name: 'red' },
  { id: 40, u: 0.52, v: 0.75, std: [0.89, 0.81, 0.20], name: 'yellow' },
  { id: 50, u: 0.51, v: 0.98, std: [0.21, 0.22, 0.22], name: 'near-black' },
  // grayscale ramp (white -> dark), sits just left of the colour column
  { id: 60, u: 0.45, v: 0.10, std: [0.91, 0.95, 0.93], name: 'gray-0' },
  { id: 70, u: 0.445, v: 0.19, std: [0.82, 0.86, 0.86], name: 'gray-1' },
  { id: 80, u: 0.44, v: 0.28, std: [0.72, 0.75, 0.73], name: 'gray-2' },
  { id: 90, u: 0.435, v: 0.37, std: [0.64, 0.67, 0.64], name: 'gray-3' },
  { id: 100, u: 0.43, v: 0.46, std: [0.57, 0.58, 0.56], name: 'gray-4' },
  { id: 110, u: 0.425, v: 0.54, std: [0.48, 0.49, 0.48], name: 'gray-5' },
  { id: 120, u: 0.42, v: 0.63, std: [0.39, 0.40, 0.39], name: 'gray-6' },
  { id: 130, u: 0.415, v: 0.72, std: [0.33, 0.32, 0.32], name: 'gray-7' },
  { id: 140, u: 0.41, v: 0.81, std: [0.27, 0.28, 0.27], name: 'gray-8' },
  { id: 150, u: 0.405, v: 0.90, std: [0.22, 0.23, 0.23], name: 'gray-9' },
];

// Physical distance between opposite corner-marker centres on the 5 cm card.
// Used to derive pixels/cm from the detected quad.
export const MARKER_SPAN_CM = 4.3;

// --- geometry ---------------------------------------------------------------

// Bilinear map from normalised (u,v) to image space given the 4 corners
// ordered [TL, TR, BR, BL].
export function quadPoint(corners: Pt[], u: number, v: number): Pt {
  const [TL, TR, BR, BL] = corners;
  const top = { x: TL.x + (TR.x - TL.x) * u, y: TL.y + (TR.y - TL.y) * u };
  const bot = { x: BL.x + (BR.x - BL.x) * u, y: BL.y + (BR.y - BL.y) * u };
  return { x: top.x + (bot.x - top.x) * v, y: top.y + (bot.y - top.y) * v };
}

// Order 4 unordered points into [TL, TR, BR, BL] by position.
export function orderCorners(pts: Pt[]): Pt[] {
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const tl = pts.find(p => p.x <= cx && p.y <= cy);
  const tr = pts.find(p => p.x > cx && p.y <= cy);
  const br = pts.find(p => p.x > cx && p.y > cy);
  const bl = pts.find(p => p.x <= cx && p.y > cy);
  // fall back to angle sort if any quadrant is empty
  if (tl && tr && br && bl) return [tl, tr, br, bl];
  const sorted = [...pts].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  return sorted;
}

export function scaleAndRotation(corners: Pt[], spanCm = MARKER_SPAN_CM): { pxPerCm: number; rotationDeg: number } {
  const [TL, TR, , BL] = corners;
  const topLen = Math.hypot(TR.x - TL.x, TR.y - TL.y);
  const leftLen = Math.hypot(BL.x - TL.x, BL.y - TL.y);
  const pxPerCm = ((topLen + leftLen) / 2) / spanCm;
  const rotationDeg = -(Math.atan2(TR.y - TL.y, TR.x - TL.x) * 180) / Math.PI;
  return { pxPerCm, rotationDeg };
}

// --- sampling ---------------------------------------------------------------

export function chipImagePoints(corners: Pt[]): Pt[] {
  return ASTRO_CHIPS.map(c => quadPoint(corners, c.u, c.v));
}

// Median RGB (0-1) in a small square patch around each chip point.
export function sampleChips(data: Uint8ClampedArray, w: number, h: number, pts: Pt[], radius = 6): number[][] {
  const med = (arr: number[]) => { arr.sort((a, b) => a - b); return arr[arr.length >> 1]; };
  return pts.map(p => {
    const R: number[] = [], G: number[] = [], B: number[] = [];
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = Math.round(p.x + dx), y = Math.round(p.y + dy);
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const i = (y * w + x) * 4;
        R.push(data[i]); G.push(data[i + 1]); B.push(data[i + 2]);
      }
    }
    if (!R.length) return [0, 0, 0];
    return [med(R) / 255, med(G) / 255, med(B) / 255];
  });
}

// --- affine colour fit ------------------------------------------------------

// Solve (A^T A) x = A^T y via Gaussian elimination. A is m x n, y is length m.
function solveNormal(A: number[][], y: number[]): number[] {
  const n = A[0].length;
  const ATA = Array.from({ length: n }, () => new Array(n).fill(0));
  const ATy = new Array(n).fill(0);
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) ATA[i][j] += A[r][i] * A[r][j];
      ATy[i] += A[r][i] * y[r];
    }
  }
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(ATA[k][i]) > Math.abs(ATA[p][i])) p = k;
    [ATA[i], ATA[p]] = [ATA[p], ATA[i]];
    [ATy[i], ATy[p]] = [ATy[p], ATy[i]];
    if (Math.abs(ATA[i][i]) < 1e-9) continue;
    for (let k = i + 1; k < n; k++) {
      const f = ATA[k][i] / ATA[i][i];
      for (let j = i; j < n; j++) ATA[k][j] -= f * ATA[i][j];
      ATy[k] -= f * ATy[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = ATy[i];
    for (let j = i + 1; j < n; j++) s -= ATA[i][j] * x[j];
    x[i] = Math.abs(ATA[i][i]) < 1e-9 ? 0 : s / ATA[i][i];
  }
  return x;
}

// Fit a 3x4 affine matrix mapping source RGB (0-1) -> target RGB (0-1).
// Row k = [a, b, c, d] with out_k = a*R + b*G + c*B + d.
export function fitAffineColor(src: number[][], tgt: number[][]): number[][] {
  const A = src.map(s => [s[0], s[1], s[2], 1]);
  return [0, 1, 2].map(k => solveNormal(A, tgt.map(t => t[k])));
}

export function applyAffine(rgb01: number[], M: number[][]): number[] {
  return M.map(m => m[0] * rgb01[0] + m[1] * rgb01[1] + m[2] * rgb01[2] + m[3]);
}

export function residualRMS(src: number[][], tgt: number[][], M: number[][]): number {
  let s = 0;
  for (let i = 0; i < src.length; i++) {
    const c = applyAffine(src[i], M);
    s += (c[0] - tgt[i][0]) ** 2 + (c[1] - tgt[i][1]) ** 2 + (c[2] - tgt[i][2]) ** 2;
  }
  return Math.sqrt(s / src.length);
}

// Convenience: build source+target from an image + quad and fit.
export function fitFromQuad(data: Uint8ClampedArray, w: number, h: number, corners: Pt[]) {
  const pts = chipImagePoints(corners);
  const src = sampleChips(data, w, h, pts);
  const tgt = ASTRO_CHIPS.map(c => c.std);
  const M = fitAffineColor(src, tgt);
  return { matrix: M, residual: residualRMS(src, tgt, M), points: pts, source: src };
}

// Apply the affine transform to a full ImageData in place (values 0-255).
export function applyAffineToImageData(data: Uint8ClampedArray, M: number[][]): void {
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const nr = M[0][0] * r + M[0][1] * g + M[0][2] * b + M[0][3] * 255;
    const ng = M[1][0] * r + M[1][1] * g + M[1][2] * b + M[1][3] * 255;
    const nb = M[2][0] * r + M[2][1] * g + M[2][2] * b + M[2][3] * 255;
    data[i] = nr < 0 ? 0 : nr > 255 ? 255 : nr;
    data[i + 1] = ng < 0 ? 0 : ng > 255 ? 255 : ng;
    data[i + 2] = nb < 0 ? 0 : nb > 255 ? 255 : nb;
  }
}

// --- marker detection (js-aruco2) -------------------------------------------

// Detect the 4 corner fiducials. Returns corners ordered [TL,TR,BR,BL], or null
// if fewer than 2 usable markers are found. When exactly 2 (common under glare)
// are found we complete the parallelogram so the user can drag the rest.
export async function detectMarkerCorners(
  data: Uint8ClampedArray, w: number, h: number
): Promise<{ corners: Pt[] | null; found: number }> {
  let AR: any;
  try {
    const mod: any = await import('js-aruco2');
    AR = (mod.default || mod).AR || mod.AR;
  } catch {
    return { corners: null, found: 0 };
  }
  if (!AR || !AR.Detector) return { corners: null, found: 0 };

  let markers: any[] = [];
  try {
    const detector = new AR.Detector({ dictionaryName: 'ARUCO_MIP_36h12' });
    markers = detector.detectImage(w, h, data) || [];
  } catch {
    return { corners: null, found: 0 };
  }

  const areaOf = (c: Pt[]) => {
    let a = 0;
    for (let i = 0; i < 4; i++) { const p = c[i], q = c[(i + 1) % 4]; a += p.x * q.y - q.x * p.y; }
    return Math.abs(a / 2);
  };
  const centres = markers
    .map(m => ({ area: areaOf(m.corners), cx: m.corners.reduce((s: number, p: Pt) => s + p.x, 0) / 4, cy: m.corners.reduce((s: number, p: Pt) => s + p.y, 0) / 4 }))
    .filter(m => m.area > (w * h) * 0.0004) // drop tiny false positives
    .sort((a, b) => b.area - a.area)
    .slice(0, 4)
    .map(m => ({ x: m.cx, y: m.cy }));

  if (centres.length >= 4) return { corners: orderCorners(centres.slice(0, 4)), found: centres.length };
  if (centres.length === 3) {
    const [a, b, c] = orderCorners([...centres, centres[0]]).slice(0, 3);
    return { corners: orderCorners([a, b, c, { x: a.x + c.x - b.x, y: a.y + c.y - b.y }]), found: 3 };
  }
  if (centres.length === 2) {
    // assume the two found are a horizontal edge; extrude downward to seed a quad
    const [p, q] = centres.sort((m, n) => m.x - n.x);
    const dyGuess = Math.hypot(q.x - p.x, q.y - p.y) * 1.25;
    const corners = [p, q, { x: q.x, y: q.y + dyGuess }, { x: p.x, y: p.y + dyGuess }];
    return { corners: orderCorners(corners), found: 2 };
  }
  return { corners: null, found: centres.length };
}

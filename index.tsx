import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  Upload, 
  Leaf, 
  Pipette, 
  FileText,
  Maximize2,
  BarChart3,
  FlaskConical,
  Square,
  Circle as CircleIcon,
  Lasso,
  Plus,
  Trash2,
  Printer,
  ChevronLeft,
  ChevronRight,
  Github,
  Save,
  CheckCircle,
  AlertCircle,
  Wand2,
  MousePointer2, 
  RefreshCw,
  Sun,
  Moon,
  Disc,
  ZoomIn,
  ZoomOut,
  Move,
  Minimize,
  RotateCw,
  Ruler,
  ScanLine,
  Undo2,
  Download
} from 'lucide-react';
import JSZip from 'jszip';
import { ASTRO_CHIPS, detectMarkerCorners, fitFromQuad, scaleAndRotation, chipImagePoints, applyAffineToImageData, type Pt } from './colorcalib';

// --- Types ---

type ToolType = 'select' | 'rect' | 'circle' | 'lasso' | 'pan';
type VisualizationMode = 'rgb' | 'ngrdi' | 'maci' | 'gi';

interface Point { x: number; y: number; }

interface Shape {
  id: string;
  type: 'rect' | 'circle' | 'lasso';
  points: Point[]; 
}

interface ROIGroup {
  id: string;
  name: string;
  shapes: Shape[];
  color: string;
  stats?: {
    pixelCount: number;
    areaCm2: number;
    meanNGRDI: number;
    meanMACI: number;
    meanGI: number;
    anthocyanin: number;
    sumR: number; sumG: number; sumB: number;
    sumNGRDI: number; sumMACI: number; sumGI: number;
  };
}

interface ImageAsset {
  id: string;
  name: string;
  url: string;
}

interface AppState {
  gallery: ImageAsset[];
  activeImageIndex: number;
  
  segmentationThreshold: number;
  activeTab: 'segmentation' | 'calibration' | 'analysis' | 'report';
  visualizationMode: VisualizationMode;
  isProcessing: boolean;
  
  // Calibration
  activeCalibrationTarget: 'astro' | 'scale' | 'gray' | 'white' | 'black';

  // Astrocalibration marker (colour correction + auto scale/rotation)
  markerCorners: Pt[] | null;        // 4 corners, image coords, order TL,TR,BR,BL
  colorMatrix: number[][] | null;    // 3x4 affine RGB correction
  colorCorrectionEnabled: boolean;
  colorResidual: number | null;      // fit quality (RMS, 0-1 space)
  markerFound: number;               // # fiducials auto-detected
  isDetectingMarker: boolean;
  calibrationColor: { r: number, g: number, b: number } | null;
  calibrationROI: Shape | null;
  whitePointColor: { r: number, g: number, b: number } | null;
  whitePointROI: Shape | null;
  blackPointColor: { r: number, g: number, b: number } | null;
  blackPointROI: Shape | null;

  // Geometric & Lens Calibration
  rotationAngle: number;
  lensCorrection: number; // Barrel/Pincushion simulation
  markerPhysicalSize: number; // cm (physical edge length of the Astrocalibration scale marker)
  scaleROI: Shape | null; // ROI drawn over the scale marker
  pixelsPerCm: number | null;

  // Segmentation Editing
  exclusionZones: Shape[];

  // Quantification
  roiGroups: ROIGroup[];
  activeGroupId: string | null;

  // UI State
  isGithubModalOpen: boolean;
  activeTool: ToolType;
  selectedShapeId: string | null;
  zoom: number;
  pan: { x: number, y: number };
  
  // Regression & Reporting
  reportSummary: string;
  processedImageURL: string | null;
  reportImages: {
    original?: string;
    segmented?: string;
    ngrdi?: string;
    maci?: string;
    gi?: string;
  };
  regressionParams: { slope: number; intercept: number; targetIndex: 'mACI' | 'NGRDI' };
}

interface DragState {
  mode: 'move' | 'resize' | 'create' | 'pan' | 'corner';
  startPoint: Point;
  startScreenPoint: Point;
  activeHandle: string | null;
  initialPoints: Point[];
  initialPan?: { x: number, y: number };
  cornerIndex?: number;
}

// --- Constants ---

// Curated demo images. The Medicago frame has a clean, well-exposed
// Astrocalibration marker (best for the colour-correction workflow). The
// fast-plants frame shows two colour morphs for colour-distinction demos. The
// ExoLab-11 GRW08 timelapse frames also carry the marker; the Hydra-1 frame is
// a germination tray for segmentation practice.
const DEMO_IMAGES = [
  {
    label: 'Medicago ground control · clean marker',
    url: 'demos/medicago_marker.jpg'
  },
  {
    label: 'ArUco marker target · tilted (fallback test)',
    url: 'demos/aruco_markers_demo.png'
  },
  {
    label: 'Fast plants · two colour morphs',
    url: 'demos/fastplants_colour.jpg'
  },
  {
    label: 'ExoLab-11 · marker (glary)',
    url: 'https://raw.githubusercontent.com/dr-richard-barker/ExoLab_11/main/grw08_images_11122024/imaging_lens_position_7.0_cam_0_1731115802.jpg'
  },
  {
    label: 'ExoLab-11 · GRW08 timelapse (early)',
    url: 'https://raw.githubusercontent.com/dr-richard-barker/ExoLab_11/main/grw08_images_11122024/imaging_lens_position_7.0_cam_0_1730496602.jpg'
  },
  {
    label: 'ExoLab-11 · GRW08 timelapse (mid)',
    url: 'https://raw.githubusercontent.com/dr-richard-barker/ExoLab_11/main/grw08_images_11122024/imaging_lens_position_7.0_cam_0_1731040202.jpg'
  },
  {
    label: 'Hydra-1 · germination tray',
    url: 'https://raw.githubusercontent.com/dr-richard-barker/Hydra1-Orbital-Greenhouse/master/Raw%20images/Ground/Img-2018-12-22%2013_48_25.492088.png'
  }
];

const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
const HANDLE_SIZE = 8;

// --- Geometry Helpers ---

const isPointInShape = (x: number, y: number, shape: Shape) => {
  const p = { x, y };
  if (shape.type === 'rect') {
    const xMin = Math.min(shape.points[0].x, shape.points[1].x), xMax = Math.max(shape.points[0].x, shape.points[1].x);
    const yMin = Math.min(shape.points[0].y, shape.points[1].y), yMax = Math.max(shape.points[0].y, shape.points[1].y);
    return p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax;
  }
  if (shape.type === 'circle') {
    const r = Math.sqrt(Math.pow(shape.points[1].x - shape.points[0].x, 2) + Math.pow(shape.points[1].y - shape.points[0].y, 2));
    const dist = Math.sqrt(Math.pow(p.x - shape.points[0].x, 2) + Math.pow(p.y - shape.points[0].y, 2));
    return dist <= r;
  }
  if (shape.type === 'lasso') {
    let inside = false;
    for (let i = 0, j = shape.points.length - 1; i < shape.points.length; j = i++) {
      const xi = shape.points[i].x, yi = shape.points[i].y;
      const xj = shape.points[j].x, yj = shape.points[j].y;
      const intersect = ((yi > p.y) !== (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }
  return false;
};

const getBoundingBox = (shape: Shape) => {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  if (shape.type === 'rect' || shape.type === 'circle') {
    if (shape.type === 'rect') {
      minX = Math.min(shape.points[0].x, shape.points[1].x); maxX = Math.max(shape.points[0].x, shape.points[1].x);
      minY = Math.min(shape.points[0].y, shape.points[1].y); maxY = Math.max(shape.points[0].y, shape.points[1].y);
    } else {
      const r = Math.sqrt(Math.pow(shape.points[1].x - shape.points[0].x, 2) + Math.pow(shape.points[1].y - shape.points[0].y, 2));
      minX = shape.points[0].x - r; maxX = shape.points[0].x + r; minY = shape.points[0].y - r; maxY = shape.points[0].y + r;
    }
  } else {
    shape.points.forEach(p => { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); });
  }
  return { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
};

// --- Deterministic Report Summary ---
// Generates the report discussion from the measured statistics only. No LLM, so
// the same inputs always yield the same text — reproducible for scientific use.
const buildReportSummary = (groups: ROIGroup[], r: AppState['regressionParams']): string => {
  const scored = groups.filter(g => g.stats && g.stats.pixelCount > 0);
  if (scored.length === 0) {
    return 'No analysis groups with segmented plant pixels were defined, so no quantitative comparison could be made. Draw one or more ROI groups over vegetated regions and regenerate the report to populate this section.';
  }

  const fmt = (n: number, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : 'n/a');
  const byArea = [...scored].sort((a, b) => (b.stats!.areaCm2) - (a.stats!.areaCm2));
  const largest = byArea[0], smallest = byArea[byArea.length - 1];
  const meanGI = scored.reduce((s, g) => s + g.stats!.meanGI, 0) / scored.length;
  const meanMACI = scored.reduce((s, g) => s + g.stats!.meanMACI, 0) / scored.length;
  const idxName = r.targetIndex;

  const p1 = `A total of ${scored.length} region-of-interest cohort${scored.length > 1 ? 's were' : ' was'} quantified. ` +
    (largest.stats!.areaCm2 > 0
      ? `Projected leaf area ranged from ${fmt(smallest.stats!.areaCm2, 2)} cm² (${smallest.name}) to ${fmt(largest.stats!.areaCm2, 2)} cm² (${largest.name}). `
      : `Areas are reported in pixels because no scale marker was calibrated; set the Astrocalibration marker to obtain cm² values. `) +
    `Mean Green Index across cohorts was ${fmt(meanGI)}, and mean modified Anthocyanin Content Index (mACI) was ${fmt(meanMACI, 3)}.`;

  const p2 = `Cohort-level detail: ` +
    scored.map(g => `${g.name} — area ${fmt(g.stats!.areaCm2, 2)} cm², NGRDI ${fmt(g.stats!.meanNGRDI)}, mACI ${fmt(g.stats!.meanMACI)}, GI ${fmt(g.stats!.meanGI)}`).join('; ') + '. ' +
    `Higher NGRDI and Green Index values indicate greater relative greenness (a proxy for chlorophyll and canopy vigour), while elevated mACI indicates stronger red/anthocyanin accumulation that is often associated with stress or maturity. ` +
    `The estimated anthocyanin content shown in the group panels is derived from the ${idxName} index via the linear model anthocyanin = ${fmt(r.slope, 2)} × ${idxName} + ${fmt(r.intercept, 2)}; recalibrate the slope and intercept against your own pigment assay for absolute values.`;

  return `${p1}\n${p2}`;
};

// --- Automatic segmentation thresholds -------------------------------------
// All operate on a 256-bin histogram of the Excess-Green index (2G-R-B, clamped
// to 0..255) and return an ExG threshold that separates background from plant.

const otsuThreshold = (hist: number[]): number => {
  const total = hist.reduce((a, b) => a + b, 0);
  if (!total) return 0;
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = -1, thresh = 0;
  for (let i = 0; i < 256; i++) {
    wB += hist[i]; if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; thresh = i; }
  }
  return thresh;
};

const triangleThreshold = (hist: number[]): number => {
  let peak = 0, peakIdx = 0, min = 255, max = 0;
  for (let i = 0; i < 256; i++) {
    if (hist[i] > peak) { peak = hist[i]; peakIdx = i; }
    if (hist[i] > 0) { if (i < min) min = i; if (i > max) max = i; }
  }
  const end = (peakIdx - min) < (max - peakIdx) ? max : min; // farther tail
  const x1 = peakIdx, y1 = peak, x2 = end, y2 = hist[end] || 0;
  const lo = Math.min(peakIdx, end), hi = Math.max(peakIdx, end);
  let best = -1, thr = peakIdx;
  for (let i = lo; i <= hi; i++) {
    const dist = Math.abs((y2 - y1) * i - (x2 - x1) * hist[i] + x2 * y1 - y2 * x1);
    if (dist > best) { best = dist; thr = i; }
  }
  return thr;
};

const meanThreshold = (hist: number[]): number => {
  let s = 0, n = 0;
  for (let i = 0; i < 256; i++) { s += i * hist[i]; n += hist[i]; }
  return n ? Math.round(s / n) : 0;
};

// Iterative isodata / intermeans threshold.
const isodataThreshold = (hist: number[]): number => {
  let t = 128;
  for (let iter = 0; iter < 100; iter++) {
    let s1 = 0, n1 = 0, s2 = 0, n2 = 0;
    for (let i = 0; i < 256; i++) { if (i <= t) { s1 += i * hist[i]; n1 += hist[i]; } else { s2 += i * hist[i]; n2 += hist[i]; } }
    const m1 = n1 ? s1 / n1 : 0, m2 = n2 ? s2 / n2 : 0;
    const nt = Math.round((m1 + m2) / 2);
    if (nt === t) break; t = nt;
  }
  return t;
};

type ThresholdMethod = 'otsu' | 'triangle' | 'isodata' | 'mean';
const THRESHOLD_FNS: Record<ThresholdMethod, (h: number[]) => number> = {
  otsu: otsuThreshold, triangle: triangleThreshold, isodata: isodataThreshold, mean: meanThreshold,
};

// --- Per-group statistics (shared by the live pipeline and batch processing) ---

interface GroupStat {
  name: string; pixelCount: number; areaCm2: number;
  meanNGRDI: number; meanMACI: number; meanGI: number; anthocyanin: number;
  meanR: number; meanG: number; meanB: number;
}
interface StatsConfig {
  roiGroups: ROIGroup[]; exclusionZones: Shape[]; segmentationThreshold: number;
  colorCorrectionEnabled: boolean; colorMatrix: number[][] | null;
  whitePointColor: { r: number; g: number; b: number } | null;
  calibrationColor: { r: number; g: number; b: number } | null;
  pixelsPerCm: number | null;
  regressionParams: { slope: number; intercept: number; targetIndex: 'mACI' | 'NGRDI' };
}

// Compute per-ROI-group statistics for one image using the same segmentation +
// index maths as the live pipeline. Used for batch processing.
const computeGroupStats = (data: Uint8ClampedArray, w: number, h: number, cfg: StatsConfig): GroupStat[] => {
  if (cfg.colorCorrectionEnabled && cfg.colorMatrix) applyAffineToImageData(data, cfg.colorMatrix);
  let rS = 1, gS = 1, bS = 1;
  if (!(cfg.colorCorrectionEnabled && cfg.colorMatrix)) {
    if (cfg.whitePointColor) { rS = 255 / cfg.whitePointColor.r; gS = 255 / cfg.whitePointColor.g; bS = 255 / cfg.whitePointColor.b; }
    else if (cfg.calibrationColor) { rS = 128 / cfg.calibrationColor.r; gS = 128 / cfg.calibrationColor.g; bS = 128 / cfg.calibrationColor.b; }
  }
  const exZones = cfg.exclusionZones.map(s => ({ s, bb: getBoundingBox(s) }));
  const groupBoxes = cfg.roiGroups.map(g => g.shapes.map(s => ({ s, bb: getBoundingBox(s) })));
  const acc = cfg.roiGroups.map(() => ({ n: 0, sNG: 0, sMA: 0, sGI: 0, sR: 0, sG: 0, sB: 0 }));
  for (let i = 0; i < data.length; i += 4) {
    const p = i >> 2, x = p % w, y = (p / w) | 0;
    const r = Math.min(255, data[i] * rS), g = Math.min(255, data[i + 1] * gS), b = Math.min(255, data[i + 2] * bS);
    if ((2 * g - r - b) <= cfg.segmentationThreshold) continue;
    let excluded = false;
    for (const z of exZones) { const bb = z.bb; if (x >= bb.minX && x <= bb.maxX && y >= bb.minY && y <= bb.maxY && isPointInShape(x, y, z.s)) { excluded = true; break; } }
    if (excluded) continue;
    const ngrdi = (g + r) === 0 ? 0 : (g - r) / (g + r);
    const maci = g === 0 ? 0 : r / g;
    const gi = (r + g + b) === 0 ? 0 : g / (r + g + b);
    for (let k = 0; k < groupBoxes.length; k++) {
      let inG = false;
      for (const sh of groupBoxes[k]) { const bb = sh.bb; if (x >= bb.minX && x <= bb.maxX && y >= bb.minY && y <= bb.maxY && isPointInShape(x, y, sh.s)) { inG = true; break; } }
      if (inG) { const a = acc[k]; a.n++; a.sNG += ngrdi; a.sMA += maci; a.sGI += gi; a.sR += r; a.sG += g; a.sB += b; }
    }
  }
  const cm2 = cfg.pixelsPerCm ? 1 / (cfg.pixelsPerCm * cfg.pixelsPerCm) : 0;
  return cfg.roiGroups.map((grp, i) => {
    const a = acc[i], c = a.n || 1;
    const mMaci = a.sMA / c, mNgrdi = a.sNG / c, mGi = a.sGI / c;
    const antho = cfg.regressionParams.slope * (cfg.regressionParams.targetIndex === 'mACI' ? mMaci : mNgrdi) + cfg.regressionParams.intercept;
    return { name: grp.name, pixelCount: a.n, areaCm2: a.n * cm2, meanNGRDI: mNgrdi, meanMACI: mMaci, meanGI: mGi, anthocyanin: antho, meanR: a.sR / c, meanG: a.sG / c, meanB: a.sB / c };
  });
};

// --- CSV helpers (shared by single-image + batch export) ---
const CSV_HEADERS = ['image', 'group', 'pixel_count', 'area_cm2', 'mean_NGRDI', 'mean_mACI', 'mean_GI',
  'est_anthocyanin', 'mean_R', 'mean_G', 'mean_B', 'px_per_cm', 'exg_threshold', 'colour_residual', 'anthocyanin_model'];
interface CsvCtx { pxPerCm: number | null; exgThreshold: number; colourResidual: number | null; model: string; }
const csvEscape = (v: any) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
const statToRow = (image: string, st: GroupStat, ctx: CsvCtx) => [
  image, st.name, st.pixelCount,
  st.areaCm2.toFixed(4), st.meanNGRDI.toFixed(5), st.meanMACI.toFixed(5), st.meanGI.toFixed(5),
  st.anthocyanin.toFixed(5), st.meanR.toFixed(2), st.meanG.toFixed(2), st.meanB.toFixed(2),
  ctx.pxPerCm ? ctx.pxPerCm.toFixed(2) : '', ctx.exgThreshold,
  ctx.colourResidual != null ? ctx.colourResidual.toFixed(4) : '', ctx.model,
];
const buildCsvText = (rows: any[][]) => [CSV_HEADERS, ...rows].map(r => r.map(csvEscape).join(',')).join('\r\n');
const triggerDownload = (text: string, filename: string) => {
  const blob = new Blob(['﻿' + text], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
};

// --- Main App ---

const App = () => {
  const [state, setState] = useState<AppState>({
    gallery: [], activeImageIndex: 0, segmentationThreshold: 20, activeTab: 'segmentation', visualizationMode: 'rgb', isProcessing: false,
    activeCalibrationTarget: 'scale', calibrationColor: null, calibrationROI: null, whitePointColor: null, whitePointROI: null, blackPointColor: null, blackPointROI: null,
    rotationAngle: 0, lensCorrection: 0, markerPhysicalSize: 2.0, scaleROI: null, pixelsPerCm: null,
    markerCorners: null, colorMatrix: null, colorCorrectionEnabled: false, colorResidual: null, markerFound: 0, isDetectingMarker: false,
    exclusionZones: [], roiGroups: [], activeGroupId: null, isGithubModalOpen: false, activeTool: 'select', selectedShapeId: null, zoom: 1, pan: { x: 0, y: 0 },
    reportSummary: '', processedImageURL: null, reportImages: {}, regressionParams: { slope: 1.5, intercept: 0.2, targetIndex: 'mACI' }
  });

  const [dragState, setDragState] = useState<DragState | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const [githubConfig, setGithubConfig] = useState({ token: '', owner: '', repo: '', path: 'biopheno-results' });
  const [githubStatus, setGithubStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  // --- Undo history + deletion ---
  const stateRef = useRef(state); stateRef.current = state; // always the latest state
  const historyRef = useRef<Partial<AppState>[]>([]);
  const [historyLen, setHistoryLen] = useState(0);
  const [batchProgress, setBatchProgress] = useState<{ done: number; total: number; running: boolean; label: string } | null>(null);
  const [batchStride, setBatchStride] = useState(20);
  const SNAP_KEYS: (keyof AppState)[] = [
    'roiGroups', 'exclusionZones', 'activeGroupId', 'selectedShapeId', 'segmentationThreshold',
    'calibrationColor', 'calibrationROI', 'whitePointColor', 'whitePointROI', 'blackPointColor', 'blackPointROI',
    'scaleROI', 'markerCorners', 'colorMatrix', 'colorCorrectionEnabled', 'colorResidual', 'markerFound',
    'pixelsPerCm', 'rotationAngle', 'lensCorrection',
  ];
  // Snapshot the editable "document" state before a mutation so Ctrl+Z can undo it.
  const pushHistory = () => {
    const s: any = stateRef.current, snap: any = {};
    SNAP_KEYS.forEach(k => { snap[k] = s[k]; });
    historyRef.current = [...historyRef.current.slice(-49), snap];
    setHistoryLen(historyRef.current.length);
  };
  const undo = () => {
    const h = historyRef.current;
    if (!h.length) return;
    const prev = h[h.length - 1];
    historyRef.current = h.slice(0, -1);
    setHistoryLen(historyRef.current.length);
    setState(s => ({ ...s, ...prev }));
  };
  const deleteSelected = () => {
    const s = stateRef.current;
    const id = s.selectedShapeId;
    if (!id) return;
    pushHistory();
    setState(cur => ({
      ...cur,
      selectedShapeId: null,
      exclusionZones: cur.exclusionZones.filter(z => z.id !== id),
      roiGroups: cur.roiGroups.map(g => ({ ...g, shapes: g.shapes.filter(sh => sh.id !== id) })),
      scaleROI: cur.scaleROI?.id === id ? null : cur.scaleROI,
      calibrationROI: cur.calibrationROI?.id === id ? null : cur.calibrationROI,
      whitePointROI: cur.whitePointROI?.id === id ? null : cur.whitePointROI,
      blackPointROI: cur.blackPointROI?.id === id ? null : cur.blackPointROI,
    }));
  };

  // Automatic ExG segmentation threshold (Otsu and friends).
  const applyAutoThreshold = (method: ThresholdMethod) => {
    const raw = rawImageData(); if (!raw) return;
    const { data, w, h } = raw;
    if (stateRef.current.colorCorrectionEnabled && stateRef.current.colorMatrix) {
      applyAffineToImageData(data, stateRef.current.colorMatrix); // match the pipeline's corrected pixels
    }
    const hist = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      let exg = 2 * data[i + 1] - data[i] - data[i + 2];
      exg = exg < 0 ? 0 : exg > 255 ? 255 : exg;
      hist[exg | 0]++;
    }
    void w; void h;
    const t = Math.max(0, Math.min(150, Math.round(THRESHOLD_FNS[method](hist))));
    pushHistory();
    setState(s => ({ ...s, segmentationThreshold: t }));
  };

  const csvCtx = (s: AppState): CsvCtx => ({
    pxPerCm: s.pixelsPerCm, exgThreshold: s.segmentationThreshold, colourResidual: s.colorResidual,
    model: `${s.regressionParams.slope}*${s.regressionParams.targetIndex}+${s.regressionParams.intercept}`,
  });

  // Export the current image's ROI-group statistics as a CSV.
  const downloadCSV = () => {
    const s = stateRef.current;
    if (!s.roiGroups.length) return;
    const img = s.gallery[s.activeImageIndex]?.name || 'image';
    const ctx = csvCtx(s);
    const rows = s.roiGroups.map(g => {
      const st = g.stats, c = st?.pixelCount || 0;
      const gs: GroupStat = {
        name: g.name, pixelCount: c, areaCm2: st?.areaCm2 || 0,
        meanNGRDI: st?.meanNGRDI || 0, meanMACI: st?.meanMACI || 0, meanGI: st?.meanGI || 0, anthocyanin: st?.anthocyanin || 0,
        meanR: c ? st!.sumR / c : 0, meanG: c ? st!.sumG / c : 0, meanB: c ? st!.sumB / c : 0,
      };
      return statToRow(img, gs, ctx);
    });
    triggerDownload(buildCsvText(rows), `phenotype_${img.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_')}.csv`);
  };

  // --- Batch processing: apply the current ROIs + calibration to many images ---
  const loadImageData = (url: string) => new Promise<{ data: Uint8ClampedArray; w: number; h: number } | null>(resolve => {
    const img = new Image(); img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      const cx = c.getContext('2d', { willReadFrequently: true });
      if (!cx) { resolve(null); return; }
      cx.drawImage(img, 0, 0);
      resolve({ data: cx.getImageData(0, 0, img.width, img.height).data, w: img.width, h: img.height });
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });

  const runBatch = async (sources: { name: string; url: string }[]) => {
    const s = stateRef.current;
    if (!s.roiGroups.length) { alert('Draw at least one ROI group on a reference frame first — batch reuses those regions and your calibration on every image.'); return; }
    if (!sources.length) { setBatchProgress(null); return; }
    const cfg: StatsConfig = {
      roiGroups: s.roiGroups, exclusionZones: s.exclusionZones, segmentationThreshold: s.segmentationThreshold,
      colorCorrectionEnabled: s.colorCorrectionEnabled, colorMatrix: s.colorMatrix,
      whitePointColor: s.whitePointColor, calibrationColor: s.calibrationColor,
      pixelsPerCm: s.pixelsPerCm, regressionParams: s.regressionParams,
    };
    const ctx = csvCtx(s);
    const rows: any[][] = [];
    for (let i = 0; i < sources.length; i++) {
      setBatchProgress({ done: i, total: sources.length, running: true, label: sources[i].name });
      const im = await loadImageData(sources[i].url);
      if (im) computeGroupStats(im.data, im.w, im.h, cfg).forEach(st => rows.push(statToRow(sources[i].name, st, ctx)));
      else rows.push([sources[i].name, 'LOAD_ERROR', '', '', '', '', '', '', '', '', '', '', ctx.exgThreshold, '', ctx.model]);
      await new Promise(r => setTimeout(r, 0)); // let the UI/progress paint
    }
    setBatchProgress({ done: sources.length, total: sources.length, running: false, label: 'Done' });
    triggerDownload(buildCsvText(rows), `phenotype_batch_${sources.length}images.csv`);
  };

  // Fetch the ExoLab-11 GRW08 timelapse frame list and batch-process a sampled subset.
  const runExolabBatch = async () => {
    if (!stateRef.current.roiGroups.length) { alert('Load an ExoLab frame, draw ROI group(s) + set your threshold/calibration, then run the batch — it reuses them on every frame.'); return; }
    setBatchProgress({ done: 0, total: 0, running: true, label: 'Fetching ExoLab frame list…' });
    try {
      const res = await fetch('https://api.github.com/repos/dr-richard-barker/ExoLab_11/contents/grw08_images_11122024');
      const files = await res.json();
      if (!Array.isArray(files)) throw new Error('bad list');
      const frames = files.filter((f: any) => /^imaging_lens.*\.jpg$/i.test(f.name)).sort((a: any, b: any) => a.name.localeCompare(b.name));
      const stride = Math.max(1, batchStride | 0);
      const sampled = frames.filter((_: any, i: number) => i % stride === 0).map((f: any) => ({ name: f.name, url: f.download_url as string }));
      await runBatch(sampled);
    } catch {
      setBatchProgress(null);
      alert('Could not fetch the ExoLab frame list (GitHub API / network). You can still batch your own images with “Add images”.');
    }
  };

  const onBatchFiles = (files: FileList | null) => {
    if (!files || !files.length) return;
    runBatch([...files].map(f => ({ name: f.name, url: URL.createObjectURL(f) })));
  };

  useEffect(() => {
    setState(s => ({ ...s, gallery: DEMO_IMAGES.map((d, i) => ({ id: `demo-${i}`, name: d.label, url: d.url })), activeImageIndex: 0 }));
  }, []);

  // Keyboard shortcuts: Esc -> select/deselect; V select, R rect, C ellipse, L lasso, H pan.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSelected(); return; }
      const map: Record<string, ToolType> = { v: 'select', r: 'rect', c: 'circle', l: 'lasso', h: 'pan' };
      if (e.key === 'Escape') setState(s => ({ ...s, activeTool: 'select', selectedShapeId: null }));
      else if (map[e.key.toLowerCase()]) setState(s => ({ ...s, activeTool: map[e.key.toLowerCase()] }));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (!state.gallery[state.activeImageIndex]) return;
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = state.gallery[state.activeImageIndex].url;
    img.onload = () => {
      loadedImageRef.current = img;
      setState(s => ({ ...s, exclusionZones: [], roiGroups: [], activeGroupId: null, rotationAngle: 0, lensCorrection: 0, scaleROI: null, pixelsPerCm: null, selectedShapeId: null, markerCorners: null, colorMatrix: null, colorCorrectionEnabled: false, colorResidual: null, markerFound: 0 }));
      setTimeout(() => fitImageToScreen(), 60);
    };
  }, [state.activeImageIndex, state.gallery]);

  // Refit when the container first gains size or the window/pane is resized, so
  // the image/overlay are always on-screen and interactive.
  useEffect(() => {
    let last = 0;
    const refit = () => {
      const c = containerRef.current; if (!c) return;
      const size = c.clientWidth + c.clientHeight;
      if (Math.abs(size - last) > 8) { last = size; fitImageToScreen(); }
    };
    window.addEventListener('resize', refit);
    let ro: ResizeObserver | undefined;
    if (typeof ResizeObserver !== 'undefined' && containerRef.current) {
      ro = new ResizeObserver(refit);
      ro.observe(containerRef.current);
    }
    return () => { window.removeEventListener('resize', refit); ro?.disconnect(); };
  }, []);

  const fitImageToScreen = (retries = 30) => {
    const img = loadedImageRef.current, container = containerRef.current;
    if (!img) return; // nothing to fit yet; the image onload will call again
    // The container may not be mounted/laid out yet (null or 0 size) when this
    // first runs — retry on the next frame instead of committing a broken
    // zoom/pan (which pushed the canvas off-screen and made drawing impossible).
    const cw = container?.clientWidth ?? 0, ch = container?.clientHeight ?? 0;
    if (!container || cw < 40 || ch < 40) { if (retries > 0) setTimeout(() => fitImageToScreen(retries - 1), 30); return; }
    let zoom = Math.min((cw - 40) / img.width, (ch - 40) / img.height);
    if (!(zoom > 0) || !isFinite(zoom)) zoom = 1;
    zoom = Math.max(0.02, Math.min(20, zoom));
    setState(s => ({ ...s, zoom, pan: { x: (cw - img.width * zoom) / 2, y: (ch - img.height * zoom) / 2 } }));
  };

  useEffect(() => { if (state.activeTab !== 'report') runPipeline(); }, [
    state.segmentationThreshold, state.activeTab, state.visualizationMode, state.calibrationColor, state.whitePointColor, state.blackPointColor,
    state.rotationAngle, state.lensCorrection, state.exclusionZones, state.roiGroups, state.pixelsPerCm,
    state.colorCorrectionEnabled, state.colorMatrix, loadedImageRef.current
  ]);

  const runPipeline = (customMode?: VisualizationMode, targetCanvas?: HTMLCanvasElement) => {
    if (!loadedImageRef.current || (!canvasRef.current && !targetCanvas)) return;
    const img = loadedImageRef.current;
    const canvas = targetCanvas || canvasRef.current!;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const mode = customMode || state.visualizationMode;

    canvas.width = img.width; canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply Rotation & (Simulated) Lens Correction
    ctx.save();
    ctx.translate(canvas.width/2, canvas.height/2);
    ctx.rotate((state.rotationAngle * Math.PI) / 180);
    const lensScale = 1 + (state.lensCorrection / 500); 
    ctx.scale(lensScale, lensScale);
    ctx.drawImage(img, -img.width/2, -img.height/2);
    ctx.restore();

    if (!targetCanvas && overlayRef.current) { overlayRef.current.width = canvas.width; overlayRef.current.height = canvas.height; }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height), data = imageData.data;
      const width = canvas.width, height = canvas.height;
      const groupStats = state.roiGroups.map(g => ({ ...g, stats: { pixelCount: 0, areaCm2: 0, meanNGRDI: 0, meanMACI: 0, meanGI: 0, anthocyanin: 0, sumR: 0, sumG: 0, sumB: 0, sumNGRDI: 0, sumMACI: 0, sumGI: 0 } }));

      // Astrocalibration affine colour correction (supersedes the 3-point white
      // balance below when active).
      const colorCorrected = state.colorCorrectionEnabled && state.colorMatrix;
      if (colorCorrected) applyAffineToImageData(data, state.colorMatrix!);

      let rS = 1, gS = 1, bS = 1;
      if (!colorCorrected) {
        if (state.whitePointColor) { rS = 255/state.whitePointColor.r; gS = 255/state.whitePointColor.g; bS = 255/state.whitePointColor.b; }
        else if (state.calibrationColor) { rS = 128/state.calibrationColor.r; gS = 128/state.calibrationColor.g; bS = 128/state.calibrationColor.b; }
      }

      for (let i = 0; i < data.length; i += 4) {
        const x = (i/4) % width, y = Math.floor((i/4) / width);
        let r = Math.min(255, data[i] * rS), g = Math.min(255, data[i+1] * gS), b = Math.min(255, data[i+2] * bS);
        
        const exg = (2 * g) - r - b;
        let isPlant = exg > state.segmentationThreshold;
        if (isPlant) {
          for (const s of state.exclusionZones) { 
            const bb = getBoundingBox(s);
            if (x >= bb.minX && x <= bb.maxX && y >= bb.minY && y <= bb.maxY && isPointInShape(x, y, s)) { isPlant = false; break; }
          }
        }

        const ngrdi = (g+r) === 0 ? 0 : (g-r)/(g+r);
        const maci = g === 0 ? 0 : r/g;
        const gi = (r+g+b) === 0 ? 0 : g/(r+g+b);

        if (isPlant) {
          state.roiGroups.forEach((group, gIdx) => {
            let inThisGroup = false;
            for (const s of group.shapes) {
              const bb = getBoundingBox(s);
              if (x >= bb.minX && x <= bb.maxX && y >= bb.minY && y <= bb.maxY && isPointInShape(x, y, s)) { inThisGroup = true; break; }
            }
            if (inThisGroup || state.roiGroups.length === 0) {
              const s = groupStats[gIdx].stats;
              s.pixelCount++; s.sumNGRDI += ngrdi; s.sumMACI += maci; s.sumGI += gi;
              s.sumR += r; s.sumG += g; s.sumB += b;
            }
          });
        }

        // Visualization
        if (state.activeTab === 'analysis' || state.activeTab === 'report' || !!targetCanvas) {
          if (isPlant) {
            if (mode === 'ngrdi') { const t = Math.max(0, Math.min(1, ngrdi/0.5)); data[i] = 220*(1-t); data[i+1] = 220*(1-t)+100*t; data[i+2] = 50*(1-t); }
            else if (mode === 'maci') { const t = Math.max(0, Math.min(1, (maci-0.5)/1.0)); data[i] = 220*t; data[i+1] = 200*(1-t); data[i+2] = 50*t; }
            else if (mode === 'gi') { const t = Math.max(0, Math.min(1, (gi-0.3)/0.4)); data[i] = 50*(1-t); data[i+1] = 255*t; data[i+2] = 100*(1-t); }
            else { data[i] = r; data[i+1] = g; data[i+2] = b; }
          } else { const gr = 0.3*r + 0.59*g + 0.11*b; data[i] = data[i+1] = data[i+2] = gr*0.2; }
        } else if (state.activeTab === 'segmentation') {
          if (isPlant) { data[i] = 0; data[i+1] = 255; data[i+2] = 0; }
          else { const gr = 0.3*r + 0.59*g + 0.11*b; data[i] = data[i+1] = data[i+2] = gr*0.3; }
        } else { data[i] = r; data[i+1] = g; data[i+2] = b; }
      }

      if (!targetCanvas) {
        const cm2Factor = state.pixelsPerCm ? 1 / (state.pixelsPerCm * state.pixelsPerCm) : 0;
        const finalGroups = groupStats.map(g => {
          const c = g.stats.pixelCount || 1;
          const mMaci = g.stats.sumMACI / c, mNgrdi = g.stats.sumNGRDI / c, mGi = g.stats.sumGI / c;
          const antho = (state.regressionParams.slope * (state.regressionParams.targetIndex === 'mACI' ? mMaci : mNgrdi)) + state.regressionParams.intercept;
          return { ...g, stats: { ...g.stats, meanMACI: mMaci, meanNGRDI: mNgrdi, meanGI: mGi, anthocyanin: antho, areaCm2: g.stats.pixelCount * cm2Factor } };
        });
        if (JSON.stringify(state.roiGroups.map(g => g.stats)) !== JSON.stringify(finalGroups.map(g => g.stats)) && !dragState) {
          setTimeout(() => setState(s => ({ ...s, roiGroups: finalGroups })), 0);
        }
      }

      ctx.putImageData(imageData, 0, 0);
      if (!targetCanvas) drawOverlay();
    } catch (e) { console.error(e); }
  };

  const drawOverlay = () => {
    const canvas = overlayRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const hS = HANDLE_SIZE / state.zoom;

    const drawShape = (shape: Shape, color: string, fill: boolean = false, isSelected: boolean = false) => {
      ctx.beginPath();
      if (shape.type === 'rect') { ctx.rect(shape.points[0].x, shape.points[0].y, shape.points[1].x - shape.points[0].x, shape.points[1].y - shape.points[0].y); }
      else if (shape.type === 'circle') { const r = Math.sqrt(Math.pow(shape.points[1].x-shape.points[0].x,2)+Math.pow(shape.points[1].y-shape.points[0].y,2)); ctx.arc(shape.points[0].x, shape.points[0].y, r, 0, 2*Math.PI); }
      else if (shape.type === 'lasso') { ctx.moveTo(shape.points[0].x, shape.points[0].y); shape.points.slice(1).forEach(p => ctx.lineTo(p.x, p.y)); ctx.closePath(); }
      ctx.lineWidth = 2 / state.zoom; ctx.strokeStyle = color; ctx.stroke();
      if (fill) { ctx.fillStyle = color + '40'; ctx.fill(); }
      if (isSelected) {
         const bb = getBoundingBox(shape); ctx.save(); ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1/state.zoom; ctx.setLineDash([4/state.zoom, 4/state.zoom]);
         ctx.strokeRect(bb.minX - 2/state.zoom, bb.minY - 2/state.zoom, bb.width + 4/state.zoom, bb.height + 4/state.zoom); ctx.restore();
         ctx.fillStyle = '#38bdf8'; const handles = [{x:bb.minX, y:bb.minY}, {x:bb.maxX, y:bb.minY}, {x:bb.minX, y:bb.maxY}, {x:bb.maxX, y:bb.maxY}];
         handles.forEach(h => ctx.fillRect(h.x - hS/2, h.y - hS/2, hS, hS));
      }
    };

    if (state.activeTab !== 'report') {
      state.exclusionZones.forEach(s => drawShape(s, '#ef4444', true, s.id === state.selectedShapeId));
      if (state.activeTab === 'calibration') {
        if (state.markerCorners) drawMarkerOverlay(ctx, state.markerCorners);
        if (state.scaleROI) drawShape(state.scaleROI, '#a855f7', true, state.scaleROI.id === state.selectedShapeId);
        if (state.calibrationROI) drawShape(state.calibrationROI, '#ffffff', false, state.calibrationROI.id === state.selectedShapeId);
        if (state.whitePointROI) drawShape(state.whitePointROI, '#06b6d4', false, state.whitePointROI.id === state.selectedShapeId);
        if (state.blackPointROI) drawShape(state.blackPointROI, '#f97316', false, state.blackPointROI.id === state.selectedShapeId);
      }
      if (state.activeTab === 'analysis') {
        state.roiGroups.forEach(g => g.shapes.forEach(s => drawShape(s, g.color, false, s.id === state.selectedShapeId)));
      }
    }
  };

  // Deterministic scale calibration: derive pixels/cm from an ROI drawn over the
  // Astrocalibration marker of known physical edge length. No network / AI required.
  const computeScaleFromROI = (shape: Shape) => {
    const bb = getBoundingBox(shape);
    const px = Math.max(bb.width, bb.height);
    if (px <= 0 || state.markerPhysicalSize <= 0) return;
    setState(s => ({ ...s, pixelsPerCm: px / s.markerPhysicalSize }));
  };

  const startScaleCalibration = () => {
    setState(s => ({ ...s, activeCalibrationTarget: 'scale', activeTool: 'rect' }));
  };

  // --- Astrocalibration marker (colour correction + auto scale/rotation) ---

  const CORNER_HIT = 14; // px (image space) hit radius for dragging corners

  const drawMarkerOverlay = (ctx: CanvasRenderingContext2D, corners: Pt[]) => {
    const z = state.zoom > 0 ? state.zoom : 1; // guard: never negative/zero radius
    ctx.save();
    // sampling quad
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    corners.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.lineWidth = 2 / z; ctx.strokeStyle = '#a855f7'; ctx.stroke();
    // chip sample points
    chipImagePoints(corners).forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 4 / z, 0, 2 * Math.PI);
      ctx.fillStyle = '#f0abfc'; ctx.fill();
      ctx.lineWidth = 1 / z; ctx.strokeStyle = '#581c87'; ctx.stroke();
    });
    // draggable corner handles
    const hs = CORNER_HIT / z;
    corners.forEach(p => { ctx.fillStyle = '#a855f7'; ctx.fillRect(p.x - hs / 2, p.y - hs / 2, hs, hs); });
    ctx.restore();
  };

  const rawImageData = (): { data: Uint8ClampedArray; w: number; h: number } | null => {
    const img = loadedImageRef.current; if (!img) return null;
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const cx = c.getContext('2d', { willReadFrequently: true }); if (!cx) return null;
    cx.drawImage(img, 0, 0);
    const id = cx.getImageData(0, 0, img.width, img.height);
    return { data: id.data, w: img.width, h: img.height };
  };

  const refitColor = (corners: Pt[]) => {
    const raw = rawImageData(); if (!raw) return;
    const fit = fitFromQuad(raw.data, raw.w, raw.h, corners);
    setState(s => ({ ...s, colorMatrix: fit.matrix, colorResidual: fit.residual, colorCorrectionEnabled: true }));
  };

  // A centred landscape quad (~card aspect) to seed manual placement when
  // auto-detection can't find the fiducials (common: the Astrobotany markers use
  // custom icons that generic ArUco decoders don't read).
  const defaultQuad = (w: number, h: number): Pt[] => {
    const cw = w * 0.4, ch = cw / 1.3, cx = w / 2, cy = h / 2;
    return [
      { x: cx - cw / 2, y: cy - ch / 2 }, { x: cx + cw / 2, y: cy - ch / 2 },
      { x: cx + cw / 2, y: cy + ch / 2 }, { x: cx - cw / 2, y: cy + ch / 2 },
    ];
  };

  const handleDetectMarker = async () => {
    const raw = rawImageData(); if (!raw) return;
    // ?detector=aruco forces the ArUco-decoder fallback (skips the geometric
    // square finder, which otherwise always wins on real ArUco marker images).
    // Handy with the "ArUco marker target" demo to test the decoder path.
    const skipGeometric = new URLSearchParams(window.location.search).get('detector') === 'aruco';
    setState(s => ({ ...s, isDetectingMarker: true, activeCalibrationTarget: 'astro' }));
    const det = await detectMarkerCorners(raw.data, raw.w, raw.h, { skipGeometric });
    // Only trust a clean 4-fiducial detection. Generic ArUco decoders give
    // unreliable partial/false matches on these custom markers, so for anything
    // less we seed a predictable centred quad for the user to place by hand.
    const trusted = det.corners && det.found >= 4;
    const corners = trusted ? det.corners! : defaultQuad(raw.w, raw.h);
    const fit = fitFromQuad(raw.data, raw.w, raw.h, corners);
    const { pxPerCm } = scaleAndRotation(corners);
    setState(s => ({
      ...s, markerCorners: corners, markerFound: det.found, isDetectingMarker: false,
      colorMatrix: fit.matrix, colorResidual: fit.residual,
      colorCorrectionEnabled: trusted,
      pixelsPerCm: trusted && pxPerCm > 0 && isFinite(pxPerCm) ? pxPerCm : s.pixelsPerCm,
    }));
    if (!trusted) {
      alert('Auto-detect isn’t reliable for the Astrobotany marker (custom fiducial icons). Drag the four purple corner handles onto the marker’s corner squares — TL on the ASTROBOTANY / plus corner — then tick “Apply colour correction”.');
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width), y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    const startPoint = { x, y };

    if (state.activeTool === 'pan' || e.button === 1) { setDragState({ mode: 'pan', startPoint, startScreenPoint: { x: e.clientX, y: e.clientY }, activeHandle: null, initialPoints: [], initialPan: { ...state.pan } }); return; }

    // Astrocalibration: drag a marker corner handle (takes priority over tools).
    if (state.activeTab === 'calibration' && state.markerCorners) {
      const hit = CORNER_HIT / (state.zoom || 1);
      const idx = state.markerCorners.findIndex(p => Math.abs(p.x - x) <= hit && Math.abs(p.y - y) <= hit);
      if (idx >= 0) { pushHistory(); setDragState({ mode: 'corner', startPoint, startScreenPoint: { x: 0, y: 0 }, activeHandle: null, initialPoints: [], cornerIndex: idx }); return; }
    }

    if (state.activeTool === 'select') {
      const hit = getShapeUnderCursor(x, y);
      if (hit) {
        pushHistory(); // about to move a shape
        setState(s => ({ ...s, selectedShapeId: hit.shape.id, activeGroupId: hit.group?.id || s.activeGroupId }));
        setDragState({ mode: 'move', startPoint, startScreenPoint: { x:0,y:0 }, activeHandle: null, initialPoints: JSON.parse(JSON.stringify(hit.shape.points)) });
      } else { setState(s => ({ ...s, selectedShapeId: null })); }
    } else {
      pushHistory(); // about to create a shape
      const newShape: Shape = { id: Math.random().toString(36), type: state.activeTool as any, points: [startPoint, startPoint] };
      if (state.activeTab === 'segmentation') setState(s => ({ ...s, exclusionZones: [...s.exclusionZones, newShape], selectedShapeId: newShape.id }));
      else if (state.activeTab === 'calibration') {
        const updates: any = { selectedShapeId: newShape.id };
        if (state.activeCalibrationTarget === 'scale') updates.scaleROI = newShape;
        else if (state.activeCalibrationTarget === 'gray') updates.calibrationROI = newShape;
        else if (state.activeCalibrationTarget === 'white') updates.whitePointROI = newShape;
        else updates.blackPointROI = newShape;
        setState(s => ({ ...s, ...updates }));
      } else if (state.activeTab === 'analysis') {
        if (state.activeGroupId) setState(s => ({ ...s, roiGroups: s.roiGroups.map(g => g.id === s.activeGroupId ? { ...g, shapes: [...g.shapes, newShape] } : g), selectedShapeId: newShape.id }));
        else { const nG = { id: Math.random().toString(36), name: `Group ${state.roiGroups.length + 1}`, color: COLORS[state.roiGroups.length%COLORS.length], shapes: [newShape] }; setState(s => ({ ...s, roiGroups: [...s.roiGroups, nG], activeGroupId: nG.id, selectedShapeId: newShape.id })); }
      }
      setDragState({ mode: 'create', startPoint, startScreenPoint: { x:0,y:0 }, activeHandle: null, initialPoints: [] });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragState || !canvasRef.current) return;
    if (dragState.mode === 'pan' && dragState.initialPan) { setState(s => ({ ...s, pan: { x: dragState.initialPan!.x + (e.clientX - dragState.startScreenPoint.x), y: dragState.initialPan!.y + (e.clientY - dragState.startScreenPoint.y) } })); return; }
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width), y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    const curr = { x, y };

    if (dragState.mode === 'corner' && dragState.cornerIndex != null && state.markerCorners) {
      const idx = dragState.cornerIndex;
      setState(s => ({ ...s, markerCorners: s.markerCorners!.map((p, i) => i === idx ? curr : p) }));
      return;
    }

    let sId = state.selectedShapeId; if (!sId) return;
    const update = (newP: Point[]) => {
      if (state.activeTab === 'segmentation') setState(s => ({ ...s, exclusionZones: s.exclusionZones.map(sh => sh.id === sId ? { ...sh, points: newP } : sh) }));
      else if (state.activeTab === 'calibration') {
        if (state.scaleROI?.id === sId) setState(s => ({ ...s, scaleROI: { ...s.scaleROI!, points: newP } }));
        else if (state.calibrationROI?.id === sId) setState(s => ({ ...s, calibrationROI: { ...s.calibrationROI!, points: newP } }));
        else if (state.whitePointROI?.id === sId) setState(s => ({ ...s, whitePointROI: { ...s.whitePointROI!, points: newP } }));
        else if (state.blackPointROI?.id === sId) setState(s => ({ ...s, blackPointROI: { ...s.blackPointROI!, points: newP } }));
      } else if (state.activeTab === 'analysis') setState(s => ({ ...s, roiGroups: s.roiGroups.map(g => ({ ...g, shapes: g.shapes.map(sh => sh.id === sId ? { ...sh, points: newP } : sh) })) }));
    };

    if (dragState.mode === 'create') {
      const shape = getSelectedShapeObj(); if (!shape) return;
      const nP = [...shape.points]; if (shape.type === 'lasso') nP.push(curr); else nP[1] = curr; update(nP);
    } else if (dragState.mode === 'move') {
      const dx = curr.x - dragState.startPoint.x, dy = curr.y - dragState.startPoint.y;
      update(dragState.initialPoints.map(p => ({ x: p.x + dx, y: p.y + dy })));
    }
  };

  const getSelectedShapeObj = () => {
    const sId = state.selectedShapeId;
    if (state.activeTab === 'segmentation') return state.exclusionZones.find(s => s.id === sId);
    if (state.activeTab === 'calibration') return [state.scaleROI, state.calibrationROI, state.whitePointROI, state.blackPointROI].find(s => s?.id === sId);
    let f; state.roiGroups.forEach(g => { const s = g.shapes.find(sh => sh.id === sId); if(s) f = s; }); return f;
  };

  const getShapeUnderCursor = (x: number, y: number) => {
    if (state.activeTab === 'segmentation') { for (const s of state.exclusionZones) if (isPointInShape(x, y, s)) return { shape: s }; }
    else if (state.activeTab === 'calibration') {
      if (state.scaleROI && isPointInShape(x, y, state.scaleROI)) return { shape: state.scaleROI, type: 'scale' };
      if (state.blackPointROI && isPointInShape(x, y, state.blackPointROI)) return { shape: state.blackPointROI, type: 'black' };
      if (state.whitePointROI && isPointInShape(x, y, state.whitePointROI)) return { shape: state.whitePointROI, type: 'white' };
      if (state.calibrationROI && isPointInShape(x, y, state.calibrationROI)) return { shape: state.calibrationROI, type: 'gray' };
    } else if (state.activeTab === 'analysis') { for (const g of state.roiGroups) for (const s of g.shapes) if (isPointInShape(x, y, s)) return { shape: s, group: g }; }
    return null;
  };

  const handleMouseUp = () => {
    if (dragState?.mode === 'corner' && state.markerCorners) {
      refitColor(state.markerCorners); // recompute colour transform from the adjusted quad
      setDragState(null);
      return;
    }
    if (state.activeTab === 'calibration' && state.selectedShapeId) {
      const s = getSelectedShapeObj();
      if (s) {
        if (state.scaleROI?.id === s.id) {
          computeScaleFromROI(s);
        } else {
          const type = state.calibrationROI?.id === s.id ? 'gray' : (state.whitePointROI?.id === s.id ? 'white' : 'black');
          calculateCalibrationFromROI(s, type as any);
        }
      }
    }
    setDragState(null);
    // Keep the drawing tool active so multiple ROIs can be drawn in a row.
    // The lasso is a one-shot gesture, so it reverts to select after each stroke;
    // calibration ROIs are single-purpose, so they revert too.
    if (state.activeTool === 'lasso' || (state.activeTab === 'calibration' && state.activeTool !== 'select' && state.activeTool !== 'pan')) {
      setState(s => ({ ...s, activeTool: 'select' }));
    }
  };

  const calculateCalibrationFromROI = (s: Shape, type: 'gray'|'white'|'black') => {
    if (!canvasRef.current) return;
    const bb = getBoundingBox(s), ctx = canvasRef.current.getContext('2d'); if (!ctx) return;
    const data = ctx.getImageData(bb.minX, bb.minY, bb.width||1, bb.height||1).data;
    let r=0, g=0, b=0, count=0;
    for (let i=0; i<data.length; i+=4) { r+=data[i]; g+=data[i+1]; b+=data[i+2]; count++; }
    const color = { r: r/count, g: g/count, b: b/count };
    if (type === 'gray') setState(s => ({ ...s, calibrationColor: color }));
    else if (type === 'white') setState(s => ({ ...s, whitePointColor: color }));
    else setState(s => ({ ...s, blackPointColor: color }));
  };

  // Append an uploaded image to the gallery and switch to it (keeps demos available).
  const handleImageUpload = (url: string, label: string) => {
    setState(s => ({ ...s, gallery: [...s.gallery, { id: `upload-${s.gallery.length}`, name: label, url }], activeImageIndex: s.gallery.length }));
  };

  const handleGenerateReport = async () => {
    if (!loadedImageRef.current) return;
    
    setState(s => ({ ...s, isProcessing: true, activeTab: 'report' }));

    const tempCanvas = document.createElement('canvas');
    const images: any = {};
    
    // Original (RGB version but with calibration applied)
    runPipeline('rgb', tempCanvas);
    images.original = tempCanvas.toDataURL('image/png');
    
    // NGRDI
    runPipeline('ngrdi', tempCanvas);
    images.ngrdi = tempCanvas.toDataURL('image/png');
    
    // mACI
    runPipeline('maci', tempCanvas);
    images.maci = tempCanvas.toDataURL('image/png');
    
    // GI
    runPipeline('gi', tempCanvas);
    images.gi = tempCanvas.toDataURL('image/png');

    // Segmented (Green mask)
    const oldTab = state.activeTab;
    const stateMock = { ...state, activeTab: 'segmentation' as const };
    // This is tricky because runPipeline depends on state. We'll manually replicate the segmentation view.
    const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
    if (ctx) {
      const imgData = ctx.getImageData(0,0, tempCanvas.width, tempCanvas.height);
      const d = imgData.data;
      for (let i=0; i<d.length; i+=4) {
        // ExG calculation simulation
        const r = d[i], g = d[i+1], b = d[i+2];
        const exg = (2*g) - r - b;
        if (exg > state.segmentationThreshold) { d[i]=0; d[i+1]=255; d[i+2]=0; }
        else { const gr = 0.3*r + 0.59*g + 0.11*b; d[i]=d[i+1]=d[i+2]=gr*0.3; }
      }
      ctx.putImageData(imgData, 0, 0);
      images.segmented = tempCanvas.toDataURL('image/png');
    }

    const sum = buildReportSummary(state.roiGroups, state.regressionParams);
    setState(s => ({ ...s, reportSummary: sum, reportImages: images, isProcessing: false }));
  };

  const NavButton = ({ active, onClick, icon, label }: any) => (
    <button onClick={onClick} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${active ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}>{icon} {label}</button>
  );

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <aside className="w-64 border-r border-slate-800 bg-slate-900 flex flex-col print:hidden">
        <div className="p-6 border-b border-slate-800">
          <h1 className="flex items-center gap-2 font-bold text-lg text-emerald-400"><Leaf className="w-6 h-6" /> BioPheno</h1>
          <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-tighter font-mono">Precision Plant Analytics</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <NavButton active={state.activeTab === 'segmentation'} onClick={() => setState(s => ({ ...s, activeTab: 'segmentation' }))} icon={<Maximize2 size={18} />} label="Segmentation" />
          <NavButton active={state.activeTab === 'calibration'} onClick={() => setState(s => ({ ...s, activeTab: 'calibration' }))} icon={<Pipette size={18} />} label="Calibration & Units" />
          <NavButton active={state.activeTab === 'analysis'} onClick={() => setState(s => ({ ...s, activeTab: 'analysis' }))} icon={<BarChart3 size={18} />} label="Results" />
          <div className="pt-4 mt-4 border-t border-slate-800 space-y-2">
            <button onClick={handleGenerateReport} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${state.activeTab === 'report' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-800'}`}><FileText size={18} /> Generate Report</button>
            <a href="guide/" target="_blank" rel="noopener" className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-all"><FileText size={18} /> User Guide ↗</a>
          </div>
        </nav>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        {state.activeTab !== 'report' && (
          <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur z-20">
            <div className="flex items-center gap-4">
              <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 shadow-inner">
                <button title="Pan (hold to move the image)" onClick={() => setState(s => ({...s, activeTool: 'pan'}))} className={`p-2 rounded ${state.activeTool === 'pan' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Move size={16} /></button>
                <div className="w-px h-4 bg-slate-700 mx-1" />
                <button title="Select / move / resize regions" onClick={() => setState(s => ({...s, activeTool: 'select'}))} className={`p-2 rounded ${state.activeTool === 'select' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><MousePointer2 size={16} /></button>
                <button title="Rectangle — drag on the image to draw a region" onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Square size={16} /></button>
                <button title="Ellipse — drag on the image to draw a region" onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><CircleIcon size={16} /></button>
                <button title="Lasso — click-drag a freehand outline" onClick={() => setState(s => ({...s, activeTool: 'lasso'}))} className={`p-2 rounded ${state.activeTool === 'lasso' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Lasso size={16} /></button>
              </div>
              <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 shadow-inner">
                <button title="Undo (Ctrl+Z)" disabled={historyLen === 0} onClick={undo} className={`p-2 rounded ${historyLen > 0 ? 'text-slate-300 hover:text-white' : 'text-slate-600 cursor-not-allowed'}`}><Undo2 size={16} /></button>
                <button title="Delete selected region (Del)" disabled={!state.selectedShapeId} onClick={deleteSelected} className={`p-2 rounded ${state.selectedShapeId ? 'text-rose-400 hover:text-rose-300' : 'text-slate-600 cursor-not-allowed'}`}><Trash2 size={16} /></button>
              </div>
              <span className="text-[10px] font-mono text-slate-500 hidden lg:inline">
                {state.activeTool === 'select' ? 'Select a region to move / delete (Del); pick ▢ ○ ⬡ to draw' :
                 state.activeTool === 'pan' ? 'Pan mode' : `Drawing: drag on the image`}
              </span>
              {state.activeTab === 'analysis' && (
                <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-2 shadow-inner">
                   <button onClick={() => setState(s => ({...s, visualizationMode: 'rgb'}))} className={`px-2 py-1 text-[10px] rounded transition-all ${state.visualizationMode === 'rgb' ? 'bg-slate-600 text-white' : 'text-slate-400'}`}>RGB</button>
                   <button onClick={() => setState(s => ({...s, visualizationMode: 'ngrdi'}))} className={`px-2 py-1 text-[10px] rounded transition-all ${state.visualizationMode === 'ngrdi' ? 'bg-emerald-600 text-white' : 'text-slate-400'}`}>NGRDI</button>
                   <button onClick={() => setState(s => ({...s, visualizationMode: 'maci'}))} className={`px-2 py-1 text-[10px] rounded transition-all ${state.visualizationMode === 'maci' ? 'bg-rose-600 text-white' : 'text-slate-400'}`}>mACI</button>
                   <button onClick={() => setState(s => ({...s, visualizationMode: 'gi'}))} className={`px-2 py-1 text-[10px] rounded transition-all ${state.visualizationMode === 'gi' ? 'bg-green-600 text-white' : 'text-slate-400'}`}>GI</button>
                </div>
              )}
            </div>
            <div className="flex items-center gap-3">
               {state.gallery.length > 0 && (
                 <select
                   value={state.activeImageIndex}
                   onChange={(e) => setState(s => ({ ...s, activeImageIndex: parseInt(e.target.value) }))}
                   title="Load a demo image or a previous upload"
                   className="bg-slate-800 border border-slate-700 rounded text-[11px] text-slate-200 px-2 py-2 max-w-[220px] focus:outline-none focus:border-emerald-500"
                 >
                   {state.gallery.map((g, i) => (
                     <option key={g.id} value={i}>{g.name}</option>
                   ))}
                 </select>
               )}
               <div className="flex items-center bg-slate-800 rounded border border-slate-700 px-2 py-1">
                  <button onClick={() => setState(s => ({...s, zoom: s.zoom*0.8}))} className="hover:text-white text-slate-400"><ZoomOut size={14}/></button>
                  <span className="text-[10px] w-12 text-center font-mono">{(state.zoom*100).toFixed(0)}%</span>
                  <button onClick={() => setState(s => ({...s, zoom: s.zoom*1.2}))} className="hover:text-white text-slate-400"><ZoomIn size={14}/></button>
                  <button className="ml-2 hover:text-white text-slate-400" onClick={fitImageToScreen}><Minimize size={14}/></button>
               </div>
               <button onClick={() => document.getElementById('f-in')?.click()} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-sm font-medium transition-colors">Upload</button>
               <input id="f-in" type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if(f) handleImageUpload(URL.createObjectURL(f), f.name); }} />
            </div>
          </header>
        )}

        <div className={`flex-1 flex overflow-hidden ${state.activeTab === 'report' ? 'hidden' : ''}`}>
          <div className="flex-1 p-6 relative overflow-hidden bg-slate-950">
            <div ref={containerRef} onWheel={(e) => { e.preventDefault(); const f = e.deltaY>0?0.9:1.1; setState(s => ({...s, zoom: Math.max(0.1, Math.min(20, s.zoom*f))})); }} className={`w-full h-full border border-slate-800 rounded-xl relative overflow-hidden shadow-2xl ${state.activeTool === 'pan' ? 'cursor-grab' : (state.activeTool === 'select' ? 'cursor-default' : 'cursor-crosshair')}`}>
              <div style={{ transform: `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`, transformOrigin: '0 0', width: '100%', height: '100%' }}>
                {/* Both canvases render at intrinsic buffer size (the image dimensions) so
                    they overlap exactly; the wrapper's scale(zoom) fits them to the view. */}
                <canvas ref={canvasRef} className="absolute top-0 left-0 block" />
                <canvas ref={overlayRef} className="absolute top-0 left-0 block pointer-events-auto" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} />
              </div>
            </div>
          </div>

          <aside className="w-80 border-l border-slate-800 bg-slate-900 p-6 overflow-y-auto">
            {state.activeTab === 'segmentation' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Maximize2 size={12}/> Extraction</h3>
                  <div className="flex justify-between text-xs mb-2"><span>ExG Threshold</span><span className="text-emerald-400 font-mono">{state.segmentationThreshold}</span></div>
                  <input type="range" min="0" max="150" value={state.segmentationThreshold} onMouseDown={pushHistory} onChange={(e) => setState(s => ({ ...s, segmentationThreshold: parseInt(e.target.value) }))} className="w-full accent-emerald-500 h-1" />
                  <div className="mt-4">
                    <div className="flex items-center gap-2 mb-2 text-[10px] text-slate-500"><Wand2 size={12}/> <span className="uppercase font-bold">Auto threshold</span></div>
                    <div className="grid grid-cols-2 gap-1.5">
                      {([['otsu','Otsu'],['triangle','Triangle'],['isodata','IsoData'],['mean','Mean']] as [ThresholdMethod,string][]).map(([m,label]) => (
                        <button key={m} onClick={() => applyAutoThreshold(m)} className="py-1.5 rounded text-[10px] font-bold bg-slate-800 border border-slate-700 text-slate-300 hover:bg-emerald-600 hover:text-white hover:border-emerald-500 transition-all">{label}</button>
                      ))}
                    </div>
                    <p className="text-[9px] text-slate-500 leading-relaxed mt-2 italic">Compute the ExG threshold automatically from the image histogram. Otsu suits bimodal plant/background scenes; Triangle handles a dominant background.</p>
                  </div>
                </div>
                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4">Exclusion Zones</h3>
                  <p className="text-[10px] text-slate-400 leading-relaxed mb-4 italic">Draw regions to mask non-plant features (soil, labels, artifacts).</p>
                  <div className="space-y-2">
                    {state.exclusionZones.map(z => (
                      <div key={z.id} className="flex items-center justify-between p-2 bg-slate-800 rounded border border-slate-700 text-[10px]">
                        <span>Excluded Area ({z.type})</span>
                        <button title="Delete this zone" onClick={() => { pushHistory(); setState(s => ({...s, exclusionZones: s.exclusionZones.filter(e => e.id !== z.id)})); }} className="text-rose-400 hover:text-rose-300"><Trash2 size={12} /></button>
                      </div>
                    ))}
                    {state.exclusionZones.length === 0 && <div className="text-[10px] text-slate-600 text-center py-4 border border-dashed border-slate-800 rounded">No zones defined</div>}
                  </div>
                </div>
              </div>
            )}

            {state.activeTab === 'calibration' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><ScanLine size={14}/> Astrocalibration Marker</h3>
                  <p className="text-[10px] text-slate-400 leading-relaxed mb-3 italic">One click auto-detects the marker's 4 corner fiducials and colour-corrects the image. On a clean, flat marker this "just works"; otherwise drag the purple corner handles onto the fiducials. For batch/timelapse processing use the <b>PlantCV Pro notebook</b>.</p>
                  <button disabled={state.isDetectingMarker} onClick={handleDetectMarker} className="w-full flex items-center justify-center gap-2 py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded text-[10px] font-bold transition-all mb-3">
                    {state.isDetectingMarker ? <RefreshCw className="animate-spin" size={14}/> : <ScanLine size={14}/>} {state.isDetectingMarker ? 'Detecting…' : 'Detect / place marker'}
                  </button>
                  {state.markerCorners && (
                    <div className="space-y-2">
                      <label className="flex items-center justify-between text-[10px] text-slate-300 bg-slate-950 p-2 rounded border border-slate-800">
                        <span>Apply colour correction</span>
                        <input type="checkbox" checked={state.colorCorrectionEnabled} onChange={(e) => setState(s => ({ ...s, colorCorrectionEnabled: e.target.checked }))} className="accent-purple-500" />
                      </label>
                      <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                        <div className="bg-slate-950 p-2 rounded border border-slate-800 text-slate-400">Fiducials <span className="text-purple-300">{state.markerFound}/4</span></div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800 text-slate-400">Residual <span className={state.colorResidual != null && state.colorResidual < 0.13 ? 'text-emerald-400' : 'text-amber-400'}>{state.colorResidual != null ? state.colorResidual.toFixed(3) : '-'}</span></div>
                      </div>
                      <p className="text-[9px] text-slate-500 leading-relaxed">Drag the 4 purple handles onto the marker's corner fiducials — <b>TL on the ASTROBOTANY / plus corner</b> — so the pink dots land on the colour patches. Lower residual = better fit (~0.12 is typical for a good print; over-exposed markers fit worse).</p>
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><RotateCw size={14}/> Geometric Alignment</h3>
                  <div className="mt-1">
                    <div className="flex justify-between text-[10px] mb-2 text-slate-400"><span>Tilt Correction</span><span className="font-mono text-indigo-400">{state.rotationAngle.toFixed(1)}°</span></div>
                    <input type="range" min="-180" max="180" step="0.1" value={state.rotationAngle} onChange={(e) => setState(s => ({ ...s, rotationAngle: parseFloat(e.target.value) }))} className="w-full accent-indigo-500 h-1" />
                  </div>
                  <div className="mt-4">
                    <div className="flex justify-between text-[10px] mb-2 text-slate-400"><span>Lens Correction</span><span className="font-mono text-indigo-400">{state.lensCorrection.toFixed(0)}</span></div>
                    <input type="range" min="-50" max="50" step="1" value={state.lensCorrection} onChange={(e) => setState(s => ({ ...s, lensCorrection: parseFloat(e.target.value) }))} className="w-full accent-indigo-500 h-1" />
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Ruler size={14}/> Scale &amp; Units</h3>
                  <p className="text-[10px] text-slate-400 leading-relaxed mb-3 italic">Enter the physical edge length of the Astrocalibration marker, then draw a box tightly over it to derive pixels/cm.</p>
                  <button onClick={startScaleCalibration} className={`w-full flex items-center justify-center gap-2 py-2 rounded text-[10px] font-bold transition-all mb-4 border ${state.activeCalibrationTarget === 'scale' ? 'bg-purple-600 border-purple-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}>
                    <Ruler size={14}/> {state.activeCalibrationTarget === 'scale' ? 'Draw box over marker…' : 'Set Scale from Marker'}
                  </button>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[9px] text-slate-500 block mb-1">Marker Size (cm)</label>
                      <input type="number" value={state.markerPhysicalSize} onChange={(e) => setState(s => ({...s, markerPhysicalSize: parseFloat(e.target.value) || 0}))} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs text-white" />
                    </div>
                    <div>
                      <label className="text-[9px] text-slate-500 block mb-1">Pixels / Cm</label>
                      <div className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs font-mono text-emerald-400">{state.pixelsPerCm?.toFixed(1) || '-'}</div>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Pipette size={14}/> Color Reference</h3>
                  <div className="grid grid-cols-3 gap-1 mb-4">
                    {['gray', 'white', 'black'].map(t => (
                      <button key={t} onClick={() => setState(s => ({...s, activeCalibrationTarget: t as any}))} className={`p-2 rounded text-[9px] uppercase border transition-all ${state.activeCalibrationTarget === t ? 'bg-slate-700 border-indigo-500 text-white shadow-sm' : 'bg-transparent border-slate-800 text-slate-500'}`}>{t}</button>
                    ))}
                  </div>
                  <div className="space-y-1">
                  {['gray', 'white', 'black'].map(t => {
                    const c = t==='gray'?state.calibrationColor : (t==='white'?state.whitePointColor : state.blackPointColor);
                    return c && (
                      <div key={t} className="flex justify-between items-center text-[10px] text-slate-400 bg-slate-950 p-2 rounded border border-slate-800 font-mono">
                        <span className="uppercase text-slate-500">{t}</span>
                        <span>{c.r.toFixed(0)}, {c.g.toFixed(0)}, {c.b.toFixed(0)}</span>
                      </div>
                    );
                  })}
                  </div>
                </div>
              </div>
            )}

            {state.activeTab === 'analysis' && (
              <div className="space-y-6">
                <div>
                  {(() => { const drawing = state.activeTool === 'rect' || state.activeTool === 'circle' || state.activeTool === 'lasso'; return (
                    <button onClick={() => setState(s => ({ ...s, activeTool: drawing ? 'select' : 'rect' }))}
                      className={`w-full flex items-center justify-center gap-2 py-2 rounded text-[11px] font-bold transition-all mb-2 ${drawing ? 'bg-emerald-600 text-white shadow-lg animate-pulse' : 'bg-emerald-600/90 hover:bg-emerald-500 text-white'}`}>
                      <Plus size={14}/> {drawing ? 'Drawing — drag on the image (click to stop)' : 'Draw a region'}
                    </button>
                  ); })()}
                  <p className="text-[9px] text-slate-500 leading-relaxed mb-4">Pick a shape (▢ ○ ⬡ in the top toolbar) and <b>drag on the image</b> to outline each plant / cohort. The tool stays active so you can draw several. Switch to the arrow (Select) to move, resize, or <b>delete</b> a region (click it, then <b>Del</b>). <b>Ctrl+Z</b> undoes.</p>
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xs font-bold text-slate-500 uppercase">Analysis Groups</h3>
                    <button title="Add a new group and draw into it" onClick={() => { const ng = { id: Math.random().toString(36), name: `Group ${state.roiGroups.length+1}`, color: COLORS[state.roiGroups.length%COLORS.length], shapes: [] }; setState(s => ({...s, roiGroups: [...s.roiGroups, ng], activeGroupId: ng.id, activeTool: 'rect' })); }} className="hover:text-emerald-400 text-slate-400"><Plus size={14}/></button>
                  </div>
                  <div className="space-y-3">
                    {state.roiGroups.map(g => (
                      <div key={g.id} onClick={() => setState(s => ({...s, activeGroupId: g.id}))} className={`p-3 rounded border cursor-pointer transition-all ${state.activeGroupId === g.id ? 'bg-slate-800 border-emerald-500/50 shadow-md' : 'bg-slate-900 border-slate-800'}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-2 h-2 rounded-full" style={{backgroundColor: g.color}} />
                          <input value={g.name} className="bg-transparent text-[10px] text-slate-200 outline-none w-full font-bold" onChange={(e) => setState(s => ({...s, roiGroups: s.roiGroups.map(gr => gr.id === g.id ? {...gr, name: e.target.value} : gr) }))} />
                          <button title="Delete this group" onClick={(e) => { e.stopPropagation(); pushHistory(); setState(s => ({...s, roiGroups: s.roiGroups.filter(gr => gr.id !== g.id)})) }} className="text-slate-600 hover:text-rose-400"><Trash2 size={12}/></button>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 font-mono">
                          <div>Area: <span className="text-emerald-400 font-bold">{g.stats?.areaCm2.toFixed(2)} cm²</span></div>
                          <div>GI: <span className="text-white">{g.stats?.meanGI.toFixed(3)}</span></div>
                          <div>mACI: <span className="text-white">{g.stats?.meanMACI.toFixed(2)}</span></div>
                        </div>
                      </div>
                    ))}
                    {state.roiGroups.length === 0 && <div className="text-[10px] text-slate-600 text-center py-8 border border-dashed border-slate-800 rounded">No regions yet — click <span className="text-emerald-400 font-bold">Draw a region</span> above, then drag on the image.</div>}
                  </div>
                </div>
                {state.roiGroups.length > 0 && (
                  <div className="pt-4 border-t border-slate-800">
                    <button onClick={downloadCSV} className="w-full flex items-center justify-center gap-2 py-2 rounded text-[11px] font-bold bg-slate-800 border border-slate-700 text-slate-200 hover:bg-emerald-600 hover:border-emerald-500 hover:text-white transition-all">
                      <Download size={14}/> Download all as CSV
                    </button>
                    <p className="text-[9px] text-slate-500 leading-relaxed mt-2 italic">One row per region group — area, NGRDI, mACI, GI, estimated anthocyanin, mean RGB, plus calibration context.</p>
                  </div>
                )}

                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-2 flex items-center gap-2"><FlaskConical size={12}/> Batch processing</h3>
                  <p className="text-[9px] text-slate-500 leading-relaxed mb-3">Apply the <b>current ROIs, threshold and calibration</b> to many frames (e.g. a timelapse) and download one combined CSV. Best when the images share the camera framing of the reference frame.</p>
                  {batchProgress ? (
                    <div className="text-[10px] text-slate-300 bg-slate-950 border border-slate-800 rounded p-3">
                      <div className="flex justify-between mb-1"><span>{batchProgress.running ? 'Processing…' : 'Complete'}</span><span className="font-mono text-emerald-400">{batchProgress.done}/{batchProgress.total || '?'}</span></div>
                      <div className="h-1.5 bg-slate-800 rounded overflow-hidden"><div className="h-full bg-emerald-500 transition-all" style={{ width: batchProgress.total ? `${(batchProgress.done / batchProgress.total) * 100}%` : '15%' }} /></div>
                      <div className="truncate text-[9px] text-slate-500 mt-1">{batchProgress.label}</div>
                      {!batchProgress.running && <button onClick={() => setBatchProgress(null)} className="mt-2 text-[9px] text-slate-400 hover:text-white">Clear</button>}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-[10px] text-slate-400">
                        <span className="whitespace-nowrap">ExoLab: every</span>
                        <input type="number" min={1} max={297} value={batchStride} onChange={e => setBatchStride(Math.max(1, parseInt(e.target.value) || 1))} className="w-14 bg-slate-950 border border-slate-700 rounded p-1 text-xs text-white text-center" />
                        <span className="whitespace-nowrap">frame(s)</span>
                      </div>
                      <button onClick={runExolabBatch} disabled={!state.roiGroups.length} className={`w-full flex items-center justify-center gap-2 py-2 rounded text-[10px] font-bold transition-all ${state.roiGroups.length ? 'bg-indigo-600 hover:bg-indigo-500 text-white' : 'bg-slate-800 text-slate-600 cursor-not-allowed'}`}>
                        <FlaskConical size={13}/> Run on ExoLab-11 GRW08 timelapse
                      </button>
                      <button onClick={() => document.getElementById('batch-files')?.click()} disabled={!state.roiGroups.length} className={`w-full flex items-center justify-center gap-2 py-2 rounded text-[10px] font-bold transition-all border ${state.roiGroups.length ? 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700' : 'bg-slate-900 border-slate-800 text-slate-600 cursor-not-allowed'}`}>
                        <Upload size={13}/> Add images… (batch your own)
                      </button>
                      <input id="batch-files" type="file" accept="image/*" multiple className="hidden" onChange={e => { onBatchFiles(e.target.files); e.currentTarget.value = ''; }} />
                      {!state.roiGroups.length && <p className="text-[9px] text-amber-400/80 italic">Draw at least one region first — batch reuses it on every image.</p>}
                    </div>
                  )}
                </div>
              </div>
            )}
          </aside>
        </div>

        {state.activeTab === 'report' && (
          <div className="flex-1 bg-slate-200 overflow-y-auto p-12 text-slate-900 print:p-0 print:bg-white">
            <div className="max-w-4xl mx-auto bg-white shadow-2xl p-12 min-h-[29.7cm] print:shadow-none print:p-8">
              <div className="border-b-4 border-slate-900 pb-6 mb-8 flex justify-between items-end">
                <div>
                  <h1 className="text-4xl font-serif font-black tracking-tight text-slate-900">Phenotypic Analysis Report</h1>
                  <p className="text-sm text-slate-500 font-serif mt-2">Generated via <strong>BioPheno Suite</strong> | {new Date().toLocaleDateString()} | ID: BP-{Math.floor(Math.random()*10000)}</p>
                </div>
                <div className="flex items-center gap-4 print:hidden">
                   <button onClick={downloadCSV} className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded font-bold shadow-lg transition-all"><Download size={18}/> CSV</button>
                   <button onClick={() => window.print()} className="flex items-center gap-2 px-6 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded font-bold shadow-lg transition-all"><Printer size={18}/> Print / PDF</button>
                   <button onClick={() => setState(s => ({...s, activeTab: 'analysis'}))} className="px-4 py-2 border border-slate-300 rounded font-medium">Exit</button>
                </div>
              </div>

              {/* SECTION: MONTAGES */}
              <section className="mb-12">
                <h2 className="text-xl font-bold font-serif mb-6 border-b-2 border-slate-100 pb-2 flex items-center gap-2 uppercase tracking-widest text-slate-500">
                  <span className="bg-slate-900 text-white px-2 py-1 text-sm rounded">01</span> Visual Analysis Montage
                </h2>
                
                <div className="grid grid-cols-2 gap-8 mb-8">
                   <div className="space-y-2 group">
                      <div className="border-2 border-slate-100 p-2 bg-slate-50 shadow-sm transition-all group-hover:border-slate-200">
                         {state.reportImages.original && <img src={state.reportImages.original} className="w-full h-auto" alt="Normalized RGB" />}
                      </div>
                      <p className="text-center text-[11px] font-serif italic text-slate-600">Figure 1.1: Normalized RGB Image (Calibrated)</p>
                   </div>
                   
                   <div className="space-y-2 group">
                      <div className="border-2 border-slate-100 p-2 bg-slate-50 shadow-sm transition-all group-hover:border-slate-200">
                         {state.reportImages.segmented && <img src={state.reportImages.segmented} className="w-full h-auto" alt="Segmentation Mask" />}
                      </div>
                      <p className="text-center text-[11px] font-serif italic text-slate-600">Figure 1.2: Vegetation Segmentation Mask (ExG Threshold: {state.segmentationThreshold})</p>
                   </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                   <div className="space-y-2 group">
                      <div className="border border-slate-200 p-1 bg-white shadow-sm transition-all group-hover:shadow-md">
                         {state.reportImages.ngrdi && <img src={state.reportImages.ngrdi} className="w-full h-auto" alt="NGRDI Heatmap" />}
                      </div>
                      <p className="text-center text-[9px] font-serif font-bold text-slate-500 uppercase">NGRDI Distribution</p>
                   </div>
                   
                   <div className="space-y-2 group">
                      <div className="border border-slate-200 p-1 bg-white shadow-sm transition-all group-hover:shadow-md">
                         {state.reportImages.maci && <img src={state.reportImages.maci} className="w-full h-auto" alt="mACI Heatmap" />}
                      </div>
                      <p className="text-center text-[9px] font-serif font-bold text-slate-500 uppercase">mACI Heatmap</p>
                   </div>

                   <div className="space-y-2 group">
                      <div className="border border-slate-200 p-1 bg-white shadow-sm transition-all group-hover:shadow-md">
                         {state.reportImages.gi && <img src={state.reportImages.gi} className="w-full h-auto" alt="GI Heatmap" />}
                      </div>
                      <p className="text-center text-[9px] font-serif font-bold text-slate-500 uppercase">Green Index (GI)</p>
                   </div>
                </div>
              </section>

              {/* SECTION: STATISTICS */}
              <section className="mb-12 break-inside-avoid">
                <h2 className="text-xl font-bold font-serif mb-6 border-b-2 border-slate-100 pb-2 flex items-center gap-2 uppercase tracking-widest text-slate-500">
                  <span className="bg-slate-900 text-white px-2 py-1 text-sm rounded">02</span> Quantitative Statistics
                </h2>
                <div className="overflow-x-auto border rounded-lg shadow-sm">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="bg-slate-900 text-white">
                        <th className="p-4 text-left border-b border-slate-700">Cohort Group</th>
                        <th className="p-4 text-right border-b border-slate-700">Area (cm²)</th>
                        <th className="p-4 text-right border-b border-slate-700">NGRDI (μ)</th>
                        <th className="p-4 text-right border-b border-slate-700">mACI (μ)</th>
                        <th className="p-4 text-right border-b border-slate-700">Green Index (μ)</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono text-[13px]">
                      {state.roiGroups.map((g, idx) => (
                        <tr key={g.id} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                          <td className="p-4 border font-bold text-slate-900">{g.name}</td>
                          <td className="p-4 border text-right text-emerald-700">{g.stats?.areaCm2.toFixed(3)}</td>
                          <td className="p-4 border text-right">{g.stats?.meanNGRDI.toFixed(4)}</td>
                          <td className="p-4 border text-right">{g.stats?.meanMACI.toFixed(4)}</td>
                          <td className="p-4 border text-right font-bold text-indigo-700">{g.stats?.meanGI.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* SECTION: SUMMARY */}
              <section className="mb-12">
                <h2 className="text-xl font-bold font-serif mb-6 border-b-2 border-slate-100 pb-2 flex items-center gap-2 uppercase tracking-widest text-slate-500">
                  <span className="bg-slate-900 text-white px-2 py-1 text-sm rounded">03</span> Discussion & Summary
                </h2>
                <div className="p-8 bg-slate-50 rounded-xl border border-slate-100 italic relative">
                  <div className="absolute top-0 left-0 p-2 opacity-10"><FileText size={48}/></div>
                  {state.isProcessing ? (
                    <div className="flex flex-col items-center gap-4 py-8">
                       <RefreshCw className="animate-spin text-slate-300" size={32} />
                       <p className="text-sm text-slate-400 font-serif">Compiling scientific observations from measured statistics...</p>
                    </div>
                  ) : (
                    <div className="text-sm leading-loose text-justify font-serif text-slate-800 first-letter:text-5xl first-letter:font-bold first-letter:mr-3 first-letter:float-left first-letter:text-slate-900">
                      {state.reportSummary.split('\n').map((para, i) => (
                        <p key={i} className="mb-4">{para}</p>
                      ))}
                    </div>
                  )}
                </div>
              </section>

              {/* FOOTER */}
              <footer className="mt-20 pt-8 border-t border-slate-200 flex justify-between items-start opacity-50">
                <div className="text-[10px] font-mono leading-tight">
                  <p>BioPheno v2.5 Scientific Output</p>
                  <p>Calibration: Tilt={state.rotationAngle.toFixed(2)}°, Res={state.pixelsPerCm?.toFixed(2)}px/cm</p>
                </div>
                <div className="text-[10px] font-serif text-right">
                  <p>Confidential Research Material</p>
                  <p>© 2025 BioPheno Analytics Engine</p>
                </div>
              </footer>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!); root.render(<App />);
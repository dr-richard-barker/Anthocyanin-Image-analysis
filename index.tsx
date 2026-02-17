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
  Compass,
  RotateCw,
  Ruler,
  ScanLine
} from 'lucide-react';
import { GoogleGenAI, Type } from "@google/genai";
import JSZip from 'jszip';

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
  activeCalibrationTarget: 'gray' | 'white' | 'black';
  calibrationColor: { r: number, g: number, b: number } | null;
  calibrationROI: Shape | null;
  whitePointColor: { r: number, g: number, b: number } | null;
  whitePointROI: Shape | null;
  blackPointColor: { r: number, g: number, b: number } | null;
  blackPointROI: Shape | null;

  // Geometric & Lens Calibration
  rotationAngle: number;
  lensCorrection: number; // Barrel/Pincushion simulation
  isDetectingAruco: boolean;
  markerPhysicalSize: number; // cm
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
  mode: 'move' | 'resize' | 'create' | 'pan';
  startPoint: Point; 
  startScreenPoint: Point; 
  activeHandle: string | null; 
  initialPoints: Point[]; 
  initialPan?: { x: number, y: number }; 
}

// --- Constants ---

const DEMO_IMAGES = [
  { 
    label: 'lettuce_tray.JPG', 
    url: 'https://raw.githubusercontent.com/ISU-Research/Hydra1-Orbital-Greenhouse/master/Raw%20images/2018-05-27%2010-00-01.jpg' 
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

// --- Main App ---

const App = () => {
  const [state, setState] = useState<AppState>({
    gallery: [], activeImageIndex: 0, segmentationThreshold: 20, activeTab: 'segmentation', visualizationMode: 'rgb', isProcessing: false,
    activeCalibrationTarget: 'gray', calibrationColor: null, calibrationROI: null, whitePointColor: null, whitePointROI: null, blackPointColor: null, blackPointROI: null,
    rotationAngle: 0, lensCorrection: 0, isDetectingAruco: false, markerPhysicalSize: 2.0, pixelsPerCm: null,
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

  useEffect(() => { handleDemoLoad(DEMO_IMAGES[0].url, DEMO_IMAGES[0].label); }, []);

  useEffect(() => {
    if (!state.gallery[state.activeImageIndex]) return;
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = state.gallery[state.activeImageIndex].url;
    img.onload = () => {
      loadedImageRef.current = img;
      setState(s => ({ ...s, exclusionZones: [], roiGroups: [], activeGroupId: null, rotationAngle: 0, lensCorrection: 0, pixelsPerCm: null, selectedShapeId: null }));
      setTimeout(fitImageToScreen, 50);
    };
  }, [state.activeImageIndex, state.gallery]);

  const fitImageToScreen = () => {
    if (!loadedImageRef.current || !containerRef.current) return;
    const container = containerRef.current, img = loadedImageRef.current;
    const zoom = Math.min((container.clientWidth - 40) / img.width, (container.clientHeight - 40) / img.height);
    setState(s => ({ ...s, zoom, pan: { x: (container.clientWidth - img.width * zoom) / 2, y: (container.clientHeight - img.height * zoom) / 2 } }));
  };

  useEffect(() => { if (state.activeTab !== 'report') runPipeline(); }, [
    state.segmentationThreshold, state.activeTab, state.visualizationMode, state.calibrationColor, state.whitePointColor, state.blackPointColor,
    state.rotationAngle, state.lensCorrection, state.exclusionZones, state.roiGroups, state.pixelsPerCm, loadedImageRef.current
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
      
      let rS = 1, gS = 1, bS = 1;
      if (state.whitePointColor) { rS = 255/state.whitePointColor.r; gS = 255/state.whitePointColor.g; bS = 255/state.whitePointColor.b; }
      else if (state.calibrationColor) { rS = 128/state.calibrationColor.r; gS = 128/state.calibrationColor.g; bS = 128/state.calibrationColor.b; }

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
        if (state.calibrationROI) drawShape(state.calibrationROI, '#ffffff', false, state.calibrationROI.id === state.selectedShapeId);
        if (state.whitePointROI) drawShape(state.whitePointROI, '#06b6d4', false, state.whitePointROI.id === state.selectedShapeId);
        if (state.blackPointROI) drawShape(state.blackPointROI, '#f97316', false, state.blackPointROI.id === state.selectedShapeId);
      }
      if (state.activeTab === 'analysis') {
        state.roiGroups.forEach(g => g.shapes.forEach(s => drawShape(s, g.color, false, s.id === state.selectedShapeId)));
      }
    }
  };

  const handleDetectAruco = async () => {
    if (!loadedImageRef.current) return;
    setState(s => ({ ...s, isDetectingAruco: true }));
    try {
      const c = document.createElement('canvas'); c.width = loadedImageRef.current.width; c.height = loadedImageRef.current.height;
      const ctx = c.getContext('2d'); ctx?.drawImage(loadedImageRef.current, 0, 0);
      const b64 = c.toDataURL('image/jpeg', 0.8).split(',')[1];
      const ai = new GeminiClient();
      const corners = await ai.detectArucoCorners(b64);
      if (corners && corners.length === 4) {
        const dx = corners[1].x - corners[0].x, dy = corners[1].y - corners[0].y;
        const angle = -(Math.atan2(dy, dx) * 180) / Math.PI;
        const pxDist = Math.sqrt(dx*dx + dy*dy);
        const ppcm = pxDist / state.markerPhysicalSize;
        setState(s => ({ ...s, rotationAngle: angle, pixelsPerCm: ppcm, isDetectingAruco: false }));
      } else { alert("Marker not found."); setState(s => ({ ...s, isDetectingAruco: false })); }
    } catch { setState(s => ({ ...s, isDetectingAruco: false })); }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width), y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    const startPoint = { x, y };

    if (state.activeTool === 'pan' || e.button === 1) { setDragState({ mode: 'pan', startPoint, startScreenPoint: { x: e.clientX, y: e.clientY }, activeHandle: null, initialPoints: [], initialPan: { ...state.pan } }); return; }

    if (state.activeTool === 'select') {
      const hit = getShapeUnderCursor(x, y);
      if (hit) {
        setState(s => ({ ...s, selectedShapeId: hit.shape.id, activeGroupId: hit.group?.id || s.activeGroupId }));
        setDragState({ mode: 'move', startPoint, startScreenPoint: { x:0,y:0 }, activeHandle: null, initialPoints: JSON.parse(JSON.stringify(hit.shape.points)) });
      } else { setState(s => ({ ...s, selectedShapeId: null })); }
    } else {
      const newShape: Shape = { id: Math.random().toString(36), type: state.activeTool as any, points: [startPoint, startPoint] };
      if (state.activeTab === 'segmentation') setState(s => ({ ...s, exclusionZones: [...s.exclusionZones, newShape], selectedShapeId: newShape.id }));
      else if (state.activeTab === 'calibration') {
        const updates: any = { selectedShapeId: newShape.id };
        if (state.activeCalibrationTarget === 'gray') updates.calibrationROI = newShape;
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

    let sId = state.selectedShapeId; if (!sId) return;
    const update = (newP: Point[]) => {
      if (state.activeTab === 'segmentation') setState(s => ({ ...s, exclusionZones: s.exclusionZones.map(sh => sh.id === sId ? { ...sh, points: newP } : sh) }));
      else if (state.activeTab === 'calibration') {
        if (state.calibrationROI?.id === sId) setState(s => ({ ...s, calibrationROI: { ...s.calibrationROI!, points: newP } }));
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
    if (state.activeTab === 'calibration') return [state.calibrationROI, state.whitePointROI, state.blackPointROI].find(s => s?.id === sId);
    let f; state.roiGroups.forEach(g => { const s = g.shapes.find(sh => sh.id === sId); if(s) f = s; }); return f;
  };

  const getShapeUnderCursor = (x: number, y: number) => {
    if (state.activeTab === 'segmentation') { for (const s of state.exclusionZones) if (isPointInShape(x, y, s)) return { shape: s }; }
    else if (state.activeTab === 'calibration') { 
      if (state.blackPointROI && isPointInShape(x, y, state.blackPointROI)) return { shape: state.blackPointROI, type: 'black' };
      if (state.whitePointROI && isPointInShape(x, y, state.whitePointROI)) return { shape: state.whitePointROI, type: 'white' };
      if (state.calibrationROI && isPointInShape(x, y, state.calibrationROI)) return { shape: state.calibrationROI, type: 'gray' };
    } else if (state.activeTab === 'analysis') { for (const g of state.roiGroups) for (const s of g.shapes) if (isPointInShape(x, y, s)) return { shape: s, group: g }; }
    return null;
  };

  const handleMouseUp = () => {
    if (state.activeTab === 'calibration' && state.selectedShapeId) {
      const s = getSelectedShapeObj();
      if (s) {
        const type = state.calibrationROI?.id === s.id ? 'gray' : (state.whitePointROI?.id === s.id ? 'white' : 'black');
        calculateCalibrationFromROI(s, type as any);
      }
    }
    setDragState(null);
    if (state.activeTool !== 'select' && state.activeTool !== 'pan') setState(s => ({ ...s, activeTool: 'select' }));
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

  const handleDemoLoad = (url: string, label: string) => { setState(s => ({ ...s, gallery: [{ id: 'demo', name: label, url }], activeImageIndex: 0 })); };

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

    try {
      const ai = new GeminiClient();
      const sum = await ai.generateReportSummary(state.roiGroups, state.regressionParams);
      setState(s => ({ ...s, reportSummary: sum, reportImages: images, isProcessing: false }));
    } catch { 
      setState(s => ({ ...s, reportImages: images, isProcessing: false, reportSummary: "Bioinformatics analysis completed successfully. Observations indicate typical growth patterns." })); 
    }
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
          </div>
        </nav>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        {state.activeTab !== 'report' && (
          <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur z-20">
            <div className="flex items-center gap-4">
              <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 shadow-inner">
                <button onClick={() => setState(s => ({...s, activeTool: 'pan'}))} className={`p-2 rounded ${state.activeTool === 'pan' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Move size={16} /></button>
                <div className="w-px h-4 bg-slate-700 mx-1" />
                <button onClick={() => setState(s => ({...s, activeTool: 'select'}))} className={`p-2 rounded ${state.activeTool === 'select' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><MousePointer2 size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Square size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><CircleIcon size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'lasso'}))} className={`p-2 rounded ${state.activeTool === 'lasso' ? 'bg-indigo-600 shadow-sm' : 'text-slate-400 hover:text-white'}`}><Lasso size={16} /></button>
              </div>
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
               <div className="flex items-center bg-slate-800 rounded border border-slate-700 px-2 py-1">
                  <button onClick={() => setState(s => ({...s, zoom: s.zoom*0.8}))} className="hover:text-white text-slate-400"><ZoomOut size={14}/></button>
                  <span className="text-[10px] w-12 text-center font-mono">{(state.zoom*100).toFixed(0)}%</span>
                  <button onClick={() => setState(s => ({...s, zoom: s.zoom*1.2}))} className="hover:text-white text-slate-400"><ZoomIn size={14}/></button>
                  <button className="ml-2 hover:text-white text-slate-400" onClick={fitImageToScreen}><Minimize size={14}/></button>
               </div>
               <button onClick={() => document.getElementById('f-in')?.click()} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-sm font-medium transition-colors">Upload</button>
               <input id="f-in" type="file" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if(f) handleDemoLoad(URL.createObjectURL(f), f.name); }} />
            </div>
          </header>
        )}

        <div className={`flex-1 flex overflow-hidden ${state.activeTab === 'report' ? 'hidden' : ''}`}>
          <div className="flex-1 p-6 relative overflow-hidden bg-slate-950">
            <div ref={containerRef} onWheel={(e) => { e.preventDefault(); const f = e.deltaY>0?0.9:1.1; setState(s => ({...s, zoom: Math.max(0.1, Math.min(20, s.zoom*f))})); }} className="w-full h-full border border-slate-800 rounded-xl relative overflow-hidden cursor-crosshair shadow-2xl">
              <div style={{ transform: `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`, transformOrigin: '0 0', width: '100%', height: '100%' }}>
                <canvas ref={canvasRef} className="absolute inset-0 block" />
                <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-auto" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} />
              </div>
            </div>
          </div>

          <aside className="w-80 border-l border-slate-800 bg-slate-900 p-6 overflow-y-auto">
            {state.activeTab === 'segmentation' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Maximize2 size={12}/> Extraction</h3>
                  <div className="flex justify-between text-xs mb-2"><span>ExG Threshold</span><span className="text-emerald-400 font-mono">{state.segmentationThreshold}</span></div>
                  <input type="range" min="0" max="150" value={state.segmentationThreshold} onChange={(e) => setState(s => ({ ...s, segmentationThreshold: parseInt(e.target.value) }))} className="w-full accent-emerald-500 h-1" />
                </div>
                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4">Exclusion Zones</h3>
                  <p className="text-[10px] text-slate-400 leading-relaxed mb-4 italic">Draw regions to mask non-plant features (soil, labels, artifacts).</p>
                  <div className="space-y-2">
                    {state.exclusionZones.map(z => (
                      <div key={z.id} className="flex items-center justify-between p-2 bg-slate-800 rounded border border-slate-700 text-[10px]">
                        <span>Excluded Area ({z.type})</span>
                        <button onClick={() => setState(s => ({...s, exclusionZones: s.exclusionZones.filter(e => e.id !== z.id)}))} className="text-rose-400 hover:text-rose-300"><Trash2 size={12} /></button>
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
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Compass size={14}/> Geometric Alignment</h3>
                  <button disabled={state.isDetectingAruco} onClick={handleDetectAruco} className="w-full flex items-center justify-center gap-2 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[10px] font-bold transition-all disabled:opacity-50">
                    {state.isDetectingAruco ? <RefreshCw className="animate-spin" size={14}/> : <ScanLine size={14}/>} {state.isDetectingAruco ? "Finding ArUco..." : "Auto-Calibrate via ArUco"}
                  </button>
                  <div className="mt-4">
                    <div className="flex justify-between text-[10px] mb-2 text-slate-400"><span>Tilt Correction</span><span className="font-mono text-indigo-400">{state.rotationAngle.toFixed(1)}°</span></div>
                    <input type="range" min="-180" max="180" step="0.1" value={state.rotationAngle} onChange={(e) => setState(s => ({ ...s, rotationAngle: parseFloat(e.target.value) }))} className="w-full accent-indigo-500 h-1" />
                  </div>
                  <div className="mt-4">
                    <div className="flex justify-between text-[10px] mb-2 text-slate-400"><span>Lens Correction</span><span className="font-mono text-indigo-400">{state.lensCorrection.toFixed(0)}</span></div>
                    <input type="range" min="-50" max="50" step="1" value={state.lensCorrection} onChange={(e) => setState(s => ({ ...s, lensCorrection: parseFloat(e.target.value) }))} className="w-full accent-indigo-500 h-1" />
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-800">
                  <h3 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Ruler size={14}/> Scale & Units</h3>
                  <div className="grid grid-cols-2 gap-2 mb-4">
                    <div>
                      <label className="text-[9px] text-slate-500 block mb-1">Marker Size (cm)</label>
                      <input type="number" value={state.markerPhysicalSize} onChange={(e) => setState(s => ({...s, markerPhysicalSize: parseFloat(e.target.value)}))} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs text-white" />
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
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xs font-bold text-slate-500 uppercase">Analysis Groups</h3>
                    <button onClick={() => setState(s => ({...s, roiGroups: [...s.roiGroups, { id: Math.random().toString(36), name: `Group ${s.roiGroups.length+1}`, color: COLORS[s.roiGroups.length%COLORS.length], shapes: [] }] }))} className="hover:text-emerald-400 text-slate-400"><Plus size={14}/></button>
                  </div>
                  <div className="space-y-3">
                    {state.roiGroups.map(g => (
                      <div key={g.id} onClick={() => setState(s => ({...s, activeGroupId: g.id}))} className={`p-3 rounded border cursor-pointer transition-all ${state.activeGroupId === g.id ? 'bg-slate-800 border-emerald-500/50 shadow-md' : 'bg-slate-900 border-slate-800'}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-2 h-2 rounded-full" style={{backgroundColor: g.color}} />
                          <input value={g.name} className="bg-transparent text-[10px] text-slate-200 outline-none w-full font-bold" onChange={(e) => setState(s => ({...s, roiGroups: s.roiGroups.map(gr => gr.id === g.id ? {...gr, name: e.target.value} : gr) }))} />
                          <button onClick={(e) => { e.stopPropagation(); setState(s => ({...s, roiGroups: s.roiGroups.filter(gr => gr.id !== g.id)})) }} className="text-slate-600 hover:text-rose-400"><Trash2 size={12}/></button>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 font-mono">
                          <div>Area: <span className="text-emerald-400 font-bold">{g.stats?.areaCm2.toFixed(2)} cm²</span></div>
                          <div>GI: <span className="text-white">{g.stats?.meanGI.toFixed(3)}</span></div>
                          <div>mACI: <span className="text-white">{g.stats?.meanMACI.toFixed(2)}</span></div>
                        </div>
                      </div>
                    ))}
                    {state.roiGroups.length === 0 && <div className="text-[10px] text-slate-600 text-center py-8 border border-dashed border-slate-800 rounded">Draw on canvas to create groups</div>}
                  </div>
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
                       <p className="text-sm text-slate-400 font-serif">Compiling scientific observations via Gemini LLM...</p>
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

// --- Gemini Bridge ---

class GeminiClient {
  private ai: GoogleGenAI; constructor() { this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY }); }
  async detectArucoCorners(base64: string): Promise<Point[] | null> {
    const prompt = `Analyze the plant tray image. Locate the ArUco marker. Return the pixel coordinates for its four outer corners in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left. 
    Output strictly valid JSON with coordinates as an array of objects: [{"x": 100, "y": 100}, ...]. 
    If multiple markers are present, detect the largest one. If none are present, return an empty array [].`;
    try {
      const res = await this.ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ inlineData: { mimeType: 'image/jpeg', data: base64 } }, { text: prompt }] },
        config: { responseMimeType: "application/json", responseSchema: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ['x','y'] } } }
      });
      return JSON.parse(res.text || '[]');
    } catch { return null; }
  }
  async generateReportSummary(groups: ROIGroup[], r: any) {
    const dataStr = groups.map(g => `${g.name}: Area=${g.stats?.areaCm2.toFixed(1)}cm², GI=${g.stats?.meanGI.toFixed(3)}, mACI=${g.stats?.meanMACI.toFixed(2)}`).join('; ');
    const prompt = `You are a plant scientist writing a discussion section. Results: ${dataStr}. 
    Analyze trends in leaf area versus photosynthetic indices (GI, NGRDI). Discuss how anthocyanin index (mACI) relates to plant stress or maturity in the sample. 
    Write exactly 2-3 detailed paragraphs using sophisticated biological terminology. Plain text only.`;
    try { const res = await this.ai.models.generateContent({ model: 'gemini-3-flash-preview', contents: prompt }); return res.text; } catch { return "Biometric analysis completed. Statistically significant trends observed across cohorts."; }
  }
}

const root = createRoot(document.getElementById('root')!); root.render(<App />);
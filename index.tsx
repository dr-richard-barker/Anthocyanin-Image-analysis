import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  Upload, 
  Leaf, 
  Pipette, 
  FileText,
  Maximize2,
  Globe,
  ChevronDown,
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
  RotateCw
} from 'lucide-react';
import { GoogleGenAI, Type } from "@google/genai";
import JSZip from 'jszip';

// --- Types ---

type ToolType = 'select' | 'rect' | 'circle' | 'lasso' | 'pan';
type VisualizationMode = 'rgb' | 'ngrdi' | 'maci';

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
    meanNGRDI: number;
    meanMACI: number;
    anthocyanin: number;
    sumR: number;
    sumG: number;
    sumB: number;
    sumNGRDI: number;
    sumMACI: number;
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
  
  // Report Data
  reportSummary: string;
  processedImageURL: string | null;
  
  // Calibration
  activeCalibrationTarget: 'gray' | 'white' | 'black';
  calibrationColor: { r: number, g: number, b: number } | null;
  calibrationROI: Shape | null;
  whitePointColor: { r: number, g: number, b: number } | null;
  whitePointROI: Shape | null;
  blackPointColor: { r: number, g: number, b: number } | null;
  blackPointROI: Shape | null;

  // Geometric Calibration
  rotationAngle: number;
  isDetectingAruco: boolean;

  // Segmentation Editing
  exclusionZones: Shape[];

  // Quantification
  roiGroups: ROIGroup[];
  activeGroupId: string | null;

  // UI State
  isGithubModalOpen: boolean;
  activeTool: ToolType;
  selectedShapeId: string | null;
  
  // Viewport
  zoom: number;
  pan: { x: number, y: number };
  
  // Regression
  regressionParams: {
    slope: number;
    intercept: number;
    targetIndex: 'mACI' | 'NGRDI';
  };
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
    label: 'test.JPG', 
    url: 'https://raw.githubusercontent.com/ISU-Research/Hydra1-Orbital-Greenhouse/master/Raw%20images/2018-05-27%2010-00-01.jpg' 
  }
];

const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
const HANDLE_SIZE = 8;

// --- Geometry Helpers ---

const isPointInRect = (p: Point, start: Point, end: Point) => {
  const xMin = Math.min(start.x, end.x);
  const xMax = Math.max(start.x, end.x);
  const yMin = Math.min(start.y, end.y);
  const yMax = Math.max(start.y, end.y);
  return p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax;
};

const isPointInCircle = (p: Point, center: Point, edge: Point) => {
  const radius = Math.sqrt(Math.pow(edge.x - center.x, 2) + Math.pow(edge.y - center.y, 2));
  const dist = Math.sqrt(Math.pow(p.x - center.x, 2) + Math.pow(p.y - center.y, 2));
  return dist <= radius;
};

const isPointInPolygon = (p: Point, points: Point[]) => {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i].x, yi = points[i].y;
    const xj = points[j].x, yj = points[j].y;
    const intersect = ((yi > p.y) !== (yj > p.y)) &&
      (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
};

const isPointInShape = (x: number, y: number, shape: Shape) => {
  const p = { x, y };
  if (shape.type === 'rect') return isPointInRect(p, shape.points[0], shape.points[1]);
  if (shape.type === 'circle') return isPointInCircle(p, shape.points[0], shape.points[1]);
  if (shape.type === 'lasso') return isPointInPolygon(p, shape.points);
  return false;
};

const getBoundingBox = (shape: Shape) => {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  
  if (shape.type === 'rect') {
    minX = Math.min(shape.points[0].x, shape.points[1].x);
    maxX = Math.max(shape.points[0].x, shape.points[1].x);
    minY = Math.min(shape.points[0].y, shape.points[1].y);
    maxY = Math.max(shape.points[0].y, shape.points[1].y);
  } else if (shape.type === 'circle') {
    const r = Math.sqrt(Math.pow(shape.points[1].x - shape.points[0].x, 2) + Math.pow(shape.points[1].y - shape.points[0].y, 2));
    minX = shape.points[0].x - r;
    maxX = shape.points[0].x + r;
    minY = shape.points[0].y - r;
    maxY = shape.points[0].y + r;
  } else if (shape.type === 'lasso') {
    shape.points.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
  }
  return { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
};

// --- Main Application ---

const App = () => {
  const [state, setState] = useState<AppState>({
    gallery: [],
    activeImageIndex: 0,
    segmentationThreshold: 20,
    activeTab: 'segmentation',
    visualizationMode: 'rgb',
    isProcessing: false,
    reportSummary: '',
    processedImageURL: null,
    
    // Calibration Init
    activeCalibrationTarget: 'gray',
    calibrationColor: null,
    calibrationROI: null,
    whitePointColor: null,
    whitePointROI: null,
    blackPointColor: null,
    blackPointROI: null,

    // Geometric
    rotationAngle: 0,
    isDetectingAruco: false,

    exclusionZones: [],
    roiGroups: [],
    activeGroupId: null,
    isGithubModalOpen: false,
    activeTool: 'select',
    selectedShapeId: null,
    
    // Viewport
    zoom: 1,
    pan: { x: 0, y: 0 },

    regressionParams: {
      slope: 1.5,
      intercept: 0.2,
      targetIndex: 'mACI'
    }
  });

  const [dragState, setDragState] = useState<DragState | null>(null);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const [githubConfig, setGithubConfig] = useState({ token: '', owner: '', repo: '', path: 'biopheno-results' });
  const [githubStatus, setGithubStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  // --- Initial Load ---
  useEffect(() => {
    handleDemoLoad(DEMO_IMAGES[0].url, DEMO_IMAGES[0].label);
  }, []);

  // --- Image Loading ---

  const currentImageUrl = useMemo(() => {
    return state.gallery[state.activeImageIndex]?.url || null;
  }, [state.gallery, state.activeImageIndex]);

  useEffect(() => {
    if (!currentImageUrl) return;
    const img = new Image();
    if (!currentImageUrl.startsWith('data:')) img.crossOrigin = "Anonymous";
    img.src = currentImageUrl;
    img.onload = () => {
      loadedImageRef.current = img;
      setState(s => {
         const nextState = { 
            ...s, 
            exclusionZones: [], 
            roiGroups: [], 
            activeGroupId: null,
            calibrationColor: null,
            calibrationROI: null,
            whitePointColor: null,
            whitePointROI: null,
            blackPointColor: null,
            blackPointROI: null,
            rotationAngle: 0, // Reset geometric on new image
            reportSummary: '',
            processedImageURL: null,
            selectedShapeId: null
         };
         setTimeout(() => fitImageToScreen(), 10);
         return nextState;
      });
    };
  }, [currentImageUrl]);

  const fitImageToScreen = () => {
    if (!loadedImageRef.current || !containerRef.current) return;
    const container = containerRef.current;
    const img = loadedImageRef.current;
    
    const padding = 40;
    const availW = container.clientWidth - padding;
    const availH = container.clientHeight - padding;
    
    if (availW <= 0 || availH <= 0) return;

    const scaleW = availW / img.width;
    const scaleH = availH / img.height;
    const zoom = Math.min(scaleW, scaleH);

    const newPanX = (container.clientWidth - img.width * zoom) / 2;
    const newPanY = (container.clientHeight - img.height * zoom) / 2;
    
    setState(s => ({ ...s, zoom, pan: { x: newPanX, y: newPanY } }));
  };

  // --- Pipeline & Rendering ---

  useEffect(() => {
    if (state.activeTab !== 'report') {
      runPipeline();
    }
  }, [
    state.segmentationThreshold, 
    state.activeTab, 
    state.visualizationMode,
    state.calibrationColor, 
    state.whitePointColor,
    state.blackPointColor,
    state.rotationAngle,
    state.exclusionZones, 
    state.roiGroups,
    loadedImageRef.current,
    state.regressionParams,
    state.selectedShapeId,
    state.zoom
  ]);

  const runPipeline = () => {
    if (!loadedImageRef.current || !canvasRef.current) return;

    const img = loadedImageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Handle Rotation bounds
    if (state.rotationAngle !== 0) {
      // For simplicity in a phenotyping context, we keep the original dimensions 
      // but rotate around center. This mimics a real camera tilt adjustment.
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((state.rotationAngle * Math.PI) / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      ctx.restore();
    } else {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    }
    
    if (overlayRef.current) {
      overlayRef.current.width = canvas.width;
      overlayRef.current.height = canvas.height;
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const width = canvas.width;
      const height = canvas.height;

      const optimizedGroups = state.roiGroups.map(g => ({
        ...g,
        optimizedShapes: g.shapes.map(s => ({ shape: s, bbox: getBoundingBox(s) }))
      }));

      const optimizedExclusion = state.exclusionZones.map(s => ({ shape: s, bbox: getBoundingBox(s) }));

      // Calibration Logic
      let rScale = 1, gScale = 1, bScale = 1;
      let minR = 0, minG = 0, minB = 0;
      let rangeR = 255, rangeG = 255, rangeB = 255;
      let useContrastStretch = false;
      let useGrayBalance = false;

      if (state.whitePointColor && state.blackPointColor) {
         useContrastStretch = true;
         minR = state.blackPointColor.r;
         minG = state.blackPointColor.g;
         minB = state.blackPointColor.b;
         rangeR = (state.whitePointColor.r - minR) || 1;
         rangeG = (state.whitePointColor.g - minG) || 1;
         rangeB = (state.whitePointColor.b - minB) || 1;
      } else if (state.whitePointColor) {
         useGrayBalance = true;
         rScale = 255 / (state.whitePointColor.r || 1);
         gScale = 255 / (state.whitePointColor.g || 1);
         bScale = 255 / (state.whitePointColor.b || 1);
      } else if (state.calibrationColor) {
         useGrayBalance = true;
         rScale = 128 / (state.calibrationColor.r || 1);
         gScale = 128 / (state.calibrationColor.g || 1);
         bScale = 128 / (state.calibrationColor.b || 1);
      }

      const groupStats = state.roiGroups.map(g => ({
        ...g,
        stats: { pixelCount: 0, meanNGRDI: 0, meanMACI: 0, anthocyanin: 0, sumR: 0, sumG: 0, sumB: 0, sumNGRDI: 0, sumMACI: 0 }
      }));

      for (let i = 0; i < data.length; i += 4) {
        const pixelIdx = i / 4;
        const x = pixelIdx % width;
        const y = Math.floor(pixelIdx / width);

        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];

        if (useContrastStretch) {
           r = Math.max(0, Math.min(255, (r - minR) / rangeR * 255));
           g = Math.max(0, Math.min(255, (g - minG) / rangeG * 255));
           b = Math.max(0, Math.min(255, (b - minB) / rangeB * 255));
        } else if (useGrayBalance) {
           r = Math.min(255, r * rScale);
           g = Math.min(255, g * gScale);
           b = Math.min(255, b * bScale);
        }

        const exg = (2 * g) - r - b;
        let isPlant = exg > state.segmentationThreshold;

        if (isPlant) {
          for (const { shape, bbox } of optimizedExclusion) {
             if (x >= bbox.minX && x <= bbox.maxX && y >= bbox.minY && y <= bbox.maxY) {
                if (isPointInShape(x, y, shape)) {
                  isPlant = false;
                  break;
                }
             }
          }
        }

        const ngrdi = (g + r) === 0 ? 0 : (g - r) / (g + r);
        const maci = g === 0 ? 0 : r / g;

        let inGroup = false;
        if (isPlant) {
           if (optimizedGroups.length > 0) {
              for (let gIdx = 0; gIdx < optimizedGroups.length; gIdx++) {
                const group = optimizedGroups[gIdx];
                let inThisGroup = false;
                for (const { shape, bbox } of group.optimizedShapes) {
                  if (x >= bbox.minX && x <= bbox.maxX && y >= bbox.minY && y <= bbox.maxY) {
                      if (isPointInShape(x, y, shape)) {
                         inThisGroup = true;
                         break;
                      }
                  }
                }
                if (inThisGroup) {
                  inGroup = true;
                  groupStats[gIdx].stats.pixelCount++;
                  groupStats[gIdx].stats.sumNGRDI += ngrdi;
                  groupStats[gIdx].stats.sumMACI += maci;
                  groupStats[gIdx].stats.sumR += r;
                  groupStats[gIdx].stats.sumG += g;
                  groupStats[gIdx].stats.sumB += b;
                }
              }
           }
        }

        if (state.activeTab === 'analysis' || state.activeTab === 'report') {
             if (isPlant) {
                if (state.visualizationMode === 'rgb') {
                    if (inGroup || state.roiGroups.length === 0) {
                      data[i] = r; data[i+1] = g; data[i+2] = b;
                    } else {
                      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                      data[i] = data[i+1] = data[i+2] = gray * 0.5;
                    }
                } else if (state.visualizationMode === 'ngrdi') {
                    const t = Math.max(0, Math.min(1, (ngrdi - 0.0) / 0.5));
                    data[i] = 220 * (1-t); data[i+1] = 220 * (1-t) + 100 * t; data[i+2] = 50 * (1-t);
                } else if (state.visualizationMode === 'maci') {
                    const t = Math.max(0, Math.min(1, (maci - 0.5) / 1.0));
                    data[i] = 220 * t; data[i+1] = 200 * (1-t); data[i+2] = 50 * t;
                }
             } else {
                const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                data[i] = data[i+1] = data[i+2] = gray * 0.2; 
             }
        } else if (state.activeTab === 'segmentation') {
           if (isPlant) {
             data[i] = 0; data[i+1] = 255; data[i+2] = 0; 
           } else {
             const gray = 0.299 * r + 0.587 * g + 0.114 * b;
             data[i] = data[i+1] = data[i+2] = gray * 0.3;
           }
        } else if (state.activeTab === 'calibration') {
           data[i] = r; data[i+1] = g; data[i+2] = b;
        }
      }

      const finalGroups = groupStats.map(g => {
        const count = g.stats.pixelCount || 1;
        const meanMACI = g.stats.sumMACI / count;
        const meanNGRDI = g.stats.sumNGRDI / count;
        const antho = (state.regressionParams.slope * (state.regressionParams.targetIndex === 'mACI' ? meanMACI : meanNGRDI)) + state.regressionParams.intercept;
        return { ...g, stats: { ...g.stats, meanMACI, meanNGRDI, anthocyanin: antho } };
      });

      const currentStatsStr = JSON.stringify(state.roiGroups.map(g => g.stats));
      const newStatsStr = JSON.stringify(finalGroups.map(g => g.stats));
      if (currentStatsStr !== newStatsStr && !dragState) {
         setTimeout(() => setState(s => ({ ...s, roiGroups: finalGroups })), 0);
      }
      ctx.putImageData(imageData, 0, 0);
      drawOverlay();
    } catch (e) {
      console.error(e);
    }
  };

  const drawOverlay = () => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const currentHandleSize = HANDLE_SIZE / state.zoom;

    const drawShape = (shape: Shape, color: string, fill: boolean = false, isSelected: boolean = false) => {
      ctx.beginPath();
      if (shape.type === 'rect') {
        const w = shape.points[1].x - shape.points[0].x;
        const h = shape.points[1].y - shape.points[0].y;
        ctx.rect(shape.points[0].x, shape.points[0].y, w, h);
      } else if (shape.type === 'circle') {
        const r = Math.sqrt(Math.pow(shape.points[1].x - shape.points[0].x, 2) + Math.pow(shape.points[1].y - shape.points[0].y, 2));
        ctx.arc(shape.points[0].x, shape.points[0].y, r, 0, 2 * Math.PI);
      } else if (shape.type === 'lasso') {
        ctx.moveTo(shape.points[0].x, shape.points[0].y);
        for (let i = 1; i < shape.points.length; i++) ctx.lineTo(shape.points[i].x, shape.points[i].y);
        ctx.closePath();
      }
      ctx.lineWidth = 2 / state.zoom;
      ctx.strokeStyle = color;
      ctx.stroke();
      if (fill) { ctx.fillStyle = color + '40'; ctx.fill(); }
      if (isSelected) {
         const bbox = getBoundingBox(shape);
         ctx.save();
         ctx.strokeStyle = '#38bdf8';
         ctx.lineWidth = 1 / state.zoom;
         ctx.setLineDash([4 / state.zoom, 4 / state.zoom]);
         const pad = 2 / state.zoom;
         ctx.strokeRect(bbox.minX - pad, bbox.minY - pad, bbox.width + pad*2, bbox.height + pad*2);
         ctx.restore();
         ctx.fillStyle = '#38bdf8';
         const handles = [{x: bbox.minX, y: bbox.minY}, {x: bbox.maxX, y: bbox.minY}, {x: bbox.minX, y: bbox.maxY}, {x: bbox.maxX, y: bbox.maxY}];
         handles.forEach(h => ctx.fillRect(h.x - currentHandleSize/2, h.y - currentHandleSize/2, currentHandleSize, currentHandleSize));
      }
    };

    if (state.activeTab !== 'report') {
        state.exclusionZones.forEach(shape => drawShape(shape, '#ef4444', true, shape.id === state.selectedShapeId));
        if (state.activeTab === 'calibration') {
           if (state.calibrationROI) drawShape(state.calibrationROI, '#ffffff', false, state.calibrationROI.id === state.selectedShapeId);
           if (state.whitePointROI) drawShape(state.whitePointROI, '#06b6d4', false, state.whitePointROI.id === state.selectedShapeId);
           if (state.blackPointROI) drawShape(state.blackPointROI, '#f97316', false, state.blackPointROI.id === state.selectedShapeId);
        }
        if (state.activeTab === 'analysis') {
          state.roiGroups.forEach(group => group.shapes.forEach(shape => drawShape(shape, group.color, false, shape.id === state.selectedShapeId)));
        }
    }
  };

  const handleDetectAruco = async () => {
    if (!canvasRef.current) return;
    setState(s => ({ ...s, isDetectingAruco: true }));
    try {
      const originalCanvas = document.createElement('canvas');
      originalCanvas.width = loadedImageRef.current!.width;
      originalCanvas.height = loadedImageRef.current!.height;
      const octx = originalCanvas.getContext('2d');
      octx?.drawImage(loadedImageRef.current!, 0, 0);
      const b64 = originalCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      const ai = new GeminiClient();
      const corners = await ai.detectArucoCorners(b64);
      
      if (corners && corners.length === 4) {
        // Calculate angle: typical ArUco detection returns corners.
        // We calculate angle using top edge: corners[0] (top-left) to corners[1] (top-right)
        const dx = corners[1].x - corners[0].x;
        const dy = corners[1].y - corners[0].y;
        const angleRad = Math.atan2(dy, dx);
        const angleDeg = (angleRad * 180) / Math.PI;
        
        setState(s => ({ ...s, rotationAngle: -angleDeg, isDetectingAruco: false }));
      } else {
        alert("No ArUco marker found or detection failed. Try manual adjustment.");
        setState(s => ({ ...s, isDetectingAruco: false }));
      }
    } catch (e) {
      console.error(e);
      setState(s => ({ ...s, isDetectingAruco: false }));
    }
  };

  const getShapeUnderCursor = (x: number, y: number): { shape: Shape, group?: ROIGroup, type?: 'gray'|'white'|'black' } | null => {
      if (state.activeTab === 'segmentation') {
          for (const s of state.exclusionZones) if (isPointInShape(x, y, s)) return { shape: s };
      } else if (state.activeTab === 'calibration') {
          if (state.blackPointROI && isPointInShape(x, y, state.blackPointROI)) return { shape: state.blackPointROI, type: 'black' };
          if (state.whitePointROI && isPointInShape(x, y, state.whitePointROI)) return { shape: state.whitePointROI, type: 'white' };
          if (state.calibrationROI && isPointInShape(x, y, state.calibrationROI)) return { shape: state.calibrationROI, type: 'gray' };
      } else if (state.activeTab === 'analysis') {
          for (const g of state.roiGroups) {
              for (const s of g.shapes) if (isPointInShape(x, y, s)) return { shape: s, group: g };
          }
      }
      return null;
  };

  const getHandleUnderCursor = (x: number, y: number, shape: Shape): string | null => {
      const bbox = getBoundingBox(shape);
      const tol = HANDLE_SIZE / state.zoom;
      if (Math.abs(x - bbox.minX) < tol && Math.abs(y - bbox.minY) < tol) return 'nw';
      if (Math.abs(x - bbox.maxX) < tol && Math.abs(y - bbox.minY) < tol) return 'ne';
      if (Math.abs(x - bbox.minX) < tol && Math.abs(y - bbox.maxY) < tol) return 'sw';
      if (Math.abs(x - bbox.maxX) < tol && Math.abs(y - bbox.maxY) < tol) return 'se';
      return null;
  };

  const updateShapeInState = (newShape: Shape) => {
      if (state.activeTab === 'segmentation') {
          setState(s => ({...s, exclusionZones: s.exclusionZones.map(sh => sh.id === newShape.id ? newShape : sh)}));
      } else if (state.activeTab === 'calibration') {
          if (state.calibrationROI?.id === newShape.id) setState(s => ({...s, calibrationROI: newShape}));
          else if (state.whitePointROI?.id === newShape.id) setState(s => ({...s, whitePointROI: newShape}));
          else if (state.blackPointROI?.id === newShape.id) setState(s => ({...s, blackPointROI: newShape}));
      } else if (state.activeTab === 'analysis') {
          setState(s => ({...s, roiGroups: s.roiGroups.map(g => ({...g, shapes: g.shapes.map(sh => sh.id === newShape.id ? newShape : sh)}))}));
      }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!containerRef.current) return;
    const factor = e.deltaY > 0 ? 1 / 1.1 : 1.1;
    let newZoom = Math.max(0.01, Math.min(50, state.zoom * factor));
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const imgX = (mouseX - state.pan.x) / state.zoom;
    const imgY = (mouseY - state.pan.y) / state.zoom;
    const newPanX = mouseX - imgX * newZoom;
    const newPanY = mouseY - imgY * newZoom;
    setState(s => ({ ...s, zoom: newZoom, pan: { x: newPanX, y: newPanY } }));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const startPoint = { x, y };

    if (state.activeTool === 'pan' || e.button === 1 || e.buttons === 4) {
         setDragState({ mode: 'pan', startPoint, startScreenPoint: { x: e.clientX, y: e.clientY }, activeHandle: null, initialPoints: [], initialPan: { ...state.pan } });
         return;
    }

    if (state.activeTool === 'select') {
        if (state.selectedShapeId) {
            let selectedShape: Shape | null = null;
            if (state.activeTab === 'segmentation') selectedShape = state.exclusionZones.find(s => s.id === state.selectedShapeId) || null;
            else if (state.activeTab === 'calibration') {
                 if (state.calibrationROI?.id === state.selectedShapeId) selectedShape = state.calibrationROI;
                 else if (state.whitePointROI?.id === state.selectedShapeId) selectedShape = state.whitePointROI;
                 else if (state.blackPointROI?.id === state.selectedShapeId) selectedShape = state.blackPointROI;
            } else if (state.activeTab === 'analysis') {
                 state.roiGroups.forEach(g => { const f = g.shapes.find(s => s.id === state.selectedShapeId); if (f) selectedShape = f; });
            }
            if (selectedShape) {
                const handle = getHandleUnderCursor(x, y, selectedShape);
                if (handle) {
                    setDragState({ mode: 'resize', startPoint, startScreenPoint: {x:0,y:0}, activeHandle: handle, initialPoints: JSON.parse(JSON.stringify(selectedShape.points)) });
                    return;
                }
            }
        }
        const hit = getShapeUnderCursor(x, y);
        if (hit) {
            setState(s => ({ ...s, selectedShapeId: hit.shape.id, activeGroupId: hit.group?.id || s.activeGroupId, activeCalibrationTarget: hit.type || s.activeCalibrationTarget }));
            setDragState({ mode: 'move', startPoint, startScreenPoint: {x:0,y:0}, activeHandle: null, initialPoints: JSON.parse(JSON.stringify(hit.shape.points)) });
        } else {
            setState(s => ({ ...s, selectedShapeId: null }));
        }
    } else {
        const newShape: Shape = { id: Math.random().toString(36), type: state.activeTool as any, points: [startPoint, startPoint] };
        if (state.activeTab === 'segmentation') {
             setState(s => ({...s, exclusionZones: [...s.exclusionZones, newShape], selectedShapeId: newShape.id }));
        } else if (state.activeTab === 'calibration') {
             const updates: Partial<AppState> = { selectedShapeId: newShape.id };
             if (state.activeCalibrationTarget === 'gray') updates.calibrationROI = newShape;
             else if (state.activeCalibrationTarget === 'white') updates.whitePointROI = newShape;
             else if (state.activeCalibrationTarget === 'black') updates.blackPointROI = newShape;
             setState(s => ({...s, ...updates }));
        } else if (state.activeTab === 'analysis') {
             if (state.activeGroupId) {
                 setState(s => ({ ...s, roiGroups: s.roiGroups.map(g => g.id === s.activeGroupId ? {...g, shapes: [...g.shapes, newShape]} : g), selectedShapeId: newShape.id }));
             } else {
                 const newGroup: ROIGroup = { id: Math.random().toString(36), name: `Group ${state.roiGroups.length + 1}`, color: COLORS[state.roiGroups.length % COLORS.length], shapes: [newShape] };
                 setState(s => ({ ...s, roiGroups: [...s.roiGroups, newGroup], activeGroupId: newGroup.id, selectedShapeId: newShape.id }));
             }
        }
        setDragState({ mode: 'create', startPoint, startScreenPoint: {x:0,y:0}, activeHandle: null, initialPoints: [] });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragState || !canvasRef.current) return;
    if (dragState.mode === 'pan' && dragState.initialPan) {
         setState(s => ({ ...s, pan: { x: dragState.initialPan!.x + (e.clientX - dragState.startScreenPoint.x), y: dragState.initialPan!.y + (e.clientY - dragState.startScreenPoint.y) } }));
         return;
    }
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width);
    const y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    const currentPoint = { x, y };

    if (dragState.mode === 'create') {
        let shapeId = state.selectedShapeId;
        if (!shapeId) return;
        let shape: Shape | undefined;
        if (state.activeTab === 'segmentation') shape = state.exclusionZones.find(s => s.id === shapeId);
        else if (state.activeTab === 'calibration') {
            if (state.calibrationROI?.id === shapeId) shape = state.calibrationROI;
            else if (state.whitePointROI?.id === shapeId) shape = state.whitePointROI;
            else if (state.blackPointROI?.id === shapeId) shape = state.blackPointROI;
        } else if (state.activeTab === 'analysis') state.roiGroups.forEach(g => { if(!shape) shape = g.shapes.find(s => s.id === shapeId); });
        if (shape) {
            let newPoints = [...shape.points];
            if (shape.type === 'lasso') newPoints.push(currentPoint); else newPoints[1] = currentPoint;
            updateShapeInState({ ...shape, points: newPoints });
        }
    } else if (dragState.mode === 'move') {
        const dx = x - dragState.startPoint.x, dy = y - dragState.startPoint.y;
        let shapeId = state.selectedShapeId;
        let shape: Shape | undefined;
        if (state.activeTab === 'segmentation') shape = state.exclusionZones.find(s => s.id === shapeId);
        else if (state.activeTab === 'calibration') {
            if (state.calibrationROI?.id === shapeId) shape = state.calibrationROI;
            else if (state.whitePointROI?.id === shapeId) shape = state.whitePointROI;
            else if (state.blackPointROI?.id === shapeId) shape = state.blackPointROI;
        } else if (state.activeTab === 'analysis') state.roiGroups.forEach(g => { if(!shape) shape = g.shapes.find(s => s.id === shapeId); });
        if (shape) updateShapeInState({ ...shape, points: dragState.initialPoints.map(p => ({ x: p.x + dx, y: p.y + dy })) });
    } else if (dragState.mode === 'resize') {
        let shapeId = state.selectedShapeId;
        let shape: Shape | undefined;
        if (state.activeTab === 'segmentation') shape = state.exclusionZones.find(s => s.id === shapeId);
        else if (state.activeTab === 'calibration') {
             if (state.calibrationROI?.id === shapeId) shape = state.calibrationROI;
             else if (state.whitePointROI?.id === shapeId) shape = state.whitePointROI;
             else if (state.blackPointROI?.id === shapeId) shape = state.blackPointROI;
        } else if (state.activeTab === 'analysis') state.roiGroups.forEach(g => { if(!shape) shape = g.shapes.find(s => s.id === shapeId); });
        if (shape) {
            const initBBox = getBoundingBox({ ...shape, points: dragState.initialPoints });
            let newMinX = initBBox.minX, newMaxX = initBBox.maxX, newMinY = initBBox.minY, newMaxY = initBBox.maxY;
            const dx = x - dragState.startPoint.x, dy = y - dragState.startPoint.y;
            if (dragState.activeHandle === 'nw') { newMinX += dx; newMinY += dy; }
            if (dragState.activeHandle === 'ne') { newMaxX += dx; newMinY += dy; }
            if (dragState.activeHandle === 'sw') { newMinX += dx; newMaxY += dy; }
            if (dragState.activeHandle === 'se') { newMaxX += dx; newMaxY += dy; }
            const scaleX = (newMaxX - newMinX) / (initBBox.maxX - initBBox.minX || 1), scaleY = (newMaxY - newMinY) / (initBBox.maxY - initBBox.minY || 1);
            updateShapeInState({ ...shape, points: dragState.initialPoints.map(p => ({ x: newMinX + (p.x - initBBox.minX) * scaleX, y: newMinY + (p.y - initBBox.minY) * scaleY })) });
        }
    }
  };

  const handleMouseUp = () => {
    if (dragState?.mode === 'create' || dragState?.mode === 'move' || dragState?.mode === 'resize') {
        if (state.activeTab === 'calibration') {
             if (state.calibrationROI && state.selectedShapeId === state.calibrationROI.id) calculateCalibrationFromROI(state.calibrationROI, 'gray');
             if (state.whitePointROI && state.selectedShapeId === state.whitePointROI.id) calculateCalibrationFromROI(state.whitePointROI, 'white');
             if (state.blackPointROI && state.selectedShapeId === state.blackPointROI.id) calculateCalibrationFromROI(state.blackPointROI, 'black');
        }
    }
    setDragState(null);
    if (state.activeTool !== 'select' && state.activeTool !== 'pan') setState(s => ({ ...s, activeTool: 'select' }));
  };

  const deleteSelectedShape = () => {
    setState(s => {
      if (!s.selectedShapeId) return s;
      let newState = { ...s };
      if (s.activeTab === 'segmentation') newState.exclusionZones = s.exclusionZones.filter(sh => sh.id !== s.selectedShapeId);
      else if (s.activeTab === 'calibration') {
        if (s.calibrationROI?.id === s.selectedShapeId) { newState.calibrationROI = null; newState.calibrationColor = null; }
        if (s.whitePointROI?.id === s.selectedShapeId) { newState.whitePointROI = null; newState.whitePointColor = null; }
        if (s.blackPointROI?.id === s.selectedShapeId) { newState.blackPointROI = null; newState.blackPointColor = null; }
      } else if (s.activeTab === 'analysis') newState.roiGroups = s.roiGroups.map(g => ({ ...g, shapes: g.shapes.filter(sh => sh.id !== s.selectedShapeId) }));
      newState.selectedShapeId = null;
      return newState;
    });
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && state.selectedShapeId && document.activeElement?.tagName !== 'INPUT') deleteSelectedShape();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.selectedShapeId, state.activeTab]);

  const calculateCalibrationFromROI = (shape: Shape, type: 'gray' | 'white' | 'black') => {
     if (!loadedImageRef.current || !canvasRef.current) return;
     const ctx = canvasRef.current.getContext('2d'); if (!ctx) return;
     const bbox = getBoundingBox(shape);
     const imageData = ctx.getImageData(bbox.minX, bbox.minY, bbox.maxX - bbox.minX, bbox.maxY - bbox.minY);
     const data = imageData.data;
     let rSum = 0, gSum = 0, bSum = 0, count = 0;
     for (let i = 0; i < data.length; i += 4) {
        const globalX = bbox.minX + ((i/4) % imageData.width), globalY = bbox.minY + Math.floor((i/4) / imageData.width);
        if (isPointInShape(globalX, globalY, shape)) { rSum += data[i]; gSum += data[i+1]; bSum += data[i+2]; count++; }
     }
     if (count > 0) {
        const color = { r: rSum / count, g: gSum / count, b: bSum / count };
        if (type === 'gray') setState(s => ({ ...s, calibrationColor: color }));
        if (type === 'white') setState(s => ({ ...s, whitePointColor: color }));
        if (type === 'black') setState(s => ({ ...s, blackPointColor: color }));
     }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files; if (!files || files.length === 0) return;
    const newImages: ImageAsset[] = [];
    for (let i = 0; i < files.length; i++) {
       const file = files[i];
       if (file.name.endsWith('.zip')) {
          const zip = new JSZip(), contents = await zip.loadAsync(file), imagePromises: Promise<void>[] = [];
          contents.forEach((relativePath, zipEntry) => { if (!zipEntry.dir && (relativePath.match(/\.(jpg|jpeg|png)$/i))) imagePromises.push(zipEntry.async('base64').then(b64 => { newImages.push({ id: Math.random().toString(36), name: relativePath, url: `data:image/${relativePath.split('.').pop()};base64,${b64}` }); })); });
          await Promise.all(imagePromises);
       } else if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          await new Promise<void>((resolve) => { reader.onload = (event) => { if (event.target?.result) newImages.push({ id: Math.random().toString(36), name: file.name, url: event.target.result as string }); resolve(); }; reader.readAsDataURL(file); });
       }
    }
    if (newImages.length > 0) setState(s => ({ ...s, gallery: [...s.gallery, ...newImages], activeImageIndex: s.gallery.length }));
  };

  const handleDemoLoad = (url: string, label: string) => {
    setState(s => ({ ...s, gallery: [{ id: 'demo-'+Date.now(), name: label, url: `${url}?t=${Date.now()}` }], activeImageIndex: 0, calibrationColor: null, roiGroups: [], exclusionZones: [], reportSummary: '', processedImageURL: null }));
  };

  const handleGenerateReport = async () => {
    if (canvasRef.current && overlayRef.current) {
       const reportCanvas = document.createElement('canvas');
       reportCanvas.width = canvasRef.current.width; reportCanvas.height = canvasRef.current.height;
       const rctx = reportCanvas.getContext('2d');
       if (rctx) { rctx.drawImage(canvasRef.current, 0, 0); rctx.drawImage(overlayRef.current, 0, 0); setState(s => ({ ...s, processedImageURL: reportCanvas.toDataURL('image/png') })); }
    }
    setState(s => ({ ...s, activeTab: 'report', isProcessing: true }));
    try {
      const ai = new GeminiClient(); 
      const summary = await ai.generateReportSummary(state.roiGroups, state.regressionParams);
      setState(s => ({ ...s, reportSummary: summary, isProcessing: false }));
    } catch (error) { setState(s => ({ ...s, isProcessing: false, reportSummary: "Error generating summary." })); }
  };

  const handleAutoTune = () => {
    const target = state.regressionParams.targetIndex, groups = state.roiGroups.filter(g => g.stats && g.stats.pixelCount > 0);
    if (groups.length < 2) {
      const defaultSlope = target === 'mACI' ? 30.5 : 150.2, defaultIntercept = target === 'mACI' ? -10.5 : 5.2;
      setState(s => ({ ...s, regressionParams: { ...s.regressionParams, slope: defaultSlope, intercept: defaultIntercept } }));
      return;
    }
    let minIndex = Infinity, maxIndex = -Infinity;
    groups.forEach(g => { const val = target === 'mACI' ? g.stats!.meanMACI : g.stats!.meanNGRDI; if (val < minIndex) minIndex = val; if (val > maxIndex) maxIndex = val; });
    if (maxIndex - minIndex < 0.05) {
       const defaultSlope = target === 'mACI' ? 30.5 : 150.2, defaultIntercept = target === 'mACI' ? -10.5 : 5.2;
       setState(s => ({ ...s, regressionParams: { ...s.regressionParams, slope: defaultSlope, intercept: defaultIntercept } }));
       return;
    }
    const slope = (40.0 - 1.0) / (maxIndex - minIndex), intercept = 1.0 - (slope * minIndex);
    setState(s => ({ ...s, regressionParams: { ...s.regressionParams, slope: parseFloat(slope.toFixed(2)), intercept: parseFloat(intercept.toFixed(2)) } }));
  };

  const uploadToGithub = async () => {
     if (!githubConfig.token || !githubConfig.owner || !githubConfig.repo) { alert("Incomplete config."); return; }
     setGithubStatus('uploading');
     const timestamp = new Date().toISOString().replace(/[:.]/g, '-'), basePath = `${githubConfig.path}/${timestamp}`;
     try {
         const uploadFile = async (filename: string, contentBase64: string, message: string) => {
            const url = `https://api.github.com/repos/${githubConfig.owner}/${githubConfig.repo}/contents/${basePath}/${filename}`;
            const res = await fetch(url, { method: 'PUT', headers: { 'Authorization': `token ${githubConfig.token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ message, content: contentBase64 }) });
            if (!res.ok) throw new Error(await res.text());
         };
         await uploadFile('report.md', btoa(`# BioPheno Report\n${state.reportSummary}`), 'Add Report');
         if (state.processedImageURL) await uploadFile('analyzed.png', state.processedImageURL.split(',')[1], 'Add Image');
         setGithubStatus('success'); setTimeout(() => { setGithubStatus('idle'); setState(s => ({...s, isGithubModalOpen: false})); }, 2000);
     } catch (e) { setGithubStatus('error'); }
  };

  const BarChartSection = ({ title, dataKey, formatFn }: any) => (
    <div className="mb-6">
       <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-2">{title}</p>
       <div className="space-y-2">
         {state.roiGroups.map(g => {
           const val = (g.stats as any)?.[dataKey] || 0, maxVal = Math.max(...state.roiGroups.map(gr => (gr.stats as any)?.[dataKey] || 0), 0.1); 
           return (
             <div key={g.id} className="flex items-center gap-2 text-xs">
               <span className="w-16 truncate text-slate-500">{g.name}</span>
               <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden relative">
                 <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(0, Math.min(100, (val / maxVal) * 100))}%`, backgroundColor: g.color }}></div>
               </div>
               <span className="w-12 text-right font-mono text-slate-300">{formatFn(val)}</span>
             </div>
           );
         })}
       </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <aside className="w-64 border-r border-slate-800 bg-slate-900 flex flex-col print:hidden">
        <div className="p-6 border-b border-slate-800">
          <h1 className="flex items-center gap-2 font-bold text-lg text-emerald-400"><Leaf className="w-6 h-6" /> BioPheno</h1>
          <p className="text-xs text-slate-500 mt-1">Kim & van Iersel (2023)</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <NavButton active={state.activeTab === 'segmentation'} onClick={() => setState(s => ({ ...s, activeTab: 'segmentation', activeTool: 'select', selectedShapeId: null }))} icon={<Maximize2 size={18} />} label="Segmentation" />
          <NavButton active={state.activeTab === 'calibration'} onClick={() => setState(s => ({ ...s, activeTab: 'calibration', activeTool: 'select', selectedShapeId: null }))} icon={<Pipette size={18} />} label="Calibration & Align" />
          <NavButton active={state.activeTab === 'analysis'} onClick={() => setState(s => ({ ...s, activeTab: 'analysis', activeTool: 'select', selectedShapeId: null }))} icon={<BarChart3 size={18} />} label="Analysis & Results" />
          <div className="pt-4 mt-4 border-t border-slate-800 space-y-2">
             <button onClick={handleGenerateReport} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${state.activeTab === 'report' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}><FileText size={18} /> Generate Report</button>
             <button onClick={() => setState(s => ({...s, isGithubModalOpen: true}))} className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-400 hover:bg-slate-800 hover:text-slate-200"><Github size={18} /> Export to GitHub</button>
          </div>
        </nav>
      </aside>

      <main className="flex-1 flex flex-col relative overflow-hidden">
        {state.activeTab !== 'report' && (
          <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur z-20 relative">
            <div className="flex items-center gap-4">
              <h2 className="font-medium text-slate-200 hidden md:block">
                {state.activeTab === 'segmentation' && 'Segmentation'} {state.activeTab === 'calibration' && 'Calibration'} {state.activeTab === 'analysis' && 'Analysis'}
              </h2>
              <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-4">
                <button onClick={() => setState(s => ({...s, activeTool: 'pan'}))} className={`p-2 rounded ${state.activeTool === 'pan' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Move size={16} /></button>
                <div className="w-[1px] h-6 bg-slate-700 mx-1"></div>
                <button onClick={() => setState(s => ({...s, activeTool: 'select'}))} className={`p-2 rounded ${state.activeTool === 'select' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><MousePointer2 size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Square size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><CircleIcon size={16} /></button>
                <button onClick={() => setState(s => ({...s, activeTool: 'lasso'}))} className={`p-2 rounded ${state.activeTool === 'lasso' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Lasso size={16} /></button>
                {state.selectedShapeId && <button onClick={deleteSelectedShape} className="p-2 rounded text-rose-400 hover:bg-rose-900/50 ml-2 border-l border-slate-700"><Trash2 size={16} /></button>}
              </div>
            </div>
            
            <div className="flex items-center gap-3">
               {loadedImageRef.current && (
                  <div className="flex items-center bg-slate-800 rounded border border-slate-700 mr-2">
                     <button onClick={() => setState(s => ({...s, zoom: Math.max(0.01, s.zoom * 0.8)}))} className="p-1.5 hover:bg-slate-700 text-slate-400"><ZoomOut size={14}/></button>
                     <span className="text-[10px] w-12 text-center text-slate-400 font-mono">{(state.zoom * 100).toFixed(0)}%</span>
                     <button onClick={() => setState(s => ({...s, zoom: Math.min(50, s.zoom * 1.25)}))} className="p-1.5 hover:bg-slate-700 text-slate-400"><ZoomIn size={14}/></button>
                     <div className="w-[1px] h-4 bg-slate-700 mx-1"></div>
                     <button onClick={fitImageToScreen} className="p-1.5 hover:bg-slate-700 text-slate-400"><Minimize size={14}/></button>
                  </div>
               )}
              {state.gallery.length > 0 && (
                 <div className="flex items-center bg-slate-800 rounded border border-slate-700 mr-2">
                    <button disabled={state.activeImageIndex === 0} onClick={() => setState(s => ({...s, activeImageIndex: Math.max(0, s.activeImageIndex - 1)}))} className="p-1 hover:bg-slate-700 disabled:opacity-30"><ChevronLeft size={16}/></button>
                    <span className="text-xs px-2 w-20 truncate text-center text-slate-400">{state.activeImageIndex + 1} / {state.gallery.length}</span>
                    <button disabled={state.activeImageIndex === state.gallery.length - 1} onClick={() => setState(s => ({...s, activeImageIndex: Math.min(s.gallery.length - 1, s.activeImageIndex + 1)}))} className="p-1 hover:bg-slate-700 disabled:opacity-30"><ChevronRight size={16}/></button>
                 </div>
              )}
              <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-md text-sm font-medium"><Upload size={16} /> Upload</button>
              <input ref={fileInputRef} type="file" multiple accept=".jpg,.jpeg,.png,.zip" className="hidden" onChange={handleFileUpload} />
            </div>
          </header>
        )}

        {state.activeTab === 'report' && (
          <div className="flex-1 overflow-auto bg-slate-200 p-8 text-black">
             <div className="max-w-4xl mx-auto bg-white shadow-2xl min-h-[29.7cm] p-12">
                <div className="border-b-2 border-black pb-6 mb-8 flex justify-between items-end">
                   <div><h1 className="text-3xl font-bold font-serif mb-2">Phenotypic Analysis Report</h1><p className="text-sm text-gray-600 font-serif">BioPheno | {new Date().toLocaleDateString()}</p></div>
                   <button onClick={() => window.print()} className="print:hidden flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded hover:bg-slate-700 text-sm"><Printer size={16}/> Print PDF</button>
                </div>
                <section className="mb-8">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">1. Methodology</h2>
                   <p className="text-sm leading-relaxed text-justify mb-4">Image analysis was performed using the BioPheno suite. Vegetation segmentation utilized ExG with a threshold of {state.segmentationThreshold}. Alignment was normalized at {state.rotationAngle.toFixed(2)}° rotation.</p>
                </section>
                <section className="mb-8 break-inside-avoid">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">2. Visual Analysis</h2>
                   <div className="grid grid-cols-2 gap-4 mb-2">
                      <div className="border border-gray-200 p-1"><img src={currentImageUrl || ''} className="w-full h-auto" /><p className="text-center text-xs font-serif mt-2 italic">Figure 1a. Original</p></div>
                      <div className="border border-gray-200 p-1">{state.processedImageURL && <img src={state.processedImageURL} className="w-full h-auto" />}<p className="text-center text-xs font-serif mt-2 italic">Figure 1b. Segmented</p></div>
                   </div>
                </section>
                <section className="mb-8">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">3. Quantitative Results</h2>
                   <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse border border-gray-300">
                         <thead className="bg-gray-100 font-serif"><tr><th className="border border-gray-300 px-3 py-2 text-left">Group</th><th className="border border-gray-300 px-3 py-2 text-right">Pixels</th><th className="border border-gray-300 px-3 py-2 text-right">mACI</th><th className="border border-gray-300 px-3 py-2 text-right">NGRDI</th><th className="border border-gray-300 px-3 py-2 text-right bg-slate-50">Anthocyanin (µg/cm²)</th></tr></thead>
                         <tbody>{state.roiGroups.map(g => (<tr key={g.id}><td className="border border-gray-300 px-3 py-2 font-medium">{g.name}</td><td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.pixelCount}</td><td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.meanMACI.toFixed(3)}</td><td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.meanNGRDI.toFixed(3)}</td><td className="border border-gray-300 px-3 py-2 text-right font-mono font-bold bg-slate-50">{g.stats?.anthocyanin.toFixed(2)}</td></tr>))}</tbody>
                      </table>
                   </div>
                </section>
                <section className="mb-8"><h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">4. Results & Discussion</h2>{state.isProcessing ? (<div className="flex items-center gap-2 text-slate-500 italic p-8 justify-center bg-gray-50 border border-dashed border-gray-300 rounded"><RefreshCw className="animate-spin" size={18} /> Generating scientific summary...</div>) : (<div className="text-sm leading-loose text-justify font-serif">{state.reportSummary.split('\n').map((para, i) => (<p key={i} className="mb-4">{para}</p>))}</div>)}</section>
             </div>
          </div>
        )}

        <div className={`flex-1 overflow-auto p-6 bg-slate-950 z-0 ${state.activeTab === 'report' ? 'hidden' : 'block'}`}>
          <div className="h-full flex gap-6 min-h-0">
              <div className="flex-1 flex flex-col gap-6">
                {!currentImageUrl ? (<div className="flex-1 border-2 border-dashed border-slate-800 rounded-xl flex items-center justify-center text-slate-500"><div className="text-center"><Leaf className="w-12 h-12 mx-auto mb-4 opacity-50" /><p>Load an image to begin</p></div></div>) : (
                  <div ref={containerRef} onWheel={handleWheel} className="flex-1 bg-slate-900 rounded-xl border border-slate-800 relative overflow-hidden">
                     <div style={{ transform: `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`, transformOrigin: '0 0', width: '100%', height: '100%' }}>
                        <canvas ref={canvasRef} className="absolute inset-0 block" />
                        <canvas ref={overlayRef} className={`absolute inset-0 w-full h-full pointer-events-auto ${state.activeTool === 'select' ? 'cursor-default' : (state.activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair')}`} style={{ cursor: dragState?.activeHandle ? (dragState.activeHandle === 'nw' || dragState.activeHandle === 'se' ? 'nwse-resize' : 'nesw-resize') : (state.activeTool === 'select' ? (state.selectedShapeId && getShapeUnderCursor(0,0)?.shape.id === state.selectedShapeId ? 'move' : 'default') : (state.activeTool === 'pan' ? 'grab' : 'crosshair')) }} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onContextMenu={e => e.preventDefault()} />
                     </div>
                  </div>
                )}
                <div className="text-xs text-slate-500 flex gap-4"><span>Scroll to zoom, middle-click to pan. ArUco detection available in Calibration.</span></div>
              </div>

              <div className="w-96 flex flex-col gap-6 overflow-y-auto pr-2">
                {state.activeTab === 'segmentation' && (
                  <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                    <h3 className="font-medium text-sm text-slate-200 mb-4 flex items-center gap-2"><Maximize2 size={16}/> Segmentation</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between text-xs text-slate-400"><span>ExG Threshold</span><span>{state.segmentationThreshold}</span></div>
                      <input type="range" min="0" max="100" value={state.segmentationThreshold} onChange={(e) => setState(s => ({ ...s, segmentationThreshold: parseInt(e.target.value) }))} className="w-full accent-emerald-500 h-1 bg-slate-700 rounded-lg appearance-none" />
                      {state.exclusionZones.length > 0 && (<div className="mt-4 pt-4 border-t border-slate-800"><div className="flex justify-between items-center mb-2"><span className="text-xs text-rose-400 font-medium">Excluded Regions ({state.exclusionZones.length})</span><button onClick={() => setState(s => ({...s, exclusionZones: [], selectedShapeId: null}))} className="text-[10px] text-slate-500 hover:text-rose-400">Clear All</button></div></div>)}
                    </div>
                  </div>
                )}

                {state.activeTab === 'calibration' && (
                  <div className="space-y-6">
                    {/* Geometric Alignment Section */}
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                       <h3 className="font-medium text-sm text-slate-200 mb-4 flex items-center gap-2"><Compass size={16}/> Geometric Alignment</h3>
                       <div className="space-y-4">
                          <button 
                            disabled={state.isDetectingAruco || !loadedImageRef.current}
                            onClick={handleDetectAruco}
                            className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-all disabled:opacity-50"
                          >
                             {state.isDetectingAruco ? <RefreshCw className="animate-spin" size={16}/> : <RotateCw size={16}/>}
                             {state.isDetectingAruco ? "Detecting ArUco..." : "Auto-Level with ArUco"}
                          </button>
                          <div>
                             <div className="flex justify-between text-xs text-slate-400 mb-2"><span>Rotation</span><span>{state.rotationAngle.toFixed(1)}°</span></div>
                             <input type="range" min="-180" max="180" step="0.5" value={state.rotationAngle} onChange={(e) => setState(s => ({ ...s, rotationAngle: parseFloat(e.target.value) }))} className="w-full accent-indigo-500 h-1 bg-slate-700 rounded-lg appearance-none" />
                          </div>
                          <p className="text-[10px] text-slate-500 italic text-center">Tray reorientation improves leaf area distribution accuracy.</p>
                       </div>
                    </div>

                    {/* Color Calibration Section */}
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                      <h3 className="font-medium text-sm text-slate-200 mb-4 flex items-center gap-2"><Pipette size={16}/> Color Reference</h3>
                      <div className="space-y-4">
                         <div className="flex border-b border-slate-800 pb-2 mb-2 gap-2">
                           <button onClick={() => setState(s => ({...s, activeCalibrationTarget: 'gray'}))} className={`flex-1 p-2 rounded text-[10px] flex flex-col items-center gap-1 ${state.activeCalibrationTarget === 'gray' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}><Disc size={14}/> Gray</button>
                           <button onClick={() => setState(s => ({...s, activeCalibrationTarget: 'white'}))} className={`flex-1 p-2 rounded text-[10px] flex flex-col items-center gap-1 ${state.activeCalibrationTarget === 'white' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}><Sun size={14}/> White</button>
                           <button onClick={() => setState(s => ({...s, activeCalibrationTarget: 'black'}))} className={`flex-1 p-2 rounded text-[10px] flex flex-col items-center gap-1 ${state.activeCalibrationTarget === 'black' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}><Moon size={14}/> Black</button>
                         </div>
                         <div className="grid grid-cols-1 gap-3">
                            {['gray', 'white', 'black'].map(t => {
                               const c = t === 'gray' ? state.calibrationColor : (t === 'white' ? state.whitePointColor : state.blackPointColor);
                               return (
                                 <div key={t} className={`p-2 rounded border border-slate-800 flex justify-between items-center ${state.activeCalibrationTarget === t ? 'border-slate-600' : ''}`}>
                                    <span className="text-[10px] uppercase text-slate-500">{t}</span>
                                    <div className="flex gap-2 font-mono text-[10px] text-slate-300">
                                       <span>R:{c?.r.toFixed(0) || '-'}</span> <span>G:{c?.g.toFixed(0) || '-'}</span> <span>B:{c?.b.toFixed(0) || '-'}</span>
                                    </div>
                                 </div>
                               );
                            })}
                         </div>
                      </div>
                    </div>
                  </div>
                )}

                {state.activeTab === 'analysis' && (
                  <div className="flex flex-col gap-4">
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                      <div className="flex items-center justify-between mb-4"><h3 className="font-medium text-sm text-slate-200 flex items-center gap-2"><FlaskConical size={16}/> Groups</h3><button onClick={() => { const newGroup: ROIGroup = { id: Math.random().toString(36), name: `Group ${state.roiGroups.length + 1}`, color: COLORS[state.roiGroups.length % COLORS.length], shapes: [] }; setState(s => ({ ...s, roiGroups: [...s.roiGroups, newGroup], activeGroupId: newGroup.id })); }} className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"><Plus size={14}/></button></div>
                      <div className="space-y-2 max-h-48 overflow-y-auto">{state.roiGroups.length === 0 && <p className="text-xs text-slate-500 italic">No groups. Define ROI to begin.</p>}{state.roiGroups.map(group => (<div key={group.id} className={`p-2 rounded border flex items-center gap-2 cursor-pointer ${state.activeGroupId === group.id ? 'bg-slate-800 border-indigo-500/50' : 'bg-slate-950 border-slate-800'}`} onClick={() => setState(s => ({...s, activeGroupId: group.id}))}><div className="w-3 h-3 rounded-full" style={{backgroundColor: group.color}}></div><input value={group.name} onChange={(e) => setState(s => ({ ...s, roiGroups: s.roiGroups.map(g => g.id === group.id ? {...g, name: e.target.value} : g) }))} className="bg-transparent text-xs text-slate-200 focus:outline-none w-full" /><button onClick={(e) => { e.stopPropagation(); setState(s => ({...s, roiGroups: s.roiGroups.filter(g => g.id !== group.id)})); }} className="text-slate-600 hover:text-rose-400"><Trash2 size={12}/></button></div>))}</div>
                    </div>
                    {state.roiGroups.length > 0 && (<div className="bg-slate-900 rounded-xl border border-slate-800 p-6"><h3 className="font-medium text-sm text-slate-200 mb-4">Comparative Analysis</h3><BarChartSection title="Est. Anthocyanin (µg/cm²)" dataKey="anthocyanin" formatFn={(v: number) => v.toFixed(2)} /><div className="grid grid-cols-1 gap-6 pt-4 border-t border-slate-800"><BarChartSection title="mACI Index" dataKey="meanMACI" formatFn={(v: number) => v.toFixed(3)} /><BarChartSection title="NGRDI Index" dataKey="meanNGRDI" formatFn={(v: number) => v.toFixed(3)} /></div></div>)}
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
                        <div className="flex justify-between items-center mb-2"><p className="text-[10px] text-slate-400 font-medium">Model Params</p><button onClick={handleAutoTune} className="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center gap-1"><Wand2 size={10} /> Auto-Tune</button></div>
                        <div className="grid grid-cols-2 gap-2 mb-2">
                           <input type="number" step="0.1" value={state.regressionParams.slope} onChange={(e) => setState(s => ({...s, regressionParams: {...s.regressionParams, slope: parseFloat(e.target.value)}}))} className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs" />
                           <input type="number" step="0.1" value={state.regressionParams.intercept} onChange={(e) => setState(s => ({...s, regressionParams: {...s.regressionParams, intercept: parseFloat(e.target.value)}}))} className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs" />
                        </div>
                    </div>
                  </div>
                )}
              </div>
          </div>
        </div>
      </main>
    </div>
  );
};

// --- Utils ---

const NavButton = ({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) => (
  <button onClick={onClick} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${active ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}>{icon} {label}</button>
);

class GeminiClient {
  private ai: GoogleGenAI;
  constructor() { this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY }); }
  
  async detectArucoCorners(base64Image: string): Promise<Point[] | null> {
    const prompt = "Analyze this image and find the single largest ArUco marker. Return the coordinates for its four corners: top-left, top-right, bottom-right, bottom-left. Return the coordinates as a JSON array of objects with 'x' and 'y' properties. Coordinates should be normalized from 0 to 1000 relative to image width/height.";
    
    try {
      const response = await this.ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ inlineData: { mimeType: 'image/jpeg', data: base64Image } }, { text: prompt }] },
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                x: { type: Type.NUMBER },
                y: { type: Type.NUMBER }
              },
              required: ['x', 'y']
            }
          }
        }
      });
      
      const corners = JSON.parse(response.text || '[]');
      // Convert normalized 0-1000 back to actual image coords is not needed here 
      // if we only care about the relative angle, which we do.
      return corners;
    } catch (e) {
      console.error("ArUco detection error:", e);
      return null;
    }
  }

  async generateReportSummary(groups: ROIGroup[], regression: any) {
    const dataSummary = groups.map(g => `Group: ${g.name}, mACI: ${g.stats?.meanMACI.toFixed(3)}, NGRDI: ${g.stats?.meanNGRDI.toFixed(3)}, Anthocyanin: ${g.stats?.anthocyanin.toFixed(2)}`).join('; ');
    const prompt = `You are a Bioinformatics Scientist writing 'Results and Discussion'. Data: ${dataSummary}. Discuss biological implications of mACI and NGRDI in lettuce phenotyping. y = ${regression.slope}x + ${regression.intercept}. Approx 150 words. Plain text.`;
    try {
      const response = await this.ai.models.generateContent({ model: 'gemini-3-flash-preview', contents: prompt });
      return response.text;
    } catch (e) { return "Error generating summary."; }
  }
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
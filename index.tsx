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
  Disc
} from 'lucide-react';
import { GoogleGenAI } from "@google/genai";
import JSZip from 'jszip';

// --- Types ---

type ToolType = 'select' | 'rect' | 'circle' | 'lasso';
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
  calibrationColor: { r: number, g: number, b: number } | null; // Gray
  calibrationROI: Shape | null;
  whitePointColor: { r: number, g: number, b: number } | null;
  whitePointROI: Shape | null;
  blackPointColor: { r: number, g: number, b: number } | null;
  blackPointROI: Shape | null;

  // Segmentation Editing
  exclusionZones: Shape[];

  // Quantification
  roiGroups: ROIGroup[];
  activeGroupId: string | null;

  // UI State
  isGithubModalOpen: boolean;
  activeTool: ToolType;
  selectedShapeId: string | null; // Added selection state
  
  // Regression
  regressionParams: {
    slope: number;
    intercept: number;
    targetIndex: 'mACI' | 'NGRDI';
  };
}

interface DragState {
  mode: 'move' | 'resize' | 'create';
  startPoint: Point;
  activeHandle: string | null; // 'nw', 'ne', 'sw', 'se'
  initialPoints: Point[]; // Snapshot of points before drag
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

    exclusionZones: [],
    roiGroups: [],
    activeGroupId: null,
    isGithubModalOpen: false,
    activeTool: 'select',
    selectedShapeId: null,
    regressionParams: {
      slope: 1.5,
      intercept: 0.2,
      targetIndex: 'mACI'
    }
  });

  const [dragState, setDragState] = useState<DragState | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);
  const [githubConfig, setGithubConfig] = useState({ token: '', owner: '', repo: '', path: 'biopheno-results' });
  const [githubStatus, setGithubStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  // --- Initial Load ---
  useEffect(() => {
    // Load default image silently without button
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
      setState(s => ({ 
        ...s, 
        exclusionZones: [], 
        roiGroups: [], 
        activeGroupId: null,
        // Reset Calibration
        calibrationColor: null,
        calibrationROI: null,
        whitePointColor: null,
        whitePointROI: null,
        blackPointColor: null,
        blackPointROI: null,
        reportSummary: '',
        processedImageURL: null,
        selectedShapeId: null
      }));
    };
  }, [currentImageUrl]);

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
    state.exclusionZones, 
    state.roiGroups,
    loadedImageRef.current,
    state.regressionParams,
    state.selectedShapeId
  ]);

  const runPipeline = () => {
    if (!loadedImageRef.current || !canvasRef.current) return;

    const img = loadedImageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    if (canvas.width !== img.width) canvas.width = img.width;
    if (canvas.height !== img.height) canvas.height = img.height;
    
    if (overlayRef.current) {
      overlayRef.current.width = canvas.width;
      overlayRef.current.height = canvas.height;
    }

    ctx.drawImage(img, 0, 0);

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

      // Calibration Parameters
      let rScale = 1, gScale = 1, bScale = 1;
      let minR = 0, minG = 0, minB = 0;
      let rangeR = 255, rangeG = 255, rangeB = 255;
      let useContrastStretch = false;
      let useGrayBalance = false;

      if (state.whitePointColor && state.blackPointColor) {
         // Priority 1: Full Range Correction if both White and Black are set
         useContrastStretch = true;
         minR = state.blackPointColor.r;
         minG = state.blackPointColor.g;
         minB = state.blackPointColor.b;
         rangeR = (state.whitePointColor.r - minR) || 1;
         rangeG = (state.whitePointColor.g - minG) || 1;
         rangeB = (state.whitePointColor.b - minB) || 1;
      } else if (state.whitePointColor) {
         // Priority 2: White Balance only (Assume White = 255, 255, 255)
         useGrayBalance = true;
         rScale = 255 / (state.whitePointColor.r || 1);
         gScale = 255 / (state.whitePointColor.g || 1);
         bScale = 255 / (state.whitePointColor.b || 1);
      } else if (state.calibrationColor) {
         // Priority 3: Gray Balance (Assume Gray = 128, 128, 128)
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

        // Apply Calibration
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
                    const val = ngrdi;
                    const minVal = 0.0, maxVal = 0.5;
                    const t = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
                    data[i] = 220 * (1-t);
                    data[i+1] = 220 * (1-t) + 100 * t;
                    data[i+2] = 50 * (1-t);
                } else if (state.visualizationMode === 'maci') {
                    const val = maci;
                    const minVal = 0.5, maxVal = 1.5;
                    const t = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
                    data[i] = 220 * t;
                    data[i+1] = 200 * (1-t);
                    data[i+2] = 50 * t;
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

        return {
          ...g,
          stats: { ...g.stats, meanMACI, meanNGRDI, anthocyanin: antho }
        };
      });

      const currentStatsStr = JSON.stringify(state.roiGroups.map(g => g.stats));
      const newStatsStr = JSON.stringify(finalGroups.map(g => g.stats));
      
      if (currentStatsStr !== newStatsStr && !dragState) {
         setTimeout(() => {
            setState(s => ({ ...s, roiGroups: finalGroups }));
         }, 0);
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

    const drawShape = (shape: Shape, color: string, fill: boolean = false, isSelected: boolean = false) => {
      ctx.beginPath();
      if (shape.type === 'rect') {
        const w = shape.points[1].x - shape.points[0].x;
        const h = shape.points[1].y - shape.points[0].y;
        ctx.rect(shape.points[0].x, shape.points[0].y, w, h);
      } else if (shape.type === 'circle') {
        const r = Math.sqrt(
          Math.pow(shape.points[1].x - shape.points[0].x, 2) + 
          Math.pow(shape.points[1].y - shape.points[0].y, 2)
        );
        ctx.arc(shape.points[0].x, shape.points[0].y, r, 0, 2 * Math.PI);
      } else if (shape.type === 'lasso') {
        ctx.moveTo(shape.points[0].x, shape.points[0].y);
        for (let i = 1; i < shape.points.length; i++) {
          ctx.lineTo(shape.points[i].x, shape.points[i].y);
        }
        ctx.closePath();
      }
      
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
      if (fill) {
        ctx.fillStyle = color + '40'; // 25% opacity
        ctx.fill();
      }

      // Draw selection UI
      if (isSelected) {
         const bbox = getBoundingBox(shape);
         ctx.save();
         ctx.strokeStyle = '#38bdf8'; // light blue
         ctx.lineWidth = 1;
         ctx.setLineDash([4, 4]);
         ctx.strokeRect(bbox.minX - 2, bbox.minY - 2, bbox.width + 4, bbox.height + 4);
         ctx.restore();

         // Handles
         ctx.fillStyle = '#38bdf8';
         const handles = [
             {x: bbox.minX, y: bbox.minY}, // nw
             {x: bbox.maxX, y: bbox.minY}, // ne
             {x: bbox.minX, y: bbox.maxY}, // sw
             {x: bbox.maxX, y: bbox.maxY}, // se
         ];
         handles.forEach(h => {
             ctx.fillRect(h.x - HANDLE_SIZE/2, h.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
         });
      }
    };

    if (state.activeTab !== 'report') {
        state.exclusionZones.forEach(shape => drawShape(shape, '#ef4444', true, shape.id === state.selectedShapeId));
        
        // Draw Calibration Shapes
        if (state.activeTab === 'calibration') {
           if (state.calibrationROI) drawShape(state.calibrationROI, '#ffffff', false, state.calibrationROI.id === state.selectedShapeId);
           if (state.whitePointROI) drawShape(state.whitePointROI, '#06b6d4', false, state.whitePointROI.id === state.selectedShapeId);
           if (state.blackPointROI) drawShape(state.blackPointROI, '#f97316', false, state.blackPointROI.id === state.selectedShapeId);
        }

        if (state.activeTab === 'analysis') {
          state.roiGroups.forEach(group => {
            group.shapes.forEach(shape => drawShape(shape, group.color, false, shape.id === state.selectedShapeId));
          });
        }
    }
  };

  const getShapeUnderCursor = (x: number, y: number): { shape: Shape, group?: ROIGroup, type?: 'gray'|'white'|'black' } | null => {
      // Check active tab content
      if (state.activeTab === 'segmentation') {
          for (const s of state.exclusionZones) {
              if (isPointInShape(x, y, s)) return { shape: s };
          }
      } else if (state.activeTab === 'calibration') {
          // Check in reverse order of drawing or importance
          if (state.blackPointROI && isPointInShape(x, y, state.blackPointROI)) return { shape: state.blackPointROI, type: 'black' };
          if (state.whitePointROI && isPointInShape(x, y, state.whitePointROI)) return { shape: state.whitePointROI, type: 'white' };
          if (state.calibrationROI && isPointInShape(x, y, state.calibrationROI)) return { shape: state.calibrationROI, type: 'gray' };
      } else if (state.activeTab === 'analysis') {
          for (const g of state.roiGroups) {
              for (const s of g.shapes) {
                  if (isPointInShape(x, y, s)) return { shape: s, group: g };
              }
          }
      }
      return null;
  };

  const getHandleUnderCursor = (x: number, y: number, shape: Shape): string | null => {
      const bbox = getBoundingBox(shape);
      const tol = HANDLE_SIZE;
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
          // Identify which one we are updating
          if (state.calibrationROI?.id === newShape.id) setState(s => ({...s, calibrationROI: newShape}));
          else if (state.whitePointROI?.id === newShape.id) setState(s => ({...s, whitePointROI: newShape}));
          else if (state.blackPointROI?.id === newShape.id) setState(s => ({...s, blackPointROI: newShape}));
      } else if (state.activeTab === 'analysis') {
          setState(s => ({
              ...s, 
              roiGroups: s.roiGroups.map(g => ({
                  ...g,
                  shapes: g.shapes.map(sh => sh.id === newShape.id ? newShape : sh)
              }))
          }));
      }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const startPoint = { x, y };

    if (state.activeTool === 'select') {
        // 1. Check handles of currently selected shape
        if (state.selectedShapeId) {
            let selectedShape: Shape | null = null;
            // Find the shape object
            if (state.activeTab === 'segmentation') selectedShape = state.exclusionZones.find(s => s.id === state.selectedShapeId) || null;
            else if (state.activeTab === 'calibration') {
                 if (state.calibrationROI?.id === state.selectedShapeId) selectedShape = state.calibrationROI;
                 else if (state.whitePointROI?.id === state.selectedShapeId) selectedShape = state.whitePointROI;
                 else if (state.blackPointROI?.id === state.selectedShapeId) selectedShape = state.blackPointROI;
            } else if (state.activeTab === 'analysis') {
                 state.roiGroups.forEach(g => {
                     const f = g.shapes.find(s => s.id === state.selectedShapeId);
                     if (f) selectedShape = f;
                 });
            }

            if (selectedShape) {
                const handle = getHandleUnderCursor(x, y, selectedShape);
                if (handle) {
                    setDragState({ mode: 'resize', startPoint, activeHandle: handle, initialPoints: JSON.parse(JSON.stringify(selectedShape.points)) });
                    return;
                }
            }
        }

        // 2. Check for new selection or move
        const hit = getShapeUnderCursor(x, y);
        if (hit) {
            setState(s => ({ 
              ...s, 
              selectedShapeId: hit.shape.id, 
              activeGroupId: hit.group?.id || s.activeGroupId,
              activeCalibrationTarget: hit.type || s.activeCalibrationTarget // Auto switch context on select
            }));
            setDragState({ mode: 'move', startPoint, activeHandle: null, initialPoints: JSON.parse(JSON.stringify(hit.shape.points)) });
        } else {
            setState(s => ({ ...s, selectedShapeId: null }));
        }

    } else {
        // Creation Mode
        const newShape: Shape = {
            id: Math.random().toString(36),
            type: state.activeTool,
            points: [startPoint, startPoint]
        };
        
        // Add immediately to state
        if (state.activeTab === 'segmentation') {
             setState(s => ({...s, exclusionZones: [...s.exclusionZones, newShape], selectedShapeId: newShape.id }));
        } else if (state.activeTab === 'calibration') {
             // Set specific calibration ROI
             const updates: Partial<AppState> = { selectedShapeId: newShape.id };
             if (state.activeCalibrationTarget === 'gray') updates.calibrationROI = newShape;
             else if (state.activeCalibrationTarget === 'white') updates.whitePointROI = newShape;
             else if (state.activeCalibrationTarget === 'black') updates.blackPointROI = newShape;
             setState(s => ({...s, ...updates }));
        } else if (state.activeTab === 'analysis') {
             if (state.activeGroupId) {
                 setState(s => ({
                     ...s,
                     roiGroups: s.roiGroups.map(g => g.id === s.activeGroupId ? {...g, shapes: [...g.shapes, newShape]} : g),
                     selectedShapeId: newShape.id
                 }));
             } else {
                 const newGroup: ROIGroup = {
                    id: Math.random().toString(36),
                    name: `Group ${state.roiGroups.length + 1}`,
                    color: COLORS[state.roiGroups.length % COLORS.length],
                    shapes: [newShape]
                 };
                 setState(s => ({ ...s, roiGroups: [...s.roiGroups, newGroup], activeGroupId: newGroup.id, selectedShapeId: newShape.id }));
             }
        }
        setDragState({ mode: 'create', startPoint, activeHandle: null, initialPoints: [] });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragState || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
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
        } else if (state.activeTab === 'analysis') {
             state.roiGroups.forEach(g => { if(!shape) shape = g.shapes.find(s => s.id === shapeId); });
        }

        if (shape) {
            let newPoints = [...shape.points];
            if (shape.type === 'lasso') {
                newPoints.push(currentPoint);
            } else {
                newPoints[1] = currentPoint;
            }
            updateShapeInState({ ...shape, points: newPoints });
        }

    } else if (dragState.mode === 'move') {
        const dx = x - dragState.startPoint.x;
        const dy = y - dragState.startPoint.y;
        
        let shapeId = state.selectedShapeId;
        let shape: Shape | undefined;
        if (state.activeTab === 'segmentation') shape = state.exclusionZones.find(s => s.id === shapeId);
        else if (state.activeTab === 'calibration') {
            if (state.calibrationROI?.id === shapeId) shape = state.calibrationROI;
            else if (state.whitePointROI?.id === shapeId) shape = state.whitePointROI;
            else if (state.blackPointROI?.id === shapeId) shape = state.blackPointROI;
        } else if (state.activeTab === 'analysis') state.roiGroups.forEach(g => { if(!shape) shape = g.shapes.find(s => s.id === shapeId); });

        if (shape) {
             const newPoints = dragState.initialPoints.map(p => ({ x: p.x + dx, y: p.y + dy }));
             updateShapeInState({ ...shape, points: newPoints });
        }

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
            let newMinX = initBBox.minX;
            let newMaxX = initBBox.maxX;
            let newMinY = initBBox.minY;
            let newMaxY = initBBox.maxY;

            const dx = x - dragState.startPoint.x;
            const dy = y - dragState.startPoint.y;

            if (dragState.activeHandle === 'nw') { newMinX += dx; newMinY += dy; }
            if (dragState.activeHandle === 'ne') { newMaxX += dx; newMinY += dy; }
            if (dragState.activeHandle === 'sw') { newMinX += dx; newMaxY += dy; }
            if (dragState.activeHandle === 'se') { newMaxX += dx; newMaxY += dy; }

            const scaleX = (newMaxX - newMinX) / (initBBox.maxX - initBBox.minX || 1);
            const scaleY = (newMaxY - newMinY) / (initBBox.maxY - initBBox.minY || 1);

            const newPoints = dragState.initialPoints.map(p => ({
                x: newMinX + (p.x - initBBox.minX) * scaleX,
                y: newMinY + (p.y - initBBox.minY) * scaleY
            }));
            updateShapeInState({ ...shape, points: newPoints });
        }
    }
  };

  const handleMouseUp = () => {
    if (dragState?.mode === 'create' || dragState?.mode === 'move' || dragState?.mode === 'resize') {
        if (state.activeTab === 'calibration') {
             // Recalculate based on which shape was being modified
             if (state.calibrationROI && state.selectedShapeId === state.calibrationROI.id) calculateCalibrationFromROI(state.calibrationROI, 'gray');
             if (state.whitePointROI && state.selectedShapeId === state.whitePointROI.id) calculateCalibrationFromROI(state.whitePointROI, 'white');
             if (state.blackPointROI && state.selectedShapeId === state.blackPointROI.id) calculateCalibrationFromROI(state.blackPointROI, 'black');
        }
    }
    setDragState(null);
    if (state.activeTool !== 'select') {
         setState(s => ({ ...s, activeTool: 'select' }));
    }
  };

  const calculateCalibrationFromROI = (shape: Shape, type: 'gray' | 'white' | 'black') => {
     if (!loadedImageRef.current || !canvasRef.current) return;
     const ctx = canvasRef.current.getContext('2d');
     if (!ctx) return;
     const bbox = getBoundingBox(shape);
     const imageData = ctx.getImageData(bbox.minX, bbox.minY, bbox.maxX - bbox.minX, bbox.maxY - bbox.minY);
     const data = imageData.data;
     let rSum = 0, gSum = 0, bSum = 0, count = 0;
     for (let i = 0; i < data.length; i += 4) {
        const localX = (i/4) % imageData.width;
        const localY = Math.floor((i/4) / imageData.width);
        const globalX = bbox.minX + localX;
        const globalY = bbox.minY + localY;
        if (isPointInShape(globalX, globalY, shape)) {
           rSum += data[i]; gSum += data[i+1]; bSum += data[i+2]; count++;
        }
     }
     if (count > 0) {
        const color = { r: rSum / count, g: gSum / count, b: bSum / count };
        if (type === 'gray') setState(s => ({ ...s, calibrationColor: color }));
        if (type === 'white') setState(s => ({ ...s, whitePointColor: color }));
        if (type === 'black') setState(s => ({ ...s, blackPointColor: color }));
     }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newImages: ImageAsset[] = [];
    
    // Process each file
    for (let i = 0; i < files.length; i++) {
       const file = files[i];
       
       if (file.name.endsWith('.zip')) {
          const zip = new JSZip();
          try {
             const contents = await zip.loadAsync(file);
             const imagePromises: Promise<void>[] = [];
             
             contents.forEach((relativePath, zipEntry) => {
                if (!zipEntry.dir && (relativePath.match(/\.(jpg|jpeg|png)$/i))) {
                   imagePromises.push(zipEntry.async('base64').then(b64 => {
                      newImages.push({
                         id: Math.random().toString(36),
                         name: relativePath,
                         url: `data:image/${relativePath.split('.').pop()};base64,${b64}`
                      });
                   }));
                }
             });
             await Promise.all(imagePromises);
          } catch (err) {
             console.error("Error reading zip", err);
          }
       } else if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          await new Promise<void>((resolve) => {
             reader.onload = (event) => {
                if (event.target?.result) {
                   newImages.push({
                      id: Math.random().toString(36),
                      name: file.name,
                      url: event.target.result as string
                   });
                }
                resolve();
             };
             reader.readAsDataURL(file);
          });
       }
    }

    if (newImages.length > 0) {
       setState(s => ({
          ...s,
          gallery: [...s.gallery, ...newImages],
          activeImageIndex: s.gallery.length // Set to first new image
       }));
    }
  };

  const handleDemoLoad = (url: string, label: string) => {
    const uniqueUrl = `${url}?t=${Date.now()}`;
    setState(s => ({
      ...s, 
      gallery: [{ id: 'demo-'+Date.now(), name: label, url: uniqueUrl }],
      activeImageIndex: 0,
      calibrationColor: null, 
      roiGroups: [], 
      exclusionZones: [], 
      reportSummary: '', 
      processedImageURL: null
    }));
  };

  const handleGenerateReport = async () => {
    if (canvasRef.current && overlayRef.current) {
       const reportCanvas = document.createElement('canvas');
       reportCanvas.width = canvasRef.current.width;
       reportCanvas.height = canvasRef.current.height;
       const rctx = reportCanvas.getContext('2d');
       if (rctx) {
          rctx.drawImage(canvasRef.current, 0, 0);
          rctx.drawImage(overlayRef.current, 0, 0);
          setState(s => ({ ...s, processedImageURL: reportCanvas.toDataURL('image/png') }));
       }
    }

    setState(s => ({ ...s, activeTab: 'report', isProcessing: true }));

    try {
      const ai = new GeminiClient(); 
      const summary = await ai.generateReportSummary(state.roiGroups, state.regressionParams);
      setState(s => ({ ...s, reportSummary: summary, isProcessing: false }));
    } catch (error) {
      console.error(error);
      setState(s => ({ ...s, isProcessing: false, reportSummary: "Error generating summary. Please check API Key." }));
    }
  };

  const handleAutoTune = () => {
    const target = state.regressionParams.targetIndex;
    const groups = state.roiGroups.filter(g => g.stats && g.stats.pixelCount > 0);
    
    // Strategy: Map observed index range to typical biological range.
    // Kim & van Iersel (2023) show Anthocyanin ranges approx 0-60 nmol/cm2 or µg/cm2.
    // mACI range approx 0.3 - 2.0.
    
    if (groups.length < 2) {
      // Fallback to Literature Defaults (Kim & van Iersel 2023 approximate)
      // Assuming mACI range 0.4-2.0 maps to 0-50 units
      const defaultSlope = target === 'mACI' ? 30.5 : 150.2; 
      const defaultIntercept = target === 'mACI' ? -10.5 : 5.2;
      setState(s => ({
        ...s,
        regressionParams: { ...s.regressionParams, slope: defaultSlope, intercept: defaultIntercept }
      }));
      return;
    }

    // Data-Driven Calculation
    // Find min and max index values among the groups
    let minIndex = Infinity;
    let maxIndex = -Infinity;

    groups.forEach(g => {
       const val = target === 'mACI' ? g.stats!.meanMACI : g.stats!.meanNGRDI;
       if (val < minIndex) minIndex = val;
       if (val > maxIndex) maxIndex = val;
    });

    if (maxIndex - minIndex < 0.05) {
       // Range too small to be reliable, use default
       const defaultSlope = target === 'mACI' ? 30.5 : 150.2;
       const defaultIntercept = target === 'mACI' ? -10.5 : 5.2;
       setState(s => ({
        ...s,
        regressionParams: { ...s.regressionParams, slope: defaultSlope, intercept: defaultIntercept }
      }));
      return;
    }

    // Assume Min Index = 1.0 µg/cm2 (Green)
    // Assume Max Index = 40.0 µg/cm2 (Red) (Typical for red lettuce)
    const targetMin = 1.0;
    const targetMax = 40.0;

    const slope = (targetMax - targetMin) / (maxIndex - minIndex);
    const intercept = targetMin - (slope * minIndex);

    setState(s => ({
      ...s,
      regressionParams: { ...s.regressionParams, slope: parseFloat(slope.toFixed(2)), intercept: parseFloat(intercept.toFixed(2)) }
    }));
  };

  // --- GitHub Export Logic ---
  
  const uploadToGithub = async () => {
     if (!githubConfig.token || !githubConfig.owner || !githubConfig.repo) {
        alert("Please fill in Token, Owner, and Repo.");
        return;
     }
     
     setGithubStatus('uploading');
     const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
     const basePath = `${githubConfig.path}/${timestamp}`;
     
     try {
         // Helper to upload file
         const uploadFile = async (filename: string, contentBase64: string, message: string) => {
            const url = `https://api.github.com/repos/${githubConfig.owner}/${githubConfig.repo}/contents/${basePath}/${filename}`;
            const res = await fetch(url, {
               method: 'PUT',
               headers: {
                  'Authorization': `token ${githubConfig.token}`,
                  'Content-Type': 'application/json',
               },
               body: JSON.stringify({
                  message: message,
                  content: contentBase64,
               })
            });
            if (!res.ok) throw new Error(await res.text());
         };

         // 1. Upload Report (Text)
         const reportContent = btoa(`
# BioPheno Analysis Report
Date: ${new Date().toLocaleDateString()}
Image: ${state.gallery[state.activeImageIndex]?.name}

## Methodology
ExG Threshold: ${state.segmentationThreshold}
Calibration: ${JSON.stringify(state.calibrationColor)}

## Results
${state.reportSummary}

## Raw Data
${JSON.stringify(state.roiGroups.map(g => ({ name: g.name, stats: g.stats })), null, 2)}
         `);
         await uploadFile('report.md', reportContent, 'Add Analysis Report');

         // 2. Upload Processed Image
         if (state.processedImageURL) {
            const base64Data = state.processedImageURL.split(',')[1];
            await uploadFile('analyzed_image.png', base64Data, 'Add Analyzed Image');
         }

         setGithubStatus('success');
         setTimeout(() => {
            setGithubStatus('idle');
            setState(s => ({...s, isGithubModalOpen: false}));
         }, 2000);

     } catch (e) {
         console.error(e);
         setGithubStatus('error');
     }
  };


  // --- Components Helpers ---

  const BarChartSection = ({ title, dataKey, formatFn }: any) => (
    <div className="mb-6">
       <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-2">{title}</p>
       <div className="space-y-2">
         {state.roiGroups.map(g => {
           const val = (g.stats as any)?.[dataKey] || 0;
           const maxVal = Math.max(...state.roiGroups.map(gr => (gr.stats as any)?.[dataKey] || 0), 0.1); 
           const pct = Math.max(0, Math.min(100, (val / maxVal) * 100));
           return (
             <div key={g.id} className="flex items-center gap-2 text-xs">
               <span className="w-16 truncate text-slate-500">{g.name}</span>
               <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden relative">
                 <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: g.color }}></div>
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
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-slate-900 flex flex-col print:hidden">
        <div className="p-6 border-b border-slate-800">
          <h1 className="flex items-center gap-2 font-bold text-lg text-emerald-400">
            <Leaf className="w-6 h-6" />
            BioPheno
          </h1>
          <p className="text-xs text-slate-500 mt-1">Kim & van Iersel (2023)</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          <NavButton 
            active={state.activeTab === 'segmentation'} 
            onClick={() => setState(s => ({ ...s, activeTab: 'segmentation', activeTool: 'select', selectedShapeId: null }))}
            icon={<Maximize2 size={18} />}
            label="Segmentation"
          />
          <NavButton 
            active={state.activeTab === 'calibration'} 
            onClick={() => setState(s => ({ ...s, activeTab: 'calibration', activeTool: 'select', selectedShapeId: null }))}
            icon={<Pipette size={18} />}
            label="Color Calibration"
          />
          <NavButton 
            active={state.activeTab === 'analysis'} 
            onClick={() => setState(s => ({ ...s, activeTab: 'analysis', activeTool: 'select', selectedShapeId: null }))}
            icon={<BarChart3 size={18} />}
            label="Analysis & Results"
          />
          <div className="pt-4 mt-4 border-t border-slate-800 space-y-2">
             <button 
                onClick={handleGenerateReport}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${state.activeTab === 'report' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}
              >
                <FileText size={18} />
                Generate Report
              </button>
             <button 
                onClick={() => setState(s => ({...s, isGithubModalOpen: true}))}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-slate-400 hover:bg-slate-800 hover:text-slate-200`}
              >
                <Github size={18} />
                Export to GitHub
              </button>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {state.activeTab !== 'report' && (
          <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur z-20">
            <div className="flex items-center gap-4">
              <h2 className="font-medium text-slate-200 hidden md:block">
                {state.activeTab === 'segmentation' && 'Vegetation Segmentation'}
                {state.activeTab === 'calibration' && 'Color Reference'}
                {state.activeTab === 'analysis' && 'Analysis & Results'}
              </h2>

              {/* Tools */}
              {(state.activeTab === 'segmentation' || state.activeTab === 'analysis') && (
                <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-4">
                  <button onClick={() => setState(s => ({...s, activeTool: 'select'}))} className={`p-2 rounded ${state.activeTool === 'select' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><MousePointer2 size={16} /></button>
                  <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Square size={16} /></button>
                  <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><CircleIcon size={16} /></button>
                  <button onClick={() => setState(s => ({...s, activeTool: 'lasso'}))} className={`p-2 rounded ${state.activeTool === 'lasso' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Lasso size={16} /></button>
                </div>
              )}
              
              {state.activeTab === 'analysis' && (
                 <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-4">
                    <button onClick={() => setState(s => ({...s, visualizationMode: 'rgb'}))} className={`px-3 py-1 text-xs rounded transition-colors ${state.visualizationMode === 'rgb' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}>RGB</button>
                    <button onClick={() => setState(s => ({...s, visualizationMode: 'ngrdi'}))} className={`px-3 py-1 text-xs rounded transition-colors ${state.visualizationMode === 'ngrdi' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'}`}>NGRDI</button>
                    <button onClick={() => setState(s => ({...s, visualizationMode: 'maci'}))} className={`px-3 py-1 text-xs rounded transition-colors ${state.visualizationMode === 'maci' ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-white'}`}>mACI</button>
                 </div>
              )}

              {state.activeTab === 'calibration' && (
                <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-4">
                   <div className="flex border-r border-slate-700 pr-2 mr-2 gap-1">
                      <button 
                        onClick={() => setState(s => ({...s, activeCalibrationTarget: 'gray'}))}
                        className={`p-1.5 rounded text-xs flex items-center gap-1 transition-colors ${state.activeCalibrationTarget === 'gray' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-300'}`}
                      ><Disc size={14}/> Gray</button>
                      <button 
                        onClick={() => setState(s => ({...s, activeCalibrationTarget: 'white'}))}
                        className={`p-1.5 rounded text-xs flex items-center gap-1 transition-colors ${state.activeCalibrationTarget === 'white' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-300'}`}
                      ><Sun size={14}/> White</button>
                      <button 
                        onClick={() => setState(s => ({...s, activeCalibrationTarget: 'black'}))}
                        className={`p-1.5 rounded text-xs flex items-center gap-1 transition-colors ${state.activeCalibrationTarget === 'black' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-300'}`}
                      ><Moon size={14}/> Black</button>
                   </div>
                   <button onClick={() => setState(s => ({...s, activeTool: 'select'}))} className={`p-2 rounded ${state.activeTool === 'select' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><MousePointer2 size={16} /></button>
                   <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`} title="Rectangle Region"><Square size={16} /></button>
                   <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`} title="Circle Region"><CircleIcon size={16} /></button>
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-3">
              {/* Image Navigation */}
              {state.gallery.length > 0 && (
                 <div className="flex items-center bg-slate-800 rounded border border-slate-700 mr-2">
                    <button 
                      disabled={state.activeImageIndex === 0}
                      onClick={() => setState(s => ({...s, activeImageIndex: Math.max(0, s.activeImageIndex - 1)}))}
                      className="p-1 hover:bg-slate-700 disabled:opacity-30"
                    ><ChevronLeft size={16}/></button>
                    <span className="text-xs px-2 w-20 truncate text-center text-slate-400">
                      {state.activeImageIndex + 1} / {state.gallery.length}
                    </span>
                    <button 
                      disabled={state.activeImageIndex === state.gallery.length - 1}
                      onClick={() => setState(s => ({...s, activeImageIndex: Math.min(s.gallery.length - 1, s.activeImageIndex + 1)}))}
                      className="p-1 hover:bg-slate-700 disabled:opacity-30"
                    ><ChevronRight size={16}/></button>
                 </div>
              )}

              <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-md text-sm font-medium"><Upload size={16} /> Upload</button>
              <input ref={fileInputRef} type="file" multiple accept=".jpg,.jpeg,.png,.zip" className="hidden" onChange={handleFileUpload} />
            </div>
          </header>
        )}

        {/* --- Github Modal --- */}
        {state.isGithubModalOpen && (
           <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
              <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-96 shadow-2xl">
                 <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><Github size={20}/> Export to GitHub</h3>
                 <div className="space-y-4">
                    <div>
                       <label className="text-xs text-slate-400 block mb-1">Personal Access Token</label>
                       <input type="password" value={githubConfig.token} onChange={e => setGithubConfig(c => ({...c, token: e.target.value}))} className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-sm text-white focus:border-indigo-500 outline-none" placeholder="ghp_..." />
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                       <div>
                          <label className="text-xs text-slate-400 block mb-1">Owner</label>
                          <input type="text" value={githubConfig.owner} onChange={e => setGithubConfig(c => ({...c, owner: e.target.value}))} className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-sm text-white" placeholder="username" />
                       </div>
                       <div>
                          <label className="text-xs text-slate-400 block mb-1">Repo Name</label>
                          <input type="text" value={githubConfig.repo} onChange={e => setGithubConfig(c => ({...c, repo: e.target.value}))} className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-sm text-white" placeholder="my-repo" />
                       </div>
                    </div>
                    <div>
                       <label className="text-xs text-slate-400 block mb-1">Folder Path</label>
                       <input type="text" value={githubConfig.path} onChange={e => setGithubConfig(c => ({...c, path: e.target.value}))} className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-sm text-white" placeholder="results/experiment-1" />
                    </div>
                    
                    {githubStatus === 'error' && <p className="text-xs text-rose-400 flex items-center gap-1"><AlertCircle size={12}/> Upload failed. Check permissions.</p>}
                    {githubStatus === 'success' && <p className="text-xs text-emerald-400 flex items-center gap-1"><CheckCircle size={12}/> Successfully uploaded!</p>}
                    
                    <div className="flex gap-2 pt-2">
                       <button onClick={() => setState(s => ({...s, isGithubModalOpen: false}))} className="flex-1 py-2 bg-slate-800 text-slate-300 rounded text-sm hover:bg-slate-700">Cancel</button>
                       <button onClick={uploadToGithub} disabled={githubStatus === 'uploading'} className="flex-1 py-2 bg-emerald-600 text-white rounded text-sm hover:bg-emerald-500 disabled:opacity-50 flex items-center justify-center gap-2">
                          {githubStatus === 'uploading' ? <RefreshCw className="animate-spin" size={14}/> : <Save size={14}/>} Upload
                       </button>
                    </div>
                 </div>
              </div>
           </div>
        )}

        {/* --- View: Report --- */}
        {state.activeTab === 'report' && (
          <div className="flex-1 overflow-auto bg-slate-200 p-8 text-black">
             <div className="max-w-4xl mx-auto bg-white shadow-2xl min-h-[29.7cm] p-12">
                {/* Header */}
                <div className="border-b-2 border-black pb-6 mb-8 flex justify-between items-end">
                   <div>
                      <h1 className="text-3xl font-bold font-serif mb-2">Phenotypic Analysis Report</h1>
                      <p className="text-sm text-gray-600 font-serif">Generated via BioPheno | {new Date().toLocaleDateString()}</p>
                   </div>
                   <button onClick={() => window.print()} className="print:hidden flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded hover:bg-slate-700 text-sm">
                      <Printer size={16}/> Print PDF
                   </button>
                </div>

                {/* 1. Methodology */}
                <section className="mb-8">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">1. Methodology</h2>
                   <p className="text-sm leading-relaxed text-justify mb-4">
                      <strong>Software & Algorithms:</strong> Image analysis was performed using the BioPheno software suite. 
                      Vegetation segmentation utilized the Excess Green Index (ExG = 2G - R - B) with a calculated threshold of 
                      <span className="font-mono bg-gray-100 px-1 mx-1 rounded">{state.segmentationThreshold}</span>. 
                      Excluded regions were manually defined to remove background artifacts.
                   </p>
                   <p className="text-sm leading-relaxed text-justify mb-4">
                      <strong>Quantification Models:</strong> Biological indices were calculated for defined Regions of Interest (ROI).
                      Normalized Green-Red Difference Index (NGRDI) and modified Anthocyanin Content Index (mACI) were computed per pixel.
                      Total anthocyanin content was estimated using a linear regression model 
                      (<em>y = {state.regressionParams.slope}x + {state.regressionParams.intercept}</em>) 
                      based on the {state.regressionParams.targetIndex} index, following protocols by Kim & van Iersel (2023).
                   </p>
                   {state.calibrationColor && (
                      <p className="text-sm leading-relaxed text-justify">
                         <strong>Color Calibration:</strong> RGB values were normalized against a neutral gray reference 
                         (R:{state.calibrationColor.r.toFixed(0)} G:{state.calibrationColor.g.toFixed(0)} B:{state.calibrationColor.b.toFixed(0)}) 
                         to ensure consistent lighting interpretation.
                      </p>
                   )}
                </section>

                {/* 2. Visual Analysis (Montage) */}
                <section className="mb-8 break-inside-avoid">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">2. Visual Analysis</h2>
                   <div className="grid grid-cols-2 gap-4 mb-2">
                      <div className="border border-gray-200 p-1">
                         <img src={currentImageUrl || ''} className="w-full h-auto object-contain" alt="Original" />
                         <p className="text-center text-xs font-serif mt-2 italic">Figure 1a. Original Specimen</p>
                      </div>
                      <div className="border border-gray-200 p-1">
                         {state.processedImageURL && <img src={state.processedImageURL} className="w-full h-auto object-contain" alt="Processed" />}
                         <p className="text-center text-xs font-serif mt-2 italic">Figure 1b. Segmented ROI Overlay</p>
                      </div>
                   </div>
                </section>

                {/* 3. Quantitative Results */}
                <section className="mb-8">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">3. Quantitative Results</h2>
                   <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse border border-gray-300">
                         <thead className="bg-gray-100 font-serif">
                            <tr>
                               <th className="border border-gray-300 px-3 py-2 text-left">Group ID</th>
                               <th className="border border-gray-300 px-3 py-2 text-right">Pixel Count</th>
                               <th className="border border-gray-300 px-3 py-2 text-right">mACI (Mean)</th>
                               <th className="border border-gray-300 px-3 py-2 text-right">NGRDI (Mean)</th>
                               <th className="border border-gray-300 px-3 py-2 text-right bg-slate-50">Est. Anthocyanin (µg/cm²)</th>
                            </tr>
                         </thead>
                         <tbody>
                            {state.roiGroups.map(g => (
                               <tr key={g.id}>
                                  <td className="border border-gray-300 px-3 py-2 font-medium">{g.name}</td>
                                  <td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.pixelCount}</td>
                                  <td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.meanMACI.toFixed(3)}</td>
                                  <td className="border border-gray-300 px-3 py-2 text-right font-mono">{g.stats?.meanNGRDI.toFixed(3)}</td>
                                  <td className="border border-gray-300 px-3 py-2 text-right font-mono font-bold bg-slate-50">{g.stats?.anthocyanin.toFixed(2)}</td>
                               </tr>
                            ))}
                         </tbody>
                      </table>
                   </div>
                </section>

                {/* 4. Results & Discussion (AI) */}
                <section className="mb-8">
                   <h2 className="text-xl font-bold font-serif mb-4 border-b border-gray-300 pb-1">4. Results & Discussion</h2>
                   {state.isProcessing ? (
                      <div className="flex items-center gap-2 text-slate-500 italic p-8 justify-center bg-gray-50 border border-dashed border-gray-300 rounded">
                         <RefreshCw className="animate-spin" size={18} /> Generating scientific summary...
                      </div>
                   ) : (
                      <div className="text-sm leading-loose text-justify font-serif">
                         {state.reportSummary.split('\n').map((para, i) => (
                            <p key={i} className="mb-4">{para}</p>
                         ))}
                      </div>
                   )}
                </section>

             </div>
          </div>
        )}

        {/* --- View: Main Canvas (Hidden when Report Active) --- */}
        <div className={`flex-1 overflow-auto p-6 bg-slate-950 z-0 ${state.activeTab === 'report' ? 'hidden' : 'block'}`}>
          <div className="h-full flex gap-6 min-h-0">
              
              {/* Left: Canvas Area */}
              <div className="flex-1 flex flex-col gap-6">
                {!currentImageUrl ? (
                   <div className="flex-1 border-2 border-dashed border-slate-800 rounded-xl flex items-center justify-center text-slate-500">
                      <div className="text-center">
                        <Leaf className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Load an image to begin</p>
                      </div>
                   </div>
                ) : (
                  <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 relative flex items-center justify-center overflow-hidden cursor-crosshair">
                    <div className="relative max-w-full max-h-full">
                      <canvas ref={canvasRef} className="block max-w-full max-h-full object-contain" />
                      <canvas 
                        ref={overlayRef} 
                        className={`absolute inset-0 w-full h-full pointer-events-auto ${state.activeTool === 'select' ? 'cursor-default' : 'cursor-crosshair'}`}
                        style={{ cursor: dragState?.activeHandle ? (dragState.activeHandle === 'nw' || dragState.activeHandle === 'se' ? 'nwse-resize' : 'nesw-resize') : (state.activeTool === 'select' ? (state.selectedShapeId && getShapeUnderCursor(0,0)?.shape.id === state.selectedShapeId ? 'move' : 'default') : 'crosshair') }}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                      />
                    </div>
                  </div>
                )}
                
                {/* Hints */}
                <div className="text-xs text-slate-500 flex gap-4">
                  {state.activeTab === 'segmentation' && <span>Use tools to erase background artifacts. Select to move/resize.</span>}
                  {state.activeTab === 'calibration' && <span>Select a target type (Gray/White/Black) and draw a region.</span>}
                  {state.activeTab === 'analysis' && <span>Use Lasso or Rect tool to define plant groups. Select to edit.</span>}
                </div>
              </div>

              {/* Right: Controls */}
              <div className="w-96 flex flex-col gap-6 overflow-y-auto pr-2">
                
                {state.activeTab === 'segmentation' && (
                  <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                    <h3 className="font-medium text-sm text-slate-200 mb-4 flex items-center gap-2"><Maximize2 size={16}/> Segmentation</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>ExG Threshold</span><span>{state.segmentationThreshold}</span>
                      </div>
                      <input 
                        type="range" min="0" max="100" 
                        value={state.segmentationThreshold}
                        onChange={(e) => setState(s => ({ ...s, segmentationThreshold: parseInt(e.target.value) }))}
                        className="w-full accent-emerald-500 h-1 bg-slate-700 rounded-lg appearance-none"
                      />
                      
                      {state.exclusionZones.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-800">
                           <div className="flex justify-between items-center mb-2">
                             <span className="text-xs text-rose-400 font-medium">Excluded Regions ({state.exclusionZones.length})</span>
                             <button 
                               onClick={() => setState(s => ({...s, exclusionZones: [], selectedShapeId: null}))}
                               className="text-[10px] text-slate-500 hover:text-rose-400"
                             >Clear All</button>
                           </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {state.activeTab === 'calibration' && (
                  <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                    <h3 className="font-medium text-sm text-slate-200 mb-4 flex items-center gap-2"><Pipette size={16}/> Calibration</h3>
                    <div className="space-y-4">
                       {/* Gray Card */}
                       <div>
                          <p className="text-xs text-slate-500 mb-1 flex items-center gap-2"><Disc size={12}/> Gray Reference (Midtones)</p>
                          <div className="p-3 bg-slate-950 rounded border border-slate-800 grid grid-cols-3 gap-2 text-center cursor-pointer hover:border-slate-600 transition-colors"
                               onClick={() => setState(s => ({...s, activeCalibrationTarget: 'gray'}))}
                               style={{borderColor: state.activeCalibrationTarget === 'gray' ? '#fff' : undefined}}
                          >
                             <div><div className="text-[10px] text-slate-500">R</div><div className="text-xs font-mono">{state.calibrationColor?.r.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">G</div><div className="text-xs font-mono">{state.calibrationColor?.g.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">B</div><div className="text-xs font-mono">{state.calibrationColor?.b.toFixed(0) || '-'}</div></div>
                          </div>
                       </div>

                       {/* White Point */}
                       <div>
                          <p className="text-xs text-slate-500 mb-1 flex items-center gap-2"><Sun size={12}/> White Point (Highlights)</p>
                          <div className="p-3 bg-slate-950 rounded border border-slate-800 grid grid-cols-3 gap-2 text-center cursor-pointer hover:border-cyan-500 transition-colors"
                               onClick={() => setState(s => ({...s, activeCalibrationTarget: 'white'}))}
                               style={{borderColor: state.activeCalibrationTarget === 'white' ? '#06b6d4' : undefined}}
                          >
                             <div><div className="text-[10px] text-slate-500">R</div><div className="text-xs font-mono">{state.whitePointColor?.r.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">G</div><div className="text-xs font-mono">{state.whitePointColor?.g.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">B</div><div className="text-xs font-mono">{state.whitePointColor?.b.toFixed(0) || '-'}</div></div>
                          </div>
                       </div>

                       {/* Black Point */}
                       <div>
                          <p className="text-xs text-slate-500 mb-1 flex items-center gap-2"><Moon size={12}/> Black Point (Shadows)</p>
                          <div className="p-3 bg-slate-950 rounded border border-slate-800 grid grid-cols-3 gap-2 text-center cursor-pointer hover:border-orange-500 transition-colors"
                               onClick={() => setState(s => ({...s, activeCalibrationTarget: 'black'}))}
                               style={{borderColor: state.activeCalibrationTarget === 'black' ? '#f97316' : undefined}}
                          >
                             <div><div className="text-[10px] text-slate-500">R</div><div className="text-xs font-mono">{state.blackPointColor?.r.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">G</div><div className="text-xs font-mono">{state.blackPointColor?.g.toFixed(0) || '-'}</div></div>
                             <div><div className="text-[10px] text-slate-500">B</div><div className="text-xs font-mono">{state.blackPointColor?.b.toFixed(0) || '-'}</div></div>
                          </div>
                       </div>
                    </div>
                    {!state.calibrationColor && !state.whitePointColor && !state.blackPointColor && <p className="text-xs text-slate-500 mt-4 italic">Tip: Setting both White and Black points enables dynamic range correction.</p>}
                  </div>
                )}

                {state.activeTab === 'analysis' && (
                  <div className="flex flex-col gap-4">
                    {/* 1. Groups Management */}
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-medium text-sm text-slate-200 flex items-center gap-2"><FlaskConical size={16}/> Groups</h3>
                        <button 
                          onClick={() => {
                             const newGroup: ROIGroup = {
                                id: Math.random().toString(36),
                                name: `Group ${state.roiGroups.length + 1}`,
                                color: COLORS[state.roiGroups.length % COLORS.length],
                                shapes: []
                             };
                             setState(s => ({ ...s, roiGroups: [...s.roiGroups, newGroup], activeGroupId: newGroup.id }));
                          }}
                          className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                        ><Plus size={14}/></button>
                      </div>

                      <div className="space-y-2 max-h-48 overflow-y-auto">
                         {state.roiGroups.length === 0 && <p className="text-xs text-slate-500 italic">No groups defined. Draw shapes to create groups.</p>}
                         {state.roiGroups.map(group => (
                           <div 
                             key={group.id} 
                             className={`p-2 rounded border flex items-center gap-2 cursor-pointer ${state.activeGroupId === group.id ? 'bg-slate-800 border-indigo-500/50' : 'bg-slate-950 border-slate-800'}`}
                             onClick={() => setState(s => ({...s, activeGroupId: group.id}))}
                           >
                             <div className="w-3 h-3 rounded-full" style={{backgroundColor: group.color}}></div>
                             <input 
                               value={group.name}
                               onChange={(e) => setState(s => ({
                                 ...s,
                                 roiGroups: s.roiGroups.map(g => g.id === group.id ? {...g, name: e.target.value} : g)
                               }))}
                               className="bg-transparent text-xs text-slate-200 focus:outline-none w-full"
                             />
                             <button 
                               onClick={(e) => {
                                 e.stopPropagation();
                                 setState(s => ({...s, roiGroups: s.roiGroups.filter(g => g.id !== group.id)}));
                               }}
                               className="text-slate-600 hover:text-rose-400"
                             ><Trash2 size={12}/></button>
                           </div>
                         ))}
                      </div>
                    </div>
                    
                    {/* 2. Charts */}
                    {state.roiGroups.length > 0 && (
                      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
                         <h3 className="font-medium text-sm text-slate-200 mb-4">Comparative Analysis</h3>
                         <BarChartSection title="Est. Anthocyanin (µg/cm²)" dataKey="anthocyanin" formatFn={(v: number) => v.toFixed(2)} />
                         <div className="grid grid-cols-1 gap-6 pt-4 border-t border-slate-800">
                            <BarChartSection title="mACI Index (Red/Green)" dataKey="meanMACI" formatFn={(v: number) => v.toFixed(3)} />
                            <BarChartSection title="NGRDI Index (Norm Diff)" dataKey="meanNGRDI" formatFn={(v: number) => v.toFixed(3)} />
                         </div>
                      </div>
                    )}
                    
                    {/* Regression Config */}
                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
                        <div className="flex justify-between items-center mb-2">
                           <p className="text-[10px] text-slate-400 font-medium">Model Params (y = mx + c)</p>
                           <button onClick={handleAutoTune} className="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
                              <Wand2 size={10} /> Auto-Tune
                           </button>
                        </div>
                        <div className="grid grid-cols-2 gap-2 mb-2">
                           <input 
                              type="number" step="0.1" placeholder="Slope"
                              value={state.regressionParams.slope}
                              onChange={(e) => setState(s => ({...s, regressionParams: {...s.regressionParams, slope: parseFloat(e.target.value)}}))}
                              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs"
                           />
                           <input 
                              type="number" step="0.1" placeholder="Intercept"
                              value={state.regressionParams.intercept}
                              onChange={(e) => setState(s => ({...s, regressionParams: {...s.regressionParams, intercept: parseFloat(e.target.value)}}))}
                              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs"
                           />
                        </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
};

// --- Utils ---

const NavButton = ({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) => (
  <button onClick={onClick} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${active ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}>
    {icon} {label}
  </button>
);

class GeminiClient {
  private ai: GoogleGenAI;
  constructor() { this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY }); }
  
  async generateReportSummary(groups: ROIGroup[], regression: any) {
    const model = this.ai.models;
    const dataSummary = groups.map(g => 
       `Group: ${g.name}, mACI: ${g.stats?.meanMACI.toFixed(3)}, NGRDI: ${g.stats?.meanNGRDI.toFixed(3)}, Anthocyanin: ${g.stats?.anthocyanin.toFixed(2)}`
    ).join('; ');

    const prompt = `
      You are a Bioinformatics Scientist writing the 'Results and Discussion' section of a research paper.
      
      Study Context: 
      - Lettuce phenotyping using RGB imagery.
      - Segmentation via Excess Green Index (ExG).
      - Quantified mACI (Modified Anthocyanin Content Index) and NGRDI (Normalized Green-Red Difference Index).
      - Estimated Anthocyanin using linear model: y = ${regression.slope}x + ${regression.intercept} (Target: ${regression.targetIndex}).

      Data: ${dataSummary}

      Task:
      Write a formal, scientific paragraph (approx 150-200 words) summarizing these results. 
      Compare the groups if multiple exist. Discuss biological implications of the indices (e.g., higher NGRDI typically indicates more green biomass/vigor, higher mACI relates to stress/anthocyanin).
      Do not use markdown formatting like bold/italics, just plain text paragraphs.
    `;
    
    try {
      const response = await model.generateContent({ model: 'gemini-3-flash-preview', contents: prompt });
      return response.text;
    } catch (e) {
      console.error(e);
      return "Error generating summary.";
    }
  }
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
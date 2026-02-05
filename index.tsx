import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  Upload, 
  Leaf, 
  Pipette, 
  FileText, // Changed from Code
  Activity, 
  RefreshCw,
  Maximize2,
  Globe,
  ChevronDown,
  BarChart3,
  FlaskConical,
  Square,
  Circle as CircleIcon,
  Lasso,
  Eraser,
  Plus,
  Trash2,
  Edit2,
  Printer
} from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

// --- Types ---

type ToolType = 'none' | 'rect' | 'circle' | 'lasso';

interface Point { x: number; y: number; }

interface Shape {
  id: string;
  type: 'rect' | 'circle' | 'lasso';
  points: Point[]; // For rect: [start, end], Circle: [center, edge], Lasso: [p1, p2, ...]
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

interface AppState {
  originalImage: string | null;
  segmentationThreshold: number;
  activeTab: 'segmentation' | 'calibration' | 'analysis' | 'report'; // Changed 'code' to 'report'
  isProcessing: boolean;
  
  // Report Data
  reportSummary: string;
  processedImageURL: string | null;
  
  // Calibration
  calibrationColor: { r: number, g: number, b: number } | null;
  calibrationROI: Shape | null;

  // Segmentation Editing
  exclusionZones: Shape[];

  // Quantification
  roiGroups: ROIGroup[];
  activeGroupId: string | null;

  // UI State
  isDemoMenuOpen: boolean;
  activeTool: ToolType;
  
  // Regression
  regressionParams: {
    slope: number;
    intercept: number;
    targetIndex: 'mACI' | 'NGRDI';
  };
}

// --- Constants ---

const DEMO_IMAGES = [
  { 
    label: 'Early Growth (Day 4)', 
    url: 'https://raw.githubusercontent.com/ISU-Research/Hydra1-Orbital-Greenhouse/master/Raw%20images/2018-05-27%2010-00-01.jpg' 
  },
  { 
    label: 'Mid Growth (Day 18)', 
    url: 'https://raw.githubusercontent.com/ISU-Research/Hydra1-Orbital-Greenhouse/master/Raw%20images/2018-06-10%2010-00-01.jpg' 
  },
  { 
    label: 'Late Growth (Day 29)', 
    url: 'https://raw.githubusercontent.com/ISU-Research/Hydra1-Orbital-Greenhouse/master/Raw%20images/2018-06-21%2010-00-01.jpg' 
  }
];

const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

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
  return { minX: Math.floor(minX), maxX: Math.ceil(maxX), minY: Math.floor(minY), maxY: Math.ceil(maxY) };
};

// --- Main Application ---

const App = () => {
  const [state, setState] = useState<AppState>({
    originalImage: null,
    segmentationThreshold: 20,
    activeTab: 'segmentation',
    isProcessing: false,
    reportSummary: '',
    processedImageURL: null,
    calibrationColor: null,
    calibrationROI: null,
    exclusionZones: [],
    roiGroups: [],
    activeGroupId: null,
    isDemoMenuOpen: false,
    activeTool: 'none',
    regressionParams: {
      slope: 1.5,
      intercept: 0.2,
      targetIndex: 'mACI'
    }
  });

  const [drawingShape, setDrawingShape] = useState<Shape | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);

  // --- Image Loading ---

  useEffect(() => {
    if (!state.originalImage) return;
    const img = new Image();
    if (!state.originalImage.startsWith('data:')) img.crossOrigin = "Anonymous";
    img.src = state.originalImage;
    img.onload = () => {
      loadedImageRef.current = img;
      // Reset ROIs on new image
      setState(s => ({ 
        ...s, 
        exclusionZones: [], 
        roiGroups: [], 
        activeGroupId: null,
        calibrationColor: null,
        calibrationROI: null,
        reportSummary: '',
        processedImageURL: null
      }));
    };
  }, [state.originalImage]);

  // --- Pipeline & Rendering ---

  // Trigger pipeline when dependencies change
  useEffect(() => {
    // Only run visual pipeline if we are in a visual tab
    if (state.activeTab !== 'report') {
      runPipeline();
    }
  }, [
    state.segmentationThreshold, 
    state.activeTab, 
    state.calibrationColor, 
    state.exclusionZones, 
    state.roiGroups,
    loadedImageRef.current
  ]);

  const runPipeline = () => {
    if (!loadedImageRef.current || !canvasRef.current) return;

    const img = loadedImageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Set dimensions
    if (canvas.width !== img.width) canvas.width = img.width;
    if (canvas.height !== img.height) canvas.height = img.height;
    
    // Sync Overlay
    if (overlayRef.current) {
      overlayRef.current.width = canvas.width;
      overlayRef.current.height = canvas.height;
      // We draw overlay separately
    }

    ctx.drawImage(img, 0, 0);

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const width = canvas.width;
      const height = canvas.height;

      // Optimization: Pre-calculate bounding boxes for shapes
      const optimizedGroups = state.roiGroups.map(g => ({
        ...g,
        optimizedShapes: g.shapes.map(s => ({ shape: s, bbox: getBoundingBox(s) }))
      }));

      const optimizedExclusion = state.exclusionZones.map(s => ({ shape: s, bbox: getBoundingBox(s) }));

      // Prepare Calibration
      let rScale = 1, gScale = 1, bScale = 1;
      if (state.calibrationColor) {
        rScale = 128 / (state.calibrationColor.r || 1);
        gScale = 128 / (state.calibrationColor.g || 1);
        bScale = 128 / (state.calibrationColor.b || 1);
      }

      // Reset Stats for Groups
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

        // 1. Calibration
        if (state.calibrationColor) {
          r = Math.min(255, r * rScale);
          g = Math.min(255, g * gScale);
          b = Math.min(255, b * bScale);
        }

        // 2. Segmentation (ExG)
        const exg = (2 * g) - r - b;
        let isPlant = exg > state.segmentationThreshold;

        // 3. Exclusion Zones (Erase)
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

        // 4. Quantification / Visualization
        // Always calculate stats if isPlant, but visual depends on tab
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

        // Rendering Logic
        if (state.activeTab === 'analysis' || state.activeTab === 'report') {
             // For Analysis/Report, we show the "Result" view
             if (isPlant) {
                if (inGroup || state.roiGroups.length === 0) {
                   // Keep original if in group or no groups defined
                   data[i] = r; data[i+1] = g; data[i+2] = b;
                } else {
                   // Plant but not in ROI
                   const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                   data[i] = data[i+1] = data[i+2] = gray * 0.5;
                }
             } else {
                // Background
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

      // Finalize Stats
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

      // Avoid infinite loop by comparing stringified stats
      const currentStatsStr = JSON.stringify(state.roiGroups.map(g => g.stats));
      const newStatsStr = JSON.stringify(finalGroups.map(g => g.stats));
      
      if (currentStatsStr !== newStatsStr) {
         setTimeout(() => {
            setState(s => ({ ...s, roiGroups: finalGroups }));
         }, 0);
      }

      ctx.putImageData(imageData, 0, 0);
      
      // Update Overlay after canvas draw
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

    const drawShape = (shape: Shape, color: string, fill: boolean = false) => {
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
    };

    // Draw Based on Tab
    if (state.activeTab !== 'report') {
        state.exclusionZones.forEach(shape => drawShape(shape, '#ef4444', true));
        if (state.calibrationROI && state.activeTab === 'calibration') {
          drawShape(state.calibrationROI, '#ffffff', false);
        }
        if (state.activeTab === 'analysis') {
          state.roiGroups.forEach(group => {
            group.shapes.forEach(shape => drawShape(shape, group.color, false));
          });
        }
        if (drawingShape) {
          drawShape(drawingShape, '#fbbf24', false);
        }
    }
  };

  // --- Interaction Handlers ---

  const handleMouseDown = (e: React.MouseEvent) => {
    if (state.activeTool === 'none') return;
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const startPoint = { x, y };

    setDrawingShape({
      id: Math.random().toString(36),
      type: state.activeTool,
      points: [startPoint, startPoint] // Init with start point
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawingShape) return;
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const currPoint = { x, y };

    if (drawingShape.type === 'lasso') {
      setDrawingShape({
        ...drawingShape,
        points: [...drawingShape.points, currPoint]
      });
    } else {
      setDrawingShape({
        ...drawingShape,
        points: [drawingShape.points[0], currPoint]
      });
    }
  };

  const handleMouseUp = () => {
    if (!drawingShape) return;

    if (state.activeTab === 'segmentation') {
      setState(s => ({ ...s, exclusionZones: [...s.exclusionZones, drawingShape] }));
    } else if (state.activeTab === 'calibration') {
      calculateCalibrationFromROI(drawingShape);
      setState(s => ({ ...s, calibrationROI: drawingShape, activeTool: 'none' }));
    } else if (state.activeTab === 'analysis') {
      if (state.activeGroupId) {
        setState(s => ({
          ...s,
          roiGroups: s.roiGroups.map(g => 
            g.id === s.activeGroupId 
            ? { ...g, shapes: [...g.shapes, drawingShape] } 
            : g
          )
        }));
      } else {
        const newGroup: ROIGroup = {
          id: Math.random().toString(36),
          name: `Group ${state.roiGroups.length + 1}`,
          color: COLORS[state.roiGroups.length % COLORS.length],
          shapes: [drawingShape]
        };
        setState(s => ({
          ...s,
          roiGroups: [...s.roiGroups, newGroup],
          activeGroupId: newGroup.id
        }));
      }
    }
    setDrawingShape(null);
  };

  const calculateCalibrationFromROI = (shape: Shape) => {
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
        setState(s => ({ ...s, calibrationColor: { r: rSum / count, g: gSum / count, b: bSum / count } }));
     }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setState(s => ({ 
          ...s, originalImage: event.target?.result as string, calibrationColor: null, roiGroups: [], exclusionZones: [], reportSummary: '', processedImageURL: null
        }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDemoLoad = (url: string) => {
    const uniqueUrl = `${url}?t=${Date.now()}`;
    setState(s => ({
      ...s, originalImage: uniqueUrl, calibrationColor: null, isDemoMenuOpen: false, roiGroups: [], exclusionZones: [], reportSummary: '', processedImageURL: null
    }));
  };

  // --- Report Generation Logic ---

  const handleGenerateReport = async () => {
    // 1. Capture current canvas state as the "Analysis" image
    if (canvasRef.current && overlayRef.current) {
       // We need to combine main canvas and overlay for the report snapshot
       const reportCanvas = document.createElement('canvas');
       reportCanvas.width = canvasRef.current.width;
       reportCanvas.height = canvasRef.current.height;
       const rctx = reportCanvas.getContext('2d');
       if (rctx) {
          // Force run pipeline once to ensure "analysis" view is what we capture
          // Actually, we are already in the render loop. If the user clicks "Generate Report"
          // We assume they are happy with current view OR we want to show the Analysis view specifically.
          // Let's draw what's on the main canvas (processed)
          rctx.drawImage(canvasRef.current, 0, 0);
          // And overlay
          rctx.drawImage(overlayRef.current, 0, 0);
          setState(s => ({ ...s, processedImageURL: reportCanvas.toDataURL('image/png') }));
       }
    }

    // 2. Switch Tab
    setState(s => ({ ...s, activeTab: 'report', isProcessing: true }));

    // 3. Generate Text with Gemini
    try {
      const ai = new GeminiClient(); 
      const summary = await ai.generateReportSummary(state.roiGroups, state.regressionParams);
      setState(s => ({ ...s, reportSummary: summary, isProcessing: false }));
    } catch (error) {
      console.error(error);
      setState(s => ({ ...s, isProcessing: false, reportSummary: "Error generating summary. Please check API Key." }));
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
            onClick={() => setState(s => ({ ...s, activeTab: 'segmentation', activeTool: 'none' }))}
            icon={<Maximize2 size={18} />}
            label="Segmentation"
          />
          <NavButton 
            active={state.activeTab === 'calibration'} 
            onClick={() => setState(s => ({ ...s, activeTab: 'calibration', activeTool: 'none' }))}
            icon={<Pipette size={18} />}
            label="Color Calibration"
          />
          <NavButton 
            active={state.activeTab === 'analysis'} 
            onClick={() => setState(s => ({ ...s, activeTab: 'analysis', activeTool: 'none' }))}
            icon={<BarChart3 size={18} />}
            label="Quantification"
          />
          <div className="pt-4 mt-4 border-t border-slate-800">
             <button 
                onClick={handleGenerateReport}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${state.activeTab === 'report' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}`}
              >
                <FileText size={18} />
                Generate Report
              </button>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {state.activeTab !== 'report' && (
          <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur z-20">
            <div className="flex items-center gap-4">
              <h2 className="font-medium text-slate-200">
                {state.activeTab === 'segmentation' && 'ExG Vegetation Segmentation'}
                {state.activeTab === 'calibration' && 'Color Reference Calibration'}
                {state.activeTab === 'analysis' && 'ROI Quantification'}
              </h2>

              {(state.activeTab === 'segmentation' || state.activeTab === 'analysis') && (
                <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-8">
                  <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Square size={16} /></button>
                  <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><CircleIcon size={16} /></button>
                  <button onClick={() => setState(s => ({...s, activeTool: 'lasso'}))} className={`p-2 rounded ${state.activeTool === 'lasso' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}><Lasso size={16} /></button>
                </div>
              )}

              {state.activeTab === 'calibration' && (
                <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700 ml-8">
                   <span className="text-xs text-slate-500 px-2">Ref Region:</span>
                   <button onClick={() => setState(s => ({...s, activeTool: 'rect'}))} className={`p-2 rounded ${state.activeTool === 'rect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`} title="Rectangle Region"><Square size={16} /></button>
                   <button onClick={() => setState(s => ({...s, activeTool: 'circle'}))} className={`p-2 rounded ${state.activeTool === 'circle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`} title="Circle Region"><CircleIcon size={16} /></button>
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-3">
              <div className="relative">
                <button onClick={() => setState(s => ({ ...s, isDemoMenuOpen: !s.isDemoMenuOpen }))} className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-sm font-medium transition-colors border border-slate-700"><Globe size={16} /> Load Demo <ChevronDown size={14} /></button>
                {state.isDemoMenuOpen && (
                  <div className="absolute top-full right-0 mt-2 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
                    {DEMO_IMAGES.map((img, idx) => (
                      <button key={idx} onClick={() => handleDemoLoad(img.url)} className="w-full text-left px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 hover:text-white transition-colors">{img.label}</button>
                    ))}
                  </div>
                )}
              </div>
              <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-md text-sm font-medium"><Upload size={16} /> Upload</button>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
            </div>
          </header>
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
                         <img src={state.originalImage || ''} className="w-full h-auto object-contain" alt="Original" />
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
                {!state.originalImage ? (
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
                        className="absolute inset-0 w-full h-full pointer-events-auto"
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                      />
                    </div>
                  </div>
                )}
                
                {/* Hints */}
                <div className="text-xs text-slate-500 flex gap-4">
                  {state.activeTab === 'segmentation' && <span>Use tools to erase background artifacts.</span>}
                  {state.activeTab === 'calibration' && <span>Draw a box/circle over a neutral gray reference card.</span>}
                  {state.activeTab === 'analysis' && <span>Use Lasso or Rect tool to define plant groups.</span>}
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
                               onClick={() => setState(s => ({...s, exclusionZones: []}))}
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
                    <div className="p-4 bg-slate-950 rounded border border-slate-800 space-y-2">
                      <div className="flex justify-between text-xs text-slate-400"><span>R</span><span className="text-slate-200 font-mono">{state.calibrationColor?.r.toFixed(1) || '-'}</span></div>
                      <div className="flex justify-between text-xs text-slate-400"><span>G</span><span className="text-slate-200 font-mono">{state.calibrationColor?.g.toFixed(1) || '-'}</span></div>
                      <div className="flex justify-between text-xs text-slate-400"><span>B</span><span className="text-slate-200 font-mono">{state.calibrationColor?.b.toFixed(1) || '-'}</span></div>
                    </div>
                    {!state.calibrationColor && <p className="text-xs text-slate-500 mt-2">Draw a region on the gray card.</p>}
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
                        <p className="text-[10px] text-slate-400 font-medium mb-2">Model Params (y = mx + c)</p>
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
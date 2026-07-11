# Anthocyanin & Leaf Image Analysis

### ▶ Use it now: **[dr-richard-barker.github.io/Anthocyanin-Image-analysis](https://dr-richard-barker.github.io/Anthocyanin-Image-analysis/)**

No install required — it runs entirely in your browser.

A browser-based image-analysis tool for plant phenotyping. Upload a photo of a
plant tray, rosette, or leaf; segment the vegetation; calibrate colour and
physical scale; and quantify **leaf area** and **vegetation / pigment indices**
(NGRDI, mACI, Green Index) per region of interest — then export a printable
report.

The whole pipeline runs **client-side in the browser**. There is no server, no
account, and **no AI / LLM dependency** — the same image and settings always
produce the same numbers, which matters for reproducible science.

> **Status:** Active refactor. Migrating from the original Google AI Studio
> prototype (which used the Gemini API) to a fully independent, GitHub
> Pages–hosted tool. See [Goals & Roadmap](#goals--roadmap) below.

---

## What it does today

| Stage | Capability |
| --- | --- |
| **Segmentation** | Excess-Green (ExG) thresholding to separate plant from background; draw exclusion zones to mask soil, labels, and artifacts. |
| **Calibration — colour** | Sample white / grey / black reference patches to white-balance the image before indices are computed. |
| **Calibration — scale** | Draw a box over an **Astrocalibration** marker of known edge length to derive pixels/cm, so areas come out in **cm²**. Manual tilt and lens-distortion sliders for geometric correction. |
| **Quantification** | Group ROIs into cohorts; per cohort compute projected leaf area, **NGRDI**, **mACI** (modified Anthocyanin Content Index), and **Green Index**, plus an estimated anthocyanin value from a user-set linear model. |
| **Visualisation** | Toggle RGB / NGRDI / mACI / GI heatmap overlays on the segmented plant. |
| **Reporting** | One-click printable report: calibrated RGB, segmentation mask, index montages, a statistics table, and a deterministic written summary generated from the measured numbers. Print to PDF. |

### Indices, defined

- **ExG** = 2G − R − B — vegetation segmentation.
- **NGRDI** = (G − R) / (G + R) — normalised green-red difference; greenness / vigour proxy.
- **mACI** = R / G — modified anthocyanin content index; higher = more red/anthocyanin (stress / maturity).
- **GI** = G / (R + G + B) — green fraction.

---

## Using the tool — a typical workflow

The app opens with a set of **demo images** (pick from the dropdown, top right) — including
ExoLab-11 frames that contain the Astrocalibration marker — or use **Upload** to load your own.
Work left-to-right through the sidebar tabs:

1. **Segmentation** — drag the **ExG Threshold** slider until the green overlay covers the
   plant and nothing else. Use the shape tools (rectangle / circle / lasso) to draw
   **exclusion zones** over soil, labels, pots, or reflections you want ignored.
2. **Calibration & Units**
   - *Astrocalibration marker (recommended):* click **Detect Marker (colour + scale)**. The tool
     finds the marker's ArUco fiducials and, in one step, sets the scale and computes a full
     **affine colour correction** (PlantCV `astro_color_matrix`-equivalent). Drag the purple corner
     handles so the pink chip dots sit on the marker's colour patches; watch the **residual** drop.
   - *Manual scale (fallback):* enter the physical **edge length (cm)** of your marker, click
     **Set Scale from Marker**, then drag a box tightly over it to derive **pixels/cm**.
   - *Colour:* pick the **grey / white / black** target and draw a box over the matching
     reference patch to white-balance the image before indices are computed.
   - *Geometry:* nudge the **Tilt** and **Lens** sliders if the image is rotated or distorted.
3. **Results** — draw one or more ROIs to create **analysis groups** (cohorts). Each group
   reports area, NGRDI, mACI, and Green Index. Rename or add groups in the sidebar, and
   switch the canvas between **RGB / NGRDI / mACI / GI** overlays.
4. **Generate Report** — produces a printable page with the calibrated image, segmentation
   mask, index montages, a statistics table, and a written summary. Use **Print / PDF** to save.

> Tip: all processing is local — nothing is uploaded — so it is safe to use on unpublished
> research imagery.

---

## Run locally

**Prerequisites:** Node.js 20+

```bash
npm install
npm run dev      # http://localhost:3000
```

Build a static bundle:

```bash
npm run build    # outputs to dist/
npm run preview  # serve the production build locally
```

No API keys or `.env` files are required.

## Deploy (GitHub Pages)

Pushes to `main` are built and published automatically by
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). To enable it once:

1. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`. The workflow runs `npm ci && npm run build` and deploys `dist/`.

`vite.config.ts` uses `base: './'` so the site works from the project subpath
`https://<user>.github.io/<repo>/` without further configuration.

---

## Tech

- **React 19 + TypeScript**, single-file app (`index.tsx`).
- **Vite** build; **Tailwind** (CDN) for styling; **lucide-react** icons; **JSZip** for exports.
- All image math is plain Canvas 2D `getImageData` pixel processing — portable and inspectable.

---

## Goals & Roadmap

The end goal is a **free, offline-capable, citable framework** for leaf/rosette/
canopy image phenotyping that slots into the **AIRI** education programme and
standardises on the **AstroBotany "Astrocalibration"** fiducial marker for scale
and colour reference.

### ✅ Done — Independence & hosting
- [x] Remove all Google GenAI / Gemini code and the `@google/genai` dependency.
- [x] Replace AI ArUco detection with **deterministic marker-scale calibration** (draw a box over the known-size marker → pixels/cm).
- [x] Replace the AI report paragraph with a **deterministic summary** built from the measured statistics (reproducible output).
- [x] Configure Vite `base` + a **GitHub Pages** deploy workflow.
- [x] Strip API-key config; the app builds and runs with zero secrets.

### 🔜 Near term — Make it a dependable measurement tool
- [ ] **CSV / JSON export** of per-cohort statistics (currently report + on-screen only).
- [ ] **Batch mode**: process a folder / gallery of images with shared calibration and combined output.
- [ ] **Persist sessions** (ROIs, calibration, settings) to `localStorage` and to a downloadable project file.
- [ ] **Pixel-accurate ROI masking** — replace bounding-box hit-tests in the stats loop with true per-pixel shape containment for lasso/circle groups.
- [ ] **Rosette-aware metrics**: convex-hull area, compactness, and per-leaf counts for Arabidopsis-style rosettes.
- [ ] **Leaf Area Index (LAI)** mode for whole-canopy / plot images (fractional green cover → LAI estimate).

### 🎯 Astrocalibration marker
Full design + PlantCV interoperability write-up: **[docs/astrocalibration.md](docs/astrocalibration.md)**.
The marker is the AIRI *Bio Imaging Spectrum 5 cm* sticker ([order here](https://www.stickermule.com/drb2025/item/19181049)):
4 corner ArUco fiducials + colour chips + grayscale ramp + 0–5 cm ruler.
- [x] Bundle demo images that contain the marker (ExoLab-11 GRW08 timelapse) + a segmentation example (Hydra-1).
- [x] Hard-code the 15-chip `astro_color_matrix()` standard and apply **affine colour correction** (PlantCV-equivalent) in the canvas pipeline ([`colorcalib.ts`](colorcalib.ts)).
- [x] **Automatic marker detection** via the 4 ArUco fiducials (pure-JS `js-aruco2`) → auto scale + colour correction, with draggable corners + a live residual readout. See **[Detect Marker]** in Calibration & Units.
- [ ] Perspective homography + auto-rotation from the fiducials (currently bilinear + manual tilt).
- [ ] Print/document the marker chip-layout spec as an SVG/PDF in this repo.
- [ ] Optional "PlantCV pro" Colab notebook sharing the same marker standard.

### 🎓 AIRI education integration
- [ ] Package as an embeddable module / iframe for AIRI lesson pages.
- [ ] Guided "lab" walkthrough mode with sample images and expected-value checks for students.
- [ ] Link outputs to the AstroBotany / OSDR data conventions used across the wider programme.

### 🧪 Quality & science
- [ ] Validate indices against a reference dataset; document accuracy and limits.
- [ ] Provide a calibration protocol so `mACI → anthocyanin` slope/intercept can be fit against a real pigment assay.
- [ ] Add a small test suite for the index math and calibration functions.
- [ ] Add `CITATION.cff` + Zenodo archiving for citability.

---

## Notes on the refactor

This tool descends from a Google AI Studio prototype ("BioPheno"). The refactor
deliberately removes every network and AI dependency so the tool is
self-contained, auditable, and safe to run on sensitive research imagery. Please
keep new features **offline-first**: if something needs a network call, make it
optional and clearly labelled.

## Licence

[MIT](LICENSE) © Richard Barker. Free to use, modify, and redistribute — including
within the AIRI education programme — provided the copyright notice is retained.

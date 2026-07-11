# Astrocalibration marker — integration design & PlantCV interoperability

This document explains **what the Astrocalibration marker is**, **how we integrate it into this
browser tool**, and **how the PlantCV library extends what we can do** with the same marker.

- **Marker / sticker:** AIRI *Bio Imaging Spectrum 5 cm* — https://astrobotany.com/product/airi-bio-imaging-spectrum-5cm/
- **Order stickers:** https://www.stickermule.com/drb2025/item/19181049
- **PlantCV reference:** [`astro_color_matrix`](https://docs.plantcv.org/en/v5.0/astro_color_matrix/) ·
  [`affine_color_correction`](https://docs.plantcv.org/en/v5.0/transform_affine_color_correction/)

---

## 1. What the marker actually is

The sticker is a **5 cm** printed fiducial + colour-reference card. From imagery (e.g. ExoLab‑11
serial `ASOP‑0006`) it carries, in one 5 cm square:

| Element | Purpose |
| --- | --- |
| **4 corner ArUco fiducials** (black squares, one per corner) | Automatic detection, orientation, perspective (homography) and **scale** — the corners are a known distance apart on a 5 cm card. |
| **Rainbow spectrum strip** | Continuous visual reference (not sampled as discrete chips). |
| **Solid colour chips** — blue, green, red, yellow, near‑black | Colour references for chromatic correction. |
| **Grayscale ramp** (white → dark, ~10 steps) | Luminance / white‑balance / gamma reference. |
| **0–5 cm ruler** | Human‑readable scale check. |
| Red border, `ASTROBOTANY.COM`, serial `ASOP‑xxxx` | Branding / provenance. |

### The standard colour matrix

`pcv.transform.astro_color_matrix()` returns a **15 × 4** matrix: `[chip_id, R, G, B]` with RGB in
`0–1`. These are the *target* values every correctly‑lit image of the marker should match:

```
chip   R     G     B      (approx. identity)
 10   0.18  0.23  0.50    blue
 20   0.34  0.62  0.25    green
 30   0.71  0.25  0.21    red
 40   0.89  0.81  0.20    yellow
 50   0.21  0.22  0.22    near‑black
 60   0.91  0.95  0.93    white          ┐
 70   0.82  0.86  0.86                    │
 80   0.72  0.75  0.73                    │
 90   0.64  0.67  0.64                    │
100   0.57  0.58  0.56    grayscale ramp  │  (luminance ladder,
110   0.48  0.49  0.48                    │   white → dark)
120   0.39  0.40  0.39                    │
130   0.33  0.32  0.32                    │
140   0.27  0.28  0.27                    │
150   0.22  0.23  0.23                   ┘
```

*(Pull the authoritative values from `astro_color_matrix()` at implementation time; the above is the
published reference and is what we will hard‑code as `ASTRO_STD_MATRIX` in the app.)*

---

## 2. The PlantCV colour-correction workflow (reference implementation)

PlantCV (Python) normalises an image against the marker like this:

```python
from plantcv import plantcv as pcv

# 1. locate the marker / colour card in the image  -> mask
mask = pcv.transform.detect_color_card(rgb_img=img)          # or find_color_card
# 2. measure the chip colours actually seen in THIS image  -> source matrix
_, src_matrix = pcv.transform.get_color_matrix(rgb_img=img, mask=mask)
# 3. the standard the marker SHOULD match  -> target matrix
tgt_matrix = pcv.transform.astro_color_matrix()
# 4. fit + apply an affine transform so source -> target
corrected = pcv.transform.affine_color_correction(rgb_img=img,
                                                  source_matrix=src_matrix,
                                                  target_matrix=tgt_matrix)
```

`affine_color_correction` finds, per output channel, the linear combination of `(R, G, B, 1)` that
minimises the Euclidean distance between the transformed measured chips and the standard chips — i.e.
a **3 × 4 affine colour matrix**. This is a strictly richer correction than our current 3‑point
(white/grey/black) white balance because it uses 15 references and corrects channel cross‑talk, not
just per‑channel gain.

---

## 3. How we integrate it into *this* tool (browser, offline)

PlantCV is Python/NumPy/OpenCV and **cannot run in the browser**. So we reimplement the essential
maths client‑side — same marker, same standard, PlantCV‑equivalent output, zero install.

**Track 1 — in‑browser Astrocalibration (the plan):**

1. **Detect the marker.** Use a pure‑JS ArUco detector (e.g. `js-aruco2`) to find the 4 corner
   fiducials → gives a homography, rotation, and scale in one step. The 5 cm card edge yields
   **pixels/cm** automatically (replacing today's manual "draw a box over the marker"), and the
   rotation feeds the existing tilt correction.
2. **Sample the chips.** With the homography, the 15 chip centres sit at *known* normalised
   positions on the card, so we read each chip's mean RGB → the **source matrix**.
3. **Fit the transform.** Solve the least‑squares affine map source → `ASTRO_STD_MATRIX`
   (a small 3×4 normal‑equation solve — trivial in JS).
4. **Apply it** to every pixel in the existing canvas pipeline, before ExG segmentation and the
   NGRDI / mACI / GI indices are computed. Result: colour‑normalised indices that are comparable
   across lighting conditions and across the whole timelapse.

This slots into the current **Calibration & Units** tab. Manual white/grey/black stays as a fallback
for images without a marker; the marker path supersedes it when present.

**Track 2 — optional "PlantCV pro" export (documented, not in‑browser):**

Provide a Colab/Jupyter notebook that runs the reference workflow above plus PlantCV's segmentation
and analysis, for users who want batch processing or publication‑grade pipelines. Our tool and the
notebook share the **same marker standard**, so results are directly comparable.

---

## 4. What PlantCV adds beyond this tool

Using the marker as the shared standard, PlantCV can take a study much further than a browser tool
should attempt:

- **Automatic colour‑card detection** (`detect_color_card` / `find_color_card`) and robust
  affine/other colour correction — no manual chip picking.
- **Advanced segmentation:** naive‑Bayes and machine‑learning classifiers, multi‑threshold, watershed
  separation of touching plants, and object counting.
- **Morphology:** skeletonisation, branch‑point and leaf‑tip detection, stem/leaf segmentation,
  and root architecture analysis.
- **Rich descriptors:** shape (area, hull, solidity, ellipse), colour histograms, and GLCM texture.
- **Multi‑plant / grid ROI workflows** and **time‑series growth analysis** across a timelapse.
- **Other modalities:** hyperspectral, thermal, NIR, and PSII/fluorescence imaging.
- **Tidy, reproducible outputs** and a documented, citable analysis ecosystem.

**Division of labour:** this tool is the accessible, zero‑install *front door* for teaching and quick
measurement (AIRI classrooms, field checks). PlantCV is the *power backend* for large or advanced
studies. The Astrocalibration marker is the common reference that makes the two interchangeable.

---

## 5. Demo images (bundled in the tool)

The image picker ships with frames that contain the marker plus a segmentation example:

- **ExoLab‑11 GRW08 timelapse** — https://github.com/dr-richard-barker/ExoLab_11/tree/main/grw08_images_11122024
  (297 frames; the fixed marker appears in every `cam_0` frame — ideal for the calibration workflow
  and time‑series growth).
- **Hydra‑1 germination trays** — https://github.com/dr-richard-barker/Hydra1-Orbital-Greenhouse/tree/master/Raw%20images
  (Flight/Ground germination carriers; no marker — used for segmentation practice).

---

## 6. Roadmap checklist

- [ ] Hard‑code `ASTRO_STD_MATRIX` (15×4) from `astro_color_matrix()`.
- [ ] Add `js-aruco2` (or equivalent) corner detection for the 4 fiducials.
- [ ] Homography → auto scale (px/cm) + rotation from the marker.
- [ ] Sample the 15 chips → source matrix.
- [ ] Least‑squares affine colour correction applied in the canvas pipeline.
- [ ] UI in **Calibration & Units**: "Detect Astrocalibration marker" with manual fallback.
- [ ] Document/print the marker spec (chip layout + positions) as an SVG/PDF in this repo.
- [ ] "PlantCV pro" Colab notebook sharing the same standard.

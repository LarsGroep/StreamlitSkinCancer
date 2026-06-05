# Design Prompt: Skin Lesion AI Research Showcase

## What this is

A Streamlit web app that presents an ISIC skin cancer classification AI system to a non-technical
audience (patients, hospital stakeholders, curious members of the public). The AI classifies
dermoscopy images into 7 classes: Melanoma (MEL), Melanocytic Nevi (NV), Basal Cell Carcinoma
(BCC), Actinic Keratosis (AKIEC), Benign Keratosis (BKL), Dermatofibroma (DF), and Vascular
Lesion (VASC). The underlying model is EfficientNet-B0 (1,280 deep features) combined with 44
hand-crafted morphology features (shape, colour, texture, border).

The app has 4 pages navigated from a left sidebar:
1. **Try It Yourself** -- image upload, GradCAM heatmap, probability bars, verdict card,
   melanoma probability gauge
2. **How the AI Sees** -- annotated lesion SVG diagram, 44 features grouped into 4 expandable
   sections with bar charts (value vs benign mole baseline)
3. **How Accurate Is It?** -- stat cards, operating point table, confusion matrix heatmap,
   ROC curve with real operating points, per-class F1 bars, threshold slider
4. **The Research Journey** -- 5-step timeline, model architecture card, failure modes,
   80-case clinician review table (filterable), example gallery, future work

## Design goal

Redesign the visual presentation so it looks **polished, clinical, and trustworthy** -- the kind
of interface a hospital innovation team would be proud to demo to clinicians. Think Apple Health
meets a radiology dashboard.

Keep it 100% Streamlit (no React, no separate frontend). All visuals must be achievable with
`st.markdown` + inline HTML/CSS, Plotly figures, and `st.image`.

## Reference design (base44 React original)

The original was built in React with Tailwind CSS and shadcn/ui. Its design language:

- **Fonts**: Sora (headings, 700-800 weight) + Inter (body, 400-600 weight) from Google Fonts
- **Colours**:
  - Primary (dark navy blue): `hsl(217, 91%, 30%)` = `#0e3fa6` approx -- use `#1e40af`
  - Background: `hsl(210, 20%, 98%)` = `#f7f9fb`
  - Card: pure white `#ffffff`
  - Border: `hsl(210, 20%, 88%)` = `#dde3ec`
  - Muted text: `hsl(220, 15%, 50%)` = `#64748b`
  - Foreground (headings): `hsl(220, 25%, 10%)` = `#0f172a`
- **Class colours** (used consistently for ISIC classes in all charts):
  - NV: `#22c55e` (green)   MEL: `#3b82f6` (blue, turns `#ef4444` red when elevated)
  - BCC: `#f97316` (orange) AKIEC: `#a855f7` (purple)
  - BKL: `#06b6d4` (cyan)   DF: `#ec4899` (pink)    VASC: `#ef4444` (red)
- **Cards**: `border-radius: 16px`, `border: 1px solid #dde3ec`, `box-shadow: 0 1px 4px rgba(15,23,42,0.04)`
- **Sidebar**: white background, 1px right border, logo with rounded-square icon in primary blue,
  radio nav items styled as rounded pill links (active = primary blue fill + white text)
- **Stat cards**: large Sora number, bold label, small grey sub-label, centred
- **Verdict card**: 2px coloured border matching the predicted class colour, light tint background
- **Melanoma gauge**: semicircle arc in green/amber/red zones with animated needle, percentage in Sora
- **Probability bars**: thin 13px rounded bars with class colour dots, MEL turns red when >30%
- **Feature bar chart**: horizontal bars, grey shaded ±1std range, grey baseline marker, red bars
  for abnormal values (outside ±2std), small "!" annotation
- **Confusion matrix**: white-to-dark-blue heatmap, bold black count annotations
- **ROC curve**: blue filled area, grey diagonal reference, scatter dots for real operating points
- **Timeline steps**: white cards with left coloured icon badge, highlighted Step 3 in light green
- **All charts**: white plot background, light `#f1f5f9` grid lines, no chart border, Plotly

## Specific improvements to make

1. **Sidebar nav**: Style the `st.radio` so active item has a solid primary-blue background and
   white text. Add icon symbols (emoji or Unicode) before each nav label. The current styling
   attempt is incomplete -- the `data-checked` CSS selector may not work reliably in all
   Streamlit versions; find the correct selector or use a custom HTML nav with `st.query_params`.

2. **Page header**: On each page, the `<h1>` should be large Sora 800, primary blue colour,
   followed immediately by a grey subtitle line -- no Streamlit default header chrome around it.

3. **GradCAM panel**: Currently uses `make_gradcam_overlay()` which blends the real `gradcam.png`
   over any uploaded image. Improve this: the gradcam.png is a standalone heatmap image from the
   Kaggle run. Instead of blending it over user uploads (wrong image), show the real gradcam.png
   on the right for example images, and for user uploads show a synthesised radial heatmap.
   Add a proper colour scale legend bar below the GradCAM panel.

4. **Probability bars**: The current layout uses HTML floats which can misalign. Replace with a
   clean Plotly horizontal bar chart (same colours, same MEL-red-if-elevated logic) so alignment
   is pixel-perfect. Sort by probability descending.

5. **Confusion matrix**: Add full class name labels (e.g. "MEL (Melanoma)") as axis tick labels,
   not just abbreviations. Add a caption below explaining the primary failure mode with specific
   numbers (51 MEL misclassified as NV).

6. **Operating point table**: Make the three rows visually distinct -- Best F1 row in a light
   blue tint, Screening row in a light red tint (high sensitivity), Balanced row in a light
   orange tint. Add a small coloured dot before each row name.

7. **Clinician review table**: Colour-code the "Cohort" column by cohort type (green for Easy
   Melanoma, orange for Missed, red for False Positive, purple for Uncertainty). Add a coloured
   "Correct" / "Missed" / "False alarm" / "Uncertain" badge in a "Result" column.

8. **Feature bar charts**: Currently uses Plotly shapes for the baseline and range. Ensure the
   grey range band is always visible (currently may be too faint). Increase bar height to 20px.
   Add a proper legend with coloured swatches at the bottom of each chart.

9. **Mobile layout**: Wrap all multi-column layouts in responsive logic: when viewport is narrow
   (use `st.columns([1])` single column), stack vertically. Streamlit does not expose viewport
   width natively, but add a CSS media query to stack `.stColumns` on small screens.

10. **Disclaimer / footer**: Every page should end with the same disclaimer card:
    "This is a research prototype built on the ISIC 2018 dataset. It is NOT a medical device.
    All outputs require clinical review." Style it as a light grey card with an info icon.

## Files in this zip

```
app.py                              -- main Streamlit app (1,514 lines)
requirements.txt                    -- dependencies
data/
  config.json                       -- model config (class names, thresholds, T=0.581)
  clinical_summary.csv              -- 3 operating points with full metrics
  clinical_safety_metrics.csv       -- confusion matrix, per-class F1/precision/recall,
                                       clinical safety rates, calibration metrics
  model_card.csv                    -- architecture, features, failure modes, limitations
  dermatologist_review_cases.csv    -- 80 cases (4 cohorts x 20), with MEL probability,
                                       calibrated probability, entropy, top 3 features
  gradcam.png                       -- real GradCAM attention map from Kaggle experiment
  __results___10_3.png              -- Kaggle cell output: training curves
  __results___10_4.png              -- Kaggle cell output: morphology extraction diagram
  __results___10_5.png              -- Kaggle cell output: threshold sweep chart
  __results___10_7.png              -- Kaggle cell output: calibration curves
  webapp_schema.json                -- typed input/output contract for the API
```

## Key real metrics to display accurately

From `clinical_safety_metrics.csv` and `clinical_summary.csv`:

| Metric | Value |
|--------|-------|
| Best macro F1 (Nonlinear Stack, argmax) | **0.7623** |
| MEL F1 at argmax | 0.5506 |
| MEL recall at argmax | 47.9% |
| MEL recall at screening (thr=0.01) | **94.4%** |
| MEL recall at balanced (thr=0.07) | **82.4%** |
| Specificity at balanced | 74.3% |
| ROC AUC | 0.9276 (Nonlinear Stack) |
| Temperature T | 0.581 |
| Primary failure | MEL misclassified as NV (n=51) and BKL (n=20) |
| NV F1 | 0.918 (best class) |
| VASC F1 | 0.929 (best class) |

Confusion matrix class order: AKIEC, BCC, BKL, DF, NV, MEL, VASC

Real confusion matrix (rows=true, cols=predicted):
```
        AKIEC  BCC  BKL  DF   NV  MEL  VASC
AKIEC:    42    6    6    0    1    3    0
BCC:       4   70    2    0    2    1    0
BKL:       9    8  126    1   32   14    0
DF:        0    0    0   10    3    0    0
NV:        1   14   37    2  917   19    2
MEL:       2    1   20    0   51   68    0
VASC:      0    0    1    0    1    0   26
```

## Run instructions

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

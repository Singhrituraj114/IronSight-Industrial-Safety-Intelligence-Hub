# IronSight Industrial Safety Intelligence Hub

IronSight is an **AI-powered PPE compliance platform** built on YOLOv8l for industrial safety monitoring. It detects safety gear (helmet, vest, gloves, shoes, harness, mask, etc.) and also identifies **missing PPE** in real-world workplace scenes. The app runs detection on **images, videos, and live webcam feeds**, then turns results into clear compliance insights with KPI cards, charts, and downloadable audit logs. The goal is simple: help safety teams spot violations fast, track compliance trends, and keep hazardous zones safer with real-time AI surveillance.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-orange)

## Highlights

- **AI PPE compliance detection** with bounding boxes, confidence scores, and class labels
- **Image, video, and webcam inference** built for real-time monitoring
- **Enterprise UI dashboard** with KPI cards, analytics charts, and audit-ready logs
- **Compliance analytics**: PPE count, missing PPE count, violations, and compliance %
- **Dataset-aware design** with full class taxonomy and augmentation strategy

## System Architecture (High-Level)

The application is a single-file Streamlit product that combines **computer vision inference**, **analytics**, and **audit logging** in one dashboard:

- **UI & orchestration**: Streamlit handles navigation, layout, and user input.
- **Inference engine**: YOLOv8l detects PPE compliance and violation classes.
- **Analytics layer**: Plotly charts and KPI cards summarize compliance health.
- **Audit layer**: Every inference session is tracked and downloadable as CSV.

### Architecture Diagram

```
User Inputs (Image / Video / Webcam)
              |
              v
       Streamlit UI Layer
              |
              v
      YOLOv8 Inference Engine
              |
              v
Post-Processing & Compliance Logic
   |                      |
   v                      v
Analytics Dashboard   Audit Log Store
   |                      |
   v                      v
  KPI + Charts       CSV Export
```

### Core Components

- **Input handlers**: file upload + webcam capture + video decode
- **Inference core**: YOLOv8 prediction on images/frames
- **Compliance engine**: converts detections into compliance %
- **Visualization**: annotated outputs + Plotly charts + KPI cards
- **Audit logging**: timestamps and exportable inference history

## How It Works

1. **Input stream** (image, video, or webcam) is ingested in the Streamlit UI.
2. **YOLOv8l inference** runs on each frame or image and returns bounding boxes, labels, and confidence scores.
3. **Compliance logic** aggregates detections into compliant vs. violation classes to compute compliance %.
4. **Analytics layer** renders KPI cards and Plotly charts (pie, bar, gauge).
5. **Audit trail** logs each inference run into a downloadable history table.

## Inference Pipeline (Detailed)

1. **Pre-processing**: Images are converted to RGB and fed into YOLOv8.
2. **Model prediction**: YOLOv8 returns per-object bounding boxes, class ids, and confidence.
3. **Post-processing**: Detections are grouped by class label and counted.
4. **Compliance scoring**: Counts are split into **compliant PPE** and **violations**.
5. **Visualization**: Annotated outputs are rendered back into the UI.

For video, frames are sampled using a configurable **frame stride** for performance.  
For webcam, inference runs in a **burst mode** to provide real-time feedback.

## App Modules & Features

| Module | What it delivers |
|---|---|
| **Landing Dashboard** | Hero section, KPI cards, live system status, enterprise UI |
| **Image Detection** | Upload image → annotated output + class counts + analytics |
| **Video Detection** | Upload video → processed output + aggregated analytics |
| **Webcam Live** | Real-time burst inference with live stats |
| **Analytics** | Compliance trend charts and global summary |
| **Dataset Overview** | Class taxonomy, augmentation, and dataset coverage |
| **Detection History** | Timestamped inference logs + CSV export |

## Compliance Analytics Logic

The compliance score is computed as:

```
compliance % = (compliant detections / total detections) * 100
```

Violation classes are detected by label keywords such as **"without"**, **"no"**, **"missing"**, and **"violation"**.  
If your dataset uses different label naming, adjust the logic in `is_violation_label()` inside `app.py`.

If there are **no detections**, compliance is set to **100%** to avoid misleading violation results.

## Output Artifacts

- **Annotated images** with bounding boxes, class labels, and confidence scores
- **Annotated videos** with tracked detections across frames
- **Live webcam feed** with real-time compliance overlay

## Analytics & KPI Dashboard

The dashboard includes:

- **Gauge chart** for compliance %
- **Pie chart** comparing compliant vs. violation detections
- **Bar chart** showing class distribution
- **Trend analytics** for compliance over time

## Detection History & Audit Logs

Every inference session is logged with:

- Timestamp
- Filename/source
- Total detections
- Compliance %
- Violations

Logs can be exported as CSV for audits or compliance reporting.

## Screenshots

Add your real UI screenshots to `/assets/` and update or replace the links below:

![Dashboard](assets/dashboard.png)
![Image Detection](assets/image-detection.png)
![Analytics](assets/analytics.png)
![Dataset Overview](assets/dataset.png)

## Dataset Overview

The PPE Detection Dataset is built for **industrial safety monitoring** and **real-time compliance detection** in environments like construction sites, factories, manufacturing plants, and hazardous work zones. It contains multi-object scenes with multiple workers per image, variable lighting, and diverse viewpoints.

**Core dataset characteristics**

- **YOLO format** annotations (`class_id, x_center, y_center, width, height`)
- **Train/validation/test splits**
- **Multi-object workplace scenes**
- **Realistic industrial conditions**

**Annotation format:** `(class_id, x_center, y_center, width, height)` using normalized coordinates.

### PPE Compliance Classes (9)

| Class | Class |
|---|---|
| Ear Protectors | Full Body Suit |
| Glasses | Gloves |
| Helmet | Mask |
| Safety Harness | Safety Shoes |
| Safety Vest |  |

### PPE Non-Compliance / Violation Classes (9)

| Class | Class |
|---|---|
| Without Ear Protectors | Without Full Body Suit |
| Without Glasses | Without Gloves |
| Without Helmet | Without Mask |
| Without Safety Harness | Without Safety Shoes |
| Without Safety Vest |  |

### Data Augmentations

Mosaic • MixUp • HSV color augmentation • Scaling • Translation • Shearing • Horizontal flipping

### Supported Use-Cases

Industrial AI safety systems • smart CCTV analytics • workplace automation • real-time compliance reporting

## UI/UX Design Goals

The interface is intentionally designed to resemble **enterprise surveillance software** with:

- Dark, futuristic theme
- KPI cards with glassmorphism
- Sidebar navigation for operational workflows
- Smooth status indicators, progress bars, and toast alerts

## Model Details

The application uses a **YOLOv8l (Large)** detection model trained on the above dataset.  
Weights are loaded from: `PPE-YOLOv8-best.pt`

## Performance & Optimization

- **Model cached** with `st.cache_resource` for fast reloads
- **Optimized image handling** for inference throughput
- **Frame stride control** for faster video processing
- **Efficient aggregation** of detections for real-time analytics

## Configuration Notes

- **Confidence/IoU thresholds** are adjustable in the sidebar.
- **Frame stride** accelerates long video processing.
- **Webcam burst length** controls real-time processing duration.

## Troubleshooting

- `ModuleNotFoundError: cv2` → install `opencv-python`
- `ModuleNotFoundError: ultralytics` → install `ultralytics`
- Webcam not found → ensure camera permissions and try restarting the app

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Webcam inference requires local device access.

## Model Weights

This app expects the model file:

```
PPE-YOLOv8-best.pt
```

For GitHub, use **Git LFS** for `.pt` files:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

If you prefer not to store weights in the repository, add the `.pt` file to `.gitignore` and provide a download link in this README.

## Project Structure

```
.
├── app.py
├── PPE-YOLOv8-best.pt
├── .gitattributes
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
└── assets/
```

## License

MIT License. See [LICENSE](LICENSE).

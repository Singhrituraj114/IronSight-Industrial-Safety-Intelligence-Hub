# IronSight Industrial Safety Intelligence Hub

Enterprise-grade PPE compliance monitoring powered by **YOLOv8l**. The system detects PPE usage and violations across industrial environments, delivers real-time compliance analytics, and provides professional dashboards for safety teams.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-orange)

## Highlights

- **AI PPE compliance detection** with bounding boxes, confidence scores, and class labels
- **Image, video, and webcam inference** built for real-time monitoring
- **Enterprise UI dashboard** with KPI cards, analytics charts, and audit-ready logs
- **Compliance analytics**: PPE count, missing PPE count, violations, and compliance %
- **Dataset-aware design** with full class taxonomy and augmentation strategy

## How It Works

1. **Input stream** (image, video, or webcam) is ingested in the Streamlit UI.
2. **YOLOv8l inference** runs on each frame or image and returns bounding boxes, labels, and confidence scores.
3. **Compliance logic** aggregates detections into compliant vs. violation classes to compute compliance %.
4. **Analytics layer** renders KPI cards and Plotly charts (pie, bar, gauge).
5. **Audit trail** logs each inference run into a downloadable history table.

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

## Model Details

The application uses a **YOLOv8l (Large)** detection model trained on the above dataset.  
Weights are loaded from: `PPE-YOLOv8-best.pt`

## Performance & Optimization

- **Model cached** with `st.cache_resource` for fast reloads
- **Optimized image handling** for inference throughput
- **Frame stride control** for faster video processing
- **Efficient aggregation** of detections for real-time analytics

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

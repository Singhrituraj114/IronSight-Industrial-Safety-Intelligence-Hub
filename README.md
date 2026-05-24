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

## Screenshots

Add your real UI screenshots to `/assets/` and update or replace the links below:

![Dashboard](assets/dashboard.png)
![Image Detection](assets/image-detection.png)
![Analytics](assets/analytics.png)
![Dataset Overview](assets/dataset.png)

## Dataset Overview

This project uses a PPE Detection Dataset designed for industrial workplace safety monitoring and real-time compliance detection. The dataset is YOLO-formatted with train/validation/test splits, multi-object scenes, and varying lighting conditions.

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
├── requirements.txt
├── README.md
└── assets/
```

## License

Choose a license (MIT or Apache-2.0) and add the LICENSE file.

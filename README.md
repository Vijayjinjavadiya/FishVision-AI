<div align="center">
<img src="assets/6eb57068-116b-11ee-a55a-9335f156a1e7.gif" alt="Fish Detection System" width="100%">

<br/>

# 🐠 Fish Detection System

**Real-time fish species identification powered by YOLOv8**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

[![mAP50](https://img.shields.io/badge/mAP50-94%25-22C55E?style=flat-square&logo=target&logoColor=white)](docs/architecture.md)
[![Classes](https://img.shields.io/badge/Species-13-6366F1?style=flat-square)](docs/architecture.md)
[![Dataset](https://img.shields.io/badge/Dataset-8%2C242%20images-F59E0B?style=flat-square)](https://universe.roboflow.com/zehra-acer/fish-detection-fztlb/dataset/5)
[![GPU](https://img.shields.io/badge/Trained%20on-RTX%203050-76B900?style=flat-square&logo=nvidia&logoColor=white)](docs/architecture.md)

</div>

---

## ✨ What it does

Upload a photo → get annotated detections in seconds.

<div align="center">

| 🔍 Detects | 📊 Reports | 📥 Exports |
|:---:|:---:|:---:|
| Bounding boxes + species labels | Species count breakdown | One-click CSV download |
| Confidence scores per fish | Average confidence score | All box coordinates included |
| 13 freshwater & marine species | Total fish count | Optional length estimation |

</div>

---

## 🐟 Detected Species

<div align="center">

| | Species | | Species | | Species |
|:---:|:---|:---:|:---|:---:|:---|
| 🐠 | AngelFish | 🐡 | BlueTang | 🦋 | ButterflyFish |
| 🤡 | ClownFish | 🟡 | GoldFish | 🌿 | Gourami |
| ⚪ | MorishIdol | 🔴 | PlatyFish | 🎀 | RibbonedSweetlips |
| 🔵 | ThreeStripedDamselfish | 🟡 | YellowCichlid | 🌙 | YellowTang |
| 🦓 | ZebraFish | | | | |

</div>

---

## 📊 Model Performance

<div align="center">

```
Model      YOLOv8n (nano)
───────────────────────────────────────
mAP50        ████████████████████  94%
Training     6,842 images
Validation   700 images   → mAP50 ~0.94
Test         700 images
───────────────────────────────────────
Hardware     NVIDIA RTX 3050 Laptop GPU
```

</div>

---

## 🚀 Quick Start

**1 — Clone**
```bash
git clone https://github.com/YOUR_USERNAME/fish-detection-system.git
cd fish-detection-system
```

**2 — Install**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**3 — Add weights**

Download `best.pt` from [Releases](https://github.com/YOUR_USERNAME/fish-detection-system/releases) → place in `models/`

**4 — Run**
```bash
streamlit run app.py
# → http://localhost:8501
```

---

## 🗂️ Project Structure

```
fish-detection-system/
│
├── 📄 app.py                   Streamlit entry point
├── 📋 requirements.txt
├── ⚙️  data.yaml               YOLO dataset config
│
├── 📦 src/
│   ├── detector.py             Model loading + inference
│   ├── predictor.py            Results → DataFrame
│   ├── utils.py                Path helpers
│   └── visualization.py        Streamlit render functions
│
├── 🧠 models/
│   └── best.pt                 Trained YOLOv8n weights (~6 MB)
│
├── 🖼️  screenshots/
├── 📚 docs/
│   └── architecture.md
└── 🎨 assets/
```

---

## 🗺️ Roadmap

<div align="center">

| Phase | Goal | Status |
|:---:|:---|:---:|
| **1** | Fish species detection — 13 classes | ✅ **Done** |
| **2** | Length estimation via depth camera | 🔵 Planned |
| **3** | Weight estimation from length | 🔵 Planned |
| **4** | Adult vs Juvenile classification | 🔵 Planned |
| **5** | OAK-D Pro camera integration | 🔵 Planned |
| **6** | Real-time monitoring dashboard | 🔵 Planned |

</div>

> The `cm_per_pixel` input in the sidebar is already wired — it's a deliberate placeholder for the OAK-D depth pipeline coming in Phase 5.

---

## ⚙️ Sidebar Controls

| Control | Default | What it does |
|---------|---------|-------------|
| Confidence threshold | `0.25` | Minimum score to show a detection |
| IoU threshold | `0.45` | Controls overlap suppression (NMS) |
| Centimetres per pixel | `0.00` | Scale for length estimation — leave at 0 until calibrated |
| Adult length threshold | `10 cm` | Fish above this length → labelled Adult |

---

## 📦 Dataset

Sourced from Roboflow Universe — **[fish-detection-fztlb v5](https://universe.roboflow.com/zehra-acer/fish-detection-fztlb/dataset/5)**

[![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-purple?style=flat-square&logo=roboflow)](https://universe.roboflow.com/zehra-acer/fish-detection-fztlb/dataset/5)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)

The dataset is **not included** in this repo (too large). Download it from Roboflow and place it under `data_sets/`.

---

## 🔁 Retrain

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=fish-detection
```

Results saved under `runs/detect/fish-detection/`.

---

## 🛠️ Tech Stack

<div align="center">

[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-FF6B35?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pillow](https://img.shields.io/badge/Pillow-Imaging-3776AB?style=for-the-badge)](https://python-pillow.org)
[![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

</div>

---

## 📄 License

This project is **MIT licensed** — see [LICENSE](LICENSE).

The dataset is licensed separately under **CC BY 4.0** by Zehra Acer via Roboflow.

---

<div align="center">

*Built as part of a larger aquatic species monitoring pipeline.*

⭐ **Star this repo** if it helped you — it means a lot.

</div>

# PasarVision AI — Traditional Market Socio-Economic Analytics Platform

**Enterprise Computer Vision Platform powered by YOLOv11 (`ultralytics`), Intel OpenVINO GPU FP16 Acceleration, ByteTrack, OpenCV Data Enhancement, and OpenPyXL Automated Reporting.**

- **Live Cloud Deployment (Pages):** https://traditionalmarket.pages.dev
- **Live Cloud Deployment (Workers):** https://traditionalmarket.bimdarmawa2.workers.dev

Built for **Deep Learning Final Project (Tema 2: Pasar Tradisional — Socio-Economic Detection)** with **100% Plug-and-Play Architecture**:
- Pre-integrated with trained **`models/best.pt`** (`cart`, `customer`, `scale`, `seller`, `vegetables`) + automatic `yolo11n.pt` fallback if removed.
- Streams real-time annotated video at **50–60 FPS** via asynchronous Intel Iris Xe GPU OpenVINO FP16 inference + binary WebSocket JPEG streaming.

---

## Quick Start (3 Exact Steps)

### Step 1 — Install Dependencies
```bash
cd pasarvision-ai
pip install -r requirements.txt
```

### Step 2 — Launch PasarVision AI (Single Command)
```bash
python run.py --open
```
*(Or run `python run.py` and open **http://127.0.0.1:8000** in Chrome / Edge.)*

### Step 3 — Plug In Your Trained `best.pt` / `best.onnx` & Market Footage (Zero Code Changes)
- **Custom YOLOv11 Weights:** Copy your trained weights to **`models/best.pt`** (or **`models/best.onnx`** on Windows 11 laptops with Smart App Control enabled — export in Colab via `YOLO("best.pt").export(format="onnx")`), or upload live from the left sidebar! The top status badge automatically turns green: **`Production Model: best.pt`** (or `best.onnx`), reading `model.names` dynamically.
- **Test Media:** Drop market images into **`samples/images/`** and market CCTV/walkthrough clips into **`samples/videos/`** (or drag-and-drop directly onto the web dashboard).

---

## Core Capabilities & Modules

| Module | File | Key Engineering Highlights |
| :--- | :--- | :--- |
| **Plug-and-Play Model Manager** | `backend/model_manager.py` | Auto-detects `models/best.pt` with `yolo11n.pt` fallback, dynamic `model.names` inspection, preset socio-economic class palette (`seller`, `customer`, `cart`, `scale`, `vegetables`), live `.pt` hot-swapping, and CUDA/MPS/CPU auto-detection. |
| **OpenCV Data Enhancement** | `backend/enhancement.py` | **CLAHE** (strictly on CIE LAB `L`-channel to prevent color shifts), **Histogram Equalization** (YCrCb `Y`-channel), **Percentile Contrast Stretching**, and **Brightness + Contrast Alpha + LUT Gamma**. |
| **ByteTrack + Interactive ROI Counter** | `backend/tracker_region.py` & `frontend/js/roi_canvas.js` | Persistent **ByteTrack** multi-object IDs (`#1, #2, ...`), fading motion trajectory trails, de-duplicated cumulative tracking, interactive Canvas ROI polygon editor (drag vertices, add edge midpoints, redraw), and `cv2.pointPolygonTest` on bottom-center `(cx, y2)`. |
| **Before vs. After Enhancement Lab** | `backend/comparator.py` | Evaluates identical YOLOv11 thresholds on **Original (Before)** vs. **Enhanced (After)** images, batches, and N-frame-sampled videos with an interactive **Before/After Swipe Slider** and Side-by-Side view. |
| **Enterprise Excel Exporter (`.xlsx`)** | `backend/excel_exporter.py` | Generates a styled 4-sheet workbook (`Executive_Summary` with native Excel `BarChart`, `Per_Class_Comparison`, `Frame_Detail_Log`, `Region_Counter_Log`) saved to `outputs/` and downloaded in 1 click. |

---

## Project Directory Structure

```text
pasarvision-ai/
├── run.py                          # Single-command Uvicorn launcher
├── requirements.txt                # Clean Python dependencies
├── README.md                       # Setup, architecture & model-swap guide
├── models/
│   └── README.txt                  # Drop trained best.pt here (auto-fallback to yolo11n.pt)
├── samples/
│   ├── images/                     # Sample test images (auto-seeded on first run)
│   └── videos/                     # Sample test videos (e.g. demo_pasar_subuh.mp4)
├── outputs/                        # Saved annotated MP4s & exported Excel (.xlsx) reports
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI routes, WebSocket/MJPEG stream & static server
│   ├── model_manager.py            # Dynamic YOLOv11 loader, hot-swap .pt & class colors
│   ├── enhancement.py              # OpenCV CLAHE, HE, Contrast Stretching & Gamma LUT
│   ├── tracker_region.py           # ByteTrack tracking, motion trails & ROI Polygon counter
│   ├── comparator.py               # Before vs. After Enhancement comparison engine
│   ├── excel_exporter.py           # 4-sheet styled OpenPyXL workbook + native BarChart
│   └── sample_seeder.py            # First-run realistic low-light sample generator
└── frontend/
    ├── index.html                  # Carbon Enterprise Dashboard UI
    ├── css/
    │   └── styles.css              # Dark Carbon (#090D16) glassmorphism & swipe slider CSS
    └── js/
        ├── app.js                  # State management, WebSocket video & API controller
        ├── roi_canvas.js           # Interactive drag/add/delete/redraw ROI polygon canvas
        └── charts.js               # Live Chart.js telemetry & Before-vs-After charts
```

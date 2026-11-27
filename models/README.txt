================================================================================
PasarVision AI — Custom YOLOv11 Model Directory
================================================================================

HOW TO PLUG IN YOUR TRAINED CUSTOM MODEL (ZERO CODE CHANGES):

1. Copy your trained YOLOv11 weights file into this folder:
       models/best.pt      (Standard PyTorch YOLOv11 weights)
       OR
       models/best.onnx    (Recommended on Windows 11 laptops with Smart App Control!)

   Tip — Exporting `best.onnx` from Google Colab / Kaggle takes 1 line:
       from ultralytics import YOLO
       YOLO("best.pt").export(format="onnx")

2. Launch (or restart) PasarVision AI:
       python run.py

   OR upload your `best.pt` / `best.onnx` file live while the app is running
   using the "Hot-Swap Upload" button in the left sidebar!

AUTOMATIC FALLBACK BEHAVIOR:
- If `models/best.pt` or `models/best.onnx` is present -> Automatically loads in
  PRODUCTION mode (Emerald badge: "Production Model: best.pt / best.onnx") and
  reads `model.names` dynamically.
- If absent -> Automatically falls back to Ultralytics `yolo11n`
  (Amber badge: "Demo Fallback: yolo11n.pt / yolo11n.onnx") so every feature
  can be tested immediately out-of-the-box.

TARGET SOCIO-ECONOMIC MARKET CLASSES (Preset Color Palette Included):
- cart        (Cyan         #06B6D4)
- customer    (Blue         #3B82F6)
- scale       (Amber/Yellow #F59E0B)
- seller      (Red/Coral    #F43F5E)
- vegetables  (Emerald      #10B981)
(All other class names in `model.names` are supported dynamically with deterministic colors.)

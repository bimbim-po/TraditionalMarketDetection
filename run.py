#!/usr/bin/env python3
"""
PasarVision AI — Single-Command Application Launcher (`run.py`).

Usage:
    python run.py
    python run.py --port 8000 --open
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn


def bootstrap_directories(root: Path) -> None:
    """Ensure all required enterprise directories exist prior to server startup."""
    required = [
        root / "models",
        root / "samples" / "images",
        root / "samples" / "videos",
        root / "outputs",
        root / "outputs" / "uploads",
    ]
    for folder in required:
        folder.mkdir(parents=True, exist_ok=True)


def print_banner(root: Path, host: str, port: int) -> None:
    best_pt = root / "models" / "best.pt"
    model_hint = (
        "PRODUCTION (models/best.pt detected)"
        if best_pt.exists()
        else "DEMO FALLBACK (yolo11n.pt — drop models/best.pt anytime!)"
    )
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"

    banner = f"""
================================================================================
  PASARVISION AI — Traditional Market Socio-Economic Analytics Platform
  Engine   : YOLOv11 (Ultralytics) + ByteTrack + OpenCV Enhancement Suite
  Model    : {model_hint}
  Dashboard: {url}  (or http://localhost:{port})
================================================================================
  -> Open {url} in Chrome / Edge to launch the Carbon Enterprise GUI.
  -> Press CTRL+C to stop the server.
================================================================================
"""
    print(banner, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PasarVision AI Local Platform")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for dev")
    parser.add_argument("--open", action="store_true", help="Automatically open default browser")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    bootstrap_directories(project_root)
    print_banner(project_root, args.host, args.port)

    if args.open:
        target_url = f"http://127.0.0.1:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(target_url)).start()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

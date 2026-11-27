"""
PasarVision AI — FastAPI Application & Real-Time Streaming Backend (`backend/main.py`).

Provides:
- Static Carbon Enterprise Dashboard hosting (`/` and `/static`)
- Plug-and-Play YOLOv11 Model Status, Hot-Swap `.pt` Upload & Switcher
- Live Enhancement, Threshold, Class Filter & Interactive ROI Polygon APIs
- Image & Multi-Image Batch Detection Endpoints
- Synchronized WebSocket Video Stream (`/ws/video`) + MJPEG Stream (`/api/video/mjpeg`)
  + Annotated `.mp4` Exporter
- Before vs. After Enhancement Lab Engine (Images & Videos)
- 4-Sheet Enterprise `.xlsx` Report Generator & Archive Downloader
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.comparator import (
    BeforeAfterComparator,
    compute_delta_pct,
    encode_bgr_to_data_url,
)
from backend.enhancement import (
    EnhancementConfig,
    apply_enhancement,
    describe_enhancement,
)
from backend.excel_exporter import generate_enterprise_excel
from backend.model_manager import ModelManager
from backend.sample_seeder import IMAGE_EXTS, VIDEO_EXTS, ensure_sample_media
from backend.tracker_region import DetectionItem, RenderOptions, TrackerRegionEngine


# ------------------------------------------------------------------------------
# Directory Bootstrap
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_IMAGES_DIR = SAMPLES_DIR / "images"
SAMPLES_VIDEOS_DIR = SAMPLES_DIR / "videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for folder in (
    MODELS_DIR,
    SAMPLES_IMAGES_DIR,
    SAMPLES_VIDEOS_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

# Seed out-of-the-box demo samples if empty
ensure_sample_media(SAMPLES_DIR)

# ------------------------------------------------------------------------------
# Core Singleton Engines
# ------------------------------------------------------------------------------
model_manager = ModelManager(models_dir=MODELS_DIR, project_root=PROJECT_ROOT)
tracker_engine = TrackerRegionEngine(model_manager=model_manager)
comparator_engine = BeforeAfterComparator(
    model_manager=model_manager,
    tracker_engine=tracker_engine,
)


class AppRuntimeState:
    """Holds live enhancement settings and the latest export-ready analytics snapshot."""

    def __init__(self) -> None:
        self.enhancement: EnhancementConfig = EnhancementConfig(
            method="none",
            clahe_clip_limit=2.5,
            clahe_tile_grid=8,
            cs_low_percentile=2.0,
            cs_high_percentile=98.0,
            brightness=18,
            contrast_alpha=1.15,
            gamma=1.25,
        )
        self.latest_report_data: Optional[Dict[str, Any]] = None
        self.latest_active_source: Optional[str] = None


runtime_state = AppRuntimeState()

# ------------------------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------------------------
app = FastAPI(
    title="PasarVision AI — Traditional Market Socio-Economic Analytics Platform",
    version="1.0.0",
    description="Enterprise YOLOv11 + ByteTrack + OpenCV Enhancement & Analytics Suite",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


# ------------------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------------------
class SwitchModelRequest(BaseModel):
    model_name: str


class UpdateSettingsRequest(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    enabled_class_ids: Optional[List[int]] = None
    all_classes_enabled: Optional[bool] = None
    enhancement: Optional[Dict[str, Any]] = None
    render_options: Optional[Dict[str, Any]] = None


class UpdateROIRequest(BaseModel):
    points: List[List[float]] = Field(..., min_length=3)


class DetectImageRequest(BaseModel):
    source_path: str


class DetectBatchRequest(BaseModel):
    source_paths: List[str]


class CompareRequest(BaseModel):
    source_path: Optional[str] = None
    source_paths: Optional[List[str]] = None
    sample_every_n: int = 10
    enhancement_override: Optional[Dict[str, Any]] = None


class VideoExportRequest(BaseModel):
    source_path: str


class ExcelExportRequest(BaseModel):
    source_path: Optional[str] = None
    prefer_mode: Optional[str] = None


# ------------------------------------------------------------------------------
# Helpers: Path Resolution & Media Listing
# ------------------------------------------------------------------------------
def resolve_safe_media_path(rel_or_name: str) -> Path:
    """Resolve a media path safely inside `samples/` or `outputs/`."""
    if not rel_or_name:
        raise HTTPException(status_code=400, detail="Media source path is required.")

    clean_rel = rel_or_name.lstrip("/\\")
    candidate = (PROJECT_ROOT / clean_rel).resolve()

    allowed_roots = (SAMPLES_DIR.resolve(), OUTPUTS_DIR.resolve())
    if not any(
        str(candidate).startswith(str(root)) for root in allowed_roots
    ):
        raise HTTPException(status_code=403, detail="Access outside media directories is denied.")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Media file not found: {rel_or_name}")
    return candidate


def list_media_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Scan `samples/images`, `samples/videos`, and `outputs/uploads`."""
    images: List[Dict[str, Any]] = []
    videos: List[Dict[str, Any]] = []

    def _scan_folder(folder: Path, origin: str) -> None:
        if not folder.exists():
            return
        for f in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if not f.is_file() or f.name.startswith("."):
                continue
            ext = f.suffix.lower()
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            size_kb = round(f.stat().st_size / 1024.0, 1)
            entry = {
                "name": f.name,
                "path": rel,
                "url": f"/{rel}",
                "origin": origin,
                "size_kb": size_kb,
            }
            if ext in IMAGE_EXTS:
                images.append(entry)
            elif ext in VIDEO_EXTS:
                videos.append(entry)

    _scan_folder(SAMPLES_IMAGES_DIR, "sample")
    _scan_folder(SAMPLES_VIDEOS_DIR, "sample")
    _scan_folder(UPLOADS_DIR, "uploaded")

    return {"images": images, "videos": videos}


def list_exported_artifacts() -> List[Dict[str, Any]]:
    """List exported `.xlsx` reports and annotated `.mp4`/`.jpg` media in `outputs/`."""
    items: List[Dict[str, Any]] = []
    if not OUTPUTS_DIR.exists():
        return items

    files = sorted(
        (p for p in OUTPUTS_DIR.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for f in files[:40]:
        ext = f.suffix.lower()
        if ext not in {".xlsx", ".mp4", ".jpg", ".png"}:
            continue
        items.append(
            {
                "filename": f.name,
                "kind": "excel" if ext == ".xlsx" else ("video" if ext == ".mp4" else "image"),
                "size_kb": round(f.stat().st_size / 1024.0, 1),
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "download_url": f"/api/download/{f.name}",
            }
        )
    return items


def build_status_payload() -> Dict[str, Any]:
    return {
        "model": model_manager.get_status(),
        "enhancement": {
            **runtime_state.enhancement.to_dict(),
            "description": describe_enhancement(runtime_state.enhancement),
        },
        "render_options": tracker_engine.render_options.to_dict(),
        "roi_polygon": tracker_engine.get_roi_polygon(),
        "samples": list_media_catalog(),
        "reports": list_exported_artifacts(),
        "has_cached_report": runtime_state.latest_report_data is not None,
    }


# ------------------------------------------------------------------------------
# Routes: Frontend & Status
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>PasarVision AI Frontend Not Found</h1>", status_code=500)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    return build_status_payload()


@app.get("/api/model/status")
async def get_model_status() -> Dict[str, Any]:
    """Single source of truth endpoint exposing dynamic `model.names` and active weights."""
    return model_manager.get_status()


@app.get("/api/samples")
async def get_samples() -> Dict[str, Any]:
    return list_media_catalog()


@app.get("/api/reports")
async def get_reports() -> Dict[str, Any]:
    return {"reports": list_exported_artifacts()}


@app.get("/api/download/{filename}")
async def download_artifact(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    target = (OUTPUTS_DIR / safe_name).resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if target.suffix.lower() == ".xlsx"
        else "application/octet-stream"
    )
    return FileResponse(path=str(target), filename=safe_name, media_type=media_type)


# ------------------------------------------------------------------------------
# Routes: YOLOv11 Model Manager & Live Settings
# ------------------------------------------------------------------------------
@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        model_manager.upload_and_swap_model(file.filename or "best.pt", content)
        tracker_engine.reset_session()
        runtime_state.latest_report_data = None
        return build_status_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model/switch")
async def switch_model(req: SwitchModelRequest) -> Dict[str, Any]:
    try:
        model_manager.switch_model(req.model_name)
        tracker_engine.reset_session()
        runtime_state.latest_report_data = None
        return build_status_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/settings")
async def update_settings(req: UpdateSettingsRequest) -> Dict[str, Any]:
    if (
        req.conf_threshold is not None
        or req.iou_threshold is not None
        or req.enabled_class_ids is not None
        or req.all_classes_enabled is not None
    ):
        model_manager.update_inference_settings(
            conf_threshold=req.conf_threshold,
            iou_threshold=req.iou_threshold,
            enabled_class_ids=req.enabled_class_ids,
            all_classes_enabled=req.all_classes_enabled,
        )
        runtime_state.latest_report_data = None

    if req.enhancement is not None:
        merged = {**runtime_state.enhancement.to_dict(), **req.enhancement}
        runtime_state.enhancement = EnhancementConfig.from_dict(merged)
        runtime_state.latest_report_data = None

    if req.render_options is not None:
        merged_opts = {
            **tracker_engine.render_options.to_dict(),
            **req.render_options,
        }
        tracker_engine.render_options = RenderOptions.from_dict(merged_opts)

    return build_status_payload()


# ------------------------------------------------------------------------------
# Routes: Interactive ROI Polygon
# ------------------------------------------------------------------------------
@app.get("/api/roi")
async def get_roi() -> Dict[str, Any]:
    return {"points": tracker_engine.get_roi_polygon()}


@app.post("/api/roi")
async def update_roi(req: UpdateROIRequest) -> Dict[str, Any]:
    try:
        pts = tracker_engine.set_roi_polygon(req.points)
        runtime_state.latest_report_data = None
        return {"points": pts}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/roi/reset")
async def reset_roi() -> Dict[str, Any]:
    pts = tracker_engine.reset_roi_polygon()
    runtime_state.latest_report_data = None
    return {"points": pts}


# ------------------------------------------------------------------------------
# Routes: Media Upload & Image / Batch Detection
# ------------------------------------------------------------------------------
@app.post("/api/upload/media")
async def upload_media(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No media files uploaded.")

    saved_items: List[Dict[str, Any]] = []
    allowed_exts = IMAGE_EXTS | VIDEO_EXTS

    for uf in files:
        raw_name = Path(uf.filename or "upload.jpg").name
        ext = Path(raw_name).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}",
            )
        stem = Path(raw_name).stem
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:50]
        target = UPLOADS_DIR / f"{safe_stem}{ext}"
        data = await uf.read()
        if not data:
            continue
        target.write_bytes(data)
        rel = target.relative_to(PROJECT_ROOT).as_posix()
        saved_items.append(
            {
                "name": target.name,
                "path": rel,
                "url": f"/{rel}",
                "media_type": "image" if ext in IMAGE_EXTS else "video",
                "size_kb": round(len(data) / 1024.0, 1),
            }
        )

    return {
        "uploaded": saved_items,
        "samples": list_media_catalog(),
    }


@app.post("/api/detect/image")
async def detect_single_image(req: DetectImageRequest) -> Dict[str, Any]:
    img_path = resolve_safe_media_path(req.source_path)
    raw_bgr = cv2.imread(str(img_path))
    if raw_bgr is None:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image: {img_path.name}")

    enh_cfg = runtime_state.enhancement.normalized()
    enhanced_bgr = apply_enhancement(raw_bgr, enh_cfg)

    # Active frame detection
    active_res = tracker_engine.process_frame(
        enhanced_bgr,
        use_tracking=False,
        timestamp_sec=0.0,
        frame_idx=1,
    )

    # Fast baseline check so Excel Export & telemetry delta are always populated
    if enh_cfg.method != "none":
        raw_res = tracker_engine.process_frame(
            raw_bgr,
            use_tracking=False,
            render_override=RenderOptions(
                show_boxes=False,
                show_track_ids=False,
                show_trails=False,
                show_roi=False,
                show_hud_badge=False,
            ),
        )
    else:
        raw_res = active_res

    # Build cached report payload for instant 1-click Excel export
    b_confs_cls: Dict[str, List[float]] = defaultdict(list)
    a_confs_cls: Dict[str, List[float]] = defaultdict(list)
    for d in raw_res["detections"]:
        b_confs_cls[d["class_name"]].append(float(d["confidence"]))
    for d in active_res["detections"]:
        a_confs_cls[d["class_name"]].append(float(d["confidence"]))

    per_cls = comparator_engine._build_per_class_table(
        before_counts=raw_res["current_counts_by_class"],
        after_counts=active_res["current_counts_by_class"],
        before_confs=b_confs_cls,
        after_confs=a_confs_cls,
    )

    b_tot = int(raw_res["total_detections"])
    a_tot = int(active_res["total_detections"])
    b_avg = float(raw_res["avg_confidence"])
    a_avg = float(active_res["avg_confidence"])

    runtime_state.latest_active_source = req.source_path
    runtime_state.latest_report_data = {
        "mode": "Image Socio-Economic Detection",
        "source_type": "image",
        "source_name": img_path.name,
        "summary": {
            "model_name": model_manager.active_model_name,
            "model_badge": model_manager.get_badge_text(),
            "device_label": model_manager.device_label,
            "enhancement_method": enh_cfg.method,
            "enhancement_description": describe_enhancement(enh_cfg),
            "conf_threshold": model_manager.conf_threshold,
            "iou_threshold": model_manager.iou_threshold,
            "roi_polygon": tracker_engine.get_roi_polygon(),
            "before_total": b_tot,
            "after_total": a_tot,
            "delta_count": a_tot - b_tot,
            "delta_pct": compute_delta_pct(b_tot, a_tot),
            "before_avg_conf": round(b_avg, 4),
            "after_avg_conf": round(a_avg, 4),
            "delta_conf": round(a_avg - b_avg, 4),
            "before_inside_roi": int(raw_res["inside_roi_now"]),
            "after_inside_roi": int(active_res["inside_roi_now"]),
            "unique_tracked_total": a_tot,
            "region_unique_entered": int(active_res["inside_roi_now"]),
            "sampled_frames": 1,
        },
        "per_class_table": per_cls,
        "frame_detail_log": [
            {
                "frame_or_file": img_path.name,
                "timestamp_sec": 0.0,
                "before_detections": b_tot,
                "after_detections": a_tot,
                "before_avg_conf": round(b_avg, 4),
                "after_avg_conf": round(a_avg, 4),
                "inside_roi_count": int(active_res["inside_roi_now"]),
            }
        ],
        "region_counter_log": active_res["region_counter_log"],
    }

    annotated_b64 = encode_bgr_to_data_url(active_res["annotated_frame"], quality=90)

    return {
        "source_name": img_path.name,
        "source_path": req.source_path,
        "annotated_image": annotated_b64,
        "width": active_res["width"],
        "height": active_res["height"],
        "latency_ms": active_res["latency_ms"],
        "enhancement_description": describe_enhancement(enh_cfg),
        "telemetry": {
            "total_detections": a_tot,
            "unique_tracked_total": a_tot,
            "inside_roi_now": active_res["inside_roi_now"],
            "inside_roi_by_class": active_res["inside_roi_by_class"],
            "region_unique_entered": active_res["region_unique_entered"],
            "region_entered_by_class": active_res["region_entered_by_class"],
            "avg_confidence": active_res["avg_confidence"],
            "conf_delta": round(a_avg - b_avg, 4),
            "delta_count": a_tot - b_tot,
            "current_counts_by_class": active_res["current_counts_by_class"],
            "class_colors": active_res["class_colors"],
            "region_counter_log": active_res["region_counter_log"],
        },
        "per_class_table": per_cls,
        "frame_detail_log": runtime_state.latest_report_data["frame_detail_log"],
    }


@app.post("/api/detect/batch")
async def detect_batch_images(req: DetectBatchRequest) -> Dict[str, Any]:
    if not req.source_paths:
        raise HTTPException(status_code=400, detail="Provide at least one image path.")
    paths = [resolve_safe_media_path(p) for p in req.source_paths]
    enh_cfg = runtime_state.enhancement.normalized()
    batch_res = await asyncio.to_thread(
        comparator_engine.compare_image_batch,
        paths,
        enh_cfg,
    )
    runtime_state.latest_report_data = copy.deepcopy(batch_res)
    return batch_res


# ------------------------------------------------------------------------------
# Routes: Before vs. After Comparison Lab
# ------------------------------------------------------------------------------
@app.post("/api/compare")
async def run_before_after_comparison(req: CompareRequest) -> Dict[str, Any]:
    enh_cfg = (
        EnhancementConfig.from_dict(req.enhancement_override)
        if req.enhancement_override is not None
        else runtime_state.enhancement.normalized()
    )
    # If user is in Before vs. After Lab and left enhancement on "none", auto-preview CLAHE
    # only if explicitly requested; otherwise respect their exact selection (frontend warns/assists).

    if req.source_paths and len(req.source_paths) > 1:
        paths = [resolve_safe_media_path(p) for p in req.source_paths]
        result = await asyncio.to_thread(
            comparator_engine.compare_image_batch,
            paths,
            enh_cfg,
        )
        runtime_state.latest_report_data = copy.deepcopy(result)
        return result

    single_rel = req.source_path or (req.source_paths[0] if req.source_paths else None)
    if not single_rel:
        raise HTTPException(status_code=400, detail="Select an image or video to compare.")

    media_path = resolve_safe_media_path(single_rel)
    ext = media_path.suffix.lower()

    if ext in IMAGE_EXTS:
        bgr = cv2.imread(str(media_path))
        if bgr is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {media_path.name}")
        result = await asyncio.to_thread(
            comparator_engine.compare_image,
            bgr,
            media_path.name,
            enh_cfg,
            True,
        )
    elif ext in VIDEO_EXTS:
        result = await asyncio.to_thread(
            comparator_engine.compare_video,
            media_path,
            enh_cfg,
            req.sample_every_n,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported media extension: {ext}")

    runtime_state.latest_active_source = single_rel
    runtime_state.latest_report_data = copy.deepcopy(result)
    return result


# ------------------------------------------------------------------------------
# Routes: Real-Time Video Streaming (WebSocket + MJPEG + Annotated MP4 Export)
# ------------------------------------------------------------------------------
@app.websocket("/ws/video")
async def websocket_video_stream(
    websocket: WebSocket,
    source: str = Query(...),
    binary: int = Query(0),
) -> None:
    await websocket.accept()

    try:
        video_path = resolve_safe_media_path(source)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "message": exc.detail})
        await websocket.close()
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json(
            {"type": "error", "message": f"Cannot open video stream: {video_path.name}"}
        )
        await websocket.close()
        return

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if src_fps <= 0.5 or src_fps > 120.0:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = round(total_frames / src_fps, 2) if total_frames > 0 else 0.0

    stream_tracker = TrackerRegionEngine(model_manager=model_manager)
    stream_tracker.reset_session()

    paused = False
    stopped = False
    seek_target: Optional[int] = None
    reset_requested = False

    frame_detail_log: List[Dict[str, Any]] = []
    agg_b_counts: Dict[str, int] = defaultdict(int)
    agg_a_counts: Dict[str, int] = defaultdict(int)
    agg_b_confs: Dict[str, List[float]] = defaultdict(list)
    agg_a_confs: Dict[str, List[float]] = defaultdict(list)
    all_b_confs: List[float] = []
    all_a_confs: List[float] = []

    async def _listen_commands() -> None:
        nonlocal paused, stopped, seek_target, reset_requested
        try:
            while not stopped:
                msg = await websocket.receive_json()
                action = str(msg.get("action", "")).lower()
                if action == "pause":
                    paused = True
                elif action in ("resume", "play"):
                    paused = False
                elif action == "stop":
                    stopped = True
                    break
                elif action == "seek":
                    seek_target = max(0, int(msg.get("frame", 0)))
                elif action == "restart":
                    seek_target = 0
                    reset_requested = True
                    paused = False
        except WebSocketDisconnect:
            stopped = True
        except Exception:
            stopped = True

    listener_task = asyncio.create_task(_listen_commands())

    def _sync_live_report_snapshot(last_inside_roi: int) -> None:
        enh_cfg = runtime_state.enhancement.normalized()
        b_tot = sum(agg_b_counts.values())
        a_tot = sum(agg_a_counts.values())
        b_avg = float(np.mean(all_b_confs)) if all_b_confs else 0.0
        a_avg = float(np.mean(all_a_confs)) if all_a_confs else 0.0
        per_cls = comparator_engine._build_per_class_table(
            before_counts=agg_b_counts,
            after_counts=agg_a_counts,
            before_confs=agg_b_confs,
            after_confs=agg_a_confs,
        )
        runtime_state.latest_active_source = source
        runtime_state.latest_report_data = {
            "mode": "Live Video Tracking & Region Analytics",
            "source_type": "video",
            "source_name": video_path.name,
            "source_rel": source,
            "has_true_baseline": enh_cfg.method == "none",
            "summary": {
                "model_name": model_manager.active_model_name,
                "model_badge": model_manager.get_badge_text(),
                "device_label": model_manager.device_label,
                "enhancement_method": enh_cfg.method,
                "enhancement_description": describe_enhancement(enh_cfg),
                "conf_threshold": model_manager.conf_threshold,
                "iou_threshold": model_manager.iou_threshold,
                "roi_polygon": tracker_engine.get_roi_polygon(),
                "before_total": b_tot,
                "after_total": a_tot,
                "delta_count": a_tot - b_tot,
                "delta_pct": compute_delta_pct(b_tot, a_tot),
                "before_avg_conf": round(b_avg, 4),
                "after_avg_conf": round(a_avg, 4),
                "delta_conf": round(a_avg - b_avg, 4),
                "before_inside_roi": stream_tracker.peak_inside_roi,
                "after_inside_roi": max(stream_tracker.peak_inside_roi, last_inside_roi),
                "unique_tracked_total": len(stream_tracker.unique_ids_all),
                "region_unique_entered": len(stream_tracker.region_entered_ids),
                "sampled_frames": len(frame_detail_log),
            },
            "per_class_table": per_cls,
            "frame_detail_log": list(frame_detail_log),
            "region_counter_log": list(stream_tracker.region_counter_log),
        }

    def _downscale_stream_frame(bgr: np.ndarray, max_w: int = 854) -> np.ndarray:
        h, w = bgr.shape[:2]
        if w <= max_w:
            return bgr
        scale = max_w / float(w)
        nw = max_w
        nh = max(2, int(round(h * scale)) & ~1)
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # ==========================================================================
    # HIGH-SPEED 55–60 FPS ASYNC GPU + BINARY STREAMING PIPELINE (`binary=1`)
    # ==========================================================================
    if binary == 1:
        state_lock = threading.Lock()
        frame_event = threading.Event()
        pending_slot: Dict[str, Any] = {"frame": None, "idx": -1, "ts": 0.0}
        shared_det: Dict[str, Any] = {
            "items": [],           # List[Dict] raw detection dicts
            "velocities": {},      # track_id -> (vx_px_s, vy_px_s)
            "prev_centers": {},    # track_id -> (cx, cy, wall_t)
            "det_wall_t": time.perf_counter(),
            "telemetry": {
                "total_detections": 0,
                "unique_tracked_total": 0,
                "inside_roi_now": 0,
                "inside_roi_by_class": {},
                "region_unique_entered": 0,
                "avg_confidence": 0.0,
                "current_counts_by_class": {},
                "class_colors": {},
                "region_counter_log": [],
            },
        }

        no_draw_opts = RenderOptions(
            show_boxes=False,
            show_track_ids=False,
            show_trails=False,
            show_roi=False,
            show_hud_badge=False,
        )

        def _gpu_detector_worker() -> None:
            last_processed_idx = -1
            while not stopped:
                frame_event.wait(timeout=0.05)
                frame_event.clear()
                if stopped:
                    break
                with state_lock:
                    work_bgr = pending_slot["frame"]
                    f_idx = pending_slot["idx"]
                    f_ts = pending_slot["ts"]
                if work_bgr is None or f_idx == last_processed_idx:
                    continue
                last_processed_idx = f_idx

                try:
                    stream_tracker.roi_points = list(tracker_engine.roi_points)
                    res = stream_tracker.process_frame(
                        work_bgr,
                        use_tracking=True,
                        timestamp_sec=f_ts,
                        frame_idx=f_idx,
                        render_override=no_draw_opts,
                    )
                    now_t = time.perf_counter()
                    with state_lock:
                        prev_c = shared_det["prev_centers"]
                        vels = shared_det["velocities"]
                        next_c = {}
                        for d in res["detections"]:
                            tid = d["track_id"]
                            if tid is not None:
                                bx1, by1, bx2, by2 = d["bbox"]
                                cx = (bx1 + bx2) * 0.5
                                cy = (by1 + by2) * 0.5
                                if tid in prev_c:
                                    px, py, pt = prev_c[tid]
                                    dt_v = max(0.02, now_t - pt)
                                    vx = (cx - px) / dt_v
                                    vy = (cy - py) / dt_v
                                    old_vx, old_vy = vels.get(tid, (0.0, 0.0))
                                    vels[tid] = (0.5 * old_vx + 0.5 * vx, 0.5 * old_vy + 0.5 * vy)
                                next_c[tid] = (cx, cy, now_t)

                        shared_det["prev_centers"] = next_c
                        shared_det["items"] = res["detections"]
                        shared_det["det_wall_t"] = now_t
                        shared_det["telemetry"] = {
                            "total_detections": res["total_detections"],
                            "unique_tracked_total": res["unique_tracked_total"],
                            "inside_roi_now": res["inside_roi_now"],
                            "inside_roi_by_class": res["inside_roi_by_class"],
                            "region_unique_entered": res["region_unique_entered"],
                            "avg_confidence": res["avg_confidence"],
                            "current_counts_by_class": res["current_counts_by_class"],
                            "class_colors": res["class_colors"],
                            "region_counter_log": res["region_counter_log"][-40:],
                        }

                        if len(frame_detail_log) == 0 or (f_idx - frame_detail_log[-1].get("_raw_idx", -99)) >= 15:
                            for d in res["detections"]:
                                agg_a_counts[d["class_name"]] += 1
                                agg_a_confs[d["class_name"]].append(float(d["confidence"]))
                                all_a_confs.append(float(d["confidence"]))
                                agg_b_counts[d["class_name"]] += 1
                                agg_b_confs[d["class_name"]].append(float(d["confidence"]))
                                all_b_confs.append(float(d["confidence"]))
                            frame_detail_log.append(
                                {
                                    "_raw_idx": f_idx,
                                    "frame_or_file": f"Frame #{f_idx}",
                                    "timestamp_sec": f_ts,
                                    "before_detections": int(res["total_detections"]),
                                    "after_detections": int(res["total_detections"]),
                                    "before_avg_conf": round(float(res["avg_confidence"]), 4),
                                    "after_avg_conf": round(float(res["avg_confidence"]), 4),
                                    "inside_roi_count": int(res["inside_roi_now"]),
                                }
                            )
                except Exception:
                    pass

        worker_thread = threading.Thread(target=_gpu_detector_worker, daemon=True)
        worker_thread.start()

        frame_idx = 0
        display_tick = 0
        target_display_fps = 58.0
        sub_ticks = max(1, min(3, int(round(target_display_fps / src_fps))))
        tick_budget = 1.0 / (src_fps * sub_ticks)
        fps_ema = 56.0
        last_tick_wall = time.perf_counter()

        try:
            while not stopped:
                if reset_requested:
                    with state_lock:
                        stream_tracker.reset_session()
                        shared_det["items"] = []
                        shared_det["velocities"].clear()
                        shared_det["prev_centers"].clear()
                        frame_detail_log.clear()
                        agg_b_counts.clear()
                        agg_a_counts.clear()
                        agg_b_confs.clear()
                        agg_a_confs.clear()
                        all_b_confs.clear()
                        all_a_confs.clear()
                    reset_requested = False

                if seek_target is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(seek_target))
                    frame_idx = seek_target
                    seek_target = None

                if paused:
                    await asyncio.sleep(0.04)
                    last_tick_wall = time.perf_counter()
                    continue

                ok, raw_bgr = cap.read()
                if not ok or raw_bgr is None:
                    _sync_live_report_snapshot(0)
                    await websocket.send_json(
                        {
                            "type": "ended",
                            "frame_idx": frame_idx,
                            "total_frames": total_frames,
                            "region_counter_log": list(stream_tracker.region_counter_log),
                            "frame_detail_log": list(frame_detail_log[-120:]),
                        }
                    )
                    paused = True
                    await asyncio.sleep(0.08)
                    continue

                ts_sec = round(frame_idx / src_fps, 2)
                enh_cfg = runtime_state.enhancement.normalized()

                # Downscale & enhance once per decoded source frame (~2ms total)
                scaled_bgr = _downscale_stream_frame(raw_bgr, max_w=854)
                enh_bgr = (
                    apply_enhancement(scaled_bgr, enh_cfg)
                    if enh_cfg.method != "none"
                    else scaled_bgr
                )

                # Feed latest frame non-blockingly to Intel Iris Xe GPU worker
                with state_lock:
                    pending_slot["frame"] = enh_bgr
                    pending_slot["idx"] = frame_idx
                    pending_slot["ts"] = ts_sec
                frame_event.set()

                fh, fw = enh_bgr.shape[:2]
                active_roi = list(tracker_engine.roi_points)
                poly_px = np.array(
                    [[int(round(px * fw)), int(round(py * fh))] for px, py in active_roi],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cur_opts = copy.copy(tracker_engine.render_options)

                # Render 1..sub_ticks smooth 60Hz interpolated frames
                for _ in range(sub_ticks):
                    if stopped or paused or seek_target is not None:
                        break
                    render_t0 = time.perf_counter()

                    with state_lock:
                        base_items = list(shared_det["items"])
                        vels = dict(shared_det["velocities"])
                        det_t0 = shared_det["det_wall_t"]
                        tel_snap = dict(shared_det["telemetry"])
                        uniq_entered = len(stream_tracker.region_entered_ids)

                    dt_interp = min(0.12, max(0.0, render_t0 - det_t0))
                    interp_dets: List[DetectionItem] = []
                    inside_now = 0

                    for d in base_items:
                        tid = d["track_id"]
                        vx, vy = vels.get(tid, (0.0, 0.0)) if tid is not None else (0.0, 0.0)
                        dx = int(round(max(-26.0, min(26.0, vx * dt_interp))))
                        dy = int(round(max(-26.0, min(26.0, vy * dt_interp))))
                        x1 = max(0, min(fw - 1, d["bbox"][0] + dx))
                        y1 = max(0, min(fh - 1, d["bbox"][1] + dy))
                        x2 = max(0, min(fw - 1, d["bbox"][2] + dx))
                        y2 = max(0, min(fh - 1, d["bbox"][3] + dy))
                        cx = int(round((x1 + x2) * 0.5))
                        in_roi = cv2.pointPolygonTest(poly_px, (float(cx), float(y2)), False) >= 0.0
                        if in_roi:
                            inside_now += 1
                        c_hex = d["color_hex"]
                        cid = int(d["class_id"])
                        c_bgr = model_manager.class_colors_bgr.get(cid, (212, 182, 6))
                        interp_dets.append(
                            DetectionItem(
                                class_id=cid,
                                class_name=d["class_name"],
                                confidence=float(d["confidence"]),
                                bbox=(x1, y1, x2, y2),
                                anchor=(cx, y2),
                                track_id=tid,
                                inside_roi=in_roi,
                                color_hex=c_hex,
                                color_bgr=c_bgr,
                            )
                        )

                    annotated = stream_tracker._render_frame(
                        frame_bgr=enh_bgr,
                        detections=interp_dets,
                        poly_px=poly_px,
                        inside_now_total=inside_now,
                        unique_entered_total=uniq_entered,
                        use_tracking=True,
                        opts=cur_opts,
                    )
                    ok_jpg, jpg_buf = cv2.imencode(
                        ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 74]
                    )
                    render_ms = (time.perf_counter() - render_t0) * 1000.0

                    if ok_jpg:
                        await websocket.send_bytes(jpg_buf.tobytes())

                    # Pace smoothly to ~58–60 FPS wall-clock
                    now_after = time.perf_counter()
                    work_dur = now_after - last_tick_wall
                    if work_dur < tick_budget:
                        await asyncio.sleep(tick_budget - work_dur)
                    tick_end = time.perf_counter()
                    real_dt = max(0.004, tick_end - last_tick_wall)
                    last_tick_wall = tick_end
                    fps_ema = 0.85 * fps_ema + 0.15 * min(62.0, 1.0 / real_dt)

                    display_tick += 1
                    if display_tick % 8 == 0:
                        tel_snap["inside_roi_now"] = inside_now
                        await websocket.send_json(
                            {
                                "type": "hud",
                                "fps": round(fps_ema, 1),
                                "latency_ms": round(render_ms, 1),
                                "frame_idx": frame_idx + 1,
                                "total_frames": total_frames,
                                "telemetry": tel_snap,
                            }
                        )

                if frame_idx % 25 == 0:
                    _sync_live_report_snapshot(0)
                frame_idx += 1

        except WebSocketDisconnect:
            pass
        finally:
            stopped = True
            frame_event.set()
            listener_task.cancel()
            cap.release()
            if frame_detail_log:
                _sync_live_report_snapshot(0)
        return

    # ==========================================================================
    # LEGACY JSON MODE (`binary=0`, preserved for automated JSON TestClient QA)
    # ==========================================================================
    frame_idx = 0
    fps_ema = src_fps

    try:
        while not stopped:
            if reset_requested:
                stream_tracker.reset_session()
                frame_detail_log.clear()
                agg_b_counts.clear()
                agg_a_counts.clear()
                agg_b_confs.clear()
                agg_a_confs.clear()
                all_b_confs.clear()
                all_a_confs.clear()
                reset_requested = False

            if seek_target is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(seek_target))
                frame_idx = seek_target
                seek_target = None

            if paused:
                await asyncio.sleep(0.06)
                continue

            loop_t0 = time.perf_counter()
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                _sync_live_report_snapshot(0)
                await websocket.send_json(
                    {
                        "type": "ended",
                        "frame_idx": frame_idx,
                        "total_frames": total_frames,
                        "region_counter_log": list(stream_tracker.region_counter_log),
                        "frame_detail_log": list(frame_detail_log[-120:]),
                    }
                )
                paused = True
                await asyncio.sleep(0.1)
                continue

            ts_sec = round(frame_idx / src_fps, 2)
            enh_cfg = runtime_state.enhancement.normalized()
            stream_tracker.roi_points = list(tracker_engine.roi_points)
            stream_tracker.render_options = copy.copy(tracker_engine.render_options)

            def _process_step() -> Dict[str, Any]:
                enh_frame = (
                    apply_enhancement(raw_frame, enh_cfg)
                    if enh_cfg.method != "none"
                    else raw_frame
                )
                res = stream_tracker.process_frame(
                    enh_frame,
                    use_tracking=True,
                    timestamp_sec=ts_sec,
                    frame_idx=frame_idx,
                )
                # Lightweight log sampling (zero extra YOLO inference in live stream!)
                if frame_idx % 10 == 0:
                    for d in res["detections"]:
                        agg_a_counts[d["class_name"]] += 1
                        agg_a_confs[d["class_name"]].append(float(d["confidence"]))
                        all_a_confs.append(float(d["confidence"]))
                        agg_b_counts[d["class_name"]] += 1
                        agg_b_confs[d["class_name"]].append(float(d["confidence"]))
                        all_b_confs.append(float(d["confidence"]))

                    frame_detail_log.append(
                        {
                            "frame_or_file": f"Frame #{frame_idx}",
                            "timestamp_sec": ts_sec,
                            "before_detections": int(res["total_detections"]),
                            "after_detections": int(res["total_detections"]),
                            "before_avg_conf": round(float(res["avg_confidence"]), 4),
                            "after_avg_conf": round(float(res["avg_confidence"]), 4),
                            "inside_roi_count": int(res["inside_roi_now"]),
                        }
                    )
                res["encoded_jpg"] = encode_bgr_to_data_url(
                    res["annotated_frame"], quality=76
                )
                return res

            step_res = await asyncio.to_thread(_process_step)

            elapsed = max(0.001, time.perf_counter() - loop_t0)
            target_interval = 1.0 / src_fps
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)
            elif elapsed > target_interval * 1.65:
                # Real-time catchup: skip 1 raw decode step so playback never lags behind
                if cap.grab():
                    frame_idx += 1
            total_step = max(0.001, time.perf_counter() - loop_t0)
            inst_fps = 1.0 / total_step
            fps_ema = 0.75 * fps_ema + 0.25 * inst_fps

            if frame_idx % 12 == 0:
                _sync_live_report_snapshot(int(step_res["inside_roi_now"]))

            await websocket.send_json(
                {
                    "type": "frame",
                    "image": step_res["encoded_jpg"],
                    "frame_idx": frame_idx + 1,
                    "total_frames": total_frames,
                    "timestamp_sec": ts_sec,
                    "duration_sec": duration_sec,
                    "fps": round(fps_ema, 1),
                    "latency_ms": step_res["latency_ms"],
                    "width": step_res["width"],
                    "height": step_res["height"],
                    "enhancement_description": describe_enhancement(enh_cfg),
                    "telemetry": {
                        "total_detections": step_res["total_detections"],
                        "unique_tracked_total": step_res["unique_tracked_total"],
                        "unique_tracked_by_class": step_res["unique_tracked_by_class"],
                        "inside_roi_now": step_res["inside_roi_now"],
                        "inside_roi_by_class": step_res["inside_roi_by_class"],
                        "region_unique_entered": step_res["region_unique_entered"],
                        "region_entered_by_class": step_res["region_entered_by_class"],
                        "avg_confidence": step_res["avg_confidence"],
                        "current_counts_by_class": step_res["current_counts_by_class"],
                        "class_colors": step_res["class_colors"],
                        "region_counter_log": step_res["region_counter_log"][-40:],
                    },
                }
            )
            frame_idx += 1

    except WebSocketDisconnect:
        pass
    finally:
        stopped = True
        listener_task.cancel()
        cap.release()
        if frame_detail_log:
            _sync_live_report_snapshot(0)


@app.get("/api/video/mjpeg")
async def stream_video_mjpeg(source: str = Query(...)) -> StreamingResponse:
    """Standards-compliant multipart MJPEG endpoint for direct browser/VLC embedding."""
    video_path = resolve_safe_media_path(source)

    def _mjpeg_generator():
        cap = cv2.VideoCapture(str(video_path))
        mjpeg_tracker = TrackerRegionEngine(model_manager=model_manager)
        mjpeg_tracker.reset_session()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                mjpeg_tracker.roi_points = list(tracker_engine.roi_points)
                mjpeg_tracker.render_options = copy.copy(tracker_engine.render_options)
                enh = apply_enhancement(frame, runtime_state.enhancement)
                res = mjpeg_tracker.process_frame(
                    enh,
                    use_tracking=True,
                    timestamp_sec=idx / max(1.0, fps),
                    frame_idx=idx,
                )
                ok_enc, buf = cv2.imencode(
                    ".jpg", res["annotated_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok_enc:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                    )
                idx += 1
        finally:
            cap.release()

    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/video/export")
async def export_annotated_mp4(req: VideoExportRequest) -> Dict[str, Any]:
    """Render and export a full annotated MP4 video with ByteTrack & ROI Counter HUD."""
    video_path = resolve_safe_media_path(req.source_path)

    def _export_worker() -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        if fps <= 0.5 or fps > 120.0:
            fps = 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 854)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"Annotated_{video_path.stem}_{ts}.mp4"
        out_file = OUTPUTS_DIR / out_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("OpenCV MP4 VideoWriter failed to initialize.")

        exp_tracker = TrackerRegionEngine(model_manager=model_manager)
        exp_tracker.roi_points = list(tracker_engine.roi_points)
        exp_tracker.render_options = copy.copy(tracker_engine.render_options)
        exp_tracker.reset_session()
        enh_cfg = runtime_state.enhancement.normalized()

        f_count = 0
        try:
            while True:
                ok, raw = cap.read()
                if not ok or raw is None:
                    break
                enh = apply_enhancement(raw, enh_cfg)
                res = exp_tracker.process_frame(
                    enh,
                    use_tracking=True,
                    timestamp_sec=f_count / fps,
                    frame_idx=f_count,
                )
                writer.write(res["annotated_frame"])
                f_count += 1
        finally:
            cap.release()
            writer.release()

        return {
            "filename": out_name,
            "download_url": f"/api/download/{out_name}",
            "frames_processed": f_count,
            "unique_tracked": len(exp_tracker.unique_ids_all),
            "region_unique_entered": len(exp_tracker.region_entered_ids),
        }

    try:
        result = await asyncio.to_thread(_export_worker)
        result["reports"] = list_exported_artifacts()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------------------
# Routes: Enterprise Excel Report Export (.xlsx)
# ------------------------------------------------------------------------------
@app.post("/api/export/excel")
async def export_excel_report(req: ExcelExportRequest) -> Dict[str, Any]:
    """
    Generate the 4-sheet Enterprise `.xlsx` workbook from the latest active
    session or run an on-demand comparison with real Before vs. After baselines.
    """
    report_payload = runtime_state.latest_report_data
    needs_true_baseline = (
        report_payload is not None
        and report_payload.get("has_true_baseline") is False
    )

    if report_payload is None or needs_true_baseline:
        prev_live = report_payload if needs_true_baseline else None
        target_rel = (
            req.source_path
            or (prev_live.get("source_rel") if prev_live else None)
            or runtime_state.latest_active_source
        )
        if not target_rel:
            catalog = list_media_catalog()
            if catalog["images"]:
                target_rel = catalog["images"][0]["path"]
            elif catalog["videos"]:
                target_rel = catalog["videos"][0]["path"]

        if not target_rel:
            raise HTTPException(
                status_code=400,
                detail="No analyzed session or sample media available to export.",
            )

        media_path = resolve_safe_media_path(target_rel)
        enh_cfg = runtime_state.enhancement.normalized()
        if media_path.suffix.lower() in IMAGE_EXTS:
            bgr = cv2.imread(str(media_path))
            if bgr is None:
                raise HTTPException(status_code=400, detail="Unreadable image source.")
            report_payload = await asyncio.to_thread(
                comparator_engine.compare_image,
                bgr,
                media_path.name,
                enh_cfg,
                False,
            )
        else:
            report_payload = await asyncio.to_thread(
                comparator_engine.compare_video,
                media_path,
                enh_cfg,
                10,
            )
            # Preserve exact full-rate live ByteTrack IDs & ROI log if available
            if prev_live and prev_live.get("region_counter_log"):
                report_payload["region_counter_log"] = prev_live["region_counter_log"]
                report_payload["summary"]["unique_tracked_total"] = max(
                    int(report_payload["summary"].get("unique_tracked_total", 0)),
                    int(prev_live["summary"].get("unique_tracked_total", 0)),
                )
                report_payload["summary"]["region_unique_entered"] = max(
                    int(report_payload["summary"].get("region_unique_entered", 0)),
                    int(prev_live["summary"].get("region_unique_entered", 0)),
                )
        runtime_state.latest_report_data = copy.deepcopy(report_payload)

    xlsx_path = await asyncio.to_thread(
        generate_enterprise_excel,
        report_payload,
        OUTPUTS_DIR,
    )

    return {
        "filename": xlsx_path.name,
        "download_url": f"/api/download/{xlsx_path.name}",
        "sheets": [
            "Executive_Summary",
            "Per_Class_Comparison",
            "Frame_Detail_Log",
            "Region_Counter_Log",
        ],
        "reports": list_exported_artifacts(),
    }

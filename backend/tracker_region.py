"""
Lightweight High-Speed ByteTrack & Interactive ROI Region Engine (`tracker_region.py`).

Optimizations:
- Probes `lap` C-extension availability ONCE at startup so CPU video tracking never
  wastes a duplicate neural network forward pass on fallback.
- Adaptive `imgsz` (`480` on CPU live video stream for ~2x higher FPS, `640` on static images).
- Clean, minimal visual overlays aligned with the Ethereal Bento UI aesthetic.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from backend.model_manager import ModelManager, hex_to_bgr, resolve_class_color_hex
from backend.onnx_yolo_engine import (
    ONNXBoxesResult,
    ONNXDetectionResult,
    PureByteTracker,
)

# Probe `lap` once at import time so we never throw/catch inside per-frame loops
LAP_AVAILABLE = False
try:
    import lap  # noqa: F401

    LAP_AVAILABLE = True
except (ImportError, OSError):
    LAP_AVAILABLE = False


DEFAULT_ROI_POLYGON: List[Tuple[float, float]] = [
    (0.20, 0.35),
    (0.80, 0.35),
    (0.88, 0.90),
    (0.12, 0.90),
]


@dataclass
class RenderOptions:
    """Visual overlay toggles controllable live from the UI."""

    show_boxes: bool = True
    show_track_ids: bool = True
    show_trails: bool = True
    show_roi: bool = True
    show_hud_badge: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RenderOptions":
        if not data:
            return cls()
        return cls(
            show_boxes=bool(data.get("show_boxes", True)),
            show_track_ids=bool(data.get("show_track_ids", True)),
            show_trails=bool(data.get("show_trails", True)),
            show_roi=bool(data.get("show_roi", True)),
            show_hud_badge=bool(data.get("show_hud_badge", True)),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "show_boxes": self.show_boxes,
            "show_track_ids": self.show_track_ids,
            "show_trails": self.show_trails,
            "show_roi": self.show_roi,
            "show_hud_badge": self.show_hud_badge,
        }


def validate_roi_polygon(points: List[Any]) -> List[Tuple[float, float]]:
    """Validate and clamp normalized `(x, y)` polygon vertices to `[0.0, 1.0]`."""
    if not points or len(points) < 3:
        raise ValueError("ROI polygon requires at least 3 vertices.")
    cleaned: List[Tuple[float, float]] = []
    for pt in points:
        if isinstance(pt, dict):
            rx, ry = float(pt["x"]), float(pt["y"])
        else:
            rx, ry = float(pt[0]), float(pt[1])
        cleaned.append(
            (
                round(max(0.0, min(1.0, rx)), 4),
                round(max(0.0, min(1.0, ry)), 4),
            )
        )
    return cleaned


def draw_dashed_segment(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1.0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist

    pos = 0.0
    step = float(dash_len + gap_len)
    while pos < dist:
        end_pos = min(pos + dash_len, dist)
        sx = int(round(x1 + dx * pos))
        sy = int(round(y1 + dy * pos))
        ex = int(round(x1 + dx * end_pos))
        ey = int(round(y1 + dy * end_pos))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += step


def draw_dashed_polygon(
    img: np.ndarray,
    poly_px: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    pts = poly_px.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[(i + 1) % n][0]), int(pts[(i + 1) % n][1]))
        draw_dashed_segment(img, p1, p2, color=color, thickness=thickness)
        cv2.circle(img, p1, 4, (20, 22, 30), -1, cv2.LINE_AA)
        cv2.circle(img, p1, 3, color, -1, cv2.LINE_AA)


@dataclass
class DetectionItem:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    anchor: Tuple[int, int]
    track_id: Optional[int]
    inside_roi: bool
    color_hex: str
    color_bgr: Tuple[int, int, int]


class TrackerRegionEngine:
    """High-efficiency multi-object tracker, ROI counter, and frame renderer."""

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager
        self.roi_points: List[Tuple[float, float]] = list(DEFAULT_ROI_POLYGON)
        self.render_options: RenderOptions = RenderOptions()

        self._first_track_frame: bool = True
        self._pure_bytetracker: PureByteTracker = PureByteTracker(max_lost=25)
        self.unique_ids_all: Set[int] = set()
        self.unique_ids_by_class: Dict[str, Set[int]] = defaultdict(set)
        self.region_entered_ids: Set[int] = set()
        self.region_entered_by_class: Dict[str, Set[int]] = defaultdict(set)
        self.region_counter_log: List[Dict[str, Any]] = []
        self.motion_trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self.peak_inside_roi: int = 0

    def get_roi_polygon(self) -> List[List[float]]:
        return [[pt[0], pt[1]] for pt in self.roi_points]

    def set_roi_polygon(self, points: List[Any]) -> List[List[float]]:
        self.roi_points = validate_roi_polygon(points)
        return self.get_roi_polygon()

    def reset_roi_polygon(self) -> List[List[float]]:
        self.roi_points = list(DEFAULT_ROI_POLYGON)
        return self.get_roi_polygon()

    def reset_session(self) -> None:
        self._first_track_frame = True
        self._pure_bytetracker.reset()
        self.unique_ids_all.clear()
        self.unique_ids_by_class.clear()
        self.region_entered_ids.clear()
        self.region_entered_by_class.clear()
        self.region_counter_log.clear()
        self.motion_trails.clear()
        self.peak_inside_roi = 0

    def _run_inference(
        self,
        frame_bgr: np.ndarray,
        use_tracking: bool,
        conf: float,
        iou: float,
        classes_arg: Optional[List[int]],
    ) -> Any:
        yolo = self.model_manager.model
        if yolo is None:
            raise RuntimeError("YOLOv11 model is not initialized.")

        is_fast_engine = self.model_manager.runtime_backend in {"onnxruntime", "openvino"}
        is_cpu = str(self.model_manager.device).startswith("cpu")
        infer_imgsz = 416 if (use_tracking and is_cpu and not is_fast_engine) else 640

        if not use_tracking:
            kwargs: Dict[str, Any] = {
                "conf": conf,
                "iou": iou,
                "classes": classes_arg,
                "device": self.model_manager.device,
                "verbose": False,
            }
            if not is_fast_engine:
                kwargs["imgsz"] = infer_imgsz
            res_list = yolo.predict(frame_bgr, **kwargs)
            return res_list[0] if res_list else None

        persist_flag = not self._first_track_frame
        self._first_track_frame = False

        # Path 1: OpenVINO GPU / ONNX Runtime has built-in zero-overhead PureByteTracker
        if is_fast_engine:
            res_list = yolo.track(
                frame_bgr,
                persist=persist_flag,
                conf=conf,
                iou=iou,
                classes=classes_arg,
                verbose=False,
            )
            return res_list[0] if res_list else None

        # Path 2: Native Ultralytics ByteTrack when `lap` is available
        if LAP_AVAILABLE:
            res_list = yolo.track(
                frame_bgr,
                persist=persist_flag,
                tracker="bytetrack.yaml",
                conf=conf,
                iou=iou,
                classes=classes_arg,
                imgsz=infer_imgsz,
                device=self.model_manager.device,
                verbose=False,
            )
            return res_list[0] if res_list else None

        # Path 3: Single-pass PyTorch `.pt` predict + PureByteTracker (zero duplicate forward pass!)
        if not persist_flag:
            self._pure_bytetracker.reset()
        low_floor = max(0.12, conf * 0.5)
        raw_list = yolo.predict(
            frame_bgr,
            conf=low_floor,
            iou=iou,
            classes=classes_arg,
            imgsz=infer_imgsz,
            device=self.model_manager.device,
            verbose=False,
        )
        r0 = raw_list[0] if raw_list else None
        if r0 is not None and r0.boxes is not None and len(r0.boxes) > 0:
            b_xyxy = r0.boxes.xyxy.cpu().numpy()
            b_conf = r0.boxes.conf.cpu().numpy()
            b_cls = r0.boxes.cls.cpu().numpy().astype(int)
            t_xyxy, t_conf, t_cls, t_ids = self._pure_bytetracker.update(
                b_xyxy, b_conf, b_cls, user_conf_thresh=conf
            )
            r_names = getattr(r0, "names", None) or self.model_manager.class_names
            return ONNXDetectionResult(
                ONNXBoxesResult(t_xyxy, t_conf, t_cls, t_ids),
                r_names,
            )

        self._pure_bytetracker.update(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            user_conf_thresh=conf,
        )
        return r0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        use_tracking: bool = True,
        timestamp_sec: float = 0.0,
        frame_idx: int = 0,
        conf_override: Optional[float] = None,
        iou_override: Optional[float] = None,
        roi_override: Optional[List[Tuple[float, float]]] = None,
        render_override: Optional[RenderOptions] = None,
        corner_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        active_roi = roi_override if roi_override is not None else self.roi_points
        poly_px = np.array(
            [[int(round(px * width)), int(round(py * height))] for px, py in active_roi],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        opts = render_override if render_override is not None else self.render_options
        conf = (
            float(conf_override)
            if conf_override is not None
            else self.model_manager.conf_threshold
        )
        iou = (
            float(iou_override)
            if iou_override is not None
            else self.model_manager.iou_threshold
        )
        classes_arg = self.model_manager.get_effective_classes_arg()

        detections: List[DetectionItem] = []

        if classes_arg is not None and len(classes_arg) == 0:
            results = None
        else:
            with self.model_manager.lock:
                results = self._run_inference(
                    frame_bgr=frame_bgr,
                    use_tracking=use_tracking,
                    conf=conf,
                    iou=iou,
                    classes_arg=classes_arg,
                )

        current_counts_by_class: Dict[str, int] = defaultdict(int)
        inside_now_by_class: Dict[str, int] = defaultdict(int)
        confidences: List[float] = []
        class_confs: Dict[str, List[float]] = defaultdict(list)

        if results is not None and results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs_np = results.boxes.conf.cpu().numpy()
            clss_np = results.boxes.cls.cpu().numpy().astype(int)
            ids_np = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else None
            )
            names_map = getattr(results, "names", None) or self.model_manager.class_names

            for idx in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[idx]]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                cid = int(clss_np[idx])
                cname = str(names_map.get(cid, f"class_{cid}"))
                c_score = float(confs_np[idx])
                tid = int(ids_np[idx]) if ids_np is not None else None

                cx = int(round((x1 + x2) * 0.5))
                anchor_pt = (cx, y2)
                is_inside = (
                    cv2.pointPolygonTest(poly_px, (float(cx), float(y2)), False) >= 0.0
                )

                c_hex = self.model_manager.class_colors_hex.get(
                    cid, resolve_class_color_hex(cid, cname)
                )
                c_bgr = self.model_manager.class_colors_bgr.get(cid, hex_to_bgr(c_hex))

                detections.append(
                    DetectionItem(
                        class_id=cid,
                        class_name=cname,
                        confidence=c_score,
                        bbox=(x1, y1, x2, y2),
                        anchor=anchor_pt,
                        track_id=tid,
                        inside_roi=is_inside,
                        color_hex=c_hex,
                        color_bgr=c_bgr,
                    )
                )

                current_counts_by_class[cname] += 1
                confidences.append(c_score)
                class_confs[cname].append(c_score)

                if is_inside:
                    inside_now_by_class[cname] += 1

                if use_tracking and tid is not None:
                    self.unique_ids_all.add(tid)
                    self.unique_ids_by_class[cname].add(tid)
                    self.motion_trails[tid].append(anchor_pt)

                    if is_inside and tid not in self.region_entered_ids:
                        self.region_entered_ids.add(tid)
                        self.region_entered_by_class[cname].add(tid)
                        self.region_counter_log.append(
                            {
                                "track_id": f"#{tid}",
                                "raw_id": tid,
                                "class_name": cname,
                                "first_entered_sec": round(float(timestamp_sec), 2),
                                "confidence": round(c_score, 4),
                                "frame_idx": int(frame_idx),
                            }
                        )

        inside_now_total = sum(inside_now_by_class.values())
        if use_tracking:
            self.peak_inside_roi = max(self.peak_inside_roi, inside_now_total)
            unique_total = len(self.unique_ids_all)
            unique_entered_total = len(self.region_entered_ids)
            unique_by_class_counts = {
                k: len(v) for k, v in self.unique_ids_by_class.items() if len(v) > 0
            }
            region_entered_by_class_counts = {
                k: len(v) for k, v in self.region_entered_by_class.items() if len(v) > 0
            }
        else:
            unique_total = len(detections)
            unique_entered_total = inside_now_total
            unique_by_class_counts = dict(current_counts_by_class)
            region_entered_by_class_counts = dict(inside_now_by_class)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        per_class_avg_conf = {
            cname: float(np.mean(vals)) for cname, vals in class_confs.items() if vals
        }

        need_draw = (
            opts.show_boxes
            or opts.show_roi
            or opts.show_trails
            or opts.show_hud_badge
            or bool(corner_tag)
        )
        annotated = (
            self._render_frame(
                frame_bgr=frame_bgr,
                detections=detections,
                poly_px=poly_px,
                inside_now_total=inside_now_total,
                unique_entered_total=unique_entered_total,
                use_tracking=use_tracking,
                opts=opts,
                corner_tag=corner_tag,
            )
            if need_draw
            else frame_bgr
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        detected_colors: Dict[str, str] = {}
        for d in detections:
            detected_colors[d.class_name] = d.color_hex

        return {
            "annotated_frame": annotated,
            "width": width,
            "height": height,
            "latency_ms": latency_ms,
            "total_detections": len(detections),
            "avg_confidence": round(avg_conf, 4),
            "inside_roi_now": inside_now_total,
            "inside_roi_by_class": dict(inside_now_by_class),
            "unique_tracked_total": unique_total,
            "unique_tracked_by_class": unique_by_class_counts,
            "region_unique_entered": unique_entered_total,
            "region_entered_by_class": region_entered_by_class_counts,
            "current_counts_by_class": dict(current_counts_by_class),
            "per_class_avg_conf": per_class_avg_conf,
            "class_colors": detected_colors,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": list(d.bbox),
                    "anchor": list(d.anchor),
                    "track_id": d.track_id,
                    "inside_roi": d.inside_roi,
                    "color_hex": d.color_hex,
                }
                for d in detections
            ],
            "region_counter_log": list(self.region_counter_log[-80:])
            if use_tracking
            else [
                {
                    "track_id": f"#IMG-{i + 1}",
                    "raw_id": i + 1,
                    "class_name": d.class_name,
                    "first_entered_sec": 0.0,
                    "confidence": round(d.confidence, 4),
                    "frame_idx": 1,
                }
                for i, d in enumerate(detections)
                if d.inside_roi
            ],
        }

    def _render_frame(
        self,
        frame_bgr: np.ndarray,
        detections: List[DetectionItem],
        poly_px: np.ndarray,
        inside_now_total: int,
        unique_entered_total: int,
        use_tracking: bool,
        opts: RenderOptions,
        corner_tag: Optional[str] = None,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]
        scale = max(0.45, min(0.95, w / 1050.0))
        box_thickness = max(1, int(round(2 * scale)))

        # 1. Soft Pastel Mint / Lavender ROI Zone
        if opts.show_roi and len(poly_px) >= 3:
            overlay = canvas.copy()
            mint_bgr = (182, 247, 158) if inside_now_total > 0 else (245, 185, 195)
            cv2.fillPoly(overlay, [poly_px], mint_bgr)
            cv2.addWeighted(overlay, 0.13, canvas, 0.87, 0, canvas)
            draw_dashed_polygon(canvas, poly_px, color=mint_bgr, thickness=max(2, box_thickness))

        # 2. Motion Trails
        if use_tracking and opts.show_trails:
            active_ids = {d.track_id for d in detections if d.track_id is not None}
            color_by_id = {
                d.track_id: d.color_bgr for d in detections if d.track_id is not None
            }
            for tid in active_ids:
                pts = self.motion_trails.get(tid)
                if not pts or len(pts) < 2:
                    continue
                c_bgr = color_by_id.get(tid, (182, 247, 158))
                n_pts = len(pts)
                for j in range(1, n_pts):
                    seg_thick = max(1, int(round((1 + 2.0 * (j / float(n_pts))) * scale)))
                    cv2.line(canvas, pts[j - 1], pts[j], c_bgr, seg_thick, cv2.LINE_AA)

        # 3. Sleek Bounding Boxes & Minimal Pill Tags
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.40, 0.48 * scale)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, ay = det.anchor
            color = det.color_bgr

            if opts.show_boxes:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

                if opts.show_roi:
                    dot_col = (158, 246, 185) if det.inside_roi else color
                    cv2.circle(canvas, (cx, ay), 5 if det.inside_roi else 3, dot_col, -1, cv2.LINE_AA)

                id_prefix = (
                    f"#{det.track_id} "
                    if (opts.show_track_ids and det.track_id is not None)
                    else ""
                )
                label = f"{id_prefix}{det.class_name} {int(round(det.confidence * 100))}%"
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
                pad_x, pad_y = 6, 4
                lbl_y2 = y1 if y1 - th - pad_y * 2 > 4 else min(h - 2, y1 + th + pad_y * 2)
                lbl_y1 = lbl_y2 - th - pad_y * 2
                lbl_x2 = min(w - 2, x1 + tw + pad_x * 2)

                cv2.rectangle(canvas, (x1, lbl_y1), (lbl_x2, lbl_y2), (18, 20, 28), -1)
                cv2.rectangle(canvas, (x1, lbl_y1), (min(w - 2, x1 + 3), lbl_y2), color, -1)
                cv2.putText(
                    canvas,
                    label,
                    (x1 + pad_x + 1, lbl_y2 - pad_y),
                    font,
                    font_scale,
                    (245, 247, 250),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Compact Minimalist On-Frame Zone Pill (Top-Right)
        if opts.show_roi and opts.show_hud_badge:
            hud_text = (
                f"Zone: {inside_now_total}  |  Total: {unique_entered_total}"
                if use_tracking
                else f"In Zone: {inside_now_total}"
            )
            fs = max(0.42, 0.48 * scale)
            (tw, th), _ = cv2.getTextSize(hud_text, font, fs, 1)
            bx2 = w - 12
            bx1 = max(10, bx2 - tw - 22)
            by1 = 12
            by2 = by1 + th + 16
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (16, 18, 26), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (158, 246, 185), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                hud_text,
                (bx1 + 11, by2 - 8),
                font,
                fs,
                (190, 248, 175),
                1,
                cv2.LINE_AA,
            )

        if corner_tag:
            fs = max(0.42, 0.48 * scale)
            (tw, th), _ = cv2.getTextSize(corner_tag, font, fs, 1)
            cv2.rectangle(canvas, (12, 12), (28 + tw, 28 + th), (16, 18, 26), -1)
            cv2.putText(
                canvas,
                corner_tag,
                (20, 22 + th),
                font,
                fs,
                (235, 240, 248),
                1,
                cv2.LINE_AA,
            )

        return canvas

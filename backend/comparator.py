"""
Before vs. After Image & Video Enhancement Comparison Engine (`comparator.py`).

Evaluates the exact same YOLOv11 weights and thresholds (`conf`, `iou`, `classes`)
on both the **Original (Before)** input and the **Enhanced (After)** input,
computing aggregate & per-class detection deltas, confidence shifts, frame-level
telemetry logs, and annotated visual comparison frames.
"""

from __future__ import annotations

import base64
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.enhancement import (
    EnhancementConfig,
    apply_enhancement,
    describe_enhancement,
)
from backend.model_manager import ModelManager, resolve_class_color_hex
from backend.onnx_yolo_engine import PureByteTracker
from backend.tracker_region import RenderOptions, TrackerRegionEngine


def encode_bgr_to_data_url(frame_bgr: np.ndarray, quality: int = 88) -> str:
    """Encode an OpenCV BGR image into a `data:image/jpeg;base64,...` string."""
    ok, buf = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode annotated frame.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def compute_delta_pct(before_val: int, after_val: int) -> float:
    """
    Compute percentage change `(after - before) / before * 100.0`.
    Handles `before == 0` gracefully (`+100.0%` if `after > 0`, else `0.0%`).
    """
    if before_val == 0:
        return 100.0 if after_val > 0 else 0.0
    return round(((after_val - before_val) / float(before_val)) * 100.0, 2)


class BeforeAfterComparator:
    """Runs rigorous Before vs. After YOLOv11 comparisons on images and videos."""

    def __init__(
        self,
        model_manager: ModelManager,
        tracker_engine: TrackerRegionEngine,
    ) -> None:
        self.model_manager = model_manager
        self.tracker_engine = tracker_engine

    def _resolve_color(self, class_name: str) -> str:
        for cid, cname in self.model_manager.class_names.items():
            if cname == class_name:
                return self.model_manager.class_colors_hex.get(cid, "#10B981")
        return resolve_class_color_hex(0, class_name)

    def _build_per_class_table(
        self,
        before_counts: Dict[str, int],
        after_counts: Dict[str, int],
        before_confs: Dict[str, List[float]],
        after_confs: Dict[str, List[float]],
    ) -> List[Dict[str, Any]]:
        all_classes = sorted(
            set(before_counts.keys()) | set(after_counts.keys()),
            key=lambda c: (-(after_counts.get(c, 0) + before_counts.get(c, 0)), c),
        )
        rows: List[Dict[str, Any]] = []
        for cname in all_classes:
            b_cnt = int(before_counts.get(cname, 0))
            a_cnt = int(after_counts.get(cname, 0))
            d_cnt = a_cnt - b_cnt
            d_pct = compute_delta_pct(b_cnt, a_cnt)

            b_c_list = before_confs.get(cname, [])
            a_c_list = after_confs.get(cname, [])
            b_avg = float(np.mean(b_c_list)) if b_c_list else 0.0
            a_avg = float(np.mean(a_c_list)) if a_c_list else 0.0
            c_delta = a_avg - b_avg

            rows.append(
                {
                    "class_name": cname,
                    "color_hex": self._resolve_color(cname),
                    "before_count": b_cnt,
                    "after_count": a_cnt,
                    "delta_count": d_cnt,
                    "delta_pct": d_pct,
                    "before_avg_conf": round(b_avg, 4),
                    "after_avg_conf": round(a_avg, 4),
                    "conf_delta": round(c_delta, 4),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Single Image Comparison
    # ------------------------------------------------------------------
    def compare_image(
        self,
        frame_bgr: np.ndarray,
        source_name: str,
        enhancement_config: EnhancementConfig,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        enh_cfg = enhancement_config.normalized()
        render_opts = RenderOptions(
            show_boxes=self.tracker_engine.render_options.show_boxes,
            show_track_ids=False,
            show_trails=False,
            show_roi=True,
            show_hud_badge=True,
        )

        # 1. Original (Before)
        before_res = self.tracker_engine.process_frame(
            frame_bgr,
            use_tracking=False,
            render_override=render_opts,
            corner_tag="BEFORE | ORIGINAL",
        )

        # 2. Enhanced (After)
        enhanced_bgr = apply_enhancement(frame_bgr, enh_cfg)
        after_tag = f"AFTER | {enh_cfg.method.upper()}"
        after_res = self.tracker_engine.process_frame(
            enhanced_bgr,
            use_tracking=False,
            render_override=render_opts,
            corner_tag=after_tag,
        )

        # Aggregate confidences per class
        b_confs_by_cls: Dict[str, List[float]] = defaultdict(list)
        a_confs_by_cls: Dict[str, List[float]] = defaultdict(list)
        all_b_confs: List[float] = []
        all_a_confs: List[float] = []

        for d in before_res["detections"]:
            b_confs_by_cls[d["class_name"]].append(float(d["confidence"]))
            all_b_confs.append(float(d["confidence"]))
        for d in after_res["detections"]:
            a_confs_by_cls[d["class_name"]].append(float(d["confidence"]))
            all_a_confs.append(float(d["confidence"]))

        b_total = int(before_res["total_detections"])
        a_total = int(after_res["total_detections"])
        d_count = a_total - b_total
        d_pct = compute_delta_pct(b_total, a_total)

        b_avg_conf = float(np.mean(all_b_confs)) if all_b_confs else 0.0
        a_avg_conf = float(np.mean(all_a_confs)) if all_a_confs else 0.0
        d_conf = a_avg_conf - b_avg_conf

        per_class_table = self._build_per_class_table(
            before_counts=before_res["current_counts_by_class"],
            after_counts=after_res["current_counts_by_class"],
            before_confs=b_confs_by_cls,
            after_confs=a_confs_by_cls,
        )

        frame_detail_log = [
            {
                "frame_or_file": source_name,
                "timestamp_sec": 0.0,
                "before_detections": b_total,
                "after_detections": a_total,
                "before_avg_conf": round(b_avg_conf, 4),
                "after_avg_conf": round(a_avg_conf, 4),
                "inside_roi_count": int(after_res["inside_roi_now"]),
            }
        ]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        payload: Dict[str, Any] = {
            "mode": "Before vs. After Lab (Image)",
            "source_type": "image",
            "source_name": source_name,
            "width": int(after_res["width"]),
            "height": int(after_res["height"]),
            "latency_ms": elapsed_ms,
            "summary": {
                "model_name": self.model_manager.active_model_name,
                "model_badge": self.model_manager.get_badge_text(),
                "device_label": self.model_manager.device_label,
                "enhancement_method": enh_cfg.method,
                "enhancement_description": describe_enhancement(enh_cfg),
                "enhancement_params": enh_cfg.to_dict(),
                "conf_threshold": self.model_manager.conf_threshold,
                "iou_threshold": self.model_manager.iou_threshold,
                "roi_polygon": self.tracker_engine.get_roi_polygon(),
                "before_total": b_total,
                "after_total": a_total,
                "delta_count": d_count,
                "delta_pct": d_pct,
                "before_avg_conf": round(b_avg_conf, 4),
                "after_avg_conf": round(a_avg_conf, 4),
                "delta_conf": round(d_conf, 4),
                "before_inside_roi": int(before_res["inside_roi_now"]),
                "after_inside_roi": int(after_res["inside_roi_now"]),
                "unique_tracked_total": a_total,
                "region_unique_entered": int(after_res["inside_roi_now"]),
                "sampled_frames": 1,
            },
            "per_class_table": per_class_table,
            "frame_detail_log": frame_detail_log,
            "region_counter_log": after_res["region_counter_log"],
        }

        if include_images:
            payload["before_image"] = encode_bgr_to_data_url(
                before_res["annotated_frame"]
            )
            payload["after_image"] = encode_bgr_to_data_url(after_res["annotated_frame"])

        return payload

    # ------------------------------------------------------------------
    # Multi-Image Batch Comparison
    # ------------------------------------------------------------------
    def compare_image_batch(
        self,
        image_paths: List[Path],
        enhancement_config: EnhancementConfig,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        enh_cfg = enhancement_config.normalized()

        agg_b_counts: Dict[str, int] = defaultdict(int)
        agg_a_counts: Dict[str, int] = defaultdict(int)
        agg_b_confs_cls: Dict[str, List[float]] = defaultdict(list)
        agg_a_confs_cls: Dict[str, List[float]] = defaultdict(list)
        all_b_confs: List[float] = []
        all_a_confs: List[float] = []

        frame_detail_log: List[Dict[str, Any]] = []
        region_counter_log: List[Dict[str, Any]] = []
        batch_items: List[Dict[str, Any]] = []

        best_pair: Optional[Tuple[str, str, str]] = None
        best_gain = -1e9
        total_b_roi = 0
        total_a_roi = 0
        last_w, last_h = 0, 0

        for idx, img_path in enumerate(image_paths):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            res = self.compare_image(
                frame_bgr=bgr,
                source_name=img_path.name,
                enhancement_config=enh_cfg,
                include_images=True,
            )
            last_w, last_h = res["width"], res["height"]
            sm = res["summary"]
            total_b_roi += sm["before_inside_roi"]
            total_a_roi += sm["after_inside_roi"]

            for row in res["per_class_table"]:
                cn = row["class_name"]
                agg_b_counts[cn] += row["before_count"]
                agg_a_counts[cn] += row["after_count"]
                if row["before_count"] > 0:
                    agg_b_confs_cls[cn].extend(
                        [row["before_avg_conf"]] * row["before_count"]
                    )
                    all_b_confs.extend([row["before_avg_conf"]] * row["before_count"])
                if row["after_count"] > 0:
                    agg_a_confs_cls[cn].extend(
                        [row["after_avg_conf"]] * row["after_count"]
                    )
                    all_a_confs.extend([row["after_avg_conf"]] * row["after_count"])

            frame_detail_log.extend(res["frame_detail_log"])
            for rlog in res["region_counter_log"]:
                entry = dict(rlog)
                entry["track_id"] = f"#{img_path.stem[:10]}-{entry.get('raw_id', idx + 1)}"
                region_counter_log.append(entry)

            gain_score = sm["delta_count"] * 2.0 + sm["delta_conf"] * 10.0
            if gain_score > best_gain:
                best_gain = gain_score
                best_pair = (res["before_image"], res["after_image"], img_path.name)

            batch_items.append(
                {
                    "filename": img_path.name,
                    "before_total": sm["before_total"],
                    "after_total": sm["after_total"],
                    "delta_count": sm["delta_count"],
                    "after_avg_conf": sm["after_avg_conf"],
                    "after_inside_roi": sm["after_inside_roi"],
                    "after_image": res["after_image"],
                    "before_image": res["before_image"],
                }
            )

        b_total = sum(agg_b_counts.values())
        a_total = sum(agg_a_counts.values())
        b_avg = float(np.mean(all_b_confs)) if all_b_confs else 0.0
        a_avg = float(np.mean(all_a_confs)) if all_a_confs else 0.0

        per_class_table = self._build_per_class_table(
            before_counts=agg_b_counts,
            after_counts=agg_a_counts,
            before_confs=agg_b_confs_cls,
            after_confs=agg_a_confs_cls,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        source_label = f"Batch ({len(batch_items)} images)"

        return {
            "mode": "Batch Image Analysis",
            "source_type": "batch",
            "source_name": source_label,
            "width": last_w,
            "height": last_h,
            "latency_ms": elapsed_ms,
            "representative_frame_label": best_pair[2] if best_pair else "",
            "before_image": best_pair[0] if best_pair else None,
            "after_image": best_pair[1] if best_pair else None,
            "batch_items": batch_items,
            "summary": {
                "model_name": self.model_manager.active_model_name,
                "model_badge": self.model_manager.get_badge_text(),
                "device_label": self.model_manager.device_label,
                "enhancement_method": enh_cfg.method,
                "enhancement_description": describe_enhancement(enh_cfg),
                "enhancement_params": enh_cfg.to_dict(),
                "conf_threshold": self.model_manager.conf_threshold,
                "iou_threshold": self.model_manager.iou_threshold,
                "roi_polygon": self.tracker_engine.get_roi_polygon(),
                "before_total": b_total,
                "after_total": a_total,
                "delta_count": a_total - b_total,
                "delta_pct": compute_delta_pct(b_total, a_total),
                "before_avg_conf": round(b_avg, 4),
                "after_avg_conf": round(a_avg, 4),
                "delta_conf": round(a_avg - b_avg, 4),
                "before_inside_roi": total_b_roi,
                "after_inside_roi": total_a_roi,
                "unique_tracked_total": a_total,
                "region_unique_entered": total_a_roi,
                "sampled_frames": len(batch_items),
            },
            "per_class_table": per_class_table,
            "frame_detail_log": frame_detail_log,
            "region_counter_log": region_counter_log,
        }

    # ------------------------------------------------------------------
    # Video Sampled Before vs. After Comparison
    # ------------------------------------------------------------------
    def compare_video(
        self,
        video_path: Path,
        enhancement_config: EnhancementConfig,
        sample_every_n: int = 10,
        max_samples: int = 90,
    ) -> Dict[str, Any]:
        """
        Process a video by sampling every `N` frames, evaluating Original (Before)
        vs. Enhanced (After) with identical thresholds, tracking unique IDs on
        the enhanced stream, and capturing the highest-impact comparison frame.
        """
        t0 = time.perf_counter()
        enh_cfg = enhancement_config.normalized()
        step = max(1, int(sample_every_n))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0.1 or fps > 240:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Use a dedicated temporary TrackerRegionEngine so live stream state isn't clobbered
        temp_tracker = TrackerRegionEngine(self.model_manager)
        temp_tracker.roi_points = list(self.tracker_engine.roi_points)
        temp_tracker.reset_session()
        post_bytetracker = PureByteTracker(max_lost=25)
        unique_ids_seen = set()
        roi_entered_ids = set()
        roi_entry_log: List[Dict[str, Any]] = []

        agg_b_counts: Dict[str, int] = defaultdict(int)
        agg_a_counts: Dict[str, int] = defaultdict(int)
        agg_b_confs_cls: Dict[str, List[float]] = defaultdict(list)
        agg_a_confs_cls: Dict[str, List[float]] = defaultdict(list)
        all_b_confs: List[float] = []
        all_a_confs: List[float] = []

        frame_detail_log: List[Dict[str, Any]] = []
        best_before_bgr: Optional[np.ndarray] = None
        best_after_bgr: Optional[np.ndarray] = None
        best_frame_label = "Frame #0"
        best_score = -1e9

        peak_b_roi = 0
        peak_a_roi = 0
        sampled_count = 0
        frame_idx = 0

        try:
            while sampled_count < max_samples:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if frame_idx % step == 0:
                    ts_sec = round(frame_idx / fps, 2)

                    # 1. Original (Before) — strict apples-to-apples stateless YOLOv11 evaluation
                    b_res = temp_tracker.process_frame(
                        frame_bgr,
                        use_tracking=False,
                        timestamp_sec=ts_sec,
                        frame_idx=frame_idx,
                        corner_tag=f"BEFORE | FRAME #{frame_idx}",
                    )

                    # 2. Enhanced (After) — identical YOLOv11 resolution & thresholds (`use_tracking=False`)
                    if enh_cfg.method == "none":
                        enh_bgr = frame_bgr
                    else:
                        enh_bgr = apply_enhancement(frame_bgr, enh_cfg)

                    a_res = temp_tracker.process_frame(
                        enh_bgr,
                        use_tracking=False,
                        timestamp_sec=ts_sec,
                        frame_idx=frame_idx,
                        corner_tag=f"AFTER ({enh_cfg.method.upper()}) | FRAME #{frame_idx}",
                    )

                    # Track unique objects & ROI entries across sampled frames without altering counts
                    dets_a = a_res["detections"]
                    if dets_a:
                        arr_xyxy = np.array([d["bbox"] for d in dets_a], dtype=np.float32)
                        arr_conf = np.array([d["confidence"] for d in dets_a], dtype=np.float32)
                        arr_cls = np.array([d["class_id"] for d in dets_a], dtype=np.int32)
                        t_xyxy, t_conf, t_cls, t_ids = post_bytetracker.update(
                            arr_xyxy,
                            arr_conf,
                            arr_cls,
                            user_conf_thresh=self.model_manager.conf_threshold,
                        )
                        poly_px = np.array(
                            [
                                [int(round(px * width)), int(round(py * height))]
                                for px, py in temp_tracker.roi_points
                            ],
                            dtype=np.int32,
                        ).reshape((-1, 1, 2))
                        for k in range(len(t_ids)):
                            tid = int(t_ids[k])
                            unique_ids_seen.add(tid)
                            bx1, by1, bx2, by2 = t_xyxy[k]
                            cx = float((bx1 + bx2) * 0.5)
                            cy2 = float(by2)
                            if cv2.pointPolygonTest(poly_px, (cx, cy2), False) >= 0.0:
                                if tid not in roi_entered_ids:
                                    roi_entered_ids.add(tid)
                                    cname = self.model_manager.class_names.get(
                                        int(t_cls[k]), f"class_{int(t_cls[k])}"
                                    )
                                    roi_entry_log.append(
                                        {
                                            "track_id": f"#{tid}",
                                            "raw_id": tid,
                                            "class_name": cname,
                                            "first_entered_sec": ts_sec,
                                            "confidence": round(float(t_conf[k]), 4),
                                            "frame_idx": int(frame_idx),
                                        }
                                    )
                    else:
                        post_bytetracker.update(
                            np.zeros((0, 4), dtype=np.float32),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=np.int32),
                            user_conf_thresh=self.model_manager.conf_threshold,
                        )

                    b_cnt = int(b_res["total_detections"])
                    a_cnt = int(a_res["total_detections"])
                    peak_b_roi = max(peak_b_roi, int(b_res["inside_roi_now"]))
                    peak_a_roi = max(peak_a_roi, int(a_res["inside_roi_now"]))

                    for d in b_res["detections"]:
                        cn = d["class_name"]
                        cf = float(d["confidence"])
                        agg_b_counts[cn] += 1
                        agg_b_confs_cls[cn].append(cf)
                        all_b_confs.append(cf)

                    for d in a_res["detections"]:
                        cn = d["class_name"]
                        cf = float(d["confidence"])
                        agg_a_counts[cn] += 1
                        agg_a_confs_cls[cn].append(cf)
                        all_a_confs.append(cf)

                    frame_detail_log.append(
                        {
                            "frame_or_file": f"Frame #{frame_idx}",
                            "timestamp_sec": ts_sec,
                            "before_detections": b_cnt,
                            "after_detections": a_cnt,
                            "before_avg_conf": round(float(b_res["avg_confidence"]), 4),
                            "after_avg_conf": round(float(a_res["avg_confidence"]), 4),
                            "inside_roi_count": int(a_res["inside_roi_now"]),
                        }
                    )

                    # Pick representative frame with greatest positive improvement & activity
                    improvement = (
                        (a_cnt - b_cnt) * 3.0
                        + (float(a_res["avg_confidence"]) - float(b_res["avg_confidence"])) * 12.0
                        + a_cnt * 0.25
                    )
                    if improvement > best_score:
                        best_score = improvement
                        best_before_bgr = b_res["annotated_frame"]
                        best_after_bgr = a_res["annotated_frame"]
                        best_frame_label = f"Frame #{frame_idx} ({ts_sec:.2f}s)"

                    sampled_count += 1

                frame_idx += 1
        finally:
            cap.release()

        b_total = sum(agg_b_counts.values())
        a_total = sum(agg_a_counts.values())
        b_avg = float(np.mean(all_b_confs)) if all_b_confs else 0.0
        a_avg = float(np.mean(all_a_confs)) if all_a_confs else 0.0

        per_class_table = self._build_per_class_table(
            before_counts=agg_b_counts,
            after_counts=agg_a_counts,
            before_confs=agg_b_confs_cls,
            after_confs=agg_a_confs_cls,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "mode": f"Before vs. After Lab (Video • every {step} frames)",
            "source_type": "video",
            "source_name": video_path.name,
            "width": width,
            "height": height,
            "total_video_frames": total_frames,
            "latency_ms": elapsed_ms,
            "representative_frame_label": best_frame_label,
            "before_image": (
                encode_bgr_to_data_url(best_before_bgr)
                if best_before_bgr is not None
                else None
            ),
            "after_image": (
                encode_bgr_to_data_url(best_after_bgr)
                if best_after_bgr is not None
                else None
            ),
            "summary": {
                "model_name": self.model_manager.active_model_name,
                "model_badge": self.model_manager.get_badge_text(),
                "device_label": self.model_manager.device_label,
                "enhancement_method": enh_cfg.method,
                "enhancement_description": describe_enhancement(enh_cfg),
                "enhancement_params": enh_cfg.to_dict(),
                "conf_threshold": self.model_manager.conf_threshold,
                "iou_threshold": self.model_manager.iou_threshold,
                "roi_polygon": temp_tracker.get_roi_polygon(),
                "before_total": b_total,
                "after_total": a_total,
                "delta_count": a_total - b_total,
                "delta_pct": compute_delta_pct(b_total, a_total),
                "before_avg_conf": round(b_avg, 4),
                "after_avg_conf": round(a_avg, 4),
                "delta_conf": round(a_avg - b_avg, 4),
                "before_inside_roi": peak_b_roi,
                "after_inside_roi": peak_a_roi,
                "unique_tracked_total": len(unique_ids_seen),
                "region_unique_entered": len(roi_entered_ids),
                "sampled_frames": sampled_count,
                "sample_every_n": step,
            },
            "per_class_table": per_class_table,
            "frame_detail_log": frame_detail_log,
            "region_counter_log": roi_entry_log,
        }

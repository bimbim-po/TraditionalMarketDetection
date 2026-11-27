"""
Ultralytics-Compatible YOLOv11 ONNX Runtime Engine & Pure-NumPy ByteTrack (`onnx_yolo_engine.py`).

Why this module exists:
On Windows 11 systems with Microsoft Smart App Control (SAC) enabled, unsigned
third-party DLLs (`torch/lib/shm.dll`, `lap/_lapjv.pyd`) are blocked by OS policy
(`[WinError 4551]`), whereas Microsoft-signed `onnxruntime` and `opencv-python`
execute natively at full SIMD hardware speed.

This module provides a 100% API-compatible YOLOv11 engine (`predict` + `track`)
that:
1. Reads `model.names` dynamically from Ultralytics ONNX metadata (`custom_metadata_map['names']`).
2. Performs exact Ultralytics `LetterBox` preprocessing and class-aware C++ NMS.
3. Implements authentic two-stage **ByteTrack** multi-object tracking (velocity
   prediction + high/low confidence IoU bipartite association) in pure NumPy.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import requests


OFFICIAL_YOLO11N_ONNX_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"
)


def download_official_yolo11n_onnx(dest_path: Path) -> Path:
    """Download official Ultralytics `yolo11n.onnx` weights if not cached locally."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        return dest_path

    tmp = dest_path.with_suffix(".downloading")
    try:
        resp = requests.get(OFFICIAL_YOLO11N_ONNX_URL, stream=True, timeout=90)
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=262144):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dest_path


# ------------------------------------------------------------------------------
# Ultralytics Result-Compatible Lightweight Containers
# ------------------------------------------------------------------------------
class _TensorLike:
    """Mimics PyTorch `.cpu().numpy()` interface expected by `TrackerRegionEngine`."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class ONNXBoxesResult:
    """Mimics `ultralytics.engine.results.Boxes`."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        track_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.xyxy = _TensorLike(xyxy.astype(np.float32))
        self.conf = _TensorLike(conf.astype(np.float32))
        self.cls = _TensorLike(cls.astype(np.float32))
        self.id = (
            _TensorLike(track_ids.astype(np.int32))
            if track_ids is not None and len(track_ids) == len(xyxy)
            else None
        )
        self._len = int(len(xyxy))

    def __len__(self) -> int:
        return self._len


class ONNXDetectionResult:
    """Mimics `ultralytics.engine.results.Results`."""

    def __init__(self, boxes: ONNXBoxesResult, names: Dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


# ------------------------------------------------------------------------------
# Pure-NumPy Two-Stage ByteTrack Multi-Object Tracker
# ------------------------------------------------------------------------------
def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized `(M, 4)` vs `(N, 4)` xyxy IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(1.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(1.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _match_by_iou(
    track_boxes: np.ndarray,
    track_classes: np.ndarray,
    det_boxes: np.ndarray,
    det_classes: np.ndarray,
    iou_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Class-consistent bipartite matching by highest IoU without external C extensions.
    Returns `(matches, unmatched_track_indices, unmatched_det_indices)`.
    """
    num_t = len(track_boxes)
    num_d = len(det_boxes)
    if num_t == 0 or num_d == 0:
        return [], list(range(num_t)), list(range(num_d))

    iou_mat = _compute_iou_matrix(track_boxes, det_boxes)
    # Penalize cross-class associations so IDs stay class-consistent
    same_cls = track_classes[:, None] == det_classes[None, :]
    iou_mat = np.where(same_cls, iou_mat, 0.0)

    matches: List[Tuple[int, int]] = []
    used_t = set()
    used_d = set()

    # Sort all candidate pairs descending by IoU
    flat_order = np.argsort(-iou_mat, axis=None)
    for flat_idx in flat_order:
        ti = int(flat_idx // num_d)
        di = int(flat_idx % num_d)
        val = float(iou_mat[ti, di])
        if val < iou_thresh:
            break
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))

    unmatched_t = [i for i in range(num_t) if i not in used_t]
    unmatched_d = [j for j in range(num_d) if j not in used_d]
    return matches, unmatched_t, unmatched_d


@dataclass
class _Tracklet:
    track_id: int
    bbox: np.ndarray          # float32 [x1, y1, x2, y2]
    velocity: np.ndarray      # float32 [vx1, vy1, vx2, vy2]
    cls_id: int
    conf: float
    hits: int = 1
    age_lost: int = 0

    def predict_next(self) -> np.ndarray:
        """Damped constant-velocity forward box prediction."""
        decay = 0.85 ** max(0, self.age_lost)
        self.bbox = self.bbox + self.velocity * decay
        return self.bbox

    def update(self, new_bbox: np.ndarray, new_conf: float, new_cls: int) -> None:
        inst_vel = new_bbox - self.bbox
        self.velocity = 0.65 * self.velocity + 0.35 * inst_vel
        self.bbox = new_bbox.astype(np.float32)
        self.conf = float(new_conf)
        self.cls_id = int(new_cls)
        self.hits += 1
        self.age_lost = 0


class PureByteTracker:
    """
    Two-stage ByteTrack multi-object tracker in pure NumPy.
    Stage 1: Associate high-confidence boxes (`>= high_thresh`) with active/lost tracks.
    Stage 2: Associate lower-confidence boxes (`low_thresh .. high_thresh`) with remaining
             active tracks to prevent ID fragmentation under partial occlusion.
    """

    def __init__(self, max_lost: int = 30) -> None:
        self.max_lost = max_lost
        self.reset()

    def reset(self) -> None:
        self._next_id: int = 1
        self._tracks: List[_Tracklet] = []

    def update(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        clss: np.ndarray,
        user_conf_thresh: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Step 0: Advance all existing track predictions
        for trk in self._tracks:
            trk.predict_next()
            trk.age_lost += 1

        if len(boxes_xyxy) == 0:
            self._tracks = [t for t in self._tracks if t.age_lost <= self.max_lost]
            empty = np.zeros((0, 4), dtype=np.float32)
            empty_1d = np.zeros((0,), dtype=np.float32)
            return empty, empty_1d, empty_1d, empty_1d.astype(np.int32)

        high_thresh = float(user_conf_thresh)
        low_thresh = max(0.10, high_thresh * 0.45)

        high_mask = confs >= high_thresh
        low_mask = (confs >= low_thresh) & (~high_mask)

        h_boxes, h_confs, h_clss = boxes_xyxy[high_mask], confs[high_mask], clss[high_mask]
        l_boxes, l_confs, l_clss = boxes_xyxy[low_mask], confs[low_mask], clss[low_mask]

        # Stage 1: Match high-confidence detections against all tracked/recently-lost objects
        t_boxes = (
            np.stack([t.bbox for t in self._tracks], axis=0)
            if self._tracks
            else np.zeros((0, 4), dtype=np.float32)
        )
        t_clss = (
            np.array([t.cls_id for t in self._tracks], dtype=np.int32)
            if self._tracks
            else np.zeros((0,), dtype=np.int32)
        )

        matches_1, unm_tracks_1, unm_dets_h = _match_by_iou(
            t_boxes, t_clss, h_boxes, h_clss, iou_thresh=0.25
        )

        output_records: List[Tuple[np.ndarray, float, int, int]] = []

        for ti, di in matches_1:
            trk = self._tracks[ti]
            trk.update(h_boxes[di], float(h_confs[di]), int(h_clss[di]))
            output_records.append((h_boxes[di], float(h_confs[di]), int(h_clss[di]), trk.track_id))

        # Stage 2: Match low-confidence detections against unmatched recently-active tracks
        rem_active_indices = [
            ti for ti in unm_tracks_1 if self._tracks[ti].age_lost <= 3
        ]
        if rem_active_indices and len(l_boxes) > 0:
            rem_boxes = np.stack([self._tracks[ti].bbox for ti in rem_active_indices], axis=0)
            rem_clss = np.array([self._tracks[ti].cls_id for ti in rem_active_indices], dtype=np.int32)
            matches_2, _, _ = _match_by_iou(
                rem_boxes, rem_clss, l_boxes, l_clss, iou_thresh=0.35
            )
            for local_ti, di in matches_2:
                real_ti = rem_active_indices[local_ti]
                trk = self._tracks[real_ti]
                trk.update(l_boxes[di], float(l_confs[di]), int(l_clss[di]))
                # Keep track alive and emit smooth box if above user threshold or high continuity
                if float(l_confs[di]) >= high_thresh * 0.85:
                    output_records.append(
                        (l_boxes[di], float(l_confs[di]), int(l_clss[di]), trk.track_id)
                    )

        # Spawn new Track IDs for unmatched high-confidence detections
        for di in unm_dets_h:
            new_id = self._next_id
            self._next_id += 1
            new_trk = _Tracklet(
                track_id=new_id,
                bbox=h_boxes[di].astype(np.float32),
                velocity=np.zeros(4, dtype=np.float32),
                cls_id=int(h_clss[di]),
                conf=float(h_confs[di]),
                hits=1,
                age_lost=0,
            )
            self._tracks.append(new_trk)
            output_records.append((h_boxes[di], float(h_confs[di]), int(h_clss[di]), new_id))

        # Prune stale lost tracks
        self._tracks = [t for t in self._tracks if t.age_lost <= self.max_lost]

        if not output_records:
            empty = np.zeros((0, 4), dtype=np.float32)
            empty_1d = np.zeros((0,), dtype=np.float32)
            return empty, empty_1d, empty_1d, empty_1d.astype(np.int32)

        out_xyxy = np.stack([r[0] for r in output_records], axis=0).astype(np.float32)
        out_conf = np.array([r[1] for r in output_records], dtype=np.float32)
        out_cls = np.array([r[2] for r in output_records], dtype=np.float32)
        out_ids = np.array([r[3] for r in output_records], dtype=np.int32)
        return out_xyxy, out_conf, out_cls, out_ids


# ------------------------------------------------------------------------------
# Ultralytics-Compatible ONNX YOLOv11 Model Wrapper
# ------------------------------------------------------------------------------
class ONNXYOLOModel:
    """Drop-in replacement for `ultralytics.YOLO` backed by `onnxruntime`."""

    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = Path(onnx_path)
        avail = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in avail
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=providers,
        )

        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        in_shape = inp.shape
        self.imgsz: Tuple[int, int] = (
            int(in_shape[2]) if isinstance(in_shape[2], int) else 640,
            int(in_shape[3]) if isinstance(in_shape[3], int) else 640,
        )

        # Dynamically read `model.names` from Ultralytics ONNX metadata
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        if "imgsz" in meta and not isinstance(in_shape[2], int):
            try:
                parsed_sz = ast.literal_eval(meta["imgsz"])
                if isinstance(parsed_sz, (list, tuple)) and len(parsed_sz) >= 2:
                    self.imgsz = (int(parsed_sz[0]), int(parsed_sz[1]))
            except Exception:
                pass

        self.names: Dict[int, str] = self._extract_dynamic_names(meta)
        self._tracker = PureByteTracker(max_lost=30)

    def _extract_dynamic_names(self, meta: Dict[str, str]) -> Dict[int, str]:
        if "names" in meta:
            raw = ast.literal_eval(meta["names"])
            if isinstance(raw, dict):
                return {int(k): str(v) for k, v in raw.items()}
            if isinstance(raw, (list, tuple)):
                return {int(i): str(v) for i, v in enumerate(raw)}

        # Infer class count dynamically from output tensor if metadata absent
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) == 3 and isinstance(out_shape[1], int) and out_shape[1] > 4:
            nc = int(out_shape[1] - 4)
            return {i: f"class_{i}" for i in range(nc)}
        return {0: "object"}

    def _letterbox(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """Exact Ultralytics `LetterBox` resize + pad."""
        h0, w0 = frame_bgr.shape[:2]
        th, tw = self.imgsz
        r = min(th / float(h0), tw / float(w0))
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        dw = (tw - new_w) * 0.5
        dh = (th - new_h) * 0.5

        if (w0, h0) != (new_w, new_h):
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_bgr

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(np.ascontiguousarray(chw), axis=0)
        return blob, r, float(left), float(top)

    def _raw_infer(
        self,
        frame_bgr: np.ndarray,
        score_floor: float,
        iou: float,
        classes: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h0, w0 = frame_bgr.shape[:2]
        blob, ratio, pad_w, pad_h = self._letterbox(frame_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        pred = outputs[0]

        # Case A: End-to-End NMS ONNX `[1, N, 6]` -> `(x1, y1, x2, y2, conf, cls)`
        if pred.ndim == 3 and pred.shape[2] == 6:
            rows = pred[0]
            mask = rows[:, 4] >= score_floor
            if classes is not None:
                mask &= np.isin(rows[:, 5].astype(int), classes)
            rows = rows[mask]
            if len(rows) == 0:
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                )
            xyxy = rows[:, :4].copy()
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / ratio
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / ratio
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w0 - 1)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h0 - 1)
            return xyxy, rows[:, 4].astype(np.float32), rows[:, 5].astype(np.int32)

        # Case B: Standard YOLOv8 / YOLOv11 `[1, 4 + nc, 8400]`
        preds = np.transpose(pred[0], (1, 0))  # `(8400, 4 + nc)`
        boxes_cxcywh = preds[:, :4]
        scores_mat = preds[:, 4:]

        if classes is not None:
            if len(classes) == 0:
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                )
            masked = np.zeros_like(scores_mat)
            valid_c = [c for c in classes if 0 <= c < scores_mat.shape[1]]
            if valid_c:
                masked[:, valid_c] = scores_mat[:, valid_c]
            scores_mat = masked

        cls_ids = np.argmax(scores_mat, axis=1)
        max_scores = scores_mat[np.arange(len(scores_mat)), cls_ids]

        keep_mask = max_scores >= score_floor
        if not np.any(keep_mask):
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes_cxcywh = boxes_cxcywh[keep_mask]
        max_scores = max_scores[keep_mask]
        cls_ids = cls_ids[keep_mask]

        # Convert cx, cy, w, h -> top-left x, y, w, h for OpenCV C++ NMS
        x_tl = boxes_cxcywh[:, 0] - 0.5 * boxes_cxcywh[:, 2]
        y_tl = boxes_cxcywh[:, 1] - 0.5 * boxes_cxcywh[:, 3]
        xywh_list = np.stack(
            [x_tl, y_tl, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1
        ).tolist()

        if hasattr(cv2.dnn, "NMSBoxesBatched"):
            nms_indices = cv2.dnn.NMSBoxesBatched(
                xywh_list,
                max_scores.tolist(),
                cls_ids.tolist(),
                float(score_floor),
                float(iou),
            )
        else:
            # Class-offset fallback
            offset_xywh = [
                [b[0] + int(c) * 4096.0, b[1] + int(c) * 4096.0, b[2], b[3]]
                for b, c in zip(xywh_list, cls_ids)
            ]
            nms_indices = cv2.dnn.NMSBoxes(
                offset_xywh, max_scores.tolist(), float(score_floor), float(iou)
            )

        if nms_indices is None or len(nms_indices) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        idx_arr = np.asarray(nms_indices).reshape(-1)
        sel_cxcywh = boxes_cxcywh[idx_arr]
        sel_scores = max_scores[idx_arr].astype(np.float32)
        sel_cls = cls_ids[idx_arr].astype(np.int32)

        x1 = (sel_cxcywh[:, 0] - 0.5 * sel_cxcywh[:, 2] - pad_w) / ratio
        y1 = (sel_cxcywh[:, 1] - 0.5 * sel_cxcywh[:, 3] - pad_h) / ratio
        x2 = (sel_cxcywh[:, 0] + 0.5 * sel_cxcywh[:, 2] - pad_w) / ratio
        y2 = (sel_cxcywh[:, 1] + 0.5 * sel_cxcywh[:, 3] - pad_h) / ratio

        xyxy = np.stack(
            [
                np.clip(x1, 0, w0 - 1),
                np.clip(y1, 0, h0 - 1),
                np.clip(x2, 0, w0 - 1),
                np.clip(y2, 0, h0 - 1),
            ],
            axis=1,
        ).astype(np.float32)
        return xyxy, sel_scores, sel_cls

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> List[ONNXDetectionResult]:
        xyxy, scores, clss = self._raw_infer(
            frame_bgr, score_floor=float(conf), iou=float(iou), classes=classes
        )
        return [ONNXDetectionResult(ONNXBoxesResult(xyxy, scores, clss, None), self.names)]

    def track(
        self,
        frame_bgr: np.ndarray,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.35,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> List[ONNXDetectionResult]:
        if not persist:
            self._tracker.reset()
        # Query down to low_thresh so Stage-2 ByteTrack association catches occluded tracks
        low_floor = max(0.10, float(conf) * 0.45)
        xyxy, scores, clss = self._raw_infer(
            frame_bgr, score_floor=low_floor, iou=float(iou), classes=classes
        )
        t_xyxy, t_scores, t_clss, t_ids = self._tracker.update(
            xyxy, scores, clss, user_conf_thresh=float(conf)
        )
        return [
            ONNXDetectionResult(
                ONNXBoxesResult(t_xyxy, t_scores, t_clss, t_ids), self.names
            )
        ]

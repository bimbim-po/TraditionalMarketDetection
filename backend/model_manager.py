"""
Plug-and-Play YOLOv11 Model Manager for PasarVision AI (`model_manager.py`).

Responsibilities:
- Auto-detect `models/best.pt` (or `models/best.onnx`) on startup; fall back
  cleanly to Ultralytics `yolo11n` (`yolo11n.pt` / `yolo11n.onnx`) out-of-the-box.
- Hybrid Dual-Runtime resilience:
  * Uses native `ultralytics.YOLO` (`.pt` & `.onnx`) whenever PyTorch is permitted.
  * Automatically engages Microsoft-signed `onnxruntime` (`ONNXYOLOModel`) when
    Windows 11 Smart App Control blocks unsigned PyTorch DLLs (`WinError 4551`).
- Dynamically inspect `model.names` without ever hardcoding class indices.
- Provide preset high-contrast colors for target socio-economic market classes
  (`seller`, `customer`, `cart`, `scale`, `vegetables` + Indonesian aliases)
  and deterministic vibrant colors for any other classes (e.g., COCO 80).
- Support thread-safe hot-swapping via `.pt` / `.onnx` upload or selector switch.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnxruntime as ort

from backend.onnx_yolo_engine import ONNXYOLOModel, download_official_yolo11n_onnx
from backend.openvino_accel import (
    OPENVINO_AVAILABLE,
    OpenVINOYOLOModel,
    ensure_openvino_for_pt,
)

# Probe PyTorch / Ultralytics availability safely (guards against Windows 11 SAC WinError 4551)
TORCH_AVAILABLE = False
TORCH_BLOCK_REASON = ""
try:
    import torch
    from ultralytics import YOLO as UltralyticsYOLO

    TORCH_AVAILABLE = True
except (ImportError, OSError) as _torch_err:
    TORCH_AVAILABLE = False
    TORCH_BLOCK_REASON = str(_torch_err)
    torch = None  # type: ignore[assignment]
    UltralyticsYOLO = None  # type: ignore[assignment]


# Strict 5-Class Target Mapping for Traditional Market (`models/best.pt`)
# ["cart", "customer", "scale", "seller", "vegetables"]
TARGET_5_CLASSES = ("cart", "customer", "scale", "seller", "vegetables")

TARGET_CLASS_HEX: Dict[str, str] = {
    "cart": "#06B6D4",         # Cyan
    "customer": "#3B82F6",     # Blue
    "scale": "#F59E0B",        # Amber / Yellow
    "seller": "#F43F5E",       # Red / Coral
    "vegetables": "#10B981",   # Emerald
    "vegetable": "#10B981",
    # Indonesian dataset aliases mapped to the exact same 5 colors
    "gerobak": "#06B6D4",
    "troly": "#06B6D4",
    "pembeli": "#3B82F6",
    "pengunjung": "#3B82F6",
    "timbangan": "#F59E0B",
    "pedagang": "#F43F5E",
    "penjual": "#F43F5E",
    "sayuran": "#10B981",
    "sayur": "#10B981",
}

FALLBACK_PALETTE_HEX: List[str] = [
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#84CC16",  # Lime
    "#3B82F6",  # Blue
    "#F97316",  # Orange
    "#14B8A6",  # Teal
    "#EAB308",  # Yellow
    "#A855F7",  # Purple
    "#EF4444",  # Red
    "#22D3EE",  # Sky Cyan
    "#4ADE80",  # Bright Green
    "#FB7185",  # Rose
    "#38BDF8",  # Light Blue
]

SUPPORTED_MODEL_EXTS = {".pt", ".onnx"}


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert `#RRGGBB` string into OpenCV `(B, G, R)` integer tuple."""
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return (129, 185, 16)
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    return (b, g, r)


def resolve_class_color_hex(class_id: int, class_name: str) -> str:
    """
    Return the preset Hex color for our 5 target socio-economic market classes,
    or a deterministic high-visibility fallback color for any other class.
    """
    key = str(class_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in TARGET_CLASS_HEX:
        return TARGET_CLASS_HEX[key]

    if class_id == 0:
        return FALLBACK_PALETTE_HEX[0]
    digest = hashlib.md5(f"{class_id}:{key}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FALLBACK_PALETTE_HEX)
    return FALLBACK_PALETTE_HEX[idx]


class ModelManager:
    """Thread-safe YOLOv11 model lifecycle and inference configuration manager."""

    FALLBACK_PT_NAME = "yolo11n.pt"
    FALLBACK_ONNX_NAME = "yolo11n.onnx"

    def __init__(self, models_dir: Path, project_root: Path) -> None:
        self.models_dir = Path(models_dir)
        self.project_root = Path(project_root)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.device, self.device_label = self._detect_device()

        self.model: Any = None
        self.runtime_backend: str = "ultralytics" if TORCH_AVAILABLE else "onnxruntime"
        self.active_model_name: str = ""
        self.active_model_path: Optional[Path] = None
        self.is_production: bool = False
        self.is_fallback: bool = True

        # Inference thresholds & class filters
        self.conf_threshold: float = 0.35
        self.iou_threshold: float = 0.45
        self.class_names: Dict[int, str] = {}
        self.class_colors_hex: Dict[int, str] = {}
        self.class_colors_bgr: Dict[int, Tuple[int, int, int]] = {}
        self.enabled_class_ids: Optional[List[int]] = None

        self.initialize_startup_model()

    @staticmethod
    def _detect_device() -> Tuple[str, str]:
        """Detect CUDA GPU, Apple MPS, or CPU runtime hardware."""
        if TORCH_AVAILABLE and torch is not None:
            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "NVIDIA CUDA"
                return "cuda:0", f"CUDA GPU • {gpu_name}"
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps", "Apple Silicon MPS"

        ort_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in ort_providers:
            return "cuda:0", "CUDA GPU (ONNX Runtime)"

        cpu_arch = platform.machine() or "x86_64"
        suffix = "" if TORCH_AVAILABLE else " • ONNX"
        return "cpu", f"CPU ({cpu_arch}{suffix})"

    @property
    def lock(self) -> threading.RLock:
        """Expose re-entrant lock for thread-safe `predict` and `track` calls."""
        return self._lock

    def initialize_startup_model(self) -> Dict[str, Any]:
        """
        Check `models/best.pt` / `models/best.onnx` first; if absent, load `yolo11n` fallback.
        """
        best_pt = self.models_dir / "best.pt"
        best_onnx = self.models_dir / "best.onnx"

        if TORCH_AVAILABLE and best_pt.exists() and best_pt.is_file():
            return self.load_model_from_path(best_pt)
        if best_onnx.exists() and best_onnx.is_file():
            return self.load_model_from_path(best_onnx)

        return self._load_fallback_model()

    def _load_fallback_model(self) -> Dict[str, Any]:
        """Load `yolo11n.pt` when PyTorch is active, or official `yolo11n.onnx` under SAC."""
        with self._lock:
            if TORCH_AVAILABLE and UltralyticsYOLO is not None:
                target_pt = self.models_dir / self.FALLBACK_PT_NAME
                root_pt = self.project_root / self.FALLBACK_PT_NAME
                if not target_pt.exists() and root_pt.exists():
                    shutil.copy2(root_pt, target_pt)

                yolo_inst = UltralyticsYOLO(str(target_pt))
                cwd_pt = Path.cwd() / self.FALLBACK_PT_NAME
                if not target_pt.exists() and cwd_pt.exists():
                    shutil.move(str(cwd_pt), str(target_pt))

                self.runtime_backend = "ultralytics"
                self._bind_model(
                    model_instance=yolo_inst,
                    model_name=self.FALLBACK_PT_NAME,
                    model_path=target_pt,
                )
                return self.get_status()

            # Windows 11 Smart App Control safe path -> official Ultralytics yolo11n.onnx
            target_onnx = self.models_dir / self.FALLBACK_ONNX_NAME
            download_official_yolo11n_onnx(target_onnx)
            onnx_inst = ONNXYOLOModel(target_onnx)
            self.runtime_backend = "onnxruntime"
            self._bind_model(
                model_instance=onnx_inst,
                model_name=self.FALLBACK_ONNX_NAME,
                model_path=target_onnx,
            )
            return self.get_status()

    def load_model_from_path(self, path: Path) -> Dict[str, Any]:
        """Validate and activate a `.pt` or `.onnx` YOLOv11 weights file from disk."""
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model file not found: {resolved}")

        ext = resolved.suffix.lower()
        if ext not in SUPPORTED_MODEL_EXTS:
            raise ValueError("Only `.pt` or `.onnx` YOLOv11 weight files are supported.")

        with self._lock:
            if ext == ".onnx":
                inst = ONNXYOLOModel(resolved)
                self.runtime_backend = "onnxruntime"
                self._bind_model(inst, resolved.name, resolved)
                return self.get_status()

            # `.pt` requested
            if TORCH_AVAILABLE and UltralyticsYOLO is not None:
                pt_inst = UltralyticsYOLO(str(resolved))
                # Accelerate `.pt` on Intel Iris Xe GPU / OpenVINO when no NVIDIA CUDA GPU
                if OPENVINO_AVAILABLE and not str(self.device).startswith("cuda"):
                    try:
                        ov_dir = ensure_openvino_for_pt(resolved, pt_inst, imgsz=416)
                        raw_names = {int(k): str(v) for k, v in dict(pt_inst.names).items()}
                        ov_inst = OpenVINOYOLOModel(
                            ov_model_dir=ov_dir,
                            names=raw_names,
                            cache_dir=self.models_dir / ".ov_cache",
                            imgsz=416,
                        )
                        self.runtime_backend = "openvino"
                        self.device_label = ov_inst.device_label
                        self._bind_model(ov_inst, resolved.name, resolved)
                        return self.get_status()
                    except Exception:
                        pass

                self.runtime_backend = "ultralytics"
                self._bind_model(pt_inst, resolved.name, resolved)
                return self.get_status()

            # If `.pt` requested on a Windows 11 SAC-locked system, check for matching `.onnx`
            companion_onnx = resolved.with_suffix(".onnx")
            if companion_onnx.exists():
                inst = ONNXYOLOModel(companion_onnx)
                self.runtime_backend = "onnxruntime"
                self._bind_model(inst, companion_onnx.name, companion_onnx)
                return self.get_status()

            raise RuntimeError(
                "Windows Smart App Control on this laptop blocks unsigned PyTorch `.pt` DLLs. "
                "Export your trained model to ONNX in 1 line (`YOLO('best.pt').export(format='onnx')`) "
                "and upload or place `models/best.onnx`!"
            )

    def _bind_model(
        self,
        model_instance: Any,
        model_name: str,
        model_path: Path,
    ) -> None:
        """Update active model reference and dynamically populate `model.names`."""
        self.model = model_instance
        self.active_model_name = model_name
        self.active_model_path = model_path

        lower_name = model_name.lower()
        self.is_fallback = lower_name in {
            self.FALLBACK_PT_NAME,
            self.FALLBACK_ONNX_NAME,
        }
        self.is_production = not self.is_fallback

        raw_names = getattr(model_instance, "names", None) or {}
        if isinstance(raw_names, list):
            parsed_names = {int(i): str(n) for i, n in enumerate(raw_names)}
        else:
            parsed_names = {int(k): str(v) for k, v in dict(raw_names).items()}

        self.class_names = dict(sorted(parsed_names.items(), key=lambda item: item[0]))
        self.class_colors_hex = {}
        self.class_colors_bgr = {}
        for cid, cname in self.class_names.items():
            c_hex = resolve_class_color_hex(cid, cname)
            self.class_colors_hex[cid] = c_hex
            self.class_colors_bgr[cid] = hex_to_bgr(c_hex)

        self.enabled_class_ids = None

    def upload_and_swap_model(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Save an uploaded `.pt` or `.onnx` file into `models/` and hot-swap immediately."""
        safe_name = Path(filename or "best.pt").name
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_MODEL_EXTS:
            raise ValueError("Uploaded file must be a `.pt` or `.onnx` YOLOv11 weights file.")
        if len(content) < 1024:
            raise ValueError("Uploaded weights file appears empty or truncated.")

        if ext == ".pt" and not TORCH_AVAILABLE:
            raise ValueError(
                "Windows Smart App Control on this laptop requires `.onnx` weights. "
                "Run `YOLO('best.pt').export(format='onnx')` in Colab and upload `best.onnx`."
            )

        temp_path = self.models_dir / f".uploading_{safe_name}"
        final_path = self.models_dir / safe_name

        try:
            temp_path.write_bytes(content)
            if ext == ".onnx":
                test_inst = ONNXYOLOModel(temp_path)
                _ = test_inst.names
            elif UltralyticsYOLO is not None:
                test_inst = UltralyticsYOLO(str(temp_path))
                _ = test_inst.names

            if final_path.exists():
                final_path.unlink()
            temp_path.replace(final_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        return self.load_model_from_path(final_path)

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to another `.pt`/`.onnx` model in `models/` or fallback `yolo11n`."""
        clean_name = Path(model_name).name
        candidate = self.models_dir / clean_name
        if candidate.exists():
            return self.load_model_from_path(candidate)
        if clean_name.lower() in {self.FALLBACK_PT_NAME, self.FALLBACK_ONNX_NAME}:
            return self._load_fallback_model()
        raise FileNotFoundError(f"Model '{clean_name}' was not found in models/.")

    def update_inference_settings(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        enabled_class_ids: Optional[List[int]] = None,
        all_classes_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update confidence, IoU NMS, and per-class visibility filters."""
        with self._lock:
            if conf_threshold is not None:
                self.conf_threshold = round(max(0.10, min(0.90, float(conf_threshold))), 2)
            if iou_threshold is not None:
                self.iou_threshold = round(max(0.10, min(0.90, float(iou_threshold))), 2)

            if all_classes_enabled is True:
                self.enabled_class_ids = None
            elif enabled_class_ids is not None:
                valid_ids = [
                    int(cid)
                    for cid in enabled_class_ids
                    if int(cid) in self.class_names
                ]
                if len(valid_ids) == len(self.class_names):
                    self.enabled_class_ids = None
                else:
                    self.enabled_class_ids = sorted(set(valid_ids))

            return self.get_status()

    def get_effective_classes_arg(self) -> Optional[List[int]]:
        if self.enabled_class_ids is None:
            return None
        return list(self.enabled_class_ids)

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all compatible `.pt` / `.onnx` files in `models/`."""
        found: Dict[str, Dict[str, Any]] = {}
        for pattern in ("*.pt", "*.onnx"):
            for f in sorted(self.models_dir.glob(pattern)):
                if f.name.startswith("."):
                    continue
                if f.suffix.lower() == ".pt" and not TORCH_AVAILABLE:
                    continue
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                found[f.name] = {
                    "name": f.name,
                    "size_mb": size_mb,
                    "is_active": f.name == self.active_model_name,
                    "is_fallback": f.name.lower()
                    in {self.FALLBACK_PT_NAME, self.FALLBACK_ONNX_NAME},
                }

        default_fb = self.FALLBACK_PT_NAME if TORCH_AVAILABLE else self.FALLBACK_ONNX_NAME
        if default_fb not in found:
            found[default_fb] = {
                "name": default_fb,
                "size_mb": None,
                "is_active": self.active_model_name == default_fb,
                "is_fallback": True,
            }

        return sorted(
            found.values(),
            key=lambda m: (
                0
                if m["name"].lower() in {"best.pt", "best.onnx"}
                else (2 if m["is_fallback"] else 1),
                m["name"].lower(),
            ),
        )

    def get_badge_text(self) -> str:
        """Return status badge string per specification."""
        lower = self.active_model_name.lower()
        if lower in {"best.pt", "best.onnx"}:
            return f"Production Model: {self.active_model_name}"
        if self.is_fallback:
            return f"Demo Fallback: {self.active_model_name}"
        return f"Production Custom: {self.active_model_name}"

    def get_classes_metadata(self) -> List[Dict[str, Any]]:
        enabled_set = (
            set(self.class_names.keys())
            if self.enabled_class_ids is None
            else set(self.enabled_class_ids)
        )
        items: List[Dict[str, Any]] = []
        for cid, cname in self.class_names.items():
            items.append(
                {
                    "id": cid,
                    "name": cname,
                    "color_hex": self.class_colors_hex.get(cid, "#10B981"),
                    "enabled": cid in enabled_set,
                    "is_target_market_class": cname.strip().lower() in TARGET_CLASS_HEX,
                }
            )
        return items

    def get_status(self) -> Dict[str, Any]:
        nc = len(self.class_names)
        fallback_warning = (
            f"Demo Fallback Active ({self.active_model_name} — {nc} COCO classes). "
            "Drop your trained best.pt into models/best.pt to switch to the 5 Traditional "
            "Market classes (cart, customer, scale, seller, vegetables)."
            if self.is_fallback
            else None
        )
        return {
            "active_model": self.active_model_name,
            "runtime_backend": self.runtime_backend,
            "torch_available": TORCH_AVAILABLE,
            "is_production": self.is_production,
            "is_fallback": self.is_fallback,
            "badge_type": "production" if self.is_production else "fallback",
            "badge_text": self.get_badge_text(),
            "fallback_warning": fallback_warning,
            "device": self.device,
            "device_label": self.device_label,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": nc,
            "names": dict(self.class_names),
            "target_market_classes": list(TARGET_5_CLASSES),
            "classes": self.get_classes_metadata(),
            "available_models": self.list_available_models(),
        }

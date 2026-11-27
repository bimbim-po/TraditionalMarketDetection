"""
OpenCV Image Data Enhancement Module for PasarVision AI.

Provides artifact-free, luminance-preserving image enhancement pipelines tailored
for traditional market computer vision scenarios (e.g., dim dawn/subuh stalls,
harsh canopy shadows, and low-contrast indoor aisles):
1. Original (None)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization on LAB L-channel)
3. Histogram Equalization (HE on YCrCb Y-channel)
4. Contrast Stretching (Percentile linear stretching on LAB L-channel)
5. Brightness, Contrast Alpha & Gamma LUT Enhancement
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import cv2
import numpy as np


VALID_ENHANCEMENT_METHODS = {
    "none",
    "clahe",
    "he",
    "cs",
    "brightness_gamma",
}


@dataclass
class EnhancementConfig:
    """Configuration state for OpenCV frame enhancement."""

    method: str = "none"
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8
    cs_low_percentile: float = 2.0
    cs_high_percentile: float = 98.0
    brightness: int = 18
    contrast_alpha: float = 1.15
    gamma: float = 1.25  # >1.0 lifts dark shadows; <1.0 deepens tones; 1.0 neutral

    def normalized(self) -> "EnhancementConfig":
        """Return a clamped, validated copy of the configuration."""
        method = str(self.method or "none").strip().lower()
        if method not in VALID_ENHANCEMENT_METHODS:
            method = "none"

        clip_limit = float(np.clip(self.clahe_clip_limit, 1.0, 6.0))
        tile_raw = int(self.clahe_tile_grid)
        tile_grid = min((4, 8, 16), key=lambda x: abs(x - tile_raw))

        low_p = float(np.clip(self.cs_low_percentile, 0.5, 20.0))
        high_p = float(np.clip(self.cs_high_percentile, 80.0, 99.5))
        if high_p <= low_p + 1.0:
            low_p, high_p = 2.0, 98.0

        brightness = int(np.clip(round(self.brightness), -50, 80))
        contrast_alpha = float(np.clip(self.contrast_alpha, 0.7, 1.8))
        gamma = float(np.clip(self.gamma, 0.5, 2.0))

        return EnhancementConfig(
            method=method,
            clahe_clip_limit=round(clip_limit, 2),
            clahe_tile_grid=tile_grid,
            cs_low_percentile=round(low_p, 1),
            cs_high_percentile=round(high_p, 1),
            brightness=brightness,
            contrast_alpha=round(contrast_alpha, 2),
            gamma=round(gamma, 2),
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "EnhancementConfig":
        """Construct a validated EnhancementConfig from an arbitrary dict."""
        if not payload:
            return cls()
        base = asdict(cls())
        for key in base:
            if key in payload and payload[key] is not None:
                base[key] = payload[key]
        return cls(**base).normalized()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-friendly dictionary."""
        return asdict(self.normalized())


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure the input numpy array is a standard 3-channel uint8 BGR image."""
    if frame is None or frame.size == 0:
        raise ValueError("Input image frame is empty or None.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def apply_clahe_lab(
    frame_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE strictly to the L-channel in CIE LAB color space.
    Preserves natural chrominance without introducing RGB hue shifts.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_histogram_equalization_ycrcb(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply global Histogram Equalization to the luminance Y-channel in YCrCb space.
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_chan)
    merged = cv2.merge((y_eq, cr_chan, cb_chan))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def apply_contrast_stretching(
    frame_bgr: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """
    Apply percentile linear contrast stretching on the LAB L-channel so outliers
    (specular bulbs or deep corners) do not prevent full dynamic range expansion.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    p_low, p_high = np.percentile(l_chan, [low_percentile, high_percentile])
    span = float(p_high - p_low)
    if span < 1e-3:
        return frame_bgr.copy()

    l_float = (l_chan.astype(np.float32) - float(p_low)) * (255.0 / span)
    l_stretched = np.clip(l_float, 0.0, 255.0).astype(np.uint8)

    merged = cv2.merge((l_stretched, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_brightness_gamma(
    frame_bgr: np.ndarray,
    brightness: int = 18,
    contrast_alpha: float = 1.15,
    gamma: float = 1.25,
) -> np.ndarray:
    """
    Apply linear contrast (alpha) & brightness (beta) followed by fast 256-entry
    LUT Gamma correction. Gamma > 1.0 lifts dark shadows; Gamma < 1.0 darkens midtones.
    """
    adjusted = cv2.convertScaleAbs(
        frame_bgr,
        alpha=float(contrast_alpha),
        beta=float(brightness),
    )
    safe_gamma = max(float(gamma), 0.1)
    if abs(safe_gamma - 1.0) < 1e-3:
        return adjusted

    inv_gamma = 1.0 / safe_gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(adjusted, lut)


def apply_enhancement(
    frame: np.ndarray,
    config: EnhancementConfig | Dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Apply the selected OpenCV image enhancement pipeline to a BGR frame.
    """
    bgr = _ensure_bgr(frame)
    cfg = (
        config.normalized()
        if isinstance(config, EnhancementConfig)
        else EnhancementConfig.from_dict(config)
    )

    if cfg.method == "none":
        return bgr.copy()
    if cfg.method == "clahe":
        return apply_clahe_lab(
            bgr,
            clip_limit=cfg.clahe_clip_limit,
            tile_grid_size=cfg.clahe_tile_grid,
        )
    if cfg.method == "he":
        return apply_histogram_equalization_ycrcb(bgr)
    if cfg.method == "cs":
        return apply_contrast_stretching(
            bgr,
            low_percentile=cfg.cs_low_percentile,
            high_percentile=cfg.cs_high_percentile,
        )
    if cfg.method == "brightness_gamma":
        return apply_brightness_gamma(
            bgr,
            brightness=cfg.brightness,
            contrast_alpha=cfg.contrast_alpha,
            gamma=cfg.gamma,
        )
    return bgr.copy()


def describe_enhancement(config: EnhancementConfig | Dict[str, Any] | None = None) -> str:
    """Return a concise, human-readable label for UI HUDs and Excel reports."""
    cfg = (
        config.normalized()
        if isinstance(config, EnhancementConfig)
        else EnhancementConfig.from_dict(config)
    )
    if cfg.method == "none":
        return "Original (No Enhancement)"
    if cfg.method == "clahe":
        return (
            f"CLAHE (LAB L-Channel | clipLimit={cfg.clahe_clip_limit:.1f}, "
            f"tileGrid={cfg.clahe_tile_grid}x{cfg.clahe_tile_grid})"
        )
    if cfg.method == "he":
        return "Histogram Equalization (YCrCb Y-Channel)"
    if cfg.method == "cs":
        return (
            f"Contrast Stretching (LAB L | {cfg.cs_low_percentile:.1f}%–"
            f"{cfg.cs_high_percentile:.1f}%)"
        )
    if cfg.method == "brightness_gamma":
        return (
            f"Brightness & Gamma (beta={cfg.brightness:+d}, "
            f"alpha={cfg.contrast_alpha:.2f}, gamma={cfg.gamma:.2f})"
        )
    return "Original (No Enhancement)"

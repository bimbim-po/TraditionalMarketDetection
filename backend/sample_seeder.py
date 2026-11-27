"""
Out-of-the-Box Sample Media Seeder (`sample_seeder.py`).

If `samples/images/` or `samples/videos/` are empty on first launch, generates
realistic photographic test samples derived from Ultralytics' bundled real-world
assets (`bus.jpg` / `zidane.jpg`) with genuine dawn ("Pasar Subuh") low-light
underexposure and natural aisle framing so:
1. Original (Before) has realistically degraded low-light visibility.
2. CLAHE / HE / Brightness-Gamma visibly recover shadowed subjects (+Δ% recall & conf).
3. Subjects' bottom-center anchors `(cx, y2)` traverse squarely inside the default
   ROI aisle polygon (`y = 0.35..0.90`) for live Region Counter verification.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _find_ultralytics_assets() -> List[Path]:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        return []
    pkg_dir = Path(spec.origin).resolve().parent
    assets_dir = pkg_dir / "assets"
    found: List[Path] = []
    if assets_dir.exists():
        for name in ("bus.jpg", "zidane.jpg"):
            p = assets_dir / name
            if p.exists():
                found.append(p)
    return found


def _simulate_dawn_lowlight(bgr: np.ndarray) -> np.ndarray:
    """
    Simulate realistic dim, low-contrast early-morning indoor market lighting
    (05:00 AM canopy shadows) where CLAHE & Gamma recover shadowed pedestrians.
    """
    h, w = bgr.shape[:2]
    f = bgr.astype(np.float32) / 255.0

    # Nonlinear shadow compression typical of underexposed dawn CCTV sensors
    dimmed = np.power(np.clip(f, 0.0, 1.0), 2.25) * 0.24

    # Subtle canopy falloff toward edges
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.52, h * 0.45
    dist_sq = ((xx - cx) / (w * 0.72)) ** 2 + ((yy - cy) / (h * 0.72)) ** 2
    vignette = np.clip(1.06 - 0.45 * dist_sq, 0.42, 1.0)[..., None]

    out = dimmed * vignette
    out[..., 2] *= 1.05
    out[..., 0] *= 0.94
    return np.clip(out * 255.0, 2.0, 220.0).astype(np.uint8)


def _frame_with_aisle_floor(
    bgr: np.ndarray,
    out_w: int = 960,
    out_h: int = 600,
) -> np.ndarray:
    """
    Fit the full photographic scene with a natural dark asphalt/aisle foreground
    band at the bottom ~14% so pedestrians' feet `(cx, y2)` land cleanly around
    `y = 0.78..0.86` (squarely inside the default ROI polygon `0.35..0.90`).
    """
    canvas = np.full((out_h, out_w, 3), (26, 28, 30), dtype=np.uint8)
    usable_h = int(round(out_h * 0.85))
    h, w = bgr.shape[:2]
    scale = min(out_w / float(w), usable_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (out_w - nw) // 2
    y0 = usable_h - nh
    canvas[max(0, y0) : max(0, y0) + nh, x0 : x0 + nw] = resized

    # Smooth floor reflection/pavement transition below y = usable_h
    if usable_h < out_h and usable_h > 20:
        bottom_strip = canvas[usable_h - 16 : usable_h, :, :]
        flipped = cv2.flip(bottom_strip, 0)
        floor = cv2.resize(
            flipped, (out_w, out_h - usable_h), interpolation=cv2.INTER_LINEAR
        )
        floor = cv2.GaussianBlur(floor, (21, 21), 0)
        floor = (floor.astype(np.float32) * 0.45).astype(np.uint8)
        canvas[usable_h:, :, :] = floor

    return canvas


def ensure_sample_media(samples_dir: Path, force_rebuild: bool = False) -> None:
    """Populate `samples/images/` and `samples/videos/` if currently empty."""
    img_dir = samples_dir / "images"
    vid_dir = samples_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    existing_imgs = [
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    existing_vids = [
        p for p in vid_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    assets = _find_ultralytics_assets()
    if not assets:
        return

    # 1. Seed Sample Images
    if force_rebuild or not existing_imgs:
        primary_bgr = cv2.imread(str(assets[0]))
        if primary_bgr is not None:
            aisle_bgr = _frame_with_aisle_floor(primary_bgr, 960, 600)
            dawn_bgr = _simulate_dawn_lowlight(aisle_bgr)
            cv2.imwrite(
                str(img_dir / "pasar_subuh_lowlight_01.jpg"),
                dawn_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            cv2.imwrite(
                str(img_dir / "aktivitas_lorong_siang_02.jpg"),
                aisle_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )

        if len(assets) > 1:
            sec_bgr = cv2.imread(str(assets[1]))
            if sec_bgr is not None:
                sec_aisle = _frame_with_aisle_floor(sec_bgr, 960, 600)
                sec_dawn = _simulate_dawn_lowlight(sec_aisle)
                cv2.imwrite(
                    str(img_dir / "interaksi_pedagang_lowlight_03.jpg"),
                    sec_dawn,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )

    # 2. Seed Sample Video (`demo_pasar_subuh.mp4`)
    if force_rebuild or not existing_vids:
        src_bgr = cv2.imread(str(assets[0]))
        if src_bgr is None:
            return

        vw, vh = 854, 480
        pano_w = 1320
        pano = np.full((vh, pano_w, 3), (24, 26, 28), dtype=np.uint8)

        # Place primary scene so pedestrians' feet sit around y = 0.80 (384 px)
        scene_main = _frame_with_aisle_floor(src_bgr, 760, vh)
        pano[:, 120:880] = scene_main

        if len(assets) > 1:
            extra = cv2.imread(str(assets[1]))
            if extra is not None:
                scene_right = _frame_with_aisle_floor(extra, 520, vh)
                pano[:, 780:1300] = cv2.addWeighted(
                    pano[:, 780:1300], 0.15, scene_right, 0.85, 0
                )

        pano_lowlight = _simulate_dawn_lowlight(pano)

        out_vid = vid_dir / "demo_pasar_subuh.mp4"
        fps = 15.0
        total_frames = 75  # 5 seconds smooth clip

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (vw, vh))
        if not writer.isOpened():
            return

        try:
            max_x = pano_w - vw
            for f_idx in range(total_frames):
                t = f_idx / float(max(1, total_frames - 1))
                smooth_t = 0.5 - 0.5 * np.cos(np.pi * t)
                x0 = int(round(smooth_t * max_x))
                crop = pano_lowlight[:, x0 : x0 + vw].copy()
                writer.write(crop)
        finally:
            writer.release()

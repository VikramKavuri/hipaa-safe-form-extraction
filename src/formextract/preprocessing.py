"""Classical computer-vision preprocessing for VLM-friendly form images.

Pipeline stages (each optional and independently ablatable):

* **OSD orientation correction** — Tesseract Orientation & Script Detection
  rotates pages that were scanned sideways / upside-down.
* **Grayscale + min-max normalization** — stabilizes contrast across scanners.
* **CLAHE** — Contrast-Limited Adaptive Histogram Equalization to recover
  faint handwriting.
* **Deskew** — Hough-line based small-angle rotation correction.

Keeping these as standalone, parameterized functions makes them easy to toggle
for the ablation study in ``eval/`` and to unit-test in isolation.
"""

from __future__ import annotations

import os
import re
import shutil

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .logging_utils import get_logger

log = get_logger(__name__)


def correct_orientation_osd(pil_img: Image.Image) -> Image.Image:
    """Rotate ``pil_img`` upright using Tesseract OSD. Returns input on failure."""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate:\s+(\d+)", str(osd))
        if not m:
            return pil_img
        rotate = int(m.group(1))
        if rotate in (90, 180, 270):
            corrected = pil_img.rotate(-rotate, expand=True)
            log.info("OSD orientation correction applied: rotate %d degrees", -rotate)
            return corrected
        return pil_img
    except Exception as e:  # noqa: BLE001
        log.warning("OSD orientation failed: %s", e)
        return pil_img


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate small skew angle (deg) from near-horizontal Hough lines.

    Returns the median line angle clamped to [-10, 10]; ``0.0`` when no usable
    lines are found. Exposed separately so it can be unit-tested deterministically.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=200, maxLineGap=10
    )
    if lines is None or len(lines) == 0:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if -45 < ang < 45:
            angles.append(ang)
    if not angles:
        return 0.0
    return float(max(min(np.median(angles), 10), -10))


def _rotate_keep_bounds(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=-angle_deg, scale=1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        gray, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=255
    )


def preprocess_form_image(
    img_path: str,
    out_dir: str,
    *,
    do_debug: bool = False,
    do_clahe: bool = True,
    do_deskew: bool = True,
    tag: str = "",
) -> str:
    """Run the CV preprocessing chain and write the final PNG. Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    base = os.path.splitext(os.path.basename(img_path))[0]
    if tag:
        base = f"{tag}_{base}"

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if do_debug:
        cv2.imwrite(os.path.join(out_dir, f"{base}_step1_gray.png"), gray)

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        if do_debug:
            cv2.imwrite(os.path.join(out_dir, f"{base}_step3_clahe.png"), gray)

    deskewed = gray
    if do_deskew:
        angle_deg = estimate_skew_angle(gray)
        if angle_deg != 0.0:
            deskewed = _rotate_keep_bounds(gray, angle_deg)
            if do_debug:
                cv2.imwrite(
                    os.path.join(out_dir, f"{base}_step4_deskewed_{angle_deg:.2f}.png"),
                    deskewed,
                )

    out_path = os.path.join(out_dir, f"{base}_FINAL_preprocessed.png")
    cv2.imwrite(out_path, deskewed)
    return out_path


def save_corrected_page_image(
    input_img_path: str, output_img_path: str, *, enable_osd: bool = True
) -> None:
    """Save an orientation-corrected copy (or a plain copy if OSD disabled)."""
    if enable_osd:
        pil_img = Image.open(input_img_path)
        pil_corr = correct_orientation_osd(pil_img)
        pil_corr.save(output_img_path)
    else:
        shutil.copy2(input_img_path, output_img_path)

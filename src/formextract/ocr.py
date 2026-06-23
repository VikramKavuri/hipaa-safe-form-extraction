"""Tesseract OCR helpers for *spatial* localization of form fields.

The VLM reads the whole page, but checkboxes are notoriously unreliable for
VLMs (they tend to transcribe every printed option rather than judge which box
is ticked). So we use Tesseract here purely to find *where* a labelled row is
on the page, crop the region to the right of its colon, and hand that tight ROI
to a dedicated checkbox classifier (see ``checkbox.py``).
"""

from __future__ import annotations

import numpy as np
import pytesseract

from .parsing import normalize_text

BBox = tuple[int, int, int, int]


def ocr_data(gray_img: np.ndarray, psm: int = 6) -> dict:
    """Return Tesseract word-level data (LSTM engine) for ``gray_img``."""
    config = f"--oem 1 --psm {psm}"
    return pytesseract.image_to_data(
        gray_img, output_type=pytesseract.Output.DICT, config=config
    )


def find_line_anchor_bbox(
    gray_page: np.ndarray,
    anchor_keywords: list[str],
    search_band: tuple[float, float] = (0.0, 0.6),
) -> BBox | None:
    """Find the bounding box of the first text line containing all keywords.

    Search is restricted to a vertical band (fractions of page height) to avoid
    matching the wrong instance of a repeated label.
    """
    H, _W = gray_page.shape[:2]
    yb1 = int(H * search_band[0])
    yb2 = int(H * search_band[1])
    band = gray_page[yb1:yb2, :]
    data = ocr_data(band, psm=6)
    n = len(data["text"])

    lines: dict[tuple, list[int]] = {}
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    keys = [normalize_text(k) for k in anchor_keywords]
    best: BBox | None = None
    for idxs in lines.values():
        line_text = " ".join(normalize_text(data["text"][j]) for j in idxs)
        if all(k in line_text for k in keys):
            lefts = [data["left"][j] for j in idxs]
            tops = [data["top"][j] for j in idxs]
            rights = [data["left"][j] + data["width"][j] for j in idxs]
            bottoms = [data["top"][j] + data["height"][j] for j in idxs]
            bbox = (min(lefts), min(tops) + yb1, max(rights), max(bottoms) + yb1)
            if best is None or bbox[1] < best[1]:
                best = bbox
    return best


def find_colon_x_in_line(
    gray_page: np.ndarray, line_bbox: BBox, scan_right_padding: int = 900
) -> int | None:
    """Find the x-coordinate just past the label's colon on a given line."""
    H, W = gray_page.shape[:2]
    x1, y1, x2, y2 = line_bbox
    pad_y = 8
    yy1 = max(y1 - pad_y, 0)
    yy2 = min(y2 + pad_y, H)
    xx1 = max(x1, 0)
    xx2 = min(x2 + scan_right_padding, W)
    line_roi = gray_page[yy1:yy2, xx1:xx2]

    data = ocr_data(line_roi, psm=7)
    for i, t in enumerate(data["text"]):
        if ":" in str(t).strip():
            return xx1 + int(data["left"][i] + data["width"][i] * 0.8)

    # Fall back to character-box detection of the colon glyph itself.
    try:
        boxes = pytesseract.image_to_boxes(line_roi, config="--oem 1 --psm 7")
        colon_candidates: list[int] = []
        for row in boxes.splitlines():
            parts = row.split(" ")
            if len(parts) < 5 or parts[0] != ":":
                continue
            left, right = int(parts[1]), int(parts[3])
            colon_candidates.append((left + right) // 2)
        if colon_candidates:
            return xx1 + min(colon_candidates)
    except Exception:  # noqa: BLE001
        pass
    return None


def build_value_roi_after_colon(
    gray_page: np.ndarray,
    line_bbox: BBox,
    colon_x: int | None,
    roi_height: int = 95,
    pad_y: int = 25,
    right_margin: int = 10,
) -> tuple[np.ndarray, BBox, BBox]:
    """Crop the value region to the right of the colon.

    Returns ``(roi, value_bbox, key_bbox)`` where ``key_bbox`` is the label side
    (used only for debug overlays).
    """
    H, W = gray_page.shape[:2]
    lx1, ly1, _lx2, _ly2 = line_bbox
    y1 = max(ly1 - pad_y, 0)
    y2 = min(y1 + roi_height, H)

    if colon_x is None:
        colon_x = min(_lx2 + 10, W - 1)

    key_bbox = (lx1, y1, min(colon_x + 5, W - 1), y2)
    vx1 = min(colon_x + 2, W - 1)
    vx2 = max(W - right_margin, vx1 + 1)
    value_bbox = (vx1, y1, vx2, y2)
    roi = gray_page[y1:y2, vx1:vx2].copy()
    return roi, value_bbox, key_bbox

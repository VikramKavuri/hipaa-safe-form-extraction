"""Unit tests for deterministic CV helpers (no external binaries needed)."""

from __future__ import annotations

import cv2
import numpy as np

from formextract.preprocessing import estimate_skew_angle


def _blank_page() -> np.ndarray:
    return np.full((600, 800), 255, dtype=np.uint8)


def test_no_lines_returns_zero():
    # A blank page has no Hough lines -> skew estimate must be 0.0.
    assert estimate_skew_angle(_blank_page()) == 0.0


def test_horizontal_lines_give_near_zero_skew():
    img = _blank_page()
    for y in (150, 300, 450):
        cv2.line(img, (50, y), (750, y), color=0, thickness=2)
    assert abs(estimate_skew_angle(img)) < 1.0


def test_skew_is_clamped_to_ten_degrees():
    # Even a steeply tilted set of lines must clamp into [-10, 10].
    img = _blank_page()
    for y0 in (120, 260, 400):
        cv2.line(img, (50, y0), (750, y0 + 200), color=0, thickness=2)
    angle = estimate_skew_angle(img)
    assert -10.0 <= angle <= 10.0

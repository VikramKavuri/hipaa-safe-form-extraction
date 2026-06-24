"""Dedicated checkbox-state classification.

Strategy: localize the checkbox row with Tesseract (``ocr.py``), crop a tight
ROI, then ask the VLM a *narrow* yes/no-per-option question about that crop.
This consistently beats letting the page-level VLM "read" checkboxes, which
tends to return every printed option regardless of which box is ticked.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .backends import VLMBackend
from .logging_utils import get_logger
from .ocr import (
    build_value_roi_after_colon,
    find_colon_x_in_line,
    find_line_anchor_bbox,
)
from .parsing import strip_code_fences

log = get_logger(__name__)

_SELECTED_SCHEMA = {
    "type": "object",
    "properties": {"selected": {"type": "array", "items": {"type": "string"}}},
    "required": ["selected"],
}


def qwen_select_checked_options_from_roi(
    roi_image_path: str,
    options: list[str],
    multi_select: bool,
    backend: VLMBackend,
) -> list[str]:
    """Classify which of ``options`` are ticked in a cropped checkbox ROI."""
    prompt = f"""
You are a checkbox state classifier.
You will be given ONE cropped ROI image that contains ONE checkbox question row.
ALLOWED OPTIONS (return only from this list): {options}
multi_select = {str(multi_select).lower()}
RULES:
1) Only judge whether each checkbox square is marked.
2) An option is SELECTED only if its checkbox square has a visible mark (X, tick, filled, scribble).
3) Do NOT return all printed option text. Return ONLY selected options from ALLOWED OPTIONS.
4) If no checkbox is marked, return ["NA"].
5) If multi_select=false and more than one box is marked, return ["NA"] (do not guess).
6) Output MUST be JSON ONLY, matching this schema:
{{"selected": ["..."]}} or {{"selected": ["NA"]}}
""".strip()

    try:
        txt = strip_code_fences(
            backend.chat(
                prompt=prompt,
                image_paths=[roi_image_path],
                json_schema=_SELECTED_SCHEMA,
                temperature=0.0,
                max_tokens=200,
            )
        )
        data = json.loads(txt)
        selected = data.get("selected", ["NA"])
        allowed = set(options) | {"NA"}
        selected = [s for s in selected if s in allowed]
        return _coerce_selection(selected, multi_select)
    except Exception as e:  # noqa: BLE001
        log.warning("Checkbox classifier failed for ROI %s: %s", roi_image_path, e)
        return ["NA"]


def _coerce_selection(selected: list[str], multi_select: bool) -> list[str]:
    """Apply the single-/multi-select rules; pure for easy unit testing."""
    if not selected:
        return ["NA"]
    if not multi_select and selected != ["NA"] and len(selected) != 1:
        return ["NA"]
    return selected


def extract_checkbox_group(
    gray_page: np.ndarray,
    anchor_keywords: list[str],
    options: list[str],
    debug_name: str,
    multi_select: bool,
    *,
    outdir: str,
    debug: bool,
    backend: VLMBackend,
    search_band: tuple[float, float] = (0.05, 0.45),
) -> list[str]:
    """Localize one checkbox row and classify its ticked option(s)."""
    if debug:
        os.makedirs(outdir, exist_ok=True)

    label_bbox = find_line_anchor_bbox(gray_page, anchor_keywords, search_band=search_band)
    if label_bbox is None:
        log.warning("Label not found for keywords: %s", anchor_keywords)
        return []

    colon_x = find_colon_x_in_line(gray_page, label_bbox)
    roi, value_bbox, key_bbox = build_value_roi_after_colon(gray_page, label_bbox, colon_x)

    roi_path = os.path.join(outdir, f"{debug_name}_value_roi.png")
    if debug:
        overlay = cv2.cvtColor(gray_page.copy(), cv2.COLOR_GRAY2BGR)
        kx1, ky1, kx2, ky2 = key_bbox
        vx1, vy1, vx2, vy2 = value_bbox
        cv2.rectangle(overlay, (kx1, ky1), (kx2, ky2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(outdir, f"{debug_name}_key_value_overlay.png"), overlay)
        cv2.imwrite(roi_path, roi)
    else:
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(roi_path, roi)

    selected = qwen_select_checked_options_from_roi(
        roi_image_path=roi_path,
        options=options,
        multi_select=multi_select,
        backend=backend,
    )
    return [] if selected == ["NA"] else selected


def checkbox_results_for_page(
    page_index: int,
    page_image_path: str,
    *,
    file_tag: str,
    outdir: str,
    debug: bool,
    backend: VLMBackend,
) -> dict[str, object]:
    """Return checkbox-derived field overrides for a given page index."""
    results: dict[str, object] = {}
    bgr = cv2.imread(page_image_path)
    if bgr is None:
        return results

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # cv2 accepts dst=None at runtime; the stubs are stricter than reality.
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-overload]

    def group(keywords, options, name, multi, band):
        return extract_checkbox_group(
            gray,
            anchor_keywords=keywords,
            options=options,
            debug_name=name,
            multi_select=multi,
            outdir=outdir,
            debug=debug,
            backend=backend,
            search_band=band,
        )

    if page_index == 1:
        band = (0.05, 0.45)
        sel = group(
            ["method", "alarm", "activ"],
            ["Pull Station", "Smoke Detector", "Other"],
            f"{file_tag}_p1_method_alarm",
            True,
            band,
        )
        results["Method_of_Alarm_Activation"] = sel if sel else ["NA"]

        sel = group(
            ["evacuation", "type"],
            ["Full Evacuation to Outside", "Other"],
            f"{file_tag}_p1_evac_type",
            False,
            band,
        )
        results["Evacuation_Type"] = sel[0] if sel else "NA"

        sel = group(
            ["type", "evacuation"],
            ["Announced", "Unannounced", "Supervised"],
            f"{file_tag}_p1_type_of_evac",
            True,
            band,
        )
        results["Type_of_Evacuation"] = "; ".join(sel) if sel else "NA"

    elif page_index == 2:
        band = (0.65, 0.98)
        sel = group(
            ["evac", "time", "meet", "locat", "require"],
            ["Yes", "No"],
            f"{file_tag}_p2_meet_requirement",
            False,
            band,
        )
        results["Did_evacuation_time_meet_location_requirement"] = sel[0] if sel else "NA"

    return results

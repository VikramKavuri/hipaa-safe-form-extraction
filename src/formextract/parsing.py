"""Pure parsing / normalization helpers (no I/O, no model calls).

These are deliberately side-effect free so they are fast and trivial to unit
test — see ``tests/test_parsing.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .logging_utils import get_logger

log = get_logger(__name__)


def safe_stem(path_str: str) -> str:
    """File-system-safe stem of a path (alphanumerics, ``_``, ``.``, ``-``)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path_str).stem)


def normalize_text(s: str | None) -> str:
    """Lowercase and collapse non-alphanumerics to single spaces (for matching)."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def strip_code_fences(text: str) -> str:
    """Remove a leading ```/```json fence and trailing ``` if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text


def parse_model_json(response_text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from raw model output.

    Handles code fences and leading/trailing prose by slicing to the outermost
    ``{ ... }``. Returns ``{}`` on failure (logged), never raises.
    """
    text = strip_code_fences(response_text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 - intentionally robust for batch runs
        log.warning("JSON parse failed: %s", e)
        log.warning("Raw model output (truncated): %s", response_text[:500])
        return {}


def normalize_for_csv(val: Any) -> str:
    """Flatten a field value into a single CSV cell.

    Lists become ``"; "``-joined strings; empty / missing values become ``"NA"``.
    """
    if val is None:
        return "NA"
    if isinstance(val, list):
        cleaned = [str(x).strip() for x in val if str(x).strip() and str(x).strip().upper() != "NA"]
        return "; ".join(cleaned) if cleaned else "NA"
    s = str(val).strip()
    return s if s else "NA"

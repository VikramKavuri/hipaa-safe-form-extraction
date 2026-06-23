"""Streamlit-free helpers for the demo.

Kept import-light (no ``streamlit``) so the demo's data/logic layer can be
unit-tested and reused. Anything that needs Ollama/Tesseract lives behind the
``run_live_extraction`` function and is only imported when actually invoked.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
LABELS_PATH = ROOT / "eval" / "ground_truth" / "labels.jsonl"
REPORT_MD = ROOT / "eval" / "reports" / "metrics.md"
REPORT_PLOT = ROOT / "eval" / "reports" / "per_field.png"

sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Sample:
    source_file: str
    pdf_path: Path
    fields: dict[str, Any]


def load_samples(labels_path: Path = LABELS_PATH, sample_dir: Path = SAMPLE_DIR) -> list[Sample]:
    """Load synthetic samples and their ground-truth records."""
    if not labels_path.exists():
        return []
    samples: list[Sample] = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec["source_file"]
            samples.append(Sample(src, sample_dir / src, rec["fields"]))
    return samples


def render_pdf_pages(pdf_path: Path, dpi: int = 110) -> list[bytes]:
    """Render each PDF page to PNG bytes (uses PyMuPDF; no model needed)."""
    import fitz

    out: list[bytes] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            out.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    finally:
        doc.close()
    return out


def split_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a record into (checkbox_fields, other_fields) for display."""
    from formextract.config import CHECKBOX_FIELDS

    checkbox = {k: v for k, v in fields.items() if k in CHECKBOX_FIELDS}
    other = {k: v for k, v in fields.items() if k not in CHECKBOX_FIELDS}
    return checkbox, other


def load_report_markdown() -> str | None:
    return REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else None


def run_live_extraction(file_path: str) -> dict[str, Any]:
    """Run the REAL pipeline on an uploaded file (requires Ollama + Tesseract)."""
    from formextract.config import get_settings
    from formextract.pipeline import _load_prompt, process_single_form

    settings = get_settings()
    prompt = _load_prompt(settings)
    return process_single_form(file_path, settings, prompt)

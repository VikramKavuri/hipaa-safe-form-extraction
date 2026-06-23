"""Ablation study: quantify each pipeline component's contribution.

Re-runs the **real** pipeline with one toggle changed at a time and tabulates
the overall exact-match. This is the table that demonstrates engineering
judgment (it requires Ollama + Tesseract, so it is not run in CI).

Usage::

    python -m eval.ablations
"""

from __future__ import annotations

from pathlib import Path

from tabulate import tabulate

from .metrics import evaluate
from .predictors import RealPredictor, load_ground_truth

ROOT = Path(__file__).resolve().parents[1]

# name -> settings overrides applied on top of defaults
ABLATIONS: dict[str, dict] = {
    "full pipeline (baseline)": {},
    "no deskew": {"do_deskew": False},
    "no CLAHE": {"use_clahe": False},
    "no checkbox pass": {"checkbox_enabled": False},
    "no OSD orientation": {"enable_osd_orientation": False},
    "300 DPI (vs 600)": {"pdf_render_dpi": 300},
    "VLM only (no CV, no checkbox)": {
        "preprocess_enabled": False,
        "checkbox_enabled": False,
    },
}


def summarize(rows: list[tuple[str, float, float]]) -> str:
    """Render the ablation result rows as a markdown table (pure → testable)."""
    table = [[name, f"{exact:.1%}", f"{delta:+.1%}"] for name, exact, delta in rows]
    return tabulate(
        table, headers=["configuration", "exact-match", "Δ vs baseline"], tablefmt="github"
    )


def run(labels: Path | None = None, sample_dir: Path | None = None) -> str:
    labels = labels or ROOT / "eval" / "ground_truth" / "labels.jsonl"
    sample_dir = sample_dir or ROOT / "data" / "sample"
    gt = load_ground_truth(labels)

    results: dict[str, float] = {}
    for name, overrides in ABLATIONS.items():
        predictor = RealPredictor(sample_dir=sample_dir, **overrides)
        preds = {src: predictor.predict(src, fields) for src, fields in gt.items()}
        results[name] = evaluate(preds, gt).overall_exact_match

    baseline = results["full pipeline (baseline)"]
    rows = [(name, val, val - baseline) for name, val in results.items()]
    md = "# Ablation study\n\n" + summarize(rows) + "\n"
    out = ROOT / "eval" / "reports" / "ablations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    run()

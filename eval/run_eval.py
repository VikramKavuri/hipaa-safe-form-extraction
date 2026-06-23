"""Run the evaluation harness and write a metrics report + plot.

Examples
--------
Run on synthetic data with the mock predictor (no model needed)::

    python -m eval.run_eval --predictor mock

Run the real pipeline (needs Ollama + Tesseract)::

    python -m eval.run_eval --predictor real
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import EvalReport, evaluate
from .predictors import MockPredictor, RealPredictor, load_ground_truth
from .reporting import render_markdown, save_field_plot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "eval" / "ground_truth" / "labels.jsonl"
DEFAULT_SAMPLE_DIR = ROOT / "data" / "sample"
DEFAULT_OUT_DIR = ROOT / "eval" / "reports"


def run(
    predictor_name: str, labels: Path, sample_dir: Path, out_dir: Path, seed: int
) -> EvalReport:
    ground_truth = load_ground_truth(labels)
    if not ground_truth:
        raise SystemExit(f"No ground truth found at {labels}. Run generate_synthetic_forms first.")

    if predictor_name == "mock":
        predictor = MockPredictor(seed=seed)
        title = "Evaluation report (mock predictor - harness demo)"
    else:
        predictor = RealPredictor(sample_dir=sample_dir)
        title = "Evaluation report (real pipeline)"

    predictions = {src: predictor.predict(src, fields) for src, fields in ground_truth.items()}
    report = evaluate(predictions, ground_truth)

    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report, title=title)
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    save_field_plot(report, out_dir / "per_field.png")

    print(md)
    print(f"\n[written] {out_dir / 'metrics.md'}")
    print(f"[written] {out_dir / 'per_field.png'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictor", choices=["mock", "real"], default="mock")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.predictor, args.labels, args.sample_dir, args.out_dir, args.seed)


if __name__ == "__main__":
    main()

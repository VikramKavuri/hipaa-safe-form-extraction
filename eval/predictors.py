"""Predictors that turn form files into field dicts for evaluation.

* ``RealPredictor`` runs the actual ``formextract`` pipeline (needs Ollama +
  Tesseract). This is what produces the *real* numbers.
* ``MockPredictor`` perturbs the ground truth with a configurable error profile.
  It lets the whole harness — metrics, reporting, plots — run and be tested in
  CI without a GPU or model server, and gives a realistic-looking demo report.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_ground_truth(labels_path: str | Path) -> dict[str, dict[str, Any]]:
    gt: dict[str, dict[str, Any]] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt[rec["source_file"]] = rec["fields"]
    return gt


class Predictor(Protocol):
    def predict(self, source_file: str, fields: dict[str, Any]) -> dict[str, Any]:
        ...


class RealPredictor:
    """Run the production pipeline on the actual form file."""

    def __init__(self, sample_dir: str | Path, **overrides: Any) -> None:
        from formextract.config import get_settings
        from formextract.pipeline import _load_prompt

        self.sample_dir = Path(sample_dir)
        self.settings = get_settings()
        for k, v in overrides.items():  # used by the ablation runner
            setattr(self.settings, k, v)
        self.prompt = _load_prompt(self.settings)

    def predict(self, source_file: str, fields: dict[str, Any]) -> dict[str, Any]:
        from formextract.pipeline import process_single_form

        path = self.sample_dir / source_file
        return process_single_form(str(path), self.settings, self.prompt)


class MockPredictor:
    """Simulate an imperfect model by perturbing ground truth deterministically."""

    def __init__(self, seed: int = 0, scalar_error: float = 0.12, list_error: float = 0.18) -> None:
        self.rng = random.Random(seed)
        self.scalar_error = scalar_error
        self.list_error = list_error

    def predict(self, source_file: str, fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in fields.items():
            if isinstance(val, list):
                out[key] = self._perturb_list(val)
            else:
                out[key] = self._perturb_scalar(str(val))
        return out

    def _perturb_scalar(self, s: str) -> str:
        r = self.rng.random()
        if r < self.scalar_error * 0.4:
            return "NA"  # missed field
        if r < self.scalar_error and len(s) > 3:
            i = self.rng.randrange(len(s))  # single-char OCR slip
            return s[:i] + self.rng.choice("aeiotrs0") + s[i + 1 :]
        return s

    def _perturb_list(self, items: list[str]) -> list[str]:
        items = list(items)
        if not items:
            return ["NA"]
        if self.rng.random() < self.list_error and len(items) > 1:
            items.pop(self.rng.randrange(len(items)))  # drop one (recall miss)
        if self.rng.random() < self.list_error * 0.5:
            items.append("spurious")  # hallucinated extra (precision miss)
        return items or ["NA"]

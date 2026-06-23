"""Field-level evaluation metrics for extracted forms.

Two field families are scored differently because they fail differently:

* **Scalar fields** (names typed as ``str``): exact match after normalization,
  plus a fuzzy *similarity* (0-1) so a one-character OCR slip is not scored the
  same as a totally wrong value.
* **Set fields** (names typed as ``list[str]``, which includes all checkbox
  fields): micro-averaged **precision / recall / F1** over the normalized option
  multiset, plus exact-set match.

Everything here is pure (no I/O, no model) so it is fully unit-testable — see
``tests/test_metrics.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make `src` importable, then reuse the project's canonical normalizer so the
# eval harness and the runtime agree on what "equal" means.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formextract.parsing import normalize_text  # noqa: E402
from formextract.schema import FireDrillFields  # noqa: E402

try:  # fast C-backed similarity; graceful fallback keeps metrics importable
    from rapidfuzz.fuzz import ratio as _fuzz_ratio

    def _similarity(a: str, b: str) -> float:
        return _fuzz_ratio(a, b) / 100.0
except Exception:  # pragma: no cover - fallback path
    from difflib import SequenceMatcher

    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


# Fields whose schema type is ``list[str]`` are scored as sets.
LIST_FIELDS: frozenset[str] = frozenset(
    name
    for name, f in FireDrillFields.model_fields.items()
    if getattr(f.annotation, "__args__", None) is not None  # list[str]
)
SCALAR_FIELDS: frozenset[str] = frozenset(FireDrillFields.model_fields) - LIST_FIELDS

_NA = "na"


def to_item_set(value: Any) -> set[str]:
    """Normalize a (possibly ``'; '``-joined or list) value to a set of items."""
    if value is None:
        return set()
    items: Iterable[str] = (
        [str(x) for x in value] if isinstance(value, list) else str(value).split(";")
    )
    out = {normalize_text(it) for it in items}
    out = {it for it in out if it and it != _NA}
    return out


def to_scalar(value: Any) -> str:
    """Normalize a scalar value to a comparable string ('' for NA/blank)."""
    if isinstance(value, list):
        value = "; ".join(str(x) for x in value)
    s = normalize_text(value)
    return "" if s == _NA else s


@dataclass
class ScalarScore:
    field: str
    exact: int = 0
    n: int = 0
    sim_sum: float = 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def mean_similarity(self) -> float:
        return self.sim_sum / self.n if self.n else 0.0


@dataclass
class SetScore:
    field: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact_sets: int = 0
    n: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact_sets / self.n if self.n else 0.0


@dataclass
class EvalReport:
    scalars: dict[str, ScalarScore] = field(default_factory=dict)
    sets: dict[str, SetScore] = field(default_factory=dict)
    n_docs: int = 0

    @property
    def overall_exact_match(self) -> float:
        """Field-weighted exact-match across every (doc, field) cell."""
        exact = sum(s.exact for s in self.scalars.values()) + sum(
            s.exact_sets for s in self.sets.values()
        )
        n = sum(s.n for s in self.scalars.values()) + sum(s.n for s in self.sets.values())
        return exact / n if n else 0.0

    @property
    def macro_f1_sets(self) -> float:
        if not self.sets:
            return 0.0
        return sum(s.f1 for s in self.sets.values()) / len(self.sets)


def evaluate(
    predictions: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> EvalReport:
    """Compare ``predictions`` to ``ground_truth`` (both keyed by source_file).

    Each inner dict maps field name -> predicted/true value. Missing predictions
    are treated as empty (a fair penalty).
    """
    report = EvalReport()
    report.scalars = {f: ScalarScore(f) for f in SCALAR_FIELDS}
    report.sets = {f: SetScore(f) for f in LIST_FIELDS}

    common = [k for k in ground_truth if k in predictions]
    report.n_docs = len(common)

    for src in common:
        pred = predictions[src]
        gold = ground_truth[src]

        for fld in SCALAR_FIELDS:
            sc = report.scalars[fld]
            p = to_scalar(pred.get(fld))
            g = to_scalar(gold.get(fld))
            sc.n += 1
            sc.sim_sum += _similarity(p, g)
            if p == g:
                sc.exact += 1

        for fld in LIST_FIELDS:
            ss = report.sets[fld]
            p_set = to_item_set(pred.get(fld))
            g_set = to_item_set(gold.get(fld))
            ss.n += 1
            ss.tp += len(p_set & g_set)
            ss.fp += len(p_set - g_set)
            ss.fn += len(g_set - p_set)
            if p_set == g_set:
                ss.exact_sets += 1

    return report

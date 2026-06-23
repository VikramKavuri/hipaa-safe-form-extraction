"""Unit tests for the evaluation metrics (pure logic, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `eval` importable as a package

from eval.metrics import (  # noqa: E402
    LIST_FIELDS,
    SCALAR_FIELDS,
    evaluate,
    to_item_set,
    to_scalar,
)


def test_field_partition_covers_schema():
    assert LIST_FIELDS and SCALAR_FIELDS
    assert LIST_FIELDS.isdisjoint(SCALAR_FIELDS)
    # Known list field and known scalar field land in the right bucket.
    assert "Method_of_Alarm_Activation" in LIST_FIELDS
    assert "Site_Address" in SCALAR_FIELDS


def test_to_item_set_handles_joined_and_na():
    assert to_item_set("Pull Station; Smoke Detector") == {"pull station", "smoke detector"}
    assert to_item_set(["A", "NA", ""]) == {"a"}
    assert to_item_set("NA") == set()
    assert to_item_set(None) == set()


def test_to_scalar_normalizes_and_blanks_na():
    assert to_scalar("  Yes ") == "yes"
    assert to_scalar("NA") == ""
    assert to_scalar(["a", "b"]) == "a b"


def _gt():
    return {
        "f1.pdf": {
            "Site_Address": "123 Maple St",
            "was_drill_observed": "Yes",
            "Method_of_Alarm_Activation": ["Pull Station", "Smoke Detector"],
        }
    }


def test_perfect_predictions_score_100():
    gt = _gt()
    report = evaluate(gt, gt)
    assert report.overall_exact_match == 1.0
    assert report.sets["Method_of_Alarm_Activation"].f1 == 1.0


def test_scalar_error_is_penalized():
    gt = _gt()
    pred = {"f1.pdf": dict(gt["f1.pdf"], was_drill_observed="No")}
    report = evaluate(pred, gt)
    assert report.scalars["was_drill_observed"].exact_rate == 0.0


def test_set_precision_recall_on_partial_match():
    gt = {"f1.pdf": {"Method_of_Alarm_Activation": ["Pull Station", "Smoke Detector"]}}
    # Predict one correct + one spurious -> recall 1/2, precision 1/2.
    pred = {"f1.pdf": {"Method_of_Alarm_Activation": ["Pull Station", "Other"]}}
    s = evaluate(pred, gt).sets["Method_of_Alarm_Activation"]
    assert s.recall == 0.5
    assert s.precision == 0.5
    assert s.exact_rate == 0.0


def test_missing_prediction_counts_as_empty():
    gt = _gt()
    report = evaluate({"f1.pdf": {}}, gt)
    # Every scalar empty-vs-nonempty -> 0 exact (except those whose GT is also empty).
    assert report.scalars["Site_Address"].exact_rate == 0.0
    assert report.sets["Method_of_Alarm_Activation"].recall == 0.0

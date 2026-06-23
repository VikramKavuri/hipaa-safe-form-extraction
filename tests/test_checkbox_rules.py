"""Unit tests for the checkbox single/multi-select coercion rules."""

from __future__ import annotations

from formextract.checkbox import _coerce_selection


def test_empty_selection_becomes_na():
    assert _coerce_selection([], multi_select=False) == ["NA"]
    assert _coerce_selection([], multi_select=True) == ["NA"]


def test_single_select_with_one_choice_passes():
    assert _coerce_selection(["Yes"], multi_select=False) == ["Yes"]


def test_single_select_with_multiple_choices_rejected():
    # Ambiguous: more than one box marked on a single-select question -> NA.
    assert _coerce_selection(["Yes", "No"], multi_select=False) == ["NA"]


def test_multi_select_allows_multiple():
    assert _coerce_selection(["Announced", "Supervised"], multi_select=True) == [
        "Announced",
        "Supervised",
    ]


def test_na_passthrough_for_single_select():
    assert _coerce_selection(["NA"], multi_select=False) == ["NA"]

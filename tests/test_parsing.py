"""Unit tests for the pure parsing / normalization helpers."""

from __future__ import annotations

import pytest

from formextract.parsing import (
    normalize_for_csv,
    normalize_text,
    parse_model_json,
    safe_stem,
    strip_code_fences,
)


class TestParseModelJson:
    def test_plain_json(self):
        assert parse_model_json('{"a": 1}') == {"a": 1}

    def test_json_with_code_fence(self):
        assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        raw = 'Sure, here is the JSON:\n{"x": "y"}\nHope that helps!'
        assert parse_model_json(raw) == {"x": "y"}

    def test_malformed_returns_empty_dict(self):
        assert parse_model_json("not json at all") == {}

    def test_non_object_json_returns_empty_dict(self):
        # A bare list is valid JSON but not a field object.
        assert parse_model_json("[1, 2, 3]") == {}


class TestNormalizeForCsv:
    def test_none_becomes_na(self):
        assert normalize_for_csv(None) == "NA"

    def test_empty_string_becomes_na(self):
        assert normalize_for_csv("   ") == "NA"

    def test_list_is_semicolon_joined(self):
        assert normalize_for_csv(["a", "b"]) == "a; b"

    def test_list_filters_na_and_blanks(self):
        assert normalize_for_csv(["a", "NA", "  ", "b"]) == "a; b"

    def test_empty_list_becomes_na(self):
        assert normalize_for_csv([]) == "NA"
        assert normalize_for_csv(["NA"]) == "NA"

    def test_scalar_is_stripped(self):
        assert normalize_for_csv("  hello  ") == "hello"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Method of Alarm Activation", "method of alarm activation"),
        ("Yes/No?", "yes no"),
        (None, ""),
        ("  Multiple   Spaces  ", "multiple spaces"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```json\n{}\n```", "{}"),
        ("```\n{}\n```", "{}"),
        ("{}", "{}"),
    ],
)
def test_strip_code_fences(raw, expected):
    assert strip_code_fences(raw) == expected


def test_safe_stem_sanitizes():
    assert safe_stem("my form (final).pdf") == "my_form_final_"
    assert safe_stem("/a/b/Report-2024.PNG") == "Report-2024"

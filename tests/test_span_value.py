from __future__ import annotations

import json

import pytest

from codebook_lab.span_value import (
    label_for_offset,
    parse_span_value,
    serialize_span_value,
    tokenize_text,
)


# --------------------------------------------------------------------------
# parse_span_value
# --------------------------------------------------------------------------

def test_parse_none_and_empty_return_empty_list():
    assert parse_span_value(None) == []
    assert parse_span_value("") == []
    assert parse_span_value("   ") == []


def test_parse_valid_json_string():
    raw = '[{"start": 0, "end": 4, "text": "fear", "label": "Emotion"}]'
    assert parse_span_value(raw) == [
        {"start": 0, "end": 4, "text": "fear", "label": "Emotion"}
    ]


def test_parse_malformed_or_non_list_returns_empty():
    assert parse_span_value("[not valid json") == []
    assert parse_span_value("{}") == []  # valid JSON, not a list


def test_parse_list_passthrough_copies_dicts_and_drops_non_dicts():
    original = [{"start": 1, "end": 2}, "junk", 5]
    parsed = parse_span_value(original)
    assert parsed == [{"start": 1, "end": 2}]
    assert parsed[0] is not original[0]


def test_parse_nan_returns_empty_list():
    pd = pytest.importorskip("pandas")
    assert parse_span_value(pd.NA) == []
    assert parse_span_value(float("nan")) == []


# --------------------------------------------------------------------------
# serialize_span_value
# --------------------------------------------------------------------------

def test_serialize_empty_returns_empty_string():
    assert serialize_span_value([]) == ""
    assert serialize_span_value(None) == ""


def test_serialize_coerces_offsets_and_keeps_optional_fields():
    out = serialize_span_value([{"start": "3", "end": "7", "text": "word", "label": "L"}])
    assert json.loads(out) == [{"start": 3, "end": 7, "text": "word", "label": "L"}]


def test_serialize_omits_empty_label_and_skips_non_dicts():
    out = serialize_span_value([{"start": 0, "end": 1, "text": "a", "label": ""}, "junk"])
    assert json.loads(out) == [{"start": 0, "end": 1, "text": "a"}]


def test_serialize_skips_uncoercible_offsets():
    assert serialize_span_value([{"start": "x", "end": 1}]) == ""


def test_serialize_preserves_unicode():
    out = serialize_span_value([{"start": 0, "end": 2, "text": "café"}])
    assert "café" in out  # ensure_ascii=False


def test_round_trip_preserves_spans():
    spans = [
        {"start": 0, "end": 4, "text": "fear", "label": "Emotion"},
        {"start": 10, "end": 15, "text": "anger"},
    ]
    assert parse_span_value(serialize_span_value(spans)) == spans


# --------------------------------------------------------------------------
# tokenize_text
# --------------------------------------------------------------------------

def test_tokenize_splits_words_and_punctuation_excluding_whitespace():
    assert tokenize_text("hello, world") == [(0, 5, "hello"), (5, 6, ","), (7, 12, "world")]


def test_tokenize_empty():
    assert tokenize_text("") == []


def test_tokenize_offsets_round_trip_to_text():
    text = "a (b) c!"
    for start, end, token in tokenize_text(text):
        assert text[start:end] == token


# --------------------------------------------------------------------------
# label_for_offset
# --------------------------------------------------------------------------

def test_label_for_offset_returns_label_empty_or_none():
    spans = [
        {"start": 0, "end": 5, "label": "joy"},
        {"start": 6, "end": 10},  # unlabeled
    ]
    assert label_for_offset(spans, 0, 5) == "joy"   # covered + labeled
    assert label_for_offset(spans, 6, 10) == ""      # covered + unlabeled
    assert label_for_offset(spans, 20, 25) is None    # not covered

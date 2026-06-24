from __future__ import annotations

import json

from codebook_lab.annotate import _extract_span_response


TEXT = "Each footstep behind her grew louder; her heart pounded as she fumbled."


def test_extracts_array_response_and_fills_text_from_offsets():
    # Model omits the text field; it should be filled from the offsets.
    response = '[{"start": 38, "end": 55}]'
    result = _extract_span_response(response, text=TEXT)
    assert result == [{"start": 38, "end": 55, "text": TEXT[38:55]}]
    assert result[0]["text"] == "her heart pounded"


def test_extracts_response_wrapper_object():
    response = json.dumps({"response": [{"start": 0, "end": 4, "text": "Each"}]})
    result = _extract_span_response(response, text=TEXT)
    assert result == [{"start": 0, "end": 4, "text": "Each"}]


def test_drops_invalid_and_out_of_range_offsets():
    response = json.dumps([
        {"start": 0, "end": 4},        # valid
        {"start": 5, "end": 5},        # end <= start -> dropped
        {"start": -1, "end": 3},       # start < 0 -> dropped
        {"start": 0, "end": 9999},     # end > len(text) -> dropped
        {"start": "x", "end": 3},      # non-int -> dropped
        {"end": 3},                    # missing start -> dropped
        "junk",                        # non-dict -> dropped
    ])
    result = _extract_span_response(response, text=TEXT)
    assert result == [{"start": 0, "end": 4, "text": "Each"}]


def test_label_kept_only_when_allowed():
    response = json.dumps([
        {"start": 0, "end": 4, "label": "fear"},
        {"start": 5, "end": 13, "label": "not_a_real_label"},
    ])
    result = _extract_span_response(response, label_options=["fear", "joy"], text=TEXT)
    # First keeps its allowed label; second is kept but the bad label is stripped.
    assert result[0] == {"start": 0, "end": 4, "text": "Each", "label": "fear"}
    assert result[1] == {"start": 5, "end": 13, "text": TEXT[5:13]}
    assert "label" not in result[1]


def test_without_text_uses_model_text_and_skips_range_check():
    response = json.dumps([{"start": 100, "end": 110, "text": "model-said"}])
    result = _extract_span_response(response)  # no text -> no upper-bound check
    assert result == [{"start": 100, "end": 110, "text": "model-said"}]


def test_garbage_response_returns_empty_list():
    assert _extract_span_response("the model refused to answer", text=TEXT) == []
    assert _extract_span_response("", text=TEXT) == []

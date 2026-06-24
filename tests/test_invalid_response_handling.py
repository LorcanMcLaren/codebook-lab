from __future__ import annotations

import pytest

from codebook_lab.annotate import (
    _RETRY_REMINDER,
    classify_text,
    extract_json_response,
    normalize_retry_strategy,
)


def _single_dropdown_codebook() -> dict:
    return {
        "header_column": "id",
        "text_column": "text",
        "section_1": {
            "section_name": "1. Relevance",
            "section_instruction": "",
            "annotations": {
                "annotation_1": {
                    "name": "is_relevant",
                    "type": "dropdown",
                    "tooltip": "",
                    "options": ["Yes", "No"],
                }
            },
        },
    }


COL = "1. Relevance_is_relevant"


# --------------------------------------------------------------------------
# extract_json_response: invalid -> None across all types (no fabricated default)
# --------------------------------------------------------------------------

def test_checkbox_invalid_returns_none_valid_returns_binary():
    assert extract_json_response('{"response": "maybe"}', "checkbox") is None
    # Note: the no-JSON fallback still matches yes/no/true/false substrings in
    # prose; only genuinely unrecognizable text falls through to None.
    assert extract_json_response("garbage", "checkbox") is None
    assert extract_json_response('{"response": true}', "checkbox") == 1
    assert extract_json_response('{"response": false}', "checkbox") == 0


def test_likert_invalid_returns_none_valid_clamps():
    assert extract_json_response('{"response": "abc"}', "likert", min_value=1, max_value=5) is None
    assert extract_json_response("no number", "likert", min_value=1, max_value=5) is None
    assert extract_json_response('{"response": 3}', "likert", min_value=1, max_value=5) == 3
    assert extract_json_response('{"response": 9}', "likert", min_value=1, max_value=5) == 5  # clamp


def test_textbox_empty_returns_none_nonempty_kept():
    assert extract_json_response('{"response": ""}', "textbox") is None
    assert extract_json_response("   ", "textbox") is None
    assert extract_json_response('{"response": "a phrase"}', "textbox") == "a phrase"


# --------------------------------------------------------------------------
# normalize_retry_strategy
# --------------------------------------------------------------------------

def test_normalize_retry_strategy_falls_back_to_identical():
    assert normalize_retry_strategy("bogus") == "identical"
    assert normalize_retry_strategy(None) == "identical"
    assert normalize_retry_strategy("REPROMPT") == "reprompt"
    assert normalize_retry_strategy("temperature") == "temperature"


# --------------------------------------------------------------------------
# retry loop in classify_text
# --------------------------------------------------------------------------

def test_retry_recovers_after_invalid_then_valid(monkeypatch):
    seq = iter(["garbage", '{"response": "Yes"}'])
    calls = []

    def fake(chain, prompt, *args, **kwargs):
        calls.append(prompt)
        return next(seq)

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake)
    result, _, _ = classify_text(
        chain=object(), text="t", codebook=_single_dropdown_codebook(), retries=1
    )
    assert result[COL] == "Yes"
    assert len(calls) == 2


def test_no_retry_when_retries_zero(monkeypatch):
    calls = []

    def fake(chain, prompt, *args, **kwargs):
        calls.append(prompt)
        return "garbage"

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake)
    result, _, _ = classify_text(
        chain=object(), text="t", codebook=_single_dropdown_codebook(), retries=0
    )
    assert result[COL] is None
    assert len(calls) == 1  # single attempt, no retry


def test_reprompt_strategy_appends_reminder_on_retry(monkeypatch):
    seq = iter(["garbage", '{"response": "Yes"}'])
    prompts = []

    def fake(chain, prompt, *args, **kwargs):
        prompts.append(prompt)
        return next(seq)

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake)
    classify_text(
        chain=object(), text="t", codebook=_single_dropdown_codebook(),
        retries=1, retry_strategy="reprompt",
    )
    assert _RETRY_REMINDER not in prompts[0]
    assert _RETRY_REMINDER in prompts[1]


def test_temperature_strategy_uses_retry_chain_on_retry(monkeypatch):
    base_chain = object()
    retry_chain = object()
    seq = iter(["garbage", '{"response": "Yes"}'])
    chains = []

    def fake(chain, prompt, *args, **kwargs):
        chains.append(chain)
        return next(seq)

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake)
    classify_text(
        chain=base_chain, text="t", codebook=_single_dropdown_codebook(),
        retries=1, retry_strategy="temperature", retry_chain=retry_chain,
    )
    assert chains[0] is base_chain
    assert chains[1] is retry_chain

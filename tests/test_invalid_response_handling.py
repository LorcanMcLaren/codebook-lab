from __future__ import annotations

import pytest

from codebook_lab.annotate import (
    _RETRY_REMINDER,
    _extract_reasoning_content,
    AnnotationResponse,
    ChatSession,
    apply_classification_to_csv,
    classify_text,
    extract_json_response,
    generate_response,
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


def test_apply_classification_chat_modes_create_expected_sessions(tmp_path, monkeypatch):
    csv_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    csv_path.write_text("id,text\n1,first\n2,second\n")
    sessions_by_mode = {}

    def fake_classify_text(*args, **kwargs):
        sessions_by_mode.setdefault(kwargs["chat_mode"], []).append(kwargs["chat_session"])
        char_counts = args[5]
        timing_data = args[6]
        return ({COL: "Yes"}, char_counts, timing_data)

    monkeypatch.setattr("codebook_lab.annotate.classify_text", fake_classify_text)

    for mode in ("per_query", "per_text", "continuous"):
        apply_classification_to_csv(
            csv_path,
            output_path,
            _single_dropdown_codebook(),
            chain=object(),
            chat_mode=mode,
        )

    assert sessions_by_mode["per_query"] == [None, None]
    assert all(session is not None for session in sessions_by_mode["per_text"])
    assert sessions_by_mode["per_text"][0] is not sessions_by_mode["per_text"][1]
    assert sessions_by_mode["continuous"][0] is sessions_by_mode["continuous"][1]


def test_generate_response_captures_reasoning_trace_with_chat_session():
    class FakeRaw:
        content = '{"response": "Yes"}'
        additional_kwargs = {"reasoning_content": "because the text matches"}

    class FakeStructuredModel:
        def invoke(self, messages):
            self.messages = messages
            return {
                "parsed": AnnotationResponse(response="Yes"),
                "raw": FakeRaw(),
            }

    class FakeChain:
        def __init__(self):
            self.structured_model = FakeStructuredModel()

        def with_structured_output(self, *args, **kwargs):
            return self.structured_model

    chain = FakeChain()
    session = ChatSession()
    traces = []
    char_counts = {"input_chars": 0, "output_chars": 0}
    timing_data = {"total_inference_time": 0, "inference_count": 0}

    response = generate_response(
        chain,
        "Prompt",
        char_counts,
        timing_data,
        row_num=1,
        annotation_name=COL,
        annotation_type="dropdown",
        chat_session=session,
        reasoning_traces=traces,
        attempt=2,
        chat_mode="per_text",
    )

    assert response == '{"response":"Yes"}'
    assert len(session.messages) == 2
    assert traces[0]["reasoning"] == "because the text matches"
    assert traces[0]["attempt"] == 2
    assert traces[0]["chat_mode"] == "per_text"


def test_extract_reasoning_content_falls_back_to_think_tags():
    class Raw:
        content = "<think>step one\nstep two</think>{\"response\": \"Yes\"}"
        additional_kwargs = {}

    assert _extract_reasoning_content(Raw()) == "step one\nstep two"

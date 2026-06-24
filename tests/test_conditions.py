from __future__ import annotations

import json

import pandas as pd

from codebook_lab.annotate import classify_text, extract_json_response
from codebook_lab.metrics import evaluate_performance, extract_column_info_from_codebook


def _conditional_codebook() -> dict:
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
        "section_2": {
            "section_name": "2. Stance",
            "section_instruction": "",
            "annotations": {
                "annotation_1": {
                    "name": "stance",
                    "type": "dropdown",
                    "tooltip": "",
                    "options": ["Positive", "Negative"],
                    "condition": {
                        "section_key": "section_1",
                        "annotation_key": "annotation_1",
                        "value": "Yes",
                    },
                }
            },
        },
    }


def test_classify_text_skips_inactive_conditional_annotations(monkeypatch):
    codebook = _conditional_codebook()
    prompts_seen: list[str] = []
    responses = iter(
        [
            '{"response": "No"}',
        ]
    )

    def fake_generate_response(*args, **kwargs):
        prompts_seen.append(kwargs.get("annotation_name", ""))
        return next(responses)

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake_generate_response)

    result, _, _ = classify_text(
        chain=object(),
        text="Example text",
        codebook=codebook,
        prompt_type="standard",
        use_examples=False,
    )

    assert prompts_seen == ["1. Relevance_is_relevant"]
    assert result["1. Relevance_is_relevant"] == "No"
    assert "2. Stance_stance" in result
    assert result["2. Stance_stance"] is None


def test_metrics_ignore_non_applicable_conditional_rows(tmp_path):
    codebook = _conditional_codebook()
    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text(json.dumps(codebook))

    column_info = extract_column_info_from_codebook(codebook_path)
    merged_df = pd.DataFrame(
        {
            "1. Relevance_is_relevant_gt": ["No", "Yes"],
            "1. Relevance_is_relevant_llm": ["No", "No"],
            "2. Stance_stance_gt": [None, "Positive"],
            "2. Stance_stance_llm": [None, None],
        }
    )

    metrics = evaluate_performance(
        merged_df=merged_df,
        columns_to_compare=["2. Stance_stance"],
        column_info=column_info,
        process_textbox=False,
    )

    accuracy_scores = metrics[0]
    percentage_agreement_scores = metrics[6]

    assert accuracy_scores["2. Stance_stance"] == 0.0
    assert percentage_agreement_scores["2. Stance_stance"] == 0.0


def test_extract_json_response_normalizes_dropdown_options():
    options = ["Yes", "No"]

    assert extract_json_response(
        '{"response": " yes\\n"}',
        "dropdown",
        options=options,
    ) == "Yes"
    assert extract_json_response("  No\n", "dropdown", options=options) == "No"
    assert extract_json_response(
        '{"response": "JSON"}',
        "dropdown",
        options=options,
    ) is None
    assert extract_json_response("JSON\n", "dropdown", options=options) is None


def test_classify_text_stores_none_for_invalid_dropdown_outputs(monkeypatch):
    codebook = _conditional_codebook()
    calls = {"count": 0}

    def fake_generate_response(*args, **kwargs):
        # Always invalid, so every attempt (including retries) fails.
        calls["count"] += 1
        return "JSON\n"

    monkeypatch.setattr("codebook_lab.annotate.generate_response", fake_generate_response)

    result, _, _ = classify_text(
        chain=object(),
        text="Example text",
        codebook=codebook,
        prompt_type="standard",
        use_examples=False,
    )

    assert result["1. Relevance_is_relevant"] is None
    assert result["2. Stance_stance"] is None
    # The single applicable annotation is attempted twice (initial + one retry).
    assert calls["count"] == 2

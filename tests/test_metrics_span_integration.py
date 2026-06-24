from __future__ import annotations

import json

import pandas as pd
import pytest

from codebook_lab.metrics import run_metrics
from codebook_lab.span_value import serialize_span_value


COLUMN = "Spans_Evidence"


def _write_task(tmp_path):
    codebook = {
        "header_column": "title",
        "text_column": "text",
        "section_1": {
            "section_name": "Spans",
            "section_instruction": "",
            "annotations": {
                "annotation_1": {
                    "name": "Evidence",
                    "type": "span",
                    "tooltip": "",
                    "label_options": [],
                }
            },
        },
    }
    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text(json.dumps(codebook))

    texts = ["her heart pounded", "a calm still day"]
    gt_spans = [
        serialize_span_value([{"start": 0, "end": 17, "text": "her heart pounded"}]),
        serialize_span_value([]),
    ]
    gt = pd.DataFrame({"sample_id": [1, 2], "title": ["a", "b"], "text": texts, COLUMN: gt_spans})
    gt_path = tmp_path / "ground-truth.csv"
    gt.to_csv(gt_path, index=False)
    return codebook_path, gt_path, texts


def _run(tmp_path, codebook_path, gt_path, llm_spans, *, process_span=True):
    llm = pd.DataFrame(
        {"sample_id": [1, 2], "title": ["a", "b"], "text": ["x", "y"], COLUMN: llm_spans}
    )
    llm_path = tmp_path / "llm.csv"
    llm.to_csv(llm_path, index=False)

    return run_metrics(
        ground_truth_csv=str(gt_path),
        llm_output_csv=str(llm_path),
        label="test",
        output_csv=str(tmp_path / "metrics.csv"),
        model_id="test-model",
        codebook_path=str(codebook_path),
        report_file=str(tmp_path / "report.txt"),
        process_span=process_span,
    )


def test_run_metrics_routes_span_column_and_scores_perfect_match(tmp_path):
    codebook_path, gt_path, _ = _write_task(tmp_path)
    perfect = [
        serialize_span_value([{"start": 0, "end": 17, "text": "her heart pounded"}]),
        serialize_span_value([]),
    ]
    result = _run(tmp_path, codebook_path, gt_path, perfect)

    col = result.metrics_by_column[COLUMN]
    assert col["annotation_type"] == "span"
    # Span-specific metrics are present (not the classification ones).
    assert col["token_f1"] == pytest.approx(1.0)
    assert col["exact_match_f1"] == pytest.approx(1.0)
    assert col["char_iou"] == pytest.approx(1.0)


def test_run_metrics_span_imperfect_prediction_lowers_exact_match(tmp_path):
    codebook_path, gt_path, _ = _write_task(tmp_path)
    wrong = [
        # Wrong boundary on row 1 (off by a word) -> not an exact match.
        serialize_span_value([{"start": 0, "end": 9, "text": "her heart"}]),
        serialize_span_value([]),
    ]
    result = _run(tmp_path, codebook_path, gt_path, wrong)

    col = result.metrics_by_column[COLUMN]
    assert col["exact_match_f1"] < 1.0
    # Partial overlap still gives non-zero char IoU, proving real computation.
    assert 0.0 < col["char_iou"] < 1.0


def test_run_metrics_skips_span_column_when_process_span_false(tmp_path):
    codebook_path, gt_path, _ = _write_task(tmp_path)
    perfect = [
        serialize_span_value([{"start": 0, "end": 17, "text": "her heart pounded"}]),
        serialize_span_value([]),
    ]
    result = _run(tmp_path, codebook_path, gt_path, perfect, process_span=False)

    assert "skipped" in result.reports[COLUMN].lower()

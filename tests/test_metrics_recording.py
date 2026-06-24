from __future__ import annotations

import pandas as pd
import pytest

from codebook_lab.metrics import (
    METRIC_COLUMNS,
    RUN_COLUMNS,
    _metric_names_for_type,
    _per_query,
    format_metrics_summary,
    write_metrics,
)


SCORE_ARG_NAMES = [
    "accuracy_scores", "precision_scores", "recall_scores", "f1_scores",
    "cohen_kappa_scores", "krippendorff_alpha_scores", "percentage_agreement_scores",
    "spearman_corr_scores", "quadratic_kappa_scores", "norm_levenshtein_scores",
    "bleu_scores", "rouge1_f_scores", "rouge2_f_scores", "rougeL_f_scores",
    "cosine_scores", "bertscore_p_scores", "bertscore_r_scores", "bertscore_f1_scores",
    "token_f1_scores", "exact_match_f1_scores", "char_iou_scores",
]


def _score_kwargs(columns, value=0.5):
    return {name: {c: value for c in columns} for name in SCORE_ARG_NAMES}


def _write(tmp_path, *, run_id, columns, column_info, **efficiency):
    metrics_csv = tmp_path / "metrics.csv"
    runs_csv = tmp_path / "metrics_runs.csv"
    write_metrics(
        str(metrics_csv), str(runs_csv), run_id, "task", "model", None, 0.0, None,
        "codebook.json", columns,
        **_unpack_scores(columns),
        column_info=column_info,
        **efficiency,
    )
    return metrics_csv, runs_csv


def _unpack_scores(columns):
    # write_metrics takes the score dicts positionally; pass them as keywords.
    return _score_kwargs(columns)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_per_query_handles_missing_denominator():
    assert _per_query(400, 4) == 100
    assert _per_query(None, 4) is None
    assert _per_query(400, None) is None
    assert _per_query(400, 0) is None


def test_metric_names_for_type():
    assert _metric_names_for_type("span") == ["token_f1", "exact_match_f1", "char_iou"]
    assert _metric_names_for_type("dropdown")[0] == "percentage_agreement"
    assert "spearman_corr" in _metric_names_for_type("likert")
    assert "bleu" in _metric_names_for_type("textbox")
    # spans do not get percentage_agreement
    assert "percentage_agreement" not in _metric_names_for_type("span")


# --------------------------------------------------------------------------
# write_metrics: two tidy tables
# --------------------------------------------------------------------------

def test_runs_table_records_totals_and_per_query_averages(tmp_path):
    columns = ["S_drop", "S_span"]
    column_info = {"S_drop": {"type": "dropdown"}, "S_span": {"type": "span"}}
    _, runs_csv = _write(
        tmp_path, run_id="r1", columns=columns, column_info=column_info,
        n_queries=4, input_chars=400, output_chars=200,
        total_inference_time=8.0, avg_inference_time=2.0,
        energy_consumed=0.01, emissions=0.005,
    )
    runs = pd.read_csv(runs_csv)
    assert list(runs.columns) == RUN_COLUMNS
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["n_queries"] == 4
    assert row["total_input_chars"] == 400
    assert row["avg_input_chars_per_query"] == pytest.approx(100.0)
    assert row["avg_output_chars_per_query"] == pytest.approx(50.0)
    assert row["avg_energy_kwh_per_query"] == pytest.approx(0.0025)
    assert row["avg_emissions_kg_per_query"] == pytest.approx(0.00125)


def test_per_query_averages_blank_when_no_query_count(tmp_path):
    columns = ["S_drop"]
    column_info = {"S_drop": {"type": "dropdown"}}
    _, runs_csv = _write(
        tmp_path, run_id="r1", columns=columns, column_info=column_info,
        n_queries=None, input_chars=400,
    )
    row = pd.read_csv(runs_csv).iloc[0]
    assert pd.isna(row["avg_input_chars_per_query"])
    assert row["total_input_chars"] == 400


def test_metrics_table_is_long_with_only_applicable_metrics(tmp_path):
    columns = ["S_drop", "S_span"]
    column_info = {"S_drop": {"type": "dropdown"}, "S_span": {"type": "span"}}
    metrics_csv, _ = _write(
        tmp_path, run_id="r1", columns=columns, column_info=column_info, n_queries=2,
    )
    metrics = pd.read_csv(metrics_csv)
    assert list(metrics.columns) == METRIC_COLUMNS
    # dropdown -> 7 metrics, span -> 3 metrics
    assert len(metrics) == 10
    drop_metrics = set(metrics[metrics["column"] == "S_drop"]["metric"])
    assert "accuracy" in drop_metrics and "percentage_agreement" in drop_metrics
    span_metrics = set(metrics[metrics["column"] == "S_span"]["metric"])
    assert span_metrics == {"token_f1", "exact_match_f1", "char_iou"}


def test_schema_is_stable_when_appending_runs_with_different_codebooks(tmp_path):
    # First run: dropdown + span. Second run (same files): a likert column.
    cols1 = ["S_drop", "S_span"]
    info1 = {"S_drop": {"type": "dropdown"}, "S_span": {"type": "span"}}
    metrics_csv, runs_csv = _write(tmp_path, run_id="r1", columns=cols1, column_info=info1, n_queries=2)

    cols2 = ["Other_likert"]
    info2 = {"Other_likert": {"type": "likert"}}
    _write(tmp_path, run_id="r2", columns=cols2, column_info=info2, n_queries=2)

    metrics = pd.read_csv(metrics_csv)
    runs = pd.read_csv(runs_csv)
    # Columns never widen; new runs simply add rows.
    assert list(metrics.columns) == METRIC_COLUMNS
    assert list(runs.columns) == RUN_COLUMNS
    assert set(metrics["run_id"]) == {"r1", "r2"}
    assert len(runs) == 2


# --------------------------------------------------------------------------
# summary: per-query lines
# --------------------------------------------------------------------------

def test_summary_includes_per_query_efficiency():
    summary = format_metrics_summary(
        {"S_drop": {"annotation_type": "dropdown", "accuracy": 0.5}},
        total_inference_time=8.0, avg_inference_time=2.0,
        input_chars=400, output_chars=200, n_queries=4,
    )
    assert "Queries (model calls): 4" in summary
    assert "avg/query: 100" in summary  # 400 input chars / 4 queries

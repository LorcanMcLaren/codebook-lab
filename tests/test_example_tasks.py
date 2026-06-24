from __future__ import annotations

import csv
import json

import pytest

from codebook_lab.conditions import get_annotation_column_name, get_annotation_entries
from codebook_lab.examples import get_example_task_files, list_example_tasks
from codebook_lab.span_metrics import compute_span_metrics
from codebook_lab.span_value import parse_span_value


def _codebook_columns(codebook):
    return [
        get_annotation_column_name(section, annotation)
        for _, section, _, annotation in get_annotation_entries(codebook)
    ]


@pytest.mark.parametrize("task_name", list_example_tasks())
def test_ground_truth_columns_match_codebook(task_name):
    """Every annotation column the codebook declares must exist in the task's
    ground-truth.csv. Guards against codebook/ground-truth drift."""
    files = get_example_task_files(task_name)
    codebook = json.loads(files["codebook_path"].read_text())
    with files["ground_truth_csv"].open() as f:
        header = next(csv.reader(f))

    missing = [col for col in _codebook_columns(codebook) if col not in header]
    assert not missing, f"{task_name} ground-truth.csv missing columns: {missing}"


def test_discrete_emotions_is_a_valid_span_task():
    """The discrete-emotions task should exercise spans: it must declare span
    annotations, and every stored span's offsets must match its source text."""
    files = get_example_task_files("discrete-emotions")
    codebook = json.loads(files["codebook_path"].read_text())

    span_columns = [
        get_annotation_column_name(section, annotation)
        for _, section, _, annotation in get_annotation_entries(codebook)
        if annotation.get("type") == "span"
    ]
    assert span_columns, "discrete-emotions should declare at least one span annotation"

    rows = list(csv.DictReader(files["ground_truth_csv"].open()))
    text_column = codebook["text_column"]
    assert rows, "discrete-emotions ground-truth.csv should not be empty"

    for row in rows:
        text = row[text_column]
        for column in span_columns:
            for span in parse_span_value(row[column]):
                assert text[span["start"]:span["end"]] == span["text"], (
                    f"offset mismatch in {column} for item {row.get('item_id')}"
                )


def test_discrete_emotions_spans_are_self_consistent():
    """gt-vs-gt span metrics must be perfect, confirming the labels form a
    coherent, scorable target."""
    files = get_example_task_files("discrete-emotions")
    codebook = json.loads(files["codebook_path"].read_text())
    rows = list(csv.DictReader(files["ground_truth_csv"].open()))
    texts = [row[codebook["text_column"]] for row in rows]

    for _, section, _, annotation in get_annotation_entries(codebook):
        if annotation.get("type") != "span":
            continue
        column = get_annotation_column_name(section, annotation)
        values = [row[column] for row in rows]
        metrics = compute_span_metrics(
            texts, values, values, label_options=annotation.get("label_options") or None
        )
        assert metrics["token_f1"] == 1.0
        assert metrics["exact_match_f1"] == 1.0
        assert metrics["char_iou"] == 1.0

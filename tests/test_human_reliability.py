from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from codebook_lab.human_reliability import (
    build_human_ground_truth,
    calculate_human_reliability,
)
from codebook_lab.metrics import run_metrics


PARENT = "1. Parent_parent"
CHILD = "2. Child_child"


def write_codebook(path: Path) -> None:
    codebook = {
        "header_column": "sample_id",
        "text_column": "text",
        "section_1": {
            "section_name": "1. Parent",
            "section_instruction": "",
            "annotations": {
                "annotation_1": {
                    "name": "parent",
                    "type": "dropdown",
                    "tooltip": "",
                    "options": ["Yes", "No"],
                }
            },
        },
        "section_2": {
            "section_name": "2. Child",
            "section_instruction": "",
            "annotations": {
                "annotation_1": {
                    "name": "child",
                    "type": "dropdown",
                    "tooltip": "",
                    "options": ["A", "B", "C"],
                    "condition": {
                        "section_key": "section_1",
                        "annotation_key": "annotation_1",
                        "value": "Yes",
                    },
                }
            },
        },
    }
    path.write_text(json.dumps(codebook))


def row(sample_id: str, parent: str, child: str = "") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "text": f"text {sample_id}",
        "source": "demo",
        PARENT: parent,
        CHILD: child,
    }


def write_coder_files(tmp_path: Path, data: dict[str, list[dict[str, str]]]) -> dict[str, Path]:
    paths = {}
    for coder_id, rows in data.items():
        path = tmp_path / f"{coder_id}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths[coder_id] = path
    return paths


def setup_task(tmp_path: Path, data: dict[str, list[dict[str, str]]]) -> tuple[Path, dict[str, Path]]:
    codebook = tmp_path / "codebook.json"
    write_codebook(codebook)
    return codebook, write_coder_files(tmp_path, data)


def test_perfect_agreement_outputs_clean_metrics(tmp_path):
    rows = [
        row("S1", "Yes", "A"),
        row("S2", "No", ""),
        row("S3", "Yes", "B"),
    ]
    codebook, coder_csvs = setup_task(tmp_path, {coder: rows for coder in ["coder1", "coder2", "coder3"]})

    result = calculate_human_reliability(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        output_dir=tmp_path / "reliability",
    )

    assert result.validation_issues.empty
    assert result.disagreements.empty
    assert result.pairwise_icr["percentage_agreement"].dropna().eq(1.0).all()
    parent_multi = result.multirater_icr[result.multirater_icr["field"].eq(PARENT)].iloc[0]
    child_multi = result.multirater_icr[result.multirater_icr["field"].eq(CHILD)].iloc[0]
    assert parent_multi["fleiss_kappa"] == 1.0
    assert child_multi["n_applicable_items"] == 2
    assert (tmp_path / "reliability" / "summary.md").exists()


def test_parent_disagreement_does_not_inflate_child_multirater_denominator(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "coder1": [row("S1", "Yes", "A")],
            "coder2": [row("S1", "Yes", "B")],
            "coder3": [row("S1", "No", "")],
            "coder4": [row("S1", "No", "")],
        },
    )

    result = calculate_human_reliability(codebook_path=codebook, coder_csvs=coder_csvs)

    child_multi = result.multirater_icr[result.multirater_icr["field"].eq(CHILD)].iloc[0]
    child_pairs = result.pairwise_icr[result.pairwise_icr["field"].eq(CHILD)]
    assert child_multi["n_applicable_items"] == 0
    assert child_pairs["n_compared"].sum() == 1


def test_validation_reports_invalid_missing_forbidden_duplicate_and_assignment_issues(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "coder1": [row("S1", "Maybe", "A"), row("S1", "Yes", "A"), row("S2", "Yes", "A")],
            "coder2": [row("S1", "Yes", ""), row("S3", "Yes", "A")],
        },
    )
    assignment = tmp_path / "assignments.csv"
    pd.DataFrame(
        [
            {"sample_id": "S1", "coder_id": "coder1"},
            {"sample_id": "S1", "coder_id": "coder2"},
            {"sample_id": "S2", "coder_id": "coder2"},
        ]
    ).to_csv(assignment, index=False)

    result = calculate_human_reliability(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        assignment_csv=assignment,
    )

    issue_types = set(result.validation_issues["issue_type"])
    assert "duplicate_coder_item" in issue_types
    assert "unexpected_coder_assignment" in issue_types
    assert "unexpected_item" in issue_types
    assert "missing_assigned_item" in issue_types
    assert "invalid_label" in issue_types
    assert "missing_required_field" in issue_types
    assert "forbidden_child_value" in issue_types


def test_wide_assignment_manifest_is_supported(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "RA1": [row("S1", "Yes", "A")],
            "RA2": [row("S1", "Yes", "A")],
        },
    )
    assignment = tmp_path / "assignments.csv"
    pd.DataFrame([{"sample_id": "S1", "ra_1": "RA1", "ra_2": "RA2"}]).to_csv(assignment, index=False)

    result = calculate_human_reliability(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        assignment_csv=assignment,
    )

    assert result.validation_issues.empty
    assert result.pairwise_icr.loc[result.pairwise_icr["field"].eq(PARENT), "n_compared"].iloc[0] == 1


def test_build_ground_truth_majority_and_adjudication_queue(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "coder1": [row("S1", "Yes", "A"), row("S2", "Yes", "A")],
            "coder2": [row("S1", "Yes", "A"), row("S2", "Yes", "B")],
            "coder3": [row("S1", "Yes", "B"), row("S2", "Yes", "C")],
        },
    )

    result = build_human_ground_truth(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        output_dir=tmp_path / "ground_truth",
    )

    assert result.ground_truth.loc[result.ground_truth["sample_id"].eq("S1"), CHILD].iloc[0] == "A"
    assert result.ground_truth.loc[result.ground_truth["sample_id"].eq("S2"), CHILD].iloc[0] == ""
    assert len(result.adjudication_queue) == 1
    assert result.adjudication_queue.iloc[0]["sample_id"] == "S2"
    assert (tmp_path / "ground_truth" / "ground-truth.csv").exists()
    assert (tmp_path / "ground_truth" / "adjudication_queue.csv").exists()

    adjudications = result.adjudication_queue.copy()
    adjudications[CHILD] = "B"
    adjudications_csv = tmp_path / "resolved.csv"
    adjudications.to_csv(adjudications_csv, index=False)

    resolved = build_human_ground_truth(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        adjudications_csv=adjudications_csv,
    )

    assert resolved.adjudication_queue.empty
    assert resolved.ground_truth.loc[resolved.ground_truth["sample_id"].eq("S2"), CHILD].iloc[0] == "B"


def test_generated_ground_truth_works_with_run_metrics(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "coder1": [row("S1", "Yes", "A"), row("S2", "No", "")],
            "coder2": [row("S1", "Yes", "A"), row("S2", "No", "")],
            "coder3": [row("S1", "Yes", "A"), row("S2", "No", "")],
        },
    )
    gt_result = build_human_ground_truth(
        codebook_path=codebook,
        coder_csvs=coder_csvs,
        output_dir=tmp_path / "ground_truth",
    )
    llm_output = tmp_path / "llm.csv"
    gt_result.ground_truth.to_csv(llm_output, index=False)

    metrics = run_metrics(
        ground_truth_csv=tmp_path / "ground_truth" / "ground-truth.csv",
        llm_output_csv=llm_output,
        label="human-test",
        output_csv=tmp_path / "metrics.csv",
        model_id="model",
        codebook_path=codebook,
        report_file=tmp_path / "report.txt",
    )

    assert metrics.metrics_by_column[PARENT]["percentage_agreement"] == 1.0
    assert metrics.metrics_by_column[CHILD]["percentage_agreement"] == 1.0


def test_missing_required_id_column_is_reported_without_row_order_fallback(tmp_path):
    codebook = tmp_path / "codebook.json"
    write_codebook(codebook)
    coder_csv = tmp_path / "coder1.csv"
    pd.DataFrame([{"text": "text", PARENT: "Yes", CHILD: "A"}]).to_csv(coder_csv, index=False)

    result = calculate_human_reliability(codebook_path=codebook, coder_csvs={"coder1": coder_csv})

    assert result.validation_issues["issue_type"].tolist() == ["missing_id_column"]


def test_invalid_adjudicated_label_raises(tmp_path):
    codebook, coder_csvs = setup_task(
        tmp_path,
        {
            "coder1": [row("S1", "Yes", "A")],
            "coder2": [row("S1", "Yes", "B")],
        },
    )
    adjudications = tmp_path / "bad_adjudications.csv"
    pd.DataFrame([{"sample_id": "S1", PARENT: "Yes", CHILD: "Not an option"}]).to_csv(adjudications, index=False)

    with pytest.raises(ValueError, match="Invalid adjudicated label"):
        build_human_ground_truth(
            codebook_path=codebook,
            coder_csvs=coder_csvs,
            adjudications_csv=adjudications,
        )

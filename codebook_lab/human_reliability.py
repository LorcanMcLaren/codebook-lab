from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .conditions import (
    get_annotation_column_name,
    get_annotation_entries,
    is_annotation_applicable,
    normalize_annotation_response_value,
)
from .types import HumanGroundTruthResult, HumanReliabilityResult


DEFAULT_ID_COLUMN = "sample_id"


@dataclass(frozen=True)
class AnnotationField:
    column: str
    section_key: str
    annotation_key: str
    annotation: dict[str, Any]
    has_condition: bool


@dataclass
class CoderResponse:
    item_id: str
    coder_id: str
    raw_values: dict[str, Any]
    values: dict[str, Any]
    applicable: dict[str, bool]


@dataclass
class _PreparedHumanCoding:
    codebook: dict[str, Any]
    fields: list[AnnotationField]
    responses: dict[str, dict[str, CoderResponse]]
    validation_issues: pd.DataFrame
    coder_ids: list[str]
    assignments_by_item: dict[str, list[str]]
    metadata_by_item: dict[str, dict[str, Any]]
    source_columns: list[str]
    id_column: str


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def majority_threshold(n_coders: int) -> int:
    return (n_coders // 2) + 1


def load_codebook_fields(codebook_path: str | Path) -> tuple[dict[str, Any], list[AnnotationField]]:
    codebook = json.loads(Path(codebook_path).read_text())
    fields = []
    for section_key, section, annotation_key, annotation in get_annotation_entries(codebook):
        fields.append(
            AnnotationField(
                column=get_annotation_column_name(section, annotation),
                section_key=section_key,
                annotation_key=annotation_key,
                annotation=annotation,
                has_condition=isinstance(annotation.get("condition"), dict),
            )
        )
    if not fields:
        raise ValueError(f"No annotation fields found in {codebook_path}.")
    return codebook, fields


def normalize_response(annotation: dict[str, Any], raw_value: Any) -> Any:
    if is_blank(raw_value):
        return None

    annotation_type = annotation.get("type", "dropdown")
    value = normalize_annotation_response_value(annotation, raw_value)

    if annotation_type == "dropdown":
        options = annotation.get("options") or []
        if options and value not in options:
            return None
        return value

    if annotation_type == "checkbox":
        return value if value in {0, 1} else None

    if annotation_type == "likert":
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        minimum = annotation.get("min_value")
        maximum = annotation.get("max_value")
        if minimum is not None and numeric < int(minimum):
            return None
        if maximum is not None and numeric > int(maximum):
            return None
        return numeric

    if annotation_type == "textbox":
        return str(value).strip()

    return str(value).strip()


def _issue_record(
    issue_type: str,
    *,
    item_id: str = "",
    coder_id: str = "",
    field: str = "",
    value: Any = "",
    message: str = "",
) -> dict[str, Any]:
    return {
        "issue_type": issue_type,
        "item_id": item_id,
        "coder_id": coder_id,
        "field": field,
        "value": "" if value is None else value,
        "message": message,
    }


def _validation_dataframe(issues: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["issue_type", "item_id", "coder_id", "field", "value", "message"]
    return pd.DataFrame(issues, columns=columns)


def _normalize_coder_csvs(coder_csvs: Mapping[str, str | Path] | list[str | Path] | tuple[str | Path, ...]) -> dict[str, Path]:
    if isinstance(coder_csvs, Mapping):
        normalized = {str(coder_id): Path(path) for coder_id, path in coder_csvs.items()}
    else:
        normalized = {}
        for path_value in coder_csvs:
            path = Path(path_value)
            coder_id = path.stem
            if coder_id in normalized:
                raise ValueError(f"Duplicate inferred coder_id {coder_id!r}. Pass coder_csvs as a mapping instead.")
            normalized[coder_id] = path

    if not normalized:
        raise ValueError("At least one coder CSV is required.")
    return normalized


def _read_assignment_csv(path: str | Path | None, id_column: str) -> dict[str, list[str]]:
    if path is None:
        return {}

    df = pd.read_csv(path, keep_default_na=False)
    if id_column not in df.columns:
        raise ValueError(f"Assignment CSV is missing required ID column {id_column!r}.")

    assignments: dict[str, list[str]] = {}
    if "coder_id" in df.columns:
        for _, row in df.iterrows():
            item_id = str(row[id_column]).strip()
            coder_id = str(row["coder_id"]).strip()
            if item_id and coder_id:
                assignments.setdefault(item_id, [])
                if coder_id not in assignments[item_id]:
                    assignments[item_id].append(coder_id)
        return assignments

    coder_columns = [
        column
        for column in df.columns
        if re.fullmatch(r"(ra|coder)_\d+", str(column), flags=re.IGNORECASE)
    ]
    if not coder_columns:
        raise ValueError("Assignment CSV must contain either a coder_id column or wide coder columns such as ra_1.")

    for _, row in df.iterrows():
        item_id = str(row[id_column]).strip()
        if not item_id:
            continue
        assignments.setdefault(item_id, [])
        for column in coder_columns:
            coder_id = str(row.get(column, "")).strip()
            if coder_id and coder_id not in assignments[item_id]:
                assignments[item_id].append(coder_id)
    return assignments


def _build_response(
    *,
    codebook: dict[str, Any],
    fields: list[AnnotationField],
    item_id: str,
    coder_id: str,
    row: pd.Series,
) -> CoderResponse:
    raw_values = {field.column: row.get(field.column, "") for field in fields}
    values = {
        field.column: normalize_response(field.annotation, raw_values[field.column])
        for field in fields
    }
    applicable = {
        field.column: is_annotation_applicable(
            codebook,
            field.section_key,
            field.annotation_key,
            raw_values,
        )
        for field in fields
    }
    return CoderResponse(
        item_id=item_id,
        coder_id=coder_id,
        raw_values=raw_values,
        values=values,
        applicable=applicable,
    )


def _load_coder_responses(
    *,
    coder_paths: dict[str, Path],
    assignments_by_item: dict[str, list[str]],
    codebook: dict[str, Any],
    fields: list[AnnotationField],
    id_column: str,
) -> tuple[dict[str, dict[str, CoderResponse]], list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    responses: dict[str, dict[str, CoderResponse]] = defaultdict(dict)
    issues: list[dict[str, Any]] = []
    metadata_by_item: dict[str, dict[str, Any]] = {}
    source_columns: list[str] = []
    expected_items = set(assignments_by_item)

    for coder_id, path in coder_paths.items():
        df = pd.read_csv(path, keep_default_na=False)
        if not source_columns:
            source_columns = df.columns.tolist()

        if id_column not in df.columns:
            issues.append(
                _issue_record(
                    "missing_id_column",
                    coder_id=coder_id,
                    message=f"{path} does not contain required ID column {id_column!r}.",
                )
            )
            continue

        for field in fields:
            if field.column not in df.columns:
                issues.append(
                    _issue_record(
                        "missing_annotation_column",
                        coder_id=coder_id,
                        field=field.column,
                        message=f"{path} does not contain annotation column {field.column!r}.",
                    )
                )
                df[field.column] = ""

        duplicate_mask = df[id_column].astype(str).duplicated(keep=False)
        for item_id in df.loc[duplicate_mask, id_column].astype(str).tolist():
            issues.append(
                _issue_record(
                    "duplicate_coder_item",
                    item_id=item_id,
                    coder_id=coder_id,
                    message="Coder file contains duplicate rows for this item ID.",
                )
            )

        df = df.drop_duplicates(subset=[id_column], keep="first")
        for _, row in df.iterrows():
            item_id = str(row[id_column]).strip()
            if not item_id:
                issues.append(
                    _issue_record(
                        "blank_item_id",
                        coder_id=coder_id,
                        message=f"Coder row has a blank {id_column!r} value.",
                    )
                )
                continue

            if assignments_by_item:
                if item_id not in expected_items:
                    issues.append(
                        _issue_record(
                            "unexpected_item",
                            item_id=item_id,
                            coder_id=coder_id,
                            message="Item ID is not present in the assignment CSV.",
                        )
                    )
                    continue
                if coder_id not in assignments_by_item[item_id]:
                    issues.append(
                        _issue_record(
                            "unexpected_coder_assignment",
                            item_id=item_id,
                            coder_id=coder_id,
                            message="Coder submitted an item they were not assigned.",
                        )
                    )
                    continue

            metadata_by_item.setdefault(item_id, row.to_dict())
            responses[item_id][coder_id] = _build_response(
                codebook=codebook,
                fields=fields,
                item_id=item_id,
                coder_id=coder_id,
                row=row,
            )

    if assignments_by_item:
        for item_id, assigned_coders in assignments_by_item.items():
            for coder_id in assigned_coders:
                if coder_id not in responses.get(item_id, {}):
                    issues.append(
                        _issue_record(
                            "missing_assigned_item",
                            item_id=item_id,
                            coder_id=coder_id,
                            message="Assigned coder did not submit this item ID.",
                        )
                    )

    return responses, issues, metadata_by_item, source_columns


def _validate_responses(
    *,
    responses: dict[str, dict[str, CoderResponse]],
    fields: list[AnnotationField],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item_responses in responses.values():
        for response in item_responses.values():
            for field in fields:
                raw_value = response.raw_values[field.column]
                value = response.values[field.column]
                blank = is_blank(raw_value)
                applicable = response.applicable[field.column]

                if not blank and value is None:
                    issues.append(
                        _issue_record(
                            "invalid_label",
                            item_id=response.item_id,
                            coder_id=response.coder_id,
                            field=field.column,
                            value=raw_value,
                            message="Response is not a valid codebook label.",
                        )
                    )

                if applicable and value is None:
                    issues.append(
                        _issue_record(
                            "missing_required_field",
                            item_id=response.item_id,
                            coder_id=response.coder_id,
                            field=field.column,
                            message="Applicable annotation field is blank or invalid.",
                        )
                    )

                if not applicable and not blank:
                    issues.append(
                        _issue_record(
                            "forbidden_child_value",
                            item_id=response.item_id,
                            coder_id=response.coder_id,
                            field=field.column,
                            value=raw_value,
                            message="Field is filled even though its condition is not satisfied.",
                        )
                    )
    return issues


def _infer_assignments(responses: dict[str, dict[str, CoderResponse]]) -> dict[str, list[str]]:
    return {
        item_id: sorted(coder_responses)
        for item_id, coder_responses in sorted(responses.items())
    }


def _prepare_human_coding(
    *,
    codebook_path: str | Path,
    coder_csvs: Mapping[str, str | Path] | list[str | Path] | tuple[str | Path, ...],
    id_column: str,
    assignment_csv: str | Path | None,
) -> _PreparedHumanCoding:
    codebook, fields = load_codebook_fields(codebook_path)
    coder_paths = _normalize_coder_csvs(coder_csvs)
    assignments_by_item = _read_assignment_csv(assignment_csv, id_column)
    responses, load_issues, metadata_by_item, source_columns = _load_coder_responses(
        coder_paths=coder_paths,
        assignments_by_item=assignments_by_item,
        codebook=codebook,
        fields=fields,
        id_column=id_column,
    )
    if not assignments_by_item:
        assignments_by_item = _infer_assignments(responses)

    all_coder_ids = sorted(set(coder_paths) | {coder for coders in assignments_by_item.values() for coder in coders})
    validation_issues = _validation_dataframe(
        load_issues + _validate_responses(responses=responses, fields=fields)
    )
    return _PreparedHumanCoding(
        codebook=codebook,
        fields=fields,
        responses=responses,
        validation_issues=validation_issues,
        coder_ids=all_coder_ids,
        assignments_by_item=assignments_by_item,
        metadata_by_item=metadata_by_item,
        source_columns=source_columns,
        id_column=id_column,
    )


def _clean_label(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _item_is_in_multirater_denominator(
    field: AnnotationField,
    responses_by_coder: dict[str, CoderResponse],
    coders_for_item: list[str],
) -> bool:
    if not field.has_condition:
        return True
    applicable_count = sum(
        1
        for coder_id in coders_for_item
        if coder_id in responses_by_coder and responses_by_coder[coder_id].applicable[field.column]
    )
    return applicable_count >= majority_threshold(len(coders_for_item))


def _labels_for_item(
    field: AnnotationField,
    responses_by_coder: dict[str, CoderResponse],
    coders_for_item: list[str],
) -> dict[str, Any]:
    labels = {}
    for coder_id in coders_for_item:
        response = responses_by_coder.get(coder_id)
        if response is None:
            labels[coder_id] = None
            continue
        if field.has_condition and not response.applicable[field.column]:
            labels[coder_id] = None
            continue
        labels[coder_id] = _clean_label(response.values[field.column])
    return labels


def _percentage_agreement(labels_a: list[Any], labels_b: list[Any]) -> float:
    if not labels_a:
        return math.nan
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


def _safe_cohen_kappa(labels_a: list[Any], labels_b: list[Any]) -> float:
    if not labels_a:
        return math.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(cohen_kappa_score(labels_a, labels_b))
    except Exception:
        return math.nan


def _safe_krippendorff_alpha(items: list[dict[str, Any]], coder_ids: list[str]) -> float:
    if not items:
        return math.nan
    labels = sorted({label for item in items for label in item.values() if label is not None}, key=str)
    if len(labels) <= 1:
        return math.nan
    label_to_int = {label: index for index, label in enumerate(labels)}
    matrix = []
    for coder_id in coder_ids:
        matrix.append([
            np.nan if item.get(coder_id) is None else label_to_int[item[coder_id]]
            for item in items
        ])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(krippendorff.alpha(reliability_data=np.array(matrix), level_of_measurement="nominal"))
    except Exception:
        return math.nan


def _fleiss_kappa(items: list[dict[str, Any]]) -> float:
    if not items:
        return math.nan
    nonmissing_counts = [sum(1 for label in item.values() if label is not None) for item in items]
    if not nonmissing_counts or min(nonmissing_counts) < 2 or len(set(nonmissing_counts)) != 1:
        return math.nan

    n_ratings = nonmissing_counts[0]
    labels = sorted({label for item in items for label in item.values() if label is not None}, key=str)
    if len(labels) <= 1:
        return math.nan

    count_rows = []
    for item in items:
        counts = Counter(label for label in item.values() if label is not None)
        count_rows.append([counts.get(label, 0) for label in labels])

    counts = np.array(count_rows, dtype=float)
    n_items = counts.shape[0]
    p_j = counts.sum(axis=0) / (n_items * n_ratings)
    p_i = ((counts * counts).sum(axis=1) - n_ratings) / (n_ratings * (n_ratings - 1))
    p_bar = p_i.mean()
    p_e = (p_j * p_j).sum()
    if p_e == 1:
        return math.nan
    return float((p_bar - p_e) / (1 - p_e))


def _calculate_pairwise_icr(prepared: _PreparedHumanCoding) -> pd.DataFrame:
    rows = []
    for field in prepared.fields:
        for coder_a, coder_b in combinations(prepared.coder_ids, 2):
            labels_a = []
            labels_b = []
            for item_id, coders_for_item in prepared.assignments_by_item.items():
                if coder_a not in coders_for_item or coder_b not in coders_for_item:
                    continue
                response_a = prepared.responses.get(item_id, {}).get(coder_a)
                response_b = prepared.responses.get(item_id, {}).get(coder_b)
                if response_a is None or response_b is None:
                    continue
                if field.has_condition and not (
                    response_a.applicable[field.column] and response_b.applicable[field.column]
                ):
                    continue
                label_a = _clean_label(response_a.values[field.column])
                label_b = _clean_label(response_b.values[field.column])
                if label_a is None or label_b is None:
                    continue
                labels_a.append(label_a)
                labels_b.append(label_b)

            rows.append(
                {
                    "field": field.column,
                    "coder_a": coder_a,
                    "coder_b": coder_b,
                    "n_compared": len(labels_a),
                    "percentage_agreement": _percentage_agreement(labels_a, labels_b),
                    "cohen_kappa": _safe_cohen_kappa(labels_a, labels_b),
                }
            )
    return pd.DataFrame(rows)


def _calculate_multirater_icr(prepared: _PreparedHumanCoding) -> pd.DataFrame:
    rows = []
    for field in prepared.fields:
        metric_items: list[dict[str, Any]] = []
        applicable_items = 0
        assigned_counts = []

        for item_id, coders_for_item in prepared.assignments_by_item.items():
            responses_by_coder = prepared.responses.get(item_id, {})
            if not _item_is_in_multirater_denominator(field, responses_by_coder, coders_for_item):
                continue

            applicable_items += 1
            assigned_counts.append(len(coders_for_item))
            labels = _labels_for_item(field, responses_by_coder, coders_for_item)
            nonmissing = {coder_id: label for coder_id, label in labels.items() if label is not None}
            if len(nonmissing) < 2:
                continue
            metric_items.append(labels)

        unanimity_count = 0
        majority_count = 0
        for item in metric_items:
            labels = [label for label in item.values() if label is not None]
            counts = Counter(labels)
            if len(counts) == 1:
                unanimity_count += 1
            if counts and max(counts.values()) > (len(labels) / 2):
                majority_count += 1

        n_items = len(metric_items)
        rows.append(
            {
                "field": field.column,
                "n_applicable_items": applicable_items,
                "n_items": n_items,
                "n_coders_assigned": ";".join(str(value) for value in sorted(set(assigned_counts))),
                "krippendorff_alpha": _safe_krippendorff_alpha(metric_items, prepared.coder_ids),
                "fleiss_kappa": _fleiss_kappa(metric_items),
                "unanimity_rate": (unanimity_count / n_items) if n_items else math.nan,
                "majority_agreement_rate": (majority_count / n_items) if n_items else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _metadata_columns(prepared: _PreparedHumanCoding) -> list[str]:
    annotation_columns = {field.column for field in prepared.fields}
    columns = [column for column in prepared.source_columns if column not in annotation_columns]
    if prepared.id_column not in columns:
        columns.insert(0, prepared.id_column)
    return columns


def _build_disagreements(prepared: _PreparedHumanCoding) -> pd.DataFrame:
    rows = []
    metadata_columns = _metadata_columns(prepared)
    for item_id, coders_for_item in prepared.assignments_by_item.items():
        responses_by_coder = prepared.responses.get(item_id, {})
        for field in prepared.fields:
            if not _item_is_in_multirater_denominator(field, responses_by_coder, coders_for_item):
                continue
            labels = _labels_for_item(field, responses_by_coder, coders_for_item)
            observed = [label for label in labels.values() if label is not None]
            unique_labels = sorted(set(observed), key=str)
            if len(unique_labels) <= 1:
                continue

            metadata = prepared.metadata_by_item.get(item_id, {})
            row = {column: metadata.get(column, "") for column in metadata_columns}
            row[prepared.id_column] = item_id
            row.update(
                {
                    "field": field.column,
                    "n_labels": len(unique_labels),
                    "labels_observed": " | ".join(str(label) for label in unique_labels),
                }
            )
            for coder_id in prepared.coder_ids:
                row[coder_id] = labels.get(coder_id, "")
            rows.append(row)
    return pd.DataFrame(rows)


def _format_float(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    return f"{float(value):.3f}"


def _build_reliability_summary(
    *,
    validation_issues: pd.DataFrame,
    pairwise_icr: pd.DataFrame,
    multirater_icr: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> str:
    lines = ["# Human Reliability Summary", ""]
    lines.append("## Validation")
    if validation_issues.empty:
        lines.append("- No validation issues found.")
    else:
        for issue_type, count in validation_issues["issue_type"].value_counts().sort_index().items():
            lines.append(f"- {issue_type}: {count}")

    lines.extend(["", "## Pairwise ICR"])
    if pairwise_icr.empty:
        lines.append("- No pairwise metrics produced.")
    else:
        grouped = pairwise_icr.groupby("field", dropna=False).agg(
            n_pairs=("n_compared", lambda series: int((series > 0).sum())),
            mean_agreement=("percentage_agreement", "mean"),
            mean_cohen_kappa=("cohen_kappa", "mean"),
        )
        for field, row in grouped.iterrows():
            lines.append(
                f"- {field}: mean agreement={_format_float(row['mean_agreement'])}, "
                f"mean Cohen's kappa={_format_float(row['mean_cohen_kappa'])}, "
                f"pairs with data={row['n_pairs']}"
            )

    lines.extend(["", "## Multi-Rater ICR"])
    if multirater_icr.empty:
        lines.append("- No multi-rater metrics produced.")
    else:
        for _, row in multirater_icr.iterrows():
            lines.append(
                f"- {row['field']}: alpha={_format_float(row['krippendorff_alpha'])}, "
                f"Fleiss' kappa={_format_float(row['fleiss_kappa'])}, "
                f"unanimity={_format_float(row['unanimity_rate'])}, "
                f"majority={_format_float(row['majority_agreement_rate'])}, "
                f"n={int(row['n_items'])}"
            )

    lines.extend(["", "## Disagreements", f"- Rows requiring review: {len(disagreements)}"])
    return "\n".join(lines) + "\n"


def calculate_human_reliability(
    *,
    codebook_path: str | Path,
    coder_csvs: Mapping[str, str | Path] | list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    assignment_csv: str | Path | None = None,
) -> HumanReliabilityResult:
    """Validate human coder CSVs and calculate inter-coder reliability metrics."""
    prepared = _prepare_human_coding(
        codebook_path=codebook_path,
        coder_csvs=coder_csvs,
        id_column=id_column,
        assignment_csv=assignment_csv,
    )
    pairwise_icr = _calculate_pairwise_icr(prepared)
    multirater_icr = _calculate_multirater_icr(prepared)
    disagreements = _build_disagreements(prepared)
    summary = _build_reliability_summary(
        validation_issues=prepared.validation_issues,
        pairwise_icr=pairwise_icr,
        multirater_icr=multirater_icr,
        disagreements=disagreements,
    )

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        prepared.validation_issues.to_csv(output_path / "validation_issues.csv", index=False)
        pairwise_icr.to_csv(output_path / "pairwise_icr.csv", index=False)
        multirater_icr.to_csv(output_path / "multirater_icr.csv", index=False)
        disagreements.to_csv(output_path / "disagreements.csv", index=False)
        (output_path / "summary.md").write_text(summary)

    return HumanReliabilityResult(
        validation_issues=prepared.validation_issues,
        pairwise_icr=pairwise_icr,
        multirater_icr=multirater_icr,
        disagreements=disagreements,
        summary_text=summary,
        output_dir=output_path,
    )


def _strict_majority_label(labels: list[Any]) -> Any | None:
    labels = [label for label in labels if label is not None]
    if not labels:
        return None
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label if count > (len(labels) / 2) else None


def _output_columns(prepared: _PreparedHumanCoding) -> list[str]:
    metadata_columns = _metadata_columns(prepared)
    annotation_columns = [field.column for field in prepared.fields]
    allowed = set(metadata_columns) | set(annotation_columns)
    columns = [column for column in prepared.source_columns if column in allowed]
    for column in metadata_columns + annotation_columns:
        if column not in columns:
            columns.append(column)
    return columns


def _load_adjudications(
    *,
    path: str | Path | None,
    fields: list[AnnotationField],
    id_column: str,
) -> dict[tuple[str, str], Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, keep_default_na=False)
    if id_column not in df.columns:
        raise ValueError(f"Adjudications file is missing required ID column {id_column!r}.")

    adjudications = {}
    for _, row in df.iterrows():
        item_id = str(row[id_column]).strip()
        if not item_id:
            continue
        for field in fields:
            if field.column not in df.columns:
                continue
            resolved_label = row.get(field.column, "")
            if is_blank(resolved_label):
                continue
            normalized = normalize_response(field.annotation, resolved_label)
            if normalized is None:
                raise ValueError(
                    f"Invalid adjudicated label for {id_column}={item_id}, field={field.column}: {resolved_label!r}"
                )
            adjudications[(item_id, field.column)] = normalized
    return adjudications


def build_human_ground_truth(
    *,
    codebook_path: str | Path,
    coder_csvs: Mapping[str, str | Path] | list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    assignment_csv: str | Path | None = None,
    adjudications_csv: str | Path | None = None,
) -> HumanGroundTruthResult:
    """Build consensus human ground truth from coder CSVs and optional adjudications."""
    prepared = _prepare_human_coding(
        codebook_path=codebook_path,
        coder_csvs=coder_csvs,
        id_column=id_column,
        assignment_csv=assignment_csv,
    )
    adjudications = _load_adjudications(path=adjudications_csv, fields=prepared.fields, id_column=id_column)
    output_columns = _output_columns(prepared)
    field_columns = {field.column for field in prepared.fields}

    ground_truth_rows = []
    queue_rows: dict[str, dict[str, Any]] = {}
    for item_id, coders_for_item in prepared.assignments_by_item.items():
        metadata = prepared.metadata_by_item.get(item_id, {})
        gt_row = {
            column: "" if column in field_columns else metadata.get(column, "")
            for column in output_columns
        }
        gt_row[id_column] = item_id
        queue_row = dict(gt_row)
        has_unresolved = False
        prior_gt_values: dict[str, Any] = {}

        for field in prepared.fields:
            field_applicable = is_annotation_applicable(
                prepared.codebook,
                field.section_key,
                field.annotation_key,
                prior_gt_values,
            )
            if not field_applicable:
                gt_row[field.column] = ""
                queue_row[field.column] = ""
                prior_gt_values[field.column] = None
                continue

            labels = _labels_for_item(field, prepared.responses.get(item_id, {}), coders_for_item)
            label = adjudications.get((item_id, field.column))
            if label is None:
                label = _strict_majority_label(list(labels.values()))

            if label is None:
                has_unresolved = True
                gt_row[field.column] = ""
                queue_row[field.column] = ""
                prior_gt_values[field.column] = None
            else:
                gt_row[field.column] = label
                queue_row[field.column] = label
                prior_gt_values[field.column] = label

        ground_truth_rows.append(gt_row)
        if has_unresolved:
            queue_rows[item_id] = queue_row

    ground_truth = pd.DataFrame(ground_truth_rows, columns=output_columns)
    adjudication_queue = pd.DataFrame(queue_rows.values(), columns=output_columns)

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        ground_truth.to_csv(output_path / "ground-truth.csv", index=False)
        adjudication_queue.to_csv(output_path / "adjudication_queue.csv", index=False)
        prepared.validation_issues.to_csv(output_path / "validation_issues.csv", index=False)

    return HumanGroundTruthResult(
        ground_truth=ground_truth,
        adjudication_queue=adjudication_queue,
        validation_issues=prepared.validation_issues,
        output_dir=output_path,
    )

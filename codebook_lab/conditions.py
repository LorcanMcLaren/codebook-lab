from __future__ import annotations

from typing import Any

import pandas as pd


def get_sorted_annotation_keys(section_content: dict[str, Any]) -> list[str]:
    """Return annotation keys in the same stable order used by CodeBook Studio."""

    def sort_key(annotation_key: str) -> tuple[int, int | str]:
        suffix = annotation_key.split("_")[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, annotation_key)

    return sorted(section_content.get("annotations", {}).keys(), key=sort_key)


def get_annotation_column_name(section_content: dict[str, Any], annotation: dict[str, Any]) -> str:
    """Return the canonical CSV column name for an annotation."""
    return f"{section_content['section_name']}_{annotation['name']}"


def get_annotation_entries(codebook: dict[str, Any]) -> list[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    """Return all section/annotation entries in display order."""
    entries: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []

    for section_key, section_content in codebook.items():
        if not section_key.startswith("section_"):
            continue
        for annotation_key in get_sorted_annotation_keys(section_content):
            annotation = section_content.get("annotations", {}).get(annotation_key, {})
            entries.append((section_key, section_content, annotation_key, annotation))

    return entries


def get_annotation_lookup(
    codebook: dict[str, Any],
) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    """Build a lookup from stable section/annotation keys to annotation metadata."""
    return {
        (section_key, annotation_key): (section_content, annotation)
        for section_key, section_content, annotation_key, annotation in get_annotation_entries(codebook)
    }


def get_annotation_condition(annotation: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized condition block when one is present."""
    condition = annotation.get("condition")
    if not isinstance(condition, dict):
        return None

    section_key = condition.get("section_key")
    annotation_key = condition.get("annotation_key")
    if not section_key or not annotation_key:
        return None

    return {
        "section_key": section_key,
        "annotation_key": annotation_key,
        "value": condition.get("value"),
    }


def normalize_annotation_response_value(annotation: dict[str, Any], value: Any) -> Any:
    """Coerce stored responses into stable comparable values."""
    if pd.isna(value):
        return None

    annotation_type = annotation.get("type", "dropdown")
    if annotation_type == "dropdown":
        normalized = str(value).strip().strip("`").strip()
        if normalized == "":
            return None

        options = annotation.get("options") or []
        if not options:
            return normalized

        option_lookup = {str(option).strip().casefold(): option for option in options}
        return option_lookup.get(normalized.casefold())

    if annotation_type == "checkbox":
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes"}:
            return 1
        if lowered in {"0", "false", "no"}:
            return 0
        return value

    if annotation_type == "likert":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    if annotation_type == "textbox":
        return str(value).strip()

    return str(value).strip()


def is_annotation_applicable(
    codebook: dict[str, Any],
    section_key: str,
    annotation_key: str,
    response_values: dict[str, Any],
    lookup: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    visited: set[tuple[str, str]] | None = None,
) -> bool:
    """Return whether an annotation should be shown/generate for the current responses."""
    lookup = lookup or get_annotation_lookup(codebook)
    current_entry = lookup.get((section_key, annotation_key))
    if not current_entry:
        return True

    _, annotation = current_entry
    condition = get_annotation_condition(annotation)
    if not condition:
        return True

    target_key = (condition["section_key"], condition["annotation_key"])
    if target_key == (section_key, annotation_key):
        return True

    target_entry = lookup.get(target_key)
    if not target_entry:
        return True

    visited = visited or set()
    if (section_key, annotation_key) in visited:
        return True

    target_section_content, target_annotation = target_entry
    if not is_annotation_applicable(
        codebook,
        condition["section_key"],
        condition["annotation_key"],
        response_values,
        lookup=lookup,
        visited=visited | {(section_key, annotation_key)},
    ):
        return False

    target_column_name = get_annotation_column_name(target_section_content, target_annotation)
    actual_value = normalize_annotation_response_value(target_annotation, response_values.get(target_column_name))
    expected_value = normalize_annotation_response_value(target_annotation, condition.get("value"))

    if actual_value is None:
        return False
    if target_annotation.get("type") == "textbox" and actual_value == "":
        return False

    return actual_value == expected_value

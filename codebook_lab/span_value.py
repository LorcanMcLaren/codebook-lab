"""Parse and serialize span annotation values shared by Lab modules.

A span annotation cell stores a JSON-encoded list of
``{"start": int, "end": int, "text": str, "label"?: str}`` objects. Empty
lists serialize to the empty string so downstream "answered?" checks treat
them as unanswered.

The helpers mirror the equivalent module in CodeBook Studio so codebooks and
labelled CSVs round-trip cleanly between the two tools.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

try:
    import pandas as pd  # only used for NaN detection
except ImportError:  # pragma: no cover - pandas is a hard dependency in Lab
    pd = None  # type: ignore[assignment]


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_span_value(value: Any) -> list[dict[str, Any]]:
    """Parse a stored span annotation value into a list of span dicts."""
    if value is None:
        return []
    if isinstance(value, list):
        return [dict(span) for span in value if isinstance(span, dict)]
    if pd is not None:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return [dict(span) for span in parsed if isinstance(span, dict)]
    return []


def serialize_span_value(spans: Iterable[dict[str, Any]] | None) -> str:
    """Serialize a list of span dicts to a JSON string for CSV storage."""
    if not spans:
        return ""
    cleaned: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            entry: dict[str, Any] = {
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
            }
        except (TypeError, ValueError):
            continue
        if "text" in span and span["text"] is not None:
            entry["text"] = str(span["text"])
        if span.get("label"):
            entry["label"] = str(span["label"])
        cleaned.append(entry)
    if not cleaned:
        return ""
    return json.dumps(cleaned, ensure_ascii=False)


def tokenize_text(text: str) -> list[tuple[int, int, str]]:
    """Tokenize a string into ``(start, end, token)`` triples.

    Uses a simple Unicode-aware split: runs of word characters become one
    token, and each non-whitespace non-word character (punctuation) becomes
    its own token. Whitespace is excluded. Stable and dependency-free so
    Lab can compute token-level F1 without forcing a tokenizer choice.
    """
    if not text:
        return []
    return [(m.start(), m.end(), m.group(0)) for m in _TOKEN_PATTERN.finditer(text)]


def label_for_offset(spans: Iterable[dict[str, Any]], start: int, end: int) -> str | None:
    """Return the label assigned to ``[start, end)`` by the first covering span.

    Returns the empty string ``""`` when an unlabeled span covers the range
    (so callers can distinguish "covered, unlabeled" from "not covered"),
    or ``None`` when no span covers the range.
    """
    for span in spans:
        try:
            sp_start = int(span.get("start", 0))
            sp_end = int(span.get("end", 0))
        except (TypeError, ValueError):
            continue
        if sp_start <= start and sp_end >= end:
            label = span.get("label")
            if label:
                return str(label)
            return ""
    return None

"""Span-annotation evaluation metrics.

Implements three complementary scores per row, then averages across the
dataset:

* ``token_f1`` — Token-level F1, label-aware (BIO-style multiclass when the
  annotation has labels, binary inside/outside otherwise). Macro-averaged
  across non-"O" classes. This is the headline number used in cross-type
  comparisons because it sits on the same 0-1 scale as classification F1.

* ``exact_match_f1`` — Strict span-level F1. A predicted span counts as a
  TP only if some human span has identical ``(start, end[, label])``. P/R
  computed across all rows.

* ``char_iou`` — Mean character-level intersection-over-union, label-aware.
  Continuous boundary measure that's forgiving on near-misses.

The module is deliberately dependency-free (no sklearn import) so it works
in slim Lab installations and is easy to unit-test.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .span_value import parse_span_value, tokenize_text


_OUTSIDE = "O"
_UNLABELED = "__UNLABELED__"


def _ensure_span_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(s) for s in value if isinstance(s, dict)]
    return parse_span_value(value)


def _token_class(spans: Sequence[dict[str, Any]], start: int, end: int) -> str:
    """Return the class for one token range.

    Returns ``"O"`` if no span covers the token; ``"__UNLABELED__"`` if a
    covering span has no label; otherwise the span's label string.
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
            return _UNLABELED
    return _OUTSIDE


def _row_token_labels(
    text: str,
    gt_spans: Sequence[dict[str, Any]],
    pred_spans: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return aligned (gt_labels, pred_labels) per token in ``text``."""
    tokens = tokenize_text(text)
    gt_labels = [_token_class(gt_spans, s, e) for s, e, _ in tokens]
    pred_labels = [_token_class(pred_spans, s, e) for s, e, _ in tokens]
    return gt_labels, pred_labels


def _macro_f1_from_counts(tp: dict[str, int], fp: dict[str, int], fn: dict[str, int]) -> float:
    classes = sorted({*tp, *fp, *fn} - {_OUTSIDE})
    if not classes:
        return float("nan")
    f1_per_class = []
    for cls in classes:
        cls_tp = tp.get(cls, 0)
        cls_fp = fp.get(cls, 0)
        cls_fn = fn.get(cls, 0)
        precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) else 0.0
        recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) else 0.0
        if precision + recall == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * precision * recall / (precision + recall))
    return sum(f1_per_class) / len(f1_per_class)


def _row_char_iou(
    text_length: int,
    gt_spans: Sequence[dict[str, Any]],
    pred_spans: Sequence[dict[str, Any]],
    label_aware: bool,
) -> float:
    """Return the label-aware char IoU for one row.

    If ``label_aware`` is True, each character is tagged with its covering
    span's label (or ``__UNLABELED__``), and we compute |intersection on
    matching label| / |union of labelled chars|, treating mismatched-label
    overlap as neither agreement nor union. If False, we treat spans as
    plain coverage masks.
    """
    if text_length <= 0:
        if not gt_spans and not pred_spans:
            return 1.0
        return 0.0

    def mask(spans: Sequence[dict[str, Any]]) -> list[str | None]:
        out: list[str | None] = [None] * text_length
        for span in spans:
            try:
                s = max(0, int(span.get("start", 0)))
                e = min(text_length, int(span.get("end", 0)))
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            label = span.get("label") if label_aware else "*"
            label = str(label) if label else _UNLABELED
            for i in range(s, e):
                if out[i] is None:
                    out[i] = label
        return out

    gt_mask = mask(gt_spans)
    pr_mask = mask(pred_spans)

    intersection = 0
    union = 0
    for g, p in zip(gt_mask, pr_mask):
        if g is None and p is None:
            continue
        union += 1
        if g is not None and p is not None and g == p:
            intersection += 1
    if union == 0:
        return 1.0
    return intersection / union


def _span_tuple(span: dict[str, Any], label_aware: bool) -> tuple[int, int, str] | None:
    try:
        s = int(span["start"])
        e = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if e <= s:
        return None
    label = span.get("label") if label_aware else "*"
    return (s, e, str(label) if label else "")


def compute_span_metrics(
    texts: Iterable[str],
    gt_values: Iterable[Any],
    pred_values: Iterable[Any],
    *,
    label_options: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute aggregate token-F1, exact-match F1, and mean char-IoU.

    Args:
        texts: Iterable of source text strings (one per row).
        gt_values: Iterable of human-labelled span values (JSON strings or
            lists of dicts).
        pred_values: Iterable of model-generated span values, same shape.
        label_options: Optional list of allowed labels. Token-F1 and exact-
            match F1 are computed in label-aware mode when this is non-empty.

    Returns:
        Dict with keys ``token_f1``, ``exact_match_f1``, ``char_iou``.
    """
    label_aware = bool(label_options)

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}

    span_tp = 0
    span_fp = 0
    span_fn = 0

    iou_values: list[float] = []
    row_count = 0

    for text, gt_value, pred_value in zip(texts, gt_values, pred_values):
        row_count += 1
        text_str = "" if text is None else str(text)
        gt_spans = _ensure_span_list(gt_value)
        pred_spans = _ensure_span_list(pred_value)

        gt_labels, pred_labels = _row_token_labels(
            text_str,
            gt_spans if label_aware else [{**s, "label": s.get("label") or "_"} for s in gt_spans],
            pred_spans if label_aware else [{**s, "label": s.get("label") or "_"} for s in pred_spans],
        )
        for gt_l, pr_l in zip(gt_labels, pred_labels):
            if gt_l == pr_l and gt_l != _OUTSIDE:
                tp[gt_l] = tp.get(gt_l, 0) + 1
            else:
                if pr_l != _OUTSIDE:
                    fp[pr_l] = fp.get(pr_l, 0) + 1
                if gt_l != _OUTSIDE:
                    fn[gt_l] = fn.get(gt_l, 0) + 1

        gt_tuples = {t for t in (_span_tuple(s, label_aware) for s in gt_spans) if t}
        pred_tuples = {t for t in (_span_tuple(s, label_aware) for s in pred_spans) if t}
        span_tp += len(gt_tuples & pred_tuples)
        span_fp += len(pred_tuples - gt_tuples)
        span_fn += len(gt_tuples - pred_tuples)

        iou_values.append(_row_char_iou(len(text_str), gt_spans, pred_spans, label_aware))

    if row_count == 0:
        nan = float("nan")
        return {"token_f1": nan, "exact_match_f1": nan, "char_iou": nan}

    token_f1 = _macro_f1_from_counts(tp, fp, fn)

    if span_tp + span_fp + span_fn == 0:
        exact_match_f1 = float("nan")
    else:
        precision = span_tp / (span_tp + span_fp) if (span_tp + span_fp) else 0.0
        recall = span_tp / (span_tp + span_fn) if (span_tp + span_fn) else 0.0
        if precision + recall == 0:
            exact_match_f1 = 0.0
        else:
            exact_match_f1 = 2 * precision * recall / (precision + recall)

    char_iou = sum(iou_values) / len(iou_values) if iou_values else float("nan")

    return {
        "token_f1": float(token_f1),
        "exact_match_f1": float(exact_match_f1),
        "char_iou": float(char_iou),
    }

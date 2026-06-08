from __future__ import annotations

import math

import pytest

from codebook_lab.span_metrics import compute_span_metrics


def test_empty_inputs_return_nan():
    result = compute_span_metrics([], [], [])
    assert math.isnan(result["token_f1"])
    assert math.isnan(result["exact_match_f1"])
    assert math.isnan(result["char_iou"])


def test_perfect_unlabeled_match_scores_one():
    text = "The quick brown fox jumps over the lazy dog."
    gt = [{"start": 4, "end": 19}]  # "quick brown fox"
    pred = [{"start": 4, "end": 19}]

    result = compute_span_metrics([text], [gt], [pred])

    assert result["exact_match_f1"] == pytest.approx(1.0)
    assert result["char_iou"] == pytest.approx(1.0)
    # Token F1 is macro across non-O classes; with a single matched class it's 1.
    assert result["token_f1"] == pytest.approx(1.0)


def test_disjoint_spans_score_zero():
    text = "abcdefghij"
    gt = [{"start": 0, "end": 3}]   # "abc"
    pred = [{"start": 5, "end": 8}]  # "fgh"

    result = compute_span_metrics([text], [gt], [pred])

    assert result["exact_match_f1"] == 0.0
    assert result["char_iou"] == 0.0


def test_partial_overlap_gives_intermediate_char_iou():
    text = "abcdefghij"
    gt = [{"start": 0, "end": 5}]   # "abcde"
    pred = [{"start": 3, "end": 8}]  # "defgh"

    result = compute_span_metrics([text], [gt], [pred])

    # Intersection 2 ("de"), union 8 ("abcdefgh") → 0.25
    assert result["char_iou"] == pytest.approx(0.25)
    # Exact span boundaries differ → no exact match TP
    assert result["exact_match_f1"] == 0.0


def test_label_aware_mismatch_does_not_count_as_match():
    text = "She felt joyful but a little anxious."
    gt = [{"start": 9, "end": 16, "label": "Joy"}]      # "joyful"
    pred = [{"start": 9, "end": 16, "label": "Fear"}]   # same span, wrong label

    label_options = ["Joy", "Sadness", "Anger", "Fear"]
    result = compute_span_metrics([text], [gt], [pred], label_options=label_options)

    # Labels disagree, so exact-match treats this as FP and FN.
    assert result["exact_match_f1"] == 0.0
    # Char-IoU is label-aware: characters are tagged but labels don't match,
    # so no agreement on those positions; the union counts both labels' chars.
    assert result["char_iou"] == 0.0


def test_label_aware_matching_label_scores_one():
    text = "She felt joyful but a little anxious."
    gt = [{"start": 9, "end": 16, "label": "Joy"}]
    pred = [{"start": 9, "end": 16, "label": "Joy"}]

    result = compute_span_metrics(
        [text], [gt], [pred], label_options=["Joy", "Fear"]
    )

    assert result["exact_match_f1"] == pytest.approx(1.0)
    assert result["char_iou"] == pytest.approx(1.0)
    assert result["token_f1"] == pytest.approx(1.0)


def test_json_string_inputs_are_parsed():
    text = "alpha beta gamma"
    gt_json = '[{"start": 0, "end": 5}]'
    pred_json = '[{"start": 0, "end": 5}]'

    result = compute_span_metrics([text], [gt_json], [pred_json])

    assert result["exact_match_f1"] == pytest.approx(1.0)


def test_no_spans_in_either_side_yields_nan_or_neutral():
    text = "nothing here"
    result = compute_span_metrics([text], [[]], [[]])

    # Nothing labelled in gold or pred → token classes set is empty,
    # exact-match has no TP/FP/FN, char-IoU treats empty/empty as neutral 1.
    assert math.isnan(result["token_f1"])
    assert math.isnan(result["exact_match_f1"])
    assert result["char_iou"] == pytest.approx(1.0)

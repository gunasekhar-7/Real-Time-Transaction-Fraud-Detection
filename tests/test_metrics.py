"""Tests for ranking and calibration metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import calculate_evaluation_metrics, calibration_report, top_k_capture


def test_top_k_capture_reviews_exactly_the_promised_share():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    capture = top_k_capture(y, scores, fraction=0.10)

    assert capture["reviewed"] == 1
    assert capture["caught"] == 1
    assert capture["recall"] == 0.5
    assert capture["precision"] == 1.0


def test_top_k_capture_does_not_inflate_on_ties():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.5)

    capture = top_k_capture(y, scores, fraction=0.10)

    assert capture["reviewed"] == 1
    assert capture["caught"] == 1


def test_metrics_expose_both_top_1_pct_measures():
    y = np.array([1, 1, 0] + [0] * 97)
    scores = np.linspace(1.0, 0.0, 100)

    metrics = calculate_evaluation_metrics(y, scores)

    assert metrics["recall_at_top_1_pct_volume"] == 0.5
    assert metrics["precision_at_top_1_pct_volume"] == 1.0


def test_calibration_flags_degenerate_scores():
    y = np.array([0] * 990 + [1] * 10)
    scores = np.zeros(1000)

    report = calibration_report(y, scores)

    assert report["exact_zero_share"] == 1.0
    assert report["brier"] == pytest.approx(0.01, abs=1e-6)
    assert report["expected_calibration_error"] == pytest.approx(0.01, abs=1e-6)
    assert report["reliability"][0]["observed_rate"] == pytest.approx(0.01, abs=1e-4)


def test_calibration_perfect_predictions_have_no_error():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])

    report = calibration_report(y, scores)

    assert report["brier"] == 0.0
    assert report["expected_calibration_error"] == 0.0

"""Tests for detection metrics and bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import bootstrap_ci, detection_metrics, paired_bootstrap_diff


def test_perfect_separation():
    metrics = detection_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        k=2,
        threshold=0.5,
        clean_scores=[0.1, 0.2],
    )
    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["n_positive"] == 2
    assert metrics["n_total"] == 4


def test_recall_at_k_with_one_hit_in_top_two():
    metrics = detection_metrics(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.7, 0.6],
        k=2,
        threshold=0.5,
        clean_scores=[0.1, 0.2],
    )
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["precision_at_k"] == pytest.approx(0.5)


def test_fpr_on_clean_scores():
    metrics = detection_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        k=2,
        threshold=0.5,
        clean_scores=[0.1, 0.6, 0.9],
    )
    assert metrics["fpr"] == pytest.approx(2.0 / 3.0)


def test_fpr_falls_back_to_negative_rows():
    metrics = detection_metrics(
        [0, 0, 1, 1],
        [0.1, 0.6, 0.8, 0.9],
        k=2,
        threshold=0.5,
    )
    assert metrics["fpr"] == pytest.approx(0.5)


def test_metric_keys():
    metrics = detection_metrics([0, 1], [0.1, 0.9])
    for key in (
        "auc_roc",
        "auc_pr",
        "recall_at_k",
        "precision_at_k",
        "fpr",
        "threshold",
        "k",
        "n_total",
        "n_positive",
        "prevalence",
    ):
        assert key in metrics


def test_invalid_inputs():
    with pytest.raises(ValueError):
        detection_metrics([0, 0], [0.1, 0.2])  # one class only
    with pytest.raises(ValueError):
        detection_metrics([0, 2], [0.1, 0.2])  # bad label
    with pytest.raises(ValueError):
        detection_metrics([0, 1], [0.1])  # length mismatch
    with pytest.raises(ValueError):
        detection_metrics([0, 1], [0.1, np.nan])  # non-finite score


def test_bootstrap_is_reproducible_and_ordered():
    values = [0.60, 0.70, 0.80, 0.90, 1.00]
    first = bootstrap_ci(values, n_boot=300, seed=7)
    second = bootstrap_ci(values, n_boot=300, seed=7)
    assert first == second
    assert first["ci_lo"] <= first["mean"] <= first["ci_hi"]
    assert first["sd"] > 0


def test_bootstrap_handles_single_value():
    result = bootstrap_ci([0.75], n_boot=100, seed=0)
    assert result["mean"] == pytest.approx(0.75)
    assert result["sd"] == pytest.approx(0.0)
    assert result["ci_lo"] == pytest.approx(0.75)
    assert result["ci_hi"] == pytest.approx(0.75)


def test_paired_bootstrap_detects_positive_difference():
    a = [0.90, 0.91, 0.89, 0.92, 0.90]
    b = [0.80, 0.81, 0.79, 0.82, 0.80]
    result = paired_bootstrap_diff(a, b, n_boot=500, seed=0)
    assert result["mean_diff"] == pytest.approx(0.10)
    assert result["ci_lo"] > 0.0
    assert result["ci_hi"] > 0.0
    assert result["p_value"] < 0.05
    assert result["n_pairs"] == 5


def test_paired_bootstrap_with_no_difference():
    a = [0.50, 0.60, 0.70]
    b = [0.50, 0.60, 0.70]
    result = paired_bootstrap_diff(a, b, n_boot=500, seed=0)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["ci_lo"] <= 0.0 <= result["ci_hi"]


def test_paired_bootstrap_rejects_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_diff([0.1], [0.2])

"""Tests for local attribution and faithfulness helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors import make_detector
from src.explain import (
    attribution_metrics,
    conditional_means,
    conditional_occlusion_contributions,
    deletion_metrics,
    mahalanobis_contributions,
    majority_agreement,
    occlusion_contributions,
    rank_features,
    true_feature_rank,
)


def test_occlusion_shape_and_known_values():
    X = np.array([[0.5, 0.1, -0.2], [0.1, -0.3, 0.2]])
    detector = make_detector("mad").fit(X)

    base, contributions = occlusion_contributions(detector, X)

    assert base.shape == (2,)
    assert contributions.shape == X.shape
    assert np.allclose(base, detector.score(X))
    # Row 0: feature 0 is the max; occluding it drops the score from 0.5 to 0.2.
    assert contributions[0, 0] == pytest.approx(0.3)
    # Row 1: feature 1 is the max; occluding it drops the score from 0.3 to 0.2.
    assert contributions[1, 1] == pytest.approx(0.1)


def test_occlusion_requires_fitted_detector():
    X = np.array([[0.5, 0.1, -0.2]])
    with pytest.raises(RuntimeError):
        occlusion_contributions(make_detector("mad"), X)


def test_mahalanobis_contributions_sum_to_squared_distance():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 4))
    rows = rng.normal(size=(10, 4))
    detector = make_detector("mahalanobis").fit(train)

    contributions = mahalanobis_contributions(detector, rows)
    squared_distance = detector.score(rows) ** 2

    assert contributions.shape == rows.shape
    assert np.allclose(contributions.sum(axis=1), squared_distance)


def test_mahalanobis_contributions_reject_other_detectors():
    X = np.array([[0.5, 0.1, -0.2]])
    detector = make_detector("mad").fit(X)
    with pytest.raises(TypeError):
        mahalanobis_contributions(detector, X)


def test_conditional_means_match_gaussian_formula():
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    precision = np.linalg.inv(covariance)
    X = np.array([[1.0, 0.5]])

    conditional = conditional_means(X, [0.0, 0.0], precision)

    # E[x0 | x1=0.5] = 0.8 * 0.5 = 0.4
    # E[x1 | x0=1.0] = 0.8 * 1.0 = 0.8
    assert conditional[0, 0] == pytest.approx(0.4)
    assert conditional[0, 1] == pytest.approx(0.8)


def test_conditional_occlusion_contributions_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    detector = make_detector("mad").fit(X)
    mean = X.mean(axis=0)
    precision = np.linalg.inv(np.cov(X, rowvar=False))

    base, contributions = conditional_occlusion_contributions(
        detector, X, mean, precision
    )

    assert base.shape == (40,)
    assert contributions.shape == X.shape
    assert np.allclose(base, detector.score(X))


def test_rank_features_by_absolute_contribution():
    contributions = np.array([[0.1, -0.9, 0.2], [0.5, 0.4, 0.3]])
    order = rank_features(contributions)
    assert order[0, 0] == 1
    assert order[1, 0] == 0


def test_true_feature_rank():
    order = np.array([[1, 0, 2], [0, 1, 2]])
    ranks = true_feature_rank(order, [1, 2])
    assert ranks.tolist() == [1, 3]


def test_attribution_metrics():
    contributions = np.array([[0.1, -0.9, 0.2], [0.5, 0.4, 0.3]])
    metrics = attribution_metrics(contributions, [1, 2])

    assert metrics["top1"].tolist() == [1.0, 0.0]
    assert metrics["top3"].tolist() == [1.0, 1.0]
    assert metrics["true_rank"].tolist() == [1, 3]
    assert metrics["mrr"][0] == pytest.approx(1.0)
    assert metrics["mrr"][1] == pytest.approx(1.0 / 3.0)


def test_deletion_metrics():
    occlusion = np.array([[0.5, 0.1, -0.2], [0.2, 0.4, 0.0]])
    result = deletion_metrics(
        occlusion, top_idx=[0, 1], true_idx=[0, 2], base_scores=[1.0, 2.0]
    )

    assert np.allclose(result["drop_top"], [0.5, 0.4])
    assert np.allclose(result["drop_true"], [0.5, 0.0])
    assert np.allclose(result["drop_random"], [-0.05, 0.1])
    assert np.allclose(result["deletion_advantage"], [0.55, 0.3])
    assert np.allclose(result["deletion_advantage_relative"], [0.55, 0.15])
    assert np.allclose(result["drop_true_vs_random"], [0.55, -0.1])


def test_majority_agreement():
    top_matrix = np.array([[0, 1, 2], [0, 1, 1], [0, 0, 2]])
    agreement = majority_agreement(top_matrix)
    assert np.allclose(agreement, [1.0, 2.0 / 3.0, 2.0 / 3.0])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        occlusion_contributions(make_detector("mad").fit(np.zeros((3, 2))), np.zeros(2))
    with pytest.raises(ValueError):
        attribution_metrics(np.zeros((2, 3)), [0])
    with pytest.raises(ValueError):
        majority_agreement(np.array([0, 1, 2]))

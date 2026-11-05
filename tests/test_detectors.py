"""Tests for the detector family."""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors import DETECTOR_NAMES, make_detector


def _data() -> tuple[np.ndarray, np.ndarray]:
    """A correlated 3-D cloud plus one point that breaks the dependence."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 2))
    normal = np.column_stack(
        [latent[:, 0], latent[:, 1], latent[:, 0] + latent[:, 1]]
    )
    normal += 0.01 * rng.normal(size=normal.shape)
    outlier = np.array([[8.0, 8.0, -8.0]])
    return normal, outlier


def _build(name: str):
    if name == "eif":
        return make_detector(name, random_seed=0)
    if name in {"iforest", "pca"}:
        return make_detector(name, random_state=0)
    return make_detector(name)


@pytest.mark.parametrize("name", ["mad", "mahalanobis", "pca", "iforest"])
def test_detector_flags_known_outlier(name):
    normal, outlier = _data()
    detector = _build(name).fit(normal)
    normal_scores = detector.score(normal)
    outlier_score = detector.score(outlier)[0]
    assert outlier_score > np.quantile(normal_scores, 0.99)


def test_extended_iforest_flags_known_outlier():
    pytest.importorskip("isotree")
    normal, outlier = _data()
    detector = make_detector("eif", random_seed=0).fit(normal)
    normal_scores = detector.score(normal)
    outlier_score = detector.score(outlier)[0]
    assert outlier_score > np.quantile(normal_scores, 0.99)


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_scores_are_finite_and_one_per_row(name):
    if name == "eif":
        pytest.importorskip("isotree")
    normal, _ = _data()
    detector = _build(name).fit(normal)
    scores = detector.score(normal)
    assert scores.shape == (len(normal),)
    assert np.isfinite(scores).all()


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        make_detector("not-a-detector")


def test_score_before_fit_raises():
    normal, _ = _data()
    detector = make_detector("mad")
    with pytest.raises(RuntimeError):
        detector.score(normal)


def test_wrong_feature_count_raises():
    normal, _ = _data()
    detector = make_detector("mad").fit(normal)
    with pytest.raises(ValueError, match="Expected 3 features"):
        detector.score(normal[:, :2])


def test_fit_score_matches_fit_then_score():
    normal, outlier = _data()
    detector = make_detector("mahalanobis")
    scores = detector.fit_score(normal, outlier)
    expected = make_detector("mahalanobis").fit(normal).score(outlier)
    assert np.allclose(scores, expected)


def test_mahalanobis_precision_is_symmetric():
    normal, _ = _data()
    detector = make_detector("mahalanobis").fit(normal)
    assert detector.precision_.shape == (3, 3)
    assert np.allclose(detector.precision_, detector.precision_.T)


def test_non_finite_input_raises():
    normal, _ = _data()
    normal[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        make_detector("mad").fit(normal)

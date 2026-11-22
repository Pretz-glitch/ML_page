"""Local attribution methods and faithfulness helpers.

Two attribution families are implemented:

occlusion
    For each feature, replace it with the station's training median (0 in the
    scaled representation), re-score the row, and take the score drop as the
    contribution. Works for any detector.

mahalanobis exact
    For a fitted :class:`~src.detectors.MahalanobisDetector`, the contribution
    of feature ``j`` is ``z_j * (Sigma^-1 z)_j``. The contributions sum exactly
    to the squared Mahalanobis distance.

Ranking uses absolute contributions. The true feature is the one deliberately
corrupted by the injector.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .detectors import BaseDetector, MahalanobisDetector


def _as_matrix(values: np.ndarray, name: str = "X") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def occlusion_contributions(
    detector: BaseDetector,
    X: np.ndarray,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute occlusion contributions for every row and feature.

    Parameters
    ----------
    detector:
        Any fitted detector with a ``score`` method.
    X:
        Rows in the within-station scaled space, shape ``(n_rows, n_features)``.
    baseline:
        Replacement value for the occluded feature. ``0`` is the station's
        training median in the scaled representation.

    Returns
    -------
    base_scores:
        Scores of the unmodified rows, shape ``(n_rows,)``.
    contributions:
        ``base_score - occluded_score`` for each feature, shape
        ``(n_rows, n_features)``. Positive means the feature pushes the score
        up.
    """
    array = _as_matrix(X)
    base_scores = np.asarray(detector.score(array), dtype=float)
    contributions = np.empty_like(array)

    for feature in range(array.shape[1]):
        occluded = array.copy()
        occluded[:, feature] = float(baseline)
        contributions[:, feature] = base_scores - detector.score(occluded)

    return base_scores, contributions


def mahalanobis_contributions(
    detector: MahalanobisDetector,
    X: np.ndarray,
) -> np.ndarray:
    """Exact additive Mahalanobis contributions.

    ``c_j = z_j * (Sigma^-1 z)_j``, so ``sum_j c_j = d^2``.
    """
    if not isinstance(detector, MahalanobisDetector):
        raise TypeError("mahalanobis_contributions requires a MahalanobisDetector")
    if detector.precision_ is None or detector.location_ is None:
        raise RuntimeError("The MahalanobisDetector must be fitted first")

    array = _as_matrix(X)
    centred = array - detector.location_
    return centred * (centred @ detector.precision_)


def conditional_means(
    X: np.ndarray,
    mean: Sequence[float],
    precision: np.ndarray,
) -> np.ndarray:
    """Gaussian conditional means ``E[x_j | x_{-j}]`` for every feature.

    Using the precision matrix ``P = Sigma^-1``:

        E[x_j | x_{-j}] = x_j - (P[:, j]^T (x - mu)) / P[j, j]
    """
    array = _as_matrix(X)
    location = np.asarray(mean, dtype=float).ravel()
    matrix = np.asarray(precision, dtype=float)

    n_features = array.shape[1]
    if location.size != n_features:
        raise ValueError("mean must have one entry per feature")
    if matrix.shape != (n_features, n_features):
        raise ValueError("precision must be square with one row per feature")

    centred = array - location
    conditional = np.empty_like(array)
    for feature in range(n_features):
        diagonal = matrix[feature, feature]
        if diagonal <= 0.0:
            raise ValueError("precision diagonal must be positive")
        conditional[:, feature] = (
            array[:, feature] - (centred @ matrix[:, feature]) / diagonal
        )
    return conditional


def conditional_occlusion_contributions(
    detector: BaseDetector,
    X: np.ndarray,
    mean: Sequence[float],
    precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Occlusion against the Gaussian conditional mean rather than a fixed value.

    Replacing a feature with its conditional mean keeps the row plausible under
    the correlation structure, which is important when features are strongly
    collinear. The reference mean and precision are fitted on training data.
    """
    array = _as_matrix(X)
    base_scores = np.asarray(detector.score(array), dtype=float)
    conditional = conditional_means(array, mean, precision)
    contributions = np.empty_like(array)

    for feature in range(array.shape[1]):
        occluded = array.copy()
        occluded[:, feature] = conditional[:, feature]
        contributions[:, feature] = base_scores - detector.score(occluded)

    return base_scores, contributions



def rank_features(contributions: np.ndarray) -> np.ndarray:
    """Return feature indices ordered by descending absolute contribution."""
    array = _as_matrix(contributions, "contributions")
    return np.argsort(-np.abs(array), axis=1, kind="stable")


def true_feature_rank(order: np.ndarray, true_idx: Sequence[int]) -> np.ndarray:
    """Return the 1-based rank of the true feature in each row."""
    order = np.asarray(order, dtype=int)
    truth = np.asarray(true_idx, dtype=int).ravel()
    if order.ndim != 2:
        raise ValueError("order must be 2-D")
    if truth.size != order.shape[0]:
        raise ValueError("true_idx must have one entry per row")

    n_rows, n_features = order.shape
    ranks = np.empty((n_rows, n_features), dtype=int)
    rows = np.arange(n_rows)[:, None]
    ranks[rows, order] = np.arange(1, n_features + 1)
    return ranks[np.arange(n_rows), truth]


def attribution_metrics(
    contributions: np.ndarray,
    true_idx: Sequence[int],
) -> dict[str, np.ndarray]:
    """Top-1, top-3, reciprocal rank and raw rank for each row."""
    array = _as_matrix(contributions, "contributions")
    truth = np.asarray(true_idx, dtype=int).ravel()
    if truth.size != array.shape[0]:
        raise ValueError("true_idx must have one entry per row")

    order = rank_features(array)
    ranks = true_feature_rank(order, truth)
    top_idx = order[:, 0]

    return {
        "order": order,
        "top_idx": top_idx,
        "true_rank": ranks,
        "top1": (top_idx == truth).astype(float),
        "top3": (ranks <= 3).astype(float),
        "mrr": 1.0 / ranks.astype(float),
    }


def deletion_metrics(
    occlusion_contributions: np.ndarray,
    top_idx: Sequence[int],
    true_idx: Sequence[int],
    base_scores: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Deletion-based faithfulness for a top-1 attribution.

    ``drop_top`` is the occlusion drop of the top-attributed feature.
    ``drop_random`` is the mean occlusion drop of the remaining features.
    ``deletion_advantage = drop_top - drop_random``. If ``base_scores`` is
    given, ``deletion_advantage_relative`` divides the advantage by the base
    score so that detectors with different score scales can be compared.
    """
    contributions = _as_matrix(occlusion_contributions, "occlusion_contributions")
    top = np.asarray(top_idx, dtype=int).ravel()
    truth = np.asarray(true_idx, dtype=int).ravel()
    if top.size != contributions.shape[0] or truth.size != contributions.shape[0]:
        raise ValueError("top_idx and true_idx must have one entry per row")

    n_rows, n_features = contributions.shape
    rows = np.arange(n_rows)
    drop_top = contributions[rows, top]
    drop_true = contributions[rows, truth]

    drop_random = np.empty(n_rows, dtype=float)
    for row in range(n_rows):
        mask = np.ones(n_features, dtype=bool)
        mask[top[row]] = False
        drop_random[row] = contributions[row, mask].mean()

    advantage = drop_top - drop_random
    if base_scores is None:
        relative = advantage.copy()
    else:
        base = np.asarray(base_scores, dtype=float).ravel()
        if base.size != n_rows:
            raise ValueError("base_scores must have one entry per row")
        relative = advantage / np.maximum(np.abs(base), 1e-12)

    return {
        "drop_top": drop_top,
        "drop_true": drop_true,
        "drop_random": drop_random,
        "deletion_advantage": advantage,
        "deletion_advantage_relative": relative,
        "drop_true_vs_random": drop_true - drop_random,
    }


def majority_agreement(top_matrix: np.ndarray) -> np.ndarray:
    """Per-row majority agreement across repeated detector fits.

    Parameters
    ----------
    top_matrix:
        Shape ``(n_repeats, n_rows)`` of top-1 feature indices.

    Returns
    -------
    agreement:
        Fraction of repeats that agree with the majority, per row.
    """
    matrix = np.asarray(top_matrix, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("top_matrix must be 2-D")
    n_repeats, n_rows = matrix.shape
    if n_repeats == 0:
        raise ValueError("top_matrix must contain at least one repeat")

    n_features = int(matrix.max()) + 1
    agreement = np.empty(n_rows, dtype=float)
    for row in range(n_rows):
        counts = np.bincount(matrix[:, row], minlength=n_features)
        agreement[row] = counts.max() / n_repeats

    return agreement

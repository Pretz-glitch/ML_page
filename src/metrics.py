"""Detection metrics and bootstrap helpers.

All metrics are computed on station-day rows. Scores are oriented so that
higher means more anomalous. ``y_true`` is 1 for injected rows and 0 for clean
rows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def detection_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    k: int | None = None,
    threshold: float | None = None,
    clean_scores: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Compute detection metrics for one detector on one injected test set.

    Parameters
    ----------
    y_true:
        Binary labels, 1 for injected rows.
    scores:
        Anomaly scores aligned with ``y_true``; higher means more anomalous.
    k:
        Cut-off for recall@k / precision@k. Defaults to the number of positives.
    threshold:
        Operating threshold. Defaults to the 99th percentile of ``scores``.
    clean_scores:
        Scores on the clean test set. If provided, FPR is computed on these
        rows; otherwise FPR is computed on ``scores`` where ``y_true == 0``.
    """
    labels = np.asarray(y_true).ravel()
    values = _as_1d(scores, "scores")

    if labels.shape != values.shape:
        raise ValueError(
            f"y_true and scores must have the same length, got "
            f"{labels.size} and {values.size}"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")

    positives = int(labels.sum())
    total = int(labels.size)
    if positives == 0 or positives == total:
        raise ValueError("y_true must contain both classes")

    cut = positives if k is None else int(k)
    cut = max(1, min(cut, total))

    order = np.argsort(-values)
    top = order[:cut]
    true_positives = int(labels[top].sum())
    recall = true_positives / positives
    precision = true_positives / cut

    if threshold is None:
        threshold = float(np.quantile(values, 0.99))
    threshold = float(threshold)

    if clean_scores is not None:
        clean = _as_1d(clean_scores, "clean_scores")
        fpr = float(np.mean(clean >= threshold))
    else:
        negatives = values[labels == 0]
        fpr = float(np.mean(negatives >= threshold))

    return {
        "auc_roc": float(roc_auc_score(labels, values)),
        "auc_pr": float(average_precision_score(labels, values)),
        "recall_at_k": float(recall),
        "precision_at_k": float(precision),
        "fpr": fpr,
        "threshold": threshold,
        "k": int(cut),
        "n_total": total,
        "n_positive": positives,
        "prevalence": float(positives / total),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap for the mean of a small set of replicate values.

    With five seeds the interval is necessarily coarse; it is reported as a
    measure of seed variability, not as a precise population interval.
    """
    array = _as_1d(values, "values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0

    if array.size == 1:
        return {"mean": mean, "sd": sd, "ci_lo": mean, "ci_hi": mean}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    boot_means = array[indices].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "mean": mean,
        "sd": sd,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
    }


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired percentile bootstrap for ``mean(a - b)``.

    ``a`` and ``b`` are paired replicate values, for example the same five
    seeds evaluated by two detectors. The p-value is a descriptive two-sided
    bootstrap tail probability with a plus-one correction; with five pairs it
    should not be read as a precise inferential test.
    """
    left = _as_1d(a, "a")
    right = _as_1d(b, "b")
    if left.size != right.size:
        raise ValueError("a and b must have the same length")
    if left.size < 2:
        raise ValueError("paired bootstrap requires at least two pairs")

    differences = left - right
    mean = float(differences.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_boot, differences.size))
    boot = differences[indices].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])

    p_left = (1 + int(np.sum(boot <= 0.0))) / (n_boot + 1)
    p_right = (1 + int(np.sum(boot >= 0.0))) / (n_boot + 1)
    p_value = min(1.0, 2.0 * min(p_left, p_right))

    return {
        "mean_diff": mean,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "n_pairs": int(left.size),
    }

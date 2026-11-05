"""Unsupervised detectors for the within-station scaled representation.

All detectors follow the same contract:

    detector = make_detector("iforest", random_state=0)
    detector.fit(X_train)
    scores = detector.score(X_test)

``X`` is a 2-D array (station-days x features) in the *within-station scaled*
space produced by :mod:`src.preprocess`. Higher scores mean more anomalous.

The detector set is deliberately small and classical. The project compares
families (univariate, covariance, reconstruction, isolation) rather than
tuning a single model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def _as_matrix(values: np.ndarray, name: str = "X") -> np.ndarray:
    """Validate a 2-D finite feature matrix."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


class BaseDetector(ABC):
    """Common interface for all detectors."""

    name: str = "base"

    def __init__(self, feature_names: Sequence[str] | None = None) -> None:
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.n_features_: int | None = None

    def _check_fitted(self) -> None:
        if self.n_features_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")

    def _check_width(self, values: np.ndarray) -> np.ndarray:
        array = _as_matrix(values)
        if array.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {array.shape[1]}"
            )
        return array

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit the detector on training rows."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row; higher means more anomalous."""

    def fit_score(self, X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Convenience method: fit on ``X_train`` and score ``X``."""
        return self.fit(X_train).score(X)

    def params(self) -> dict[str, Any]:
        """Return a small, JSON-serialisable description for run logs."""
        return {"detector": self.name, "n_features": self.n_features_}


class MADDetector(BaseDetector):
    """Univariate baseline: the maximum absolute robust z-score across features.

    The input is already within-station scaled, so this is equivalent to
    flagging a station-day when any single measurement is far from that
    station's training median. It cannot detect a correlation break.
    """

    name = "mad"

    def fit(self, X: np.ndarray) -> "MADDetector":
        array = _as_matrix(X)
        self.n_features_ = array.shape[1]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = self._check_width(X)
        return np.max(np.abs(array), axis=1)


class MahalanobisDetector(BaseDetector):
    """Covariance baseline: distance under a multivariate Gaussian.

    The default uses the sample covariance with a pseudo-inverse. This
    preserves the off-diagonal dependence structure that correlation-break
    faults violate. Ledoit-Wolf shrinkage is available as a sensitivity
    variant (``shrinkage=True``), but it is not the default: on this dataset it
    shrinks the off-diagonals heavily and substantially weakens detection of
    dependence anomalies.
    """

    name = "mahalanobis"

    def __init__(
        self,
        shrinkage: bool = False,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.shrinkage = shrinkage
        self.location_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        array = _as_matrix(X)
        self.n_features_ = array.shape[1]

        if self.shrinkage:
            estimator = LedoitWolf().fit(array)
            self.location_ = np.asarray(estimator.location_, dtype=float)
            self.covariance_ = np.asarray(estimator.covariance_, dtype=float)
            self.precision_ = np.asarray(estimator.precision_, dtype=float)
        else:
            self.location_ = array.mean(axis=0)
            self.covariance_ = np.atleast_2d(np.cov(array, rowvar=False))
            self.precision_ = np.linalg.pinv(self.covariance_)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = self._check_width(X)
        difference = array - self.location_
        squared = np.einsum("ij,jk,ik->i", difference, self.precision_, difference)
        return np.sqrt(np.maximum(squared, 0.0))

    def params(self) -> dict[str, Any]:
        payload = super().params()
        payload["shrinkage"] = self.shrinkage
        return payload


class PCAReconstructionDetector(BaseDetector):
    """Linear reconstruction baseline: squared error after PCA projection."""

    name = "pca"

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 0,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.n_components = n_components
        self.random_state = random_state
        self.model_: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAReconstructionDetector":
        array = _as_matrix(X)
        self.n_features_ = array.shape[1]
        components = min(self.n_components, array.shape[1], array.shape[0])
        self.model_ = PCA(n_components=components, random_state=self.random_state)
        self.model_.fit(array)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = self._check_width(X)
        reconstructed = self.model_.inverse_transform(self.model_.transform(array))
        return np.sum((array - reconstructed) ** 2, axis=1)

    def params(self) -> dict[str, Any]:
        payload = super().params()
        payload["n_components"] = int(self.model_.n_components_) if self.model_ else None
        return payload


class IsolationForestDetector(BaseDetector):
    """Standard Isolation Forest with axis-parallel splits."""

    name = "iforest"

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 0,
        n_jobs: int = -1,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        array = _as_matrix(X)
        self.n_features_ = array.shape[1]
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(array)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = self._check_width(X)
        # sklearn returns higher values for more normal rows.
        return -self.model_.score_samples(array)

    def params(self) -> dict[str, Any]:
        payload = super().params()
        payload["n_estimators"] = self.n_estimators
        return payload


class ExtendedIsolationForestDetector(BaseDetector):
    """Extended Isolation Forest with hyperplane splits (``isotree``).

    This is an optional detector. It requires the ``isotree`` package.
    """

    name = "eif"

    def __init__(
        self,
        ntrees: int = 300,
        random_seed: int = 0,
        nthreads: int = -1,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self.ntrees = ntrees
        self.random_seed = random_seed
        self.nthreads = nthreads
        self.model_: Any = None

    def fit(self, X: np.ndarray) -> "ExtendedIsolationForestDetector":
        try:
            import isotree
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "Extended Isolation Forest requires the 'isotree' package. "
                "Install it with: pip install isotree"
            ) from error

        array = _as_matrix(X)
        self.n_features_ = array.shape[1]
        self.model_ = isotree.IsolationForest(
            ntrees=self.ntrees,
            ndim=self.n_features_,
            random_seed=self.random_seed,
            nthreads=self.nthreads,
        )
        self.model_.fit(array)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = self._check_width(X)
        return np.asarray(self.model_.predict(array), dtype=float)

    def params(self) -> dict[str, Any]:
        payload = super().params()
        payload["ntrees"] = self.ntrees
        return payload


DETECTORS: dict[str, type[BaseDetector]] = {
    "mad": MADDetector,
    "mahalanobis": MahalanobisDetector,
    "pca": PCAReconstructionDetector,
    "iforest": IsolationForestDetector,
    "eif": ExtendedIsolationForestDetector,
}

DETECTOR_NAMES: tuple[str, ...] = tuple(DETECTORS)


def make_detector(name: str, **kwargs: Any) -> BaseDetector:
    """Create a detector by short name.

    Parameters
    ----------
    name:
        One of ``mad``, ``mahalanobis``, ``pca``, ``iforest``, ``eif``.
    **kwargs:
        Passed to the detector constructor.
    """
    key = name.lower()
    if key not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}. Choose from {DETECTOR_NAMES}")
    return DETECTORS[key](**kwargs)

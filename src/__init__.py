"""Anomaly Page pipeline package."""

from .detectors import DETECTOR_NAMES, BaseDetector, make_detector
from .faults import (
    FAULT_TYPES,
    FaultInjector,
    InjectionProfile,
    InjectionResult,
    build_injection_profile,
)
from .io import Dataset, load_dataset
from .metrics import bootstrap_ci, detection_metrics, paired_bootstrap_diff
from .preprocess import (
    Scaler,
    build_raw_frame,
    build_scaled_frame,
    fit_scaler,
    transform,
)

__all__ = [
    "Dataset",
    "load_dataset",
    "Scaler",
    "build_raw_frame",
    "build_scaled_frame",
    "fit_scaler",
    "transform",
    "BaseDetector",
    "DETECTOR_NAMES",
    "make_detector",
    "FAULT_TYPES",
    "FaultInjector",
    "InjectionProfile",
    "InjectionResult",
    "build_injection_profile",
    "bootstrap_ci",
    "detection_metrics",
    "paired_bootstrap_diff",
]

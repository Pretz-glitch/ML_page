"""Flatten the archive into a station-day table and fit leakage-safe scaling.

The modelling representation is the *within-station* robust z-score:

    z[t, s, f] = (x[t, s, f] - median_train[s, f]) / iqr_train[s, f]

Medians and IQRs are computed on training days only. This is mandatory for
this dataset: more than 90% of the variance in several features is between
stations, so a pooled model would otherwise learn station identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import Dataset, N_FEATURES, N_STATIONS, TEST_DAYS, TRAIN_DAYS

META_COLUMNS: tuple[str, ...] = (
    "day_index",
    "date",
    "station_id",
    "station_group",
    "split",
)

EPS = 1e-9


@dataclass(frozen=True)
class Scaler:
    """Per-station, per-feature robust scaler parameters."""

    station_ids: np.ndarray  # (37,)
    feature_names: tuple[str, ...]  # (11,)
    median: np.ndarray  # (37, 11)
    scale: np.ndarray  # (37, 11)
    fallback: np.ndarray  # (37, 11), bool: IQR was too small, used SD
    constant: np.ndarray  # (37, 11), bool: SD was also too small

    def summary(self) -> dict[str, int]:
        return {
            "fallback_cells": int(self.fallback.sum()),
            "constant_cells": int(self.constant.sum()),
        }


def fit_scaler(
    train_tensor: np.ndarray,
    station_ids: np.ndarray,
    feature_names: tuple[str, ...],
    eps: float = EPS,
) -> Scaler:
    """Fit the robust scaler on training days only.

    Parameters
    ----------
    train_tensor:
        Array of shape ``(n_train_days, 37, 11)`` containing training values.
    station_ids:
        Station identifiers, length 37.
    feature_names:
        Feature names, length 11.
    """
    expected = (TRAIN_DAYS, N_STATIONS, N_FEATURES)
    if train_tensor.shape != expected:
        raise ValueError(f"Expected training tensor {expected}, got {train_tensor.shape}")
    if len(station_ids) != N_STATIONS:
        raise ValueError(f"Expected {N_STATIONS} station IDs")
    if len(feature_names) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} feature names")

    median = np.nanmedian(train_tensor, axis=0)
    q25 = np.nanpercentile(train_tensor, 25, axis=0)
    q75 = np.nanpercentile(train_tensor, 75, axis=0)
    iqr = q75 - q25
    sd = np.nanstd(train_tensor, axis=0, ddof=0)

    fallback = iqr < eps
    constant = fallback & (sd < eps)
    scale = np.where(constant, 1.0, np.where(fallback, sd, iqr))
    scale = np.maximum(scale, eps)

    return Scaler(
        station_ids=np.asarray(station_ids, dtype=int),
        feature_names=tuple(feature_names),
        median=median,
        scale=scale,
        fallback=fallback,
        constant=constant,
    )


def transform(tensor: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Apply a fitted scaler to an array of shape ``(days, 37, 11)``."""
    if tensor.shape[1:] != (N_STATIONS, N_FEATURES):
        raise ValueError(
            f"Expected trailing shape {(N_STATIONS, N_FEATURES)}, got {tensor.shape[1:]}"
        )
    return (tensor - scaler.median[None, :, :]) / scaler.scale[None, :, :]


def _metadata(ds: Dataset) -> dict[str, np.ndarray]:
    day_index = np.repeat(np.arange(ds.n_days), ds.n_stations)
    return {
        "day_index": day_index,
        "date": np.repeat(ds.dates, ds.n_stations),
        "station_id": np.tile(ds.station_ids, ds.n_days),
        "station_group": np.tile(ds.station_groups, ds.n_days),
        "split": np.where(day_index < TRAIN_DAYS, "train", "test"),
    }


def build_raw_frame(ds: Dataset) -> pd.DataFrame:
    """Build the station-day table with the original (normalised) values."""
    meta = _metadata(ds)
    values = ds.tensor.reshape(-1, ds.n_features)

    frame = pd.DataFrame(meta)
    for index, name in enumerate(ds.feature_names):
        frame[name] = values[:, index]

    frame["date"] = pd.to_datetime(frame["date"])
    return frame.loc[:, list(META_COLUMNS) + list(ds.feature_names)]


def build_scaled_frame(ds: Dataset, scaler: Scaler) -> pd.DataFrame:
    """Build the station-day table with within-station robust z-scores."""
    if tuple(scaler.feature_names) != tuple(ds.feature_names):
        raise ValueError("Scaler feature names do not match the dataset")

    meta = _metadata(ds)
    values = transform(ds.tensor, scaler).reshape(-1, ds.n_features)

    frame = pd.DataFrame(meta)
    for index, name in enumerate(ds.feature_names):
        frame[f"z_{name}"] = values[:, index]

    frame["date"] = pd.to_datetime(frame["date"])
    return frame.loc[:, list(META_COLUMNS) + [f"z_{n}" for n in ds.feature_names]]


def scaler_frame(scaler: Scaler) -> pd.DataFrame:
    """Flatten the scaler parameters into a tidy table for inspection."""
    rows: list[dict[str, object]] = []
    for station_index, station_id in enumerate(scaler.station_ids):
        for feature_index, feature in enumerate(scaler.feature_names):
            rows.append(
                {
                    "station_id": int(station_id),
                    "feature": feature,
                    "median": float(scaler.median[station_index, feature_index]),
                    "scale": float(scaler.scale[station_index, feature_index]),
                    "fallback_sd": bool(scaler.fallback[station_index, feature_index]),
                    "constant": bool(scaler.constant[station_index, feature_index]),
                }
            )
    return pd.DataFrame(rows)


def validate_frame(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> None:
    """Assert the invariants required by the Phase 1 definition of done."""
    expected_rows = TRAIN_DAYS * N_STATIONS + TEST_DAYS * N_STATIONS
    if len(frame) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(frame)}")

    if len(feature_names) != N_FEATURES:
        raise AssertionError(f"Expected {N_FEATURES} features, got {len(feature_names)}")
    for name in feature_names:
        if name not in frame.columns:
            raise AssertionError(f"Missing feature column: {name}")

    values = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("Feature matrix contains NaN or infinite values")

    train_days = set(frame.loc[frame["split"] == "train", "day_index"].unique())
    test_days = set(frame.loc[frame["split"] == "test", "day_index"].unique())
    if train_days != set(range(TRAIN_DAYS)):
        raise AssertionError("Training day indices are not 0..422")
    if test_days != set(range(TRAIN_DAYS, TRAIN_DAYS + TEST_DAYS)):
        raise AssertionError("Test day indices are not 423..704")
    if train_days & test_days:
        raise AssertionError("Train and test day indices overlap")

    if frame["station_id"].nunique() != N_STATIONS:
        raise AssertionError("Unexpected number of stations")

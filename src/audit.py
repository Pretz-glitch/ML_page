"""Audit computations for Phase 1.

All functions are pure: they take a station-day frame and return DataFrames.
The script layer is responsible for writing files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FEATURE_FAMILIES

FLOOR = 1e-3 + 1e-9


def variance_table(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Between-station vs within-station variance for each feature."""
    rows: list[dict[str, object]] = []

    for feature in features:
        pivot = frame.pivot(index="day_index", columns="station_id", values=feature)
        values = pivot.to_numpy(dtype=float)

        station_means = np.nanmean(values, axis=0)
        station_vars = np.nanvar(values, axis=0, ddof=0)

        between = float(np.nanvar(station_means, ddof=0))
        within = float(np.nanmean(station_vars))
        total = float(np.nanvar(values, ddof=0))

        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        median_within_iqr = float(np.nanmedian(q75 - q25))

        rows.append(
            {
                "feature": feature,
                "family": FEATURE_FAMILIES.get(feature, "unknown"),
                "n_values": int(values.size),
                "missing": int(np.isnan(values).sum()),
                "overall_min": float(np.nanmin(values)),
                "overall_max": float(np.nanmax(values)),
                "overall_mean": float(np.nanmean(values)),
                "overall_std": float(np.nanstd(values, ddof=0)),
                "between_variance": between,
                "within_variance": within,
                "total_variance": total,
                "between_fraction": between / total if total > 0 else np.nan,
                "median_within_iqr": median_within_iqr,
                "floor_fraction": float(np.nanmean(values <= FLOOR)),
                "n_unique": int(len(np.unique(values[np.isfinite(values)]))),
            }
        )

    return pd.DataFrame(rows)


def correlation_table(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Pearson correlation matrix with feature names as the index."""
    return frame.loc[:, list(features)].corr()


def top_correlations(
    corr: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Return unique feature pairs sorted by absolute correlation."""
    features = list(corr.columns)
    rows: list[dict[str, object]] = []
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            rows.append(
                {
                    "feature_a": features[left],
                    "feature_b": features[right],
                    "r": float(corr.iloc[left, right]),
                }
            )

    table = pd.DataFrame(rows)
    table["abs_r"] = table["r"].abs()
    table = table.sort_values("abs_r", ascending=False).drop(columns="abs_r")
    if limit is not None:
        table = table.head(limit)
    return table.reset_index(drop=True)


def floor_table(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Check for missing values and values at the normalisation floor."""
    rows: list[dict[str, object]] = []
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "n_values": int(values.size),
                "missing": int(np.isnan(values).sum()),
                "n_at_floor": int(np.sum(values <= FLOOR)),
                "floor_fraction": float(np.mean(values <= FLOOR)),
                "n_exact_zero": int(np.sum(values == 0.0)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "n_unique": int(len(np.unique(values[np.isfinite(values)]))),
            }
        )
    return pd.DataFrame(rows)


def station_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Per-station mean and standard deviation for every feature."""
    summary = frame.groupby(["station_id", "station_group"])[list(features)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{feature}_{statistic}" for feature, statistic in summary.columns]
    return summary.reset_index()


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Row, day and date coverage for each split."""
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        subset = frame.loc[frame["split"] == split]
        rows.append(
            {
                "split": split,
                "rows": int(len(subset)),
                "days": int(subset["day_index"].nunique()),
                "stations": int(subset["station_id"].nunique()),
                "start": str(subset["date"].min().date()),
                "end": str(subset["date"].max().date()),
            }
        )
    return pd.DataFrame(rows)


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Station and row counts for the three connected station groups."""
    summary = (
        frame.groupby("station_group")
        .agg(stations=("station_id", "nunique"), rows=("day_index", "size"))
        .reset_index()
    )
    return summary

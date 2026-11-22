"""Helpers for the unlabelled real-record analysis.

The functions here are descriptive only. They summarise flags produced by the
detectors on the clean test period; there is no ground truth to score against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def pairwise_overlap(
    flags: pd.DataFrame,
    roles: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pairwise overlap between binary detector flags.

    Parameters
    ----------
    flags:
        Boolean columns named after the detectors.
    roles:
        Optional mapping from detector name to ``primary``/``secondary``.
    """
    columns = list(flags.columns)
    if len(columns) < 2:
        raise ValueError("At least two flag columns are required")

    n_rows = len(flags)
    rows: list[dict[str, object]] = []

    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            first = columns[left]
            second = columns[right]
            a = flags[first].to_numpy(dtype=bool)
            b = flags[second].to_numpy(dtype=bool)

            n_a = int(a.sum())
            n_b = int(b.sum())
            intersection = int(np.sum(a & b))
            union = int(np.sum(a | b))
            both_false = int(np.sum(~a & ~b))

            if roles is None:
                pair_type = "all"
            else:
                first_role = roles.get(first, "unknown")
                second_role = roles.get(second, "unknown")
                pair_type = "-".join(sorted((first_role, second_role)))

            rows.append(
                {
                    "detector_a": first,
                    "detector_b": second,
                    "pair_type": pair_type,
                    "n_a": n_a,
                    "n_b": n_b,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else np.nan,
                    "agreement": (intersection + both_false) / n_rows,
                }
            )

    return pd.DataFrame(rows)


def concentration_summary(
    frame: pd.DataFrame,
    detectors: Sequence[str],
    group_column: str,
) -> pd.DataFrame:
    """Summarise flags by station or date."""
    required = ["flag_" + detector for detector in detectors]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing flag columns: {missing}")
    if group_column not in frame.columns:
        raise KeyError(f"Missing group column: {group_column}")

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_column, sort=False):
        any_flag = group[required].any(axis=1)
        row: dict[str, object] = {
            group_column: key,
            "n_rows": int(len(group)),
            "n_any": int(any_flag.sum()),
            "rate_any": float(any_flag.mean()),
        }
        for detector in detectors:
            row[f"n_{detector}"] = int(group[f"flag_{detector}"].sum())
        if "consensus_primary" in group.columns:
            row["n_consensus_primary"] = int(group["consensus_primary"].sum())
        rows.append(row)

    return pd.DataFrame(rows)


def flag_summary(
    frame: pd.DataFrame,
    detectors: Sequence[str],
    roles: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-detector flag counts and concentration."""
    rows: list[dict[str, object]] = []
    for detector in detectors:
        flag = frame[f"flag_{detector}"].to_numpy(dtype=bool)
        score = frame[f"score_{detector}"].to_numpy(dtype=float)
        threshold = float(frame[f"threshold_{detector}"].iloc[0])

        rows.append(
            {
                "detector": detector,
                "role": roles.get(detector, "unknown") if roles else "unknown",
                "threshold": threshold,
                "n_flags": int(flag.sum()),
                "flag_rate": float(flag.mean()),
                "n_unique_stations": int(frame.loc[flag, "station_id"].nunique()),
                "n_unique_dates": int(frame.loc[flag, "date"].nunique()),
                "max_score": float(score.max()),
                "median_score_flagged": (
                    float(np.median(score[flag])) if flag.any() else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def max_score_ratio(frame: pd.DataFrame, detectors: Sequence[str]) -> np.ndarray:
    """Maximum score/threshold ratio across detectors for each row."""
    ratios = np.vstack(
        [
            frame[f"score_{detector}"].to_numpy(dtype=float)
            / frame[f"threshold_{detector}"].to_numpy(dtype=float)
            for detector in detectors
        ]
    )
    return ratios.max(axis=0)


def select_case_studies(
    frame: pd.DataFrame,
    n_cases: int = 3,
    severity_column: str = "profile_severity",
    diversity_column: str = "top_feature",
) -> pd.DataFrame:
    """Select high-consensus cases from distinct stations and profiles.

    Rows are ordered by primary-detector consensus, total flags, eligible-feature
    severity (if present) and maximum score ratio. The greedy pass prefers
    different stations and, when available, different top features so the case
    studies are not three views of the same pattern.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    if frame.empty:
        return frame.assign(case_id=pd.Series(dtype=int))

    sort_columns = ["n_primary_flags", "n_flags"]
    if severity_column in frame.columns:
        sort_columns.append(severity_column)
    sort_columns.append("max_ratio")

    ordered = frame.sort_values(sort_columns, ascending=False)

    selected: list[object] = []
    used_stations: set[object] = set()
    used_features: set[object] = set()

    if diversity_column in frame.columns:
        for index, row in ordered.iterrows():
            if row["station_id"] in used_stations:
                continue
            feature = row[diversity_column]
            if pd.notna(feature) and feature in used_features:
                continue
            selected.append(index)
            used_stations.add(row["station_id"])
            if pd.notna(feature):
                used_features.add(feature)
            if len(selected) == n_cases:
                break

    if len(selected) < n_cases:
        for index, row in ordered.iterrows():
            if index in selected or row["station_id"] in used_stations:
                continue
            selected.append(index)
            used_stations.add(row["station_id"])
            if len(selected) == n_cases:
                break

    if len(selected) < n_cases:
        for index in ordered.index:
            if index in selected:
                continue
            selected.append(index)
            if len(selected) == n_cases:
                break

    result = frame.loc[selected].copy()
    result["case_id"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)

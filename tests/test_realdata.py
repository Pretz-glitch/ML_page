"""Tests for the real-record analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.realdata import (
    concentration_summary,
    flag_summary,
    max_score_ratio,
    pairwise_overlap,
    select_case_studies,
)


def test_pairwise_overlap():
    flags = pd.DataFrame(
        {
            "a": [True, True, False, False],
            "b": [True, False, True, False],
        }
    )
    result = pairwise_overlap(flags, {"a": "primary", "b": "primary"})
    row = result.iloc[0]

    assert row["n_a"] == 2
    assert row["n_b"] == 2
    assert row["intersection"] == 1
    assert row["union"] == 3
    assert row["jaccard"] == pytest.approx(1.0 / 3.0)
    assert row["agreement"] == pytest.approx(0.5)
    assert row["pair_type"] == "primary-primary"


def test_pairwise_overlap_requires_two_columns():
    with pytest.raises(ValueError):
        pairwise_overlap(pd.DataFrame({"a": [True, False]}))


def test_concentration_summary():
    frame = pd.DataFrame(
        {
            "station_id": [1, 1, 2, 2],
            "flag_a": [True, False, True, False],
            "flag_b": [False, False, True, True],
            "consensus_primary": [False, False, True, False],
        }
    )
    result = concentration_summary(frame, ["a", "b"], "station_id").set_index(
        "station_id"
    )

    assert result.loc[1, "n_any"] == 1
    assert result.loc[1, "n_a"] == 1
    assert result.loc[1, "n_b"] == 0
    assert result.loc[2, "n_any"] == 2
    assert result.loc[2, "n_b"] == 2
    assert result.loc[2, "n_consensus_primary"] == 1


def test_flag_summary():
    frame = pd.DataFrame(
        {
            "flag_a": [True, False, True],
            "score_a": [0.9, 0.2, 0.8],
            "threshold_a": [0.5, 0.5, 0.5],
            "station_id": [1, 2, 1],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
        }
    )
    result = flag_summary(frame, ["a"], {"a": "primary"}).iloc[0]

    assert result["n_flags"] == 2
    assert result["flag_rate"] == pytest.approx(2.0 / 3.0)
    assert result["n_unique_stations"] == 1
    assert result["n_unique_dates"] == 2
    assert result["threshold"] == pytest.approx(0.5)
    assert result["median_score_flagged"] == pytest.approx(0.85)


def test_max_score_ratio():
    frame = pd.DataFrame(
        {
            "score_a": [1.0, 2.0],
            "threshold_a": [1.0, 1.0],
            "score_b": [0.5, 4.0],
            "threshold_b": [1.0, 2.0],
        }
    )
    result = max_score_ratio(frame, ["a", "b"])
    assert np.allclose(result, [1.0, 2.0])


def test_select_case_studies_prefers_distinct_stations():
    frame = pd.DataFrame(
        {
            "station_id": [1, 1, 2, 3, 4],
            "n_primary_flags": [3, 3, 2, 1, 0],
            "n_flags": [5, 5, 4, 3, 1],
            "max_ratio": [5.0, 4.5, 4.0, 3.0, 1.0],
        }
    )
    result = select_case_studies(frame, n_cases=3)

    assert len(result) == 3
    assert result["station_id"].is_unique
    assert result["station_id"].tolist() == [1, 2, 3]
    assert result["case_id"].tolist() == [1, 2, 3]


def test_select_case_studies_fills_when_stations_run_out():
    frame = pd.DataFrame(
        {
            "station_id": [1, 1, 1],
            "n_primary_flags": [3, 2, 1],
            "n_flags": [5, 4, 3],
            "max_ratio": [5.0, 4.0, 3.0],
        }
    )
    result = select_case_studies(frame, n_cases=2)
    assert len(result) == 2
    assert result["case_id"].tolist() == [1, 2]


def test_select_case_studies_prefers_distinct_features():
    frame = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "top_feature": ["pH_max", "pH_max", "DO_min", "T_mean"],
            "n_primary_flags": [3, 3, 2, 1],
            "n_flags": [5, 5, 4, 3],
            "max_ratio": [5.0, 4.5, 4.0, 3.0],
        }
    )
    result = select_case_studies(frame, n_cases=3)
    assert result["top_feature"].tolist() == ["pH_max", "DO_min", "T_mean"]
    assert result["station_id"].tolist() == [1, 3, 4]


def test_select_case_studies_rejects_bad_count():
    with pytest.raises(ValueError):
        select_case_studies(pd.DataFrame(), n_cases=0)

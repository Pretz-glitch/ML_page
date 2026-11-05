"""Tests for fault injection: labels, masks, prevalence and data integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.faults import (
    FAULT_TYPES,
    LABEL_COLUMNS,
    FaultInjector,
    build_injection_profile,
)

FEATURES = ("alpha", "beta", "gamma")
N_DAYS = 60
N_STATIONS = 5
N_FEATURES = 3
DAY_OFFSET = 423


def _clean(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 0.85, size=(N_DAYS, N_STATIONS, N_FEATURES))


def _injector(clean: np.ndarray, seed: int = 0) -> FaultInjector:
    profile = build_injection_profile(clean, FEATURES, min_median_iqr=0.01)
    dates = np.datetime64("2020-01-01") + np.arange(N_DAYS).astype("timedelta64[D]")
    return FaultInjector(
        profile=profile,
        station_ids=np.arange(N_STATIONS),
        dates=dates,
        seed=seed,
    )


@pytest.mark.parametrize("fault", FAULT_TYPES)
def test_injection_invariants(fault):
    clean = _clean()
    result = _injector(clean).inject(
        clean,
        fault,
        prevalence=0.1,
        window_range=(3, 8),
        day_offset=DAY_OFFSET,
    )

    assert result.corrupted.shape == clean.shape
    assert result.station_day_mask.shape == (N_DAYS, N_STATIONS)
    assert result.cell_mask.shape == (N_DAYS, N_STATIONS, N_FEATURES)
    assert result.corrupted.min() >= 0.0
    assert result.corrupted.max() <= 1.0

    # Clean cells must be byte-identical.
    assert np.array_equal(
        result.corrupted[~result.cell_mask], clean[~result.cell_mask]
    )

    # Every marked cell must actually change.
    assert np.all(
        ~np.isclose(
            result.corrupted[result.cell_mask],
            clean[result.cell_mask],
            atol=1e-12,
            rtol=0.0,
        )
    )

    assert len(result.labels) > 0
    assert list(result.labels.columns) == list(LABEL_COLUMNS)
    assert result.cell_mask.sum() == len(result.labels)
    assert result.n_events > 0
    assert result.actual_prevalence > 0.0
    assert abs(result.actual_prevalence - result.target_prevalence) <= 0.06

    # Labels are unique at the cell level.
    assert not result.labels.duplicated(
        ["day_index", "station_id", "feature"]
    ).any()

    # row_id must match the flattened station-day table convention.
    expected_row_id = (
        result.labels["day_index"] * N_STATIONS + result.labels["station_index"]
    )
    assert np.array_equal(result.labels["row_id"].to_numpy(), expected_row_id.to_numpy())

    # Station-day mask counts unique station-days in the label table.
    unique_station_days = result.labels[["day_index", "station_id"]].drop_duplicates()
    assert result.station_day_mask.sum() == len(unique_station_days)

    # Labels must point at marked cells.
    for row in result.labels.itertuples(index=False):
        day = row.day_index - DAY_OFFSET
        assert result.cell_mask[day, row.station_id, row.feature_index]


def test_input_tensor_is_not_mutated():
    clean = _clean()
    before = clean.copy()
    _injector(clean).inject(clean, "spike", prevalence=0.1)
    assert np.array_equal(clean, before)


def test_spike_labels_are_single_day():
    clean = _clean()
    result = _injector(clean).inject(clean, "spike", prevalence=0.1)
    assert (
        result.labels["window_start_day"] == result.labels["window_end_day"]
    ).all()


@pytest.mark.parametrize("fault", ["drift", "correlation_break", "stuck"])
def test_temporal_faults_span_multiple_days(fault):
    clean = _clean()
    result = _injector(clean).inject(
        clean, fault, prevalence=0.1, window_range=(3, 8)
    )
    spans = result.labels["window_end_day"] - result.labels["window_start_day"]
    assert (spans >= 1).all()


def test_stuck_fault_is_constant_within_event():
    clean = _clean()
    result = _injector(clean).inject(
        clean, "stuck", prevalence=0.1, window_range=(3, 8)
    )

    for _, group in result.labels.groupby("event_id"):
        values = []
        for row in group.itertuples(index=False):
            day = row.day_index - DAY_OFFSET
            values.append(result.corrupted[day, row.station_id, row.feature_index])
        assert len(np.unique(np.round(values, 12))) == 1


def test_drift_magnitude_is_constant_within_event():
    clean = _clean()
    result = _injector(clean).inject(
        clean, "drift", prevalence=0.1, window_range=(3, 8)
    )

    for _, group in result.labels.groupby("event_id"):
        assert group["magnitude"].nunique() == 1
        assert np.isfinite(group["magnitude"].iloc[0])


@pytest.mark.parametrize(
    ("fault", "expected_unit"),
    [
        ("spike", "iqr"),
        ("drift", "iqr"),
        ("correlation_break", "not_applicable"),
        ("stuck", "not_applicable"),
    ],
)
def test_magnitude_unit(fault, expected_unit):
    clean = _clean()
    result = _injector(clean).inject(clean, fault, prevalence=0.1)
    assert set(result.labels["magnitude_unit"]) == {expected_unit}


def test_same_seed_is_deterministic():
    clean = _clean()
    first = _injector(clean, seed=7).inject(clean, "drift", prevalence=0.1)
    second = _injector(clean, seed=7).inject(clean, "drift", prevalence=0.1)
    assert np.array_equal(first.corrupted, second.corrupted)
    pd.testing.assert_frame_equal(first.labels, second.labels)


def test_different_seed_changes_the_injection():
    clean = _clean()
    first = _injector(clean, seed=0).inject(clean, "spike", prevalence=0.1)
    second = _injector(clean, seed=1).inject(clean, "spike", prevalence=0.1)
    assert not np.array_equal(first.corrupted, second.corrupted)


def test_invalid_fault_type_raises():
    clean = _clean()
    with pytest.raises(ValueError, match="Unknown fault type"):
        _injector(clean).inject(clean, "not-a-fault")


@pytest.mark.parametrize("prevalence", [0.0, -0.1, 1.5])
def test_invalid_prevalence_raises(prevalence):
    clean = _clean()
    with pytest.raises(ValueError, match="prevalence"):
        _injector(clean).inject(clean, "spike", prevalence=prevalence)


def test_scaled_input_is_rejected():
    clean = _clean() * 10.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _injector(_clean()).inject(clean, "spike", prevalence=0.1)


def test_profile_excludes_constant_feature():
    clean = _clean()
    clean[:, :, 0] = 0.5
    profile = build_injection_profile(clean, FEATURES, min_median_iqr=0.01)
    assert 0 not in profile.eligible_features
    assert profile.iqr.shape == (N_STATIONS, N_FEATURES)
    assert profile.median_iqr.shape == (N_FEATURES,)


def test_profile_requires_matching_feature_names():
    clean = _clean()
    with pytest.raises(ValueError, match="feature names"):
        build_injection_profile(clean, ("only-one",))

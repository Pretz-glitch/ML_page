"""Tests for the archive loader."""

from __future__ import annotations

import numpy as np

from src.io import (
    END_DATE,
    EXPECTED_FEATURES,
    FEATURE_FAMILIES,
    N_DAYS,
    N_FEATURES,
    N_STATIONS,
    START_DATE,
)


def test_tensor_shape(dataset):
    assert dataset.tensor.shape == (N_DAYS, N_STATIONS, N_FEATURES)
    assert dataset.n_station_days == 26_085


def test_no_missing_or_infinite_values(dataset):
    assert np.isfinite(dataset.tensor).all()


def test_station_ids_are_unique(dataset):
    assert len(dataset.station_ids) == N_STATIONS
    assert len(np.unique(dataset.station_ids)) == N_STATIONS


def test_dates_are_contiguous(dataset):
    assert str(dataset.dates[0]) == START_DATE
    assert str(dataset.dates[-1]) == END_DATE
    gaps = np.diff(dataset.dates.astype("datetime64[D]").astype(int))
    assert np.all(gaps == 1)


def test_feature_names_are_expected(dataset):
    assert dataset.feature_names == EXPECTED_FEATURES
    assert len(dataset.feature_labels) == N_FEATURES
    assert all(feature in FEATURE_FAMILIES for feature in dataset.feature_names)


def test_station_groups_cover_every_station(dataset):
    assert dataset.station_groups.shape == (N_STATIONS,)
    assert set(np.unique(dataset.station_groups)) == {0, 1, 2}
    sizes = sorted(
        int((dataset.station_groups == group).sum()) for group in (0, 1, 2)
    )
    assert sizes == [1, 8, 28]


def test_values_are_in_the_normalised_range(dataset):
    assert dataset.tensor.min() >= 0.0
    assert dataset.tensor.max() <= 1.0

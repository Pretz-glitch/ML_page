"""Tests for the station-day table, temporal split and leakage-safe scaling."""

from __future__ import annotations

import numpy as np
import pytest

from src.io import N_FEATURES, N_STATIONS, TEST_DAYS, TRAIN_DAYS
from src.preprocess import (
    build_raw_frame,
    build_scaled_frame,
    fit_scaler,
    transform,
    validate_frame,
)


def test_table_has_expected_rows_and_columns(dataset):
    frame = build_raw_frame(dataset)
    assert len(frame) == (TRAIN_DAYS + TEST_DAYS) * N_STATIONS
    assert frame["station_id"].nunique() == N_STATIONS
    assert len(dataset.feature_names) == N_FEATURES
    assert frame.loc[:, list(dataset.feature_names)].isna().sum().sum() == 0


def test_split_boundary(dataset):
    frame = build_raw_frame(dataset)
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]

    assert set(train["day_index"].unique()) == set(range(TRAIN_DAYS))
    assert set(test["day_index"].unique()) == set(range(TRAIN_DAYS, TRAIN_DAYS + TEST_DAYS))
    assert not set(train["day_index"].unique()) & set(test["day_index"].unique())
    assert len(train) == TRAIN_DAYS * N_STATIONS
    assert len(test) == TEST_DAYS * N_STATIONS


def test_validate_frame_accepts_the_built_table(dataset):
    validate_frame(build_raw_frame(dataset), dataset.feature_names)


def test_scaler_is_fitted_on_training_days_only(dataset):
    base = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )

    # The stored median must equal the training median, not a pooled median.
    manual_train_median = np.median(dataset.tensor[:TRAIN_DAYS, 0, 0])
    assert base.median[0, 0] == pytest.approx(manual_train_median)

    # The test is meaningful only if the pooled median differs somewhere.
    train_medians = np.median(dataset.tensor[:TRAIN_DAYS], axis=0)
    pooled_medians = np.median(dataset.tensor, axis=0)
    assert not np.allclose(train_medians, pooled_medians)

    corrupted = dataset.tensor.copy()
    corrupted[TRAIN_DAYS:] = corrupted[TRAIN_DAYS:] + 0.5
    same = fit_scaler(
        corrupted[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )

    assert np.array_equal(base.median, same.median)
    assert np.array_equal(base.scale, same.scale)


def test_scaler_rejects_the_wrong_number_of_days(dataset):
    with pytest.raises(ValueError):
        fit_scaler(dataset.tensor, dataset.station_ids, dataset.feature_names)


def test_training_medians_are_zero_after_scaling(dataset):
    scaler = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )
    scaled = transform(dataset.tensor[:TRAIN_DAYS], scaler)
    medians = np.median(scaled, axis=0)
    assert np.max(np.abs(medians)) < 1e-10


def test_training_iqr_is_one_where_iqr_was_used(dataset):
    scaler = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )
    scaled = transform(dataset.tensor[:TRAIN_DAYS], scaler)
    q25 = np.percentile(scaled, 25, axis=0)
    q75 = np.percentile(scaled, 75, axis=0)
    iqr = q75 - q25

    usable = ~scaler.fallback
    if usable.any():
        assert np.allclose(iqr[usable], 1.0, atol=1e-6)


def test_scaled_frame_matches_transform(dataset):
    scaler = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )
    frame = build_scaled_frame(dataset, scaler)
    train = frame.loc[frame["split"] == "train"]

    z_columns = [f"z_{name}" for name in dataset.feature_names]
    assert len(z_columns) == N_FEATURES
    assert train.loc[:, z_columns].isna().sum().sum() == 0

    expected = transform(dataset.tensor[:TRAIN_DAYS], scaler)[0, 0, 0]
    actual = train.loc[
        (train["day_index"] == 0) & (train["station_id"] == dataset.station_ids[0]),
        "z_SC_max",
    ].iloc[0]
    assert actual == pytest.approx(expected)


def test_test_rows_are_transformed_with_train_parameters(dataset):
    scaler = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )
    frame = build_scaled_frame(dataset, scaler)

    day = TRAIN_DAYS
    station_id = dataset.station_ids[0]
    expected = (dataset.tensor[day, 0, 0] - scaler.median[0, 0]) / scaler.scale[0, 0]
    actual = frame.loc[
        (frame["day_index"] == day) & (frame["station_id"] == station_id),
        "z_SC_max",
    ].iloc[0]

    assert actual == pytest.approx(expected)

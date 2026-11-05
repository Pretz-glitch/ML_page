"""Load the UCI 733 water-quality archive into a clean tensor.

The distributed file stores the data as MATLAB cell arrays:

    X_tr: 1 x 423, each cell is 37 x 11 (stations x features) for one day
    X_te: 1 x 282, each cell is 37 x 11

This module stitches them into a single ``(705, 37, 11)`` array and exposes
station IDs, station groups, feature names and dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from .utils import project_root

# ---------------------------------------------------------------------------
# Constants that describe the distributed archive
# ---------------------------------------------------------------------------

N_DAYS = 705
N_STATIONS = 37
N_FEATURES = 11
TRAIN_DAYS = 423
TEST_DAYS = 282
START_DATE = "2016-01-28"
END_DATE = "2018-01-01"

EXPECTED_FEATURES: tuple[str, ...] = (
    "SC_max",
    "pH_max",
    "pH_min",
    "SC_min",
    "SC_mean",
    "DO_max",
    "DO_mean",
    "DO_min",
    "T_mean",
    "T_min",
    "T_max",
)

FEATURE_FAMILIES: dict[str, str] = {
    "SC_max": "specific_conductance",
    "pH_max": "ph",
    "pH_min": "ph",
    "SC_min": "specific_conductance",
    "SC_mean": "specific_conductance",
    "DO_max": "dissolved_oxygen",
    "DO_mean": "dissolved_oxygen",
    "DO_min": "dissolved_oxygen",
    "T_mean": "temperature",
    "T_min": "temperature",
    "T_max": "temperature",
}


@dataclass(frozen=True)
class Dataset:
    """Container for the stitched UCI 733 archive."""

    tensor: np.ndarray  # (705, 37, 11), float64
    station_ids: np.ndarray  # (37,), int
    station_groups: np.ndarray  # (37,), int, 0..2
    feature_names: tuple[str, ...]  # 11 short names
    feature_labels: tuple[str, ...]  # original long names
    dates: np.ndarray  # (705,), datetime64[D]

    @property
    def n_days(self) -> int:
        return self.tensor.shape[0]

    @property
    def n_stations(self) -> int:
        return self.tensor.shape[1]

    @property
    def n_features(self) -> int:
        return self.tensor.shape[2]

    @property
    def n_station_days(self) -> int:
        return self.n_days * self.n_stations


def _short_name(long_name: str) -> str:
    """Convert a UCI feature description into a stable short name."""
    text = long_name.lower()

    if "specific conductance" in text:
        variable = "SC"
    elif "dissolved oxygen" in text:
        variable = "DO"
    elif "temperature" in text:
        variable = "T"
    elif re.search(r"\bph\b", text):
        variable = "pH"
    else:  # pragma: no cover - defensive
        raise ValueError(f"Cannot infer variable from feature name: {long_name!r}")

    if "maximum" in text:
        statistic = "max"
    elif "minimum" in text:
        statistic = "min"
    elif "mean" in text:
        statistic = "mean"
    else:  # pragma: no cover - defensive
        raise ValueError(f"Cannot infer statistic from feature name: {long_name!r}")

    return f"{variable}_{statistic}"


def _parse_groups(raw: np.ndarray, n_stations: int = N_STATIONS) -> np.ndarray:
    """Map the 1-based ``location_group`` cell array to 0-based station groups."""
    groups = np.full(n_stations, -1, dtype=int)
    cells = np.asarray(raw).ravel()

    for group_index, cell in enumerate(cells):
        indices = np.asarray(cell).ravel().astype(int) - 1
        if indices.size == 0:
            continue
        if indices.min() < 0 or indices.max() >= n_stations:
            raise ValueError(f"Station group {group_index} contains an out-of-range index")
        if (groups[indices] != -1).any():
            raise ValueError(f"Station group {group_index} overlaps another group")
        groups[indices] = group_index

    if (groups < 0).any():
        missing = np.flatnonzero(groups < 0).tolist()
        raise ValueError(f"Stations without a group assignment: {missing}")

    return groups


def _stack_cells(cell_array: np.ndarray, expected_days: int) -> np.ndarray:
    """Stack a MATLAB cell array of ``(37, 11)`` matrices into a tensor."""
    if cell_array.ndim != 2 or cell_array.shape[0] != 1:
        raise ValueError(f"Expected a 1 x N cell array, got shape {cell_array.shape}")
    if cell_array.shape[1] != expected_days:
        raise ValueError(
            f"Expected {expected_days} daily cells, got {cell_array.shape[1]}"
        )

    parts: list[np.ndarray] = []
    for day in range(cell_array.shape[1]):
        cell = np.asarray(cell_array[0, day], dtype=float)
        if cell.shape != (N_STATIONS, N_FEATURES):
            raise ValueError(
                f"Day {day}: expected shape {(N_STATIONS, N_FEATURES)}, got {cell.shape}"
            )
        parts.append(cell)

    tensor = np.stack(parts, axis=0)
    if not np.isfinite(tensor).all():
        raise ValueError("The archive contains NaN or infinite values")
    return tensor


def load_dataset(path: str | Path | None = None) -> Dataset:
    """Load and validate the UCI 733 archive.

    Parameters
    ----------
    path:
        Path to ``water_dataset.mat``. Defaults to
        ``data/raw/water_dataset.mat`` inside the project root.
    """
    if path is None:
        path = project_root() / "data" / "raw" / "water_dataset.mat"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw archive not found at {path}. "
            "Download it from the UCI repository; see data/README.md."
        )

    mat = sio.loadmat(path)
    required = {"X_tr", "X_te", "features", "location_ids", "location_group"}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f"Archive is missing required variables: {sorted(missing)}")

    train = _stack_cells(mat["X_tr"], TRAIN_DAYS)
    test = _stack_cells(mat["X_te"], TEST_DAYS)
    tensor = np.concatenate([train, test], axis=0)

    station_ids = np.asarray(mat["location_ids"]).ravel().astype(int)
    if station_ids.size != N_STATIONS:
        raise ValueError(f"Expected {N_STATIONS} station IDs, got {station_ids.size}")
    if len(np.unique(station_ids)) != station_ids.size:
        raise ValueError("Station IDs are not unique")

    labels = tuple(
        str(np.asarray(item).ravel()[0]) for item in np.asarray(mat["features"]).ravel()
    )
    names = tuple(_short_name(label) for label in labels)
    if names != EXPECTED_FEATURES:
        raise ValueError(
            "Parsed feature names do not match the expected order.\n"
            f"Parsed:   {names}\nExpected: {EXPECTED_FEATURES}"
        )

    groups = _parse_groups(mat["location_group"])

    dates = np.datetime64(START_DATE) + np.arange(tensor.shape[0]).astype(
        "timedelta64[D]"
    )
    if str(dates[-1]) != END_DATE:
        raise ValueError(f"Expected the series to end on {END_DATE}, got {dates[-1]}")

    return Dataset(
        tensor=tensor,
        station_ids=station_ids,
        station_groups=groups,
        feature_names=names,
        feature_labels=labels,
        dates=dates,
    )

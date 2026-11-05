"""Controlled fault injection for the multi-station water-quality record.

The injector works on **raw normalised values in [0, 1]**, not on the scaled
representation. The caller injects into the clean test tensor, then applies the
train-fitted scaler. This keeps the scaler blind to corrupted values.

Fault types
-----------
spike
    One station-day, one feature, shifted by +/- ``k * IQR``.
drift
    A contiguous window with a linear ramp ending at +/- ``k * IQR``.
correlation_break
    A contiguous window whose values are replaced by values of the same
    feature drawn from other station-days. Marginals are preserved; dependence
    is broken.
stuck
    A contiguous window held at its first value. Optional diagnostic; a
    cross-sectional detector should not be able to see it reliably.

The return object contains the corrupted tensor, a tidy label table, and two
boolean masks (station-day and cell level) aligned with the tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

FAULT_TYPES: tuple[str, ...] = ("spike", "drift", "correlation_break", "stuck")

LABEL_COLUMNS: tuple[str, ...] = (
    "event_id",
    "row_id",
    "day_index",
    "date",
    "station_id",
    "station_index",
    "feature",
    "feature_index",
    "fault_type",
    "magnitude",
    "magnitude_unit",
    "window_start_day",
    "window_end_day",
    "seed",
)


@dataclass(frozen=True)
class InjectionProfile:
    """Which features are eligible, and their training IQRs."""

    feature_names: tuple[str, ...]
    eligible_features: tuple[int, ...]
    iqr: np.ndarray  # (n_stations, n_features)
    median_iqr: np.ndarray  # (n_features,)
    threshold: float

    @property
    def eligible_feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_names[index] for index in self.eligible_features)


@dataclass(frozen=True)
class InjectionResult:
    """Corrupted data plus the ground truth needed for scoring."""

    corrupted: np.ndarray
    labels: pd.DataFrame
    station_day_mask: np.ndarray  # (n_days, n_stations)
    cell_mask: np.ndarray  # (n_days, n_stations, n_features)
    fault_type: str
    seed: int
    target_prevalence: float
    actual_prevalence: float
    n_events: int
    day_offset: int
    eligible_feature_names: tuple[str, ...]


def build_injection_profile(
    train_tensor: np.ndarray,
    feature_names: Sequence[str],
    min_median_iqr: float = 0.01,
) -> InjectionProfile:
    """Select features with enough within-station variation to inject into.

    Parameters
    ----------
    train_tensor:
        Training data of shape ``(n_train_days, n_stations, n_features)``.
    feature_names:
        Feature names, length ``n_features``.
    min_median_iqr:
        A feature is eligible if its median within-station IQR exceeds this
        value. The default 0.01 is in the raw normalised units of the dataset.
    """
    tensor = np.asarray(train_tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(f"train_tensor must be 3-D, got shape {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("train_tensor contains NaN or infinite values")

    names = tuple(feature_names)
    if len(names) != tensor.shape[2]:
        raise ValueError(
            f"Expected {tensor.shape[2]} feature names, got {len(names)}"
        )

    q25 = np.percentile(tensor, 25, axis=0)
    q75 = np.percentile(tensor, 75, axis=0)
    iqr = q75 - q25
    median_iqr = np.median(iqr, axis=0)
    eligible = tuple(
        int(index) for index in np.flatnonzero(median_iqr > float(min_median_iqr))
    )

    if not eligible:
        raise ValueError(
            "No feature has enough within-station variation for injection. "
            "Lower min_median_iqr or inspect the data."
        )

    return InjectionProfile(
        feature_names=names,
        eligible_features=eligible,
        iqr=iqr,
        median_iqr=median_iqr,
        threshold=float(min_median_iqr),
    )


@dataclass
class FaultInjector:
    """Inject one fault type into a clean raw test tensor."""

    profile: InjectionProfile
    station_ids: np.ndarray
    dates: np.ndarray
    seed: int = 0

    def inject(
        self,
        clean: np.ndarray,
        fault_type: str,
        prevalence: float = 0.02,
        magnitude: float | tuple[float, float] = (4.0, 8.0),
        window_range: tuple[int, int] = (5, 20),
        day_offset: int = 423,
    ) -> InjectionResult:
        """Return a corrupted copy, labels and masks.

        Parameters
        ----------
        clean:
            Raw normalised test tensor, shape ``(n_days, n_stations, n_features)``.
        fault_type:
            One of ``spike``, ``drift``, ``correlation_break``, ``stuck``.
        prevalence:
            Target fraction of station-days to corrupt.
        magnitude:
            Scalar or ``(low, high)`` range, in units of the feature's training
            IQR. Ignored for ``correlation_break`` and ``stuck``.
        window_range:
            Inclusive ``(min_days, max_days)`` for temporal faults.
        day_offset:
            Global day index of the first row in ``clean``. The test period
            starts at 423.
        """
        tensor = self._validate_clean(clean)
        fault = self._validate_fault_type(fault_type)
        target = self._validate_prevalence(prevalence, tensor.shape)
        low, high = self._validate_window(window_range, tensor.shape[0])

        if fault == "spike":
            return self._inject_spikes(
                tensor=tensor,
                target=target,
                magnitude=magnitude,
                day_offset=day_offset,
            )
        return self._inject_windows(
            tensor=tensor,
            fault=fault,
            target=target,
            magnitude=magnitude,
            window_range=(low, high),
            day_offset=day_offset,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_clean(self, clean: np.ndarray) -> np.ndarray:
        tensor = np.asarray(clean, dtype=float)
        if tensor.ndim != 3:
            raise ValueError(f"clean must be 3-D, got shape {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError("clean contains NaN or infinite values")
        if tensor.min() < -1e-9 or tensor.max() > 1.0 + 1e-9:
            raise ValueError(
                "clean must contain raw normalised values in [0, 1]. "
                "Inject before scaling, not after."
            )

        n_days, n_stations, n_features = tensor.shape
        if len(self.station_ids) != n_stations:
            raise ValueError(
                f"Expected {n_stations} station IDs, got {len(self.station_ids)}"
            )
        if len(self.dates) != n_days:
            raise ValueError(f"Expected {n_days} dates, got {len(self.dates)}")
        if self.profile.iqr.shape != (n_stations, n_features):
            raise ValueError(
                "Injection profile shape does not match the clean tensor: "
                f"{self.profile.iqr.shape} vs {(n_stations, n_features)}"
            )
        if len(self.profile.feature_names) != n_features:
            raise ValueError(
                f"Expected {n_features} feature names, got "
                f"{len(self.profile.feature_names)}"
            )

        return tensor

    @staticmethod
    def _validate_fault_type(fault_type: str) -> str:
        key = fault_type.lower()
        if key not in FAULT_TYPES:
            raise ValueError(
                f"Unknown fault type {fault_type!r}. Choose from {FAULT_TYPES}"
            )
        return key

    @staticmethod
    def _validate_prevalence(prevalence: float, shape: tuple[int, ...]) -> int:
        value = float(prevalence)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"prevalence must be in (0, 1], got {prevalence}")
        n_days, n_stations = shape[0], shape[1]
        cells = n_days * n_stations
        return max(1, min(cells, int(round(value * cells))))

    @staticmethod
    def _validate_window(
        window_range: tuple[int, int], n_days: int
    ) -> tuple[int, int]:
        low, high = int(window_range[0]), int(window_range[1])
        if low < 1 or high < low:
            raise ValueError(f"Invalid window_range: {window_range}")
        if high > n_days:
            raise ValueError(
                f"window_range upper bound {high} exceeds the series length {n_days}"
            )
        return low, high

    @staticmethod
    def _draw_magnitude(
        rng: np.random.Generator, magnitude: float | tuple[float, float]
    ) -> float:
        if isinstance(magnitude, (tuple, list, np.ndarray)):
            if len(magnitude) != 2:
                raise ValueError("magnitude must be a scalar or a (low, high) pair")
            low, high = float(magnitude[0]), float(magnitude[1])
            if low <= 0.0 or high < low:
                raise ValueError(f"Invalid magnitude range: {magnitude}")
            return float(rng.uniform(low, high))

        value = float(magnitude)
        if value <= 0.0:
            raise ValueError(f"magnitude must be positive, got {magnitude}")
        return value

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def _inject_spikes(
        self,
        tensor: np.ndarray,
        target: int,
        magnitude: float | tuple[float, float],
        day_offset: int,
    ) -> InjectionResult:
        n_days, n_stations, n_features = tensor.shape
        rng = np.random.default_rng(self.seed)

        corrupted = tensor.copy()
        station_mask = np.zeros((n_days, n_stations), dtype=bool)
        cell_mask = np.zeros((n_days, n_stations, n_features), dtype=bool)
        labels: list[dict[str, object]] = []

        for flat_index in rng.permutation(n_days * n_stations):
            if len(labels) >= target:
                break
            day = int(flat_index // n_stations)
            station = int(flat_index % n_stations)
            feature = int(rng.choice(self.profile.eligible_features))
            k = self._draw_magnitude(rng, magnitude)
            delta = k * self.profile.iqr[station, feature]
            sign = -1.0 if rng.random() < 0.5 else 1.0

            original = tensor[day, station, feature]
            updated = float(np.clip(original + sign * delta, 0.0, 1.0))
            if np.isclose(updated, original, atol=1e-12, rtol=0.0):
                updated = float(np.clip(original - sign * delta, 0.0, 1.0))
            if np.isclose(updated, original, atol=1e-12, rtol=0.0):
                continue

            corrupted[day, station, feature] = updated
            station_mask[day, station] = True
            cell_mask[day, station, feature] = True
            labels.append(
                self._label(
                    event_id=len(labels),
                    day_local=day,
                    station=station,
                    feature=feature,
                    fault_type="spike",
                    magnitude=k,
                    magnitude_unit="iqr",
                    window_start_local=day,
                    window_end_local=day,
                    day_offset=day_offset,
                )
            )

        return self._result(
            corrupted=corrupted,
            labels=labels,
            station_mask=station_mask,
            cell_mask=cell_mask,
            fault_type="spike",
            target=target,
            n_events=len(labels),
            day_offset=day_offset,
        )

    def _inject_windows(
        self,
        tensor: np.ndarray,
        fault: str,
        target: int,
        magnitude: float | tuple[float, float],
        window_range: tuple[int, int],
        day_offset: int,
    ) -> InjectionResult:
        n_days, n_stations, n_features = tensor.shape
        rng = np.random.default_rng(self.seed)

        corrupted = tensor.copy()
        station_mask = np.zeros((n_days, n_stations), dtype=bool)
        cell_mask = np.zeros((n_days, n_stations, n_features), dtype=bool)
        labels: list[dict[str, object]] = []
        used: set[tuple[int, int, int]] = set()

        affected = 0
        event_id = 0
        attempts = 0
        max_attempts = 10_000

        while affected < target and attempts < max_attempts:
            attempts += 1
            station = int(rng.integers(0, n_stations))
            feature = int(rng.choice(self.profile.eligible_features))
            width = int(rng.integers(window_range[0], window_range[1] + 1))
            start = int(rng.integers(0, n_days - width + 1))

            cells = [(station, feature, start + offset) for offset in range(width)]
            if any(cell in used for cell in cells):
                continue

            if fault == "drift":
                changes, event_magnitude = self._drift_changes(
                    tensor, station, feature, start, width, magnitude, rng
                )
            elif fault == "correlation_break":
                changes, event_magnitude = self._break_changes(
                    tensor, station, feature, start, width, rng
                )
            else:  # stuck
                changes, event_magnitude = self._stuck_changes(
                    tensor, station, feature, start, width
                )

            if not changes:
                continue

            used.update(cells)
            affected += len(changes)
            event_id += 1

            for offset, updated in changes.items():
                day = start + offset
                corrupted[day, station, feature] = updated
                station_mask[day, station] = True
                cell_mask[day, station, feature] = True
                labels.append(
                    self._label(
                        event_id=event_id - 1,
                        day_local=day,
                        station=station,
                        feature=feature,
                        fault_type=fault,
                        magnitude=event_magnitude,
                        magnitude_unit=(
                            "iqr" if fault == "drift" else "not_applicable"
                        ),
                        window_start_local=start,
                        window_end_local=start + width - 1,
                        day_offset=day_offset,
                    )
                )

        return self._result(
            corrupted=corrupted,
            labels=labels,
            station_mask=station_mask,
            cell_mask=cell_mask,
            fault_type=fault,
            target=target,
            n_events=event_id,
            day_offset=day_offset,
        )

    def _drift_changes(
        self,
        tensor: np.ndarray,
        station: int,
        feature: int,
        start: int,
        width: int,
        magnitude: float | tuple[float, float],
        rng: np.random.Generator,
    ) -> tuple[dict[int, float], float]:
        original = tensor[start : start + width, station, feature]
        k = self._draw_magnitude(rng, magnitude)
        delta = k * self.profile.iqr[station, feature]
        sign = -1.0 if rng.random() < 0.5 else 1.0

        # Start at a non-zero increment so every day is actually changed.
        ramp = np.linspace(sign * delta / width, sign * delta, width)
        updated = np.clip(original + ramp, 0.0, 1.0)
        if np.allclose(updated, original, atol=1e-12, rtol=0.0):
            updated = np.clip(original - ramp, 0.0, 1.0)
        if np.allclose(updated, original, atol=1e-12, rtol=0.0):
            return {}, np.nan

        changes = {
            offset: float(updated[offset])
            for offset in range(width)
            if not np.isclose(updated[offset], original[offset], atol=1e-12, rtol=0.0)
        }
        return changes, k

    def _break_changes(
        self,
        tensor: np.ndarray,
        station: int,
        feature: int,
        start: int,
        width: int,
        rng: np.random.Generator,
    ) -> tuple[dict[int, float], float]:
        pool = tensor[:, :, feature].ravel()
        changes: dict[int, float] = {}

        for offset in range(width):
            day = start + offset
            original = tensor[day, station, feature]
            candidates = pool[~np.isclose(pool, original, atol=1e-12, rtol=0.0)]
            if candidates.size == 0:
                continue
            donor = float(rng.choice(candidates))
            changes[offset] = float(np.clip(donor, 0.0, 1.0))

        return changes, np.nan

    def _stuck_changes(
        self,
        tensor: np.ndarray,
        station: int,
        feature: int,
        start: int,
        width: int,
    ) -> tuple[dict[int, float], float]:
        stuck_value = float(tensor[start, station, feature])
        changes: dict[int, float] = {}
        for offset in range(1, width):
            day = start + offset
            original = tensor[day, station, feature]
            if not np.isclose(original, stuck_value, atol=1e-12, rtol=0.0):
                changes[offset] = stuck_value
        return changes, np.nan

    # ------------------------------------------------------------------
    # Labels and result assembly
    # ------------------------------------------------------------------

    def _label(
        self,
        event_id: int,
        day_local: int,
        station: int,
        feature: int,
        fault_type: str,
        magnitude: float,
        magnitude_unit: str,
        window_start_local: int,
        window_end_local: int,
        day_offset: int,
    ) -> dict[str, object]:
        day_global = day_offset + day_local
        return {
            "event_id": int(event_id),
            "row_id": int(day_global * len(self.station_ids) + station),
            "day_index": int(day_global),
            "date": pd.Timestamp(np.asarray(self.dates)[day_local]),
            "station_id": int(self.station_ids[station]),
            "station_index": int(station),
            "feature": self.profile.feature_names[feature],
            "feature_index": int(feature),
            "fault_type": fault_type,
            "magnitude": float(magnitude) if np.isfinite(magnitude) else np.nan,
            "magnitude_unit": magnitude_unit,
            "window_start_day": int(day_offset + window_start_local),
            "window_end_day": int(day_offset + window_end_local),
            "seed": int(self.seed),
        }

    def _result(
        self,
        corrupted: np.ndarray,
        labels: list[dict[str, object]],
        station_mask: np.ndarray,
        cell_mask: np.ndarray,
        fault_type: str,
        target: int,
        n_events: int,
        day_offset: int,
    ) -> InjectionResult:
        n_days, n_stations = station_mask.shape
        frame = pd.DataFrame(labels, columns=list(LABEL_COLUMNS))
        actual = float(station_mask.sum() / (n_days * n_stations))
        return InjectionResult(
            corrupted=corrupted,
            labels=frame,
            station_day_mask=station_mask,
            cell_mask=cell_mask,
            fault_type=fault_type,
            seed=int(self.seed),
            target_prevalence=float(target / (n_days * n_stations)),
            actual_prevalence=actual,
            n_events=int(n_events),
            day_offset=int(day_offset),
            eligible_feature_names=self.profile.eligible_feature_names,
        )

#!/usr/bin/env python3
"""Phase 1, step 1: build the station-day table and the within-station scaler.

Outputs
-------
data/processed/station_day.csv          raw normalised values + metadata
data/processed/station_day_scaled.csv   within-station robust z-scores + metadata
data/processed/scaler_params.csv        per-station, per-feature median and scale
results/logs/01_build_dataset.json      run log with versions and checksums
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import TRAIN_DAYS, load_dataset  # noqa: E402
from src.preprocess import (  # noqa: E402
    build_raw_frame,
    build_scaled_frame,
    fit_scaler,
    scaler_frame,
    validate_frame,
)
from src.utils import git_commit, package_versions, sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "water_dataset.mat",
        help="Path to water_dataset.mat",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory for generated tables",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results" / "logs",
        help="Directory for the JSON run log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.raw}")
    dataset = load_dataset(args.raw)
    print(
        f"  tensor={dataset.tensor.shape}  stations={dataset.n_stations}  "
        f"features={dataset.n_features}  days={dataset.n_days}"
    )

    train_tensor = dataset.tensor[:TRAIN_DAYS]
    scaler = fit_scaler(train_tensor, dataset.station_ids, dataset.feature_names)
    scaler_stats = scaler.summary()
    print(
        f"  scaler fitted on {TRAIN_DAYS} training days "
        f"(fallback={scaler_stats['fallback_cells']}, "
        f"constant={scaler_stats['constant_cells']})"
    )

    raw_frame = build_raw_frame(dataset)
    scaled_frame = build_scaled_frame(dataset, scaler)
    validate_frame(raw_frame, dataset.feature_names)
    validate_frame(scaled_frame, tuple(f"z_{name}" for name in dataset.feature_names))

    raw_path = args.out_dir / "station_day.csv"
    scaled_path = args.out_dir / "station_day_scaled.csv"
    scaler_path = args.out_dir / "scaler_params.csv"

    raw_frame.to_csv(raw_path, index=False)
    scaled_frame.to_csv(scaled_path, index=False)
    scaler_frame(scaler).to_csv(scaler_path, index=False)

    split_counts = raw_frame["split"].value_counts().to_dict()
    missing = int(raw_frame.loc[:, list(dataset.feature_names)].isna().sum().sum())

    print()
    print(f"Rows:     {len(raw_frame):,}")
    print(f"  train:  {split_counts.get('train', 0):,}")
    print(f"  test:   {split_counts.get('test', 0):,}")
    print(f"Dates:    {raw_frame['date'].min().date()} -> {raw_frame['date'].max().date()}")
    print(f"Stations: {raw_frame['station_id'].nunique()}")
    print(f"Features: {len(dataset.feature_names)}")
    print(f"Missing:  {missing}")

    payload = {
        "phase": 1,
        "step": "build_dataset",
        "git_commit": git_commit(),
        "versions": package_versions(),
        "inputs": {
            "raw": str(args.raw),
            "raw_sha256": sha256(args.raw),
        },
        "outputs": {
            "station_day": str(raw_path),
            "station_day_sha256": sha256(raw_path),
            "station_day_scaled": str(scaled_path),
            "station_day_scaled_sha256": sha256(scaled_path),
            "scaler_params": str(scaler_path),
            "scaler_params_sha256": sha256(scaler_path),
        },
        "shape": {
            "rows": int(len(raw_frame)),
            "train_rows": int(split_counts.get("train", 0)),
            "test_rows": int(split_counts.get("test", 0)),
            "stations": int(raw_frame["station_id"].nunique()),
            "features": int(len(dataset.feature_names)),
            "days": int(raw_frame["day_index"].nunique()),
        },
        "missing_values": missing,
        "date_range": {
            "start": str(raw_frame["date"].min().date()),
            "end": str(raw_frame["date"].max().date()),
        },
        "scaler": {
            "fit_days": TRAIN_DAYS,
            **scaler_stats,
        },
    }
    log_path = args.log_dir / "01_build_dataset.json"
    write_json(log_path, payload)

    print()
    print(f"Wrote {raw_path}")
    print(f"Wrote {scaled_path}")
    print(f"Wrote {scaler_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

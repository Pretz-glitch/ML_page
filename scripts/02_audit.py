#!/usr/bin/env python3
"""Phase 1, step 2: audit the station-day table and write the Phase 1 figures.

Outputs
-------
results/tables/audit_variance.csv
results/tables/audit_correlations.csv
results/tables/audit_top_correlations.csv
results/tables/audit_floor_values.csv
results/tables/audit_station_summary.csv
results/tables/audit_split_summary.csv
results/tables/audit_group_summary.csv
results/tables/audit_summary.json
results/figures/audit_correlation.png
results/figures/audit_variance.png
results/figures/audit_coverage.png
results/figures/audit_distributions.png
results/phase1_summary.md
results/logs/02_audit.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import (  # noqa: E402
    FLOOR,
    correlation_table,
    floor_table,
    group_summary,
    split_summary,
    station_summary,
    top_correlations,
    variance_table,
)
from src.io import (  # noqa: E402
    FEATURE_FAMILIES,
    N_FEATURES,
    N_STATIONS,
    TEST_DAYS,
    TRAIN_DAYS,
)
from src.plots import (  # noqa: E402
    save_correlation_heatmap,
    save_coverage_figure,
    save_distributions,
    save_variance_bars,
)
from src.preprocess import META_COLUMNS  # noqa: E402
from src.utils import (  # noqa: E402
    git_commit,
    package_versions,
    sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory containing station_day.csv and station_day_scaled.csv",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "results" / "tables",
        help="Directory for audit tables",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "results" / "figures",
        help="Directory for audit figures",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results" / "logs",
        help="Directory for the run log",
    )
    return parser.parse_args()


def md_table(headers: list[str], rows: list[tuple[object, ...]]) -> str:
    """Render a small GitHub-flavoured Markdown table without extra dependencies."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.processed_dir / "station_day.csv"
    scaled_path = args.processed_dir / "station_day_scaled.csv"
    for path in (raw_path, scaled_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/01_build_dataset.py first.")

    raw = pd.read_csv(raw_path, parse_dates=["date"])
    scaled = pd.read_csv(scaled_path, parse_dates=["date"])

    features = tuple(
        column for column in raw.columns if column not in META_COLUMNS
    )
    if len(features) != N_FEATURES:
        raise AssertionError(f"Expected {N_FEATURES} feature columns, got {len(features)}")
    if len(raw) != TRAIN_DAYS * N_STATIONS + TEST_DAYS * N_STATIONS:
        raise AssertionError(f"Unexpected row count: {len(raw)}")
    if raw.loc[:, list(features)].isna().any().any():
        raise AssertionError("Raw table contains missing values")
    if set(raw["split"].unique()) != {"train", "test"}:
        raise AssertionError("Unexpected split labels")

    z_features = tuple(f"z_{feature}" for feature in features)
    missing_z = [feature for feature in z_features if feature not in scaled.columns]
    if missing_z:
        raise AssertionError(f"Scaled table is missing columns: {missing_z}")

    print("Computing audit tables...")
    families = pd.DataFrame(
        {
            "feature": list(features),
            "family": [FEATURE_FAMILIES[feature] for feature in features],
        }
    )
    variance = variance_table(raw, features)
    correlation = correlation_table(raw, features)
    top = top_correlations(correlation)
    floor = floor_table(raw, features)
    stations = station_summary(raw, features)
    splits = split_summary(raw)
    groups = group_summary(raw)

    variance_path = args.table_dir / "audit_variance.csv"
    correlation_path = args.table_dir / "audit_correlations.csv"
    top_path = args.table_dir / "audit_top_correlations.csv"
    floor_path = args.table_dir / "audit_floor_values.csv"
    family_path = args.table_dir / "audit_feature_families.csv"
    station_path = args.table_dir / "audit_station_summary.csv"
    split_path = args.table_dir / "audit_split_summary.csv"
    group_path = args.table_dir / "audit_group_summary.csv"

    variance.to_csv(variance_path, index=False)
    correlation.to_csv(correlation_path, index_label="feature")
    top.to_csv(top_path, index=False)
    floor.to_csv(floor_path, index=False)
    families.to_csv(family_path, index=False)
    stations.to_csv(station_path, index=False)
    splits.to_csv(split_path, index=False)
    groups.to_csv(group_path, index=False)

    print("Checking the scaled training representation...")
    train_scaled = scaled.loc[scaled["split"] == "train"]
    scaled_median = train_scaled.groupby("station_id")[list(z_features)].median()
    q25 = train_scaled.groupby("station_id")[list(z_features)].quantile(0.25)
    q75 = train_scaled.groupby("station_id")[list(z_features)].quantile(0.75)
    scaled_iqr = q75 - q25

    scaled_median_values = scaled_median.to_numpy(dtype=float)
    scaled_iqr_values = scaled_iqr.to_numpy(dtype=float)
    nonzero_iqr = scaled_iqr_values[scaled_iqr_values > 1e-9]
    scaled_check = {
        "max_abs_train_median": float(np.max(np.abs(scaled_median_values))),
        "median_nonzero_train_iqr": float(np.median(nonzero_iqr)) if nonzero_iqr.size else 0.0,
        "min_nonzero_train_iqr": float(np.min(nonzero_iqr)) if nonzero_iqr.size else 0.0,
        "max_train_iqr": float(np.max(scaled_iqr_values)),
        "cells_with_zero_train_iqr": int(np.sum(scaled_iqr_values <= 1e-9)),
        "cells_checked": int(scaled_iqr_values.size),
    }

    print("Writing figures...")
    correlation_figure = args.figure_dir / "audit_correlation.png"
    variance_figure = args.figure_dir / "audit_variance.png"
    coverage_figure = args.figure_dir / "audit_coverage.png"
    distributions_figure = args.figure_dir / "audit_distributions.png"

    save_correlation_heatmap(correlation, correlation_figure)
    save_variance_bars(variance, variance_figure)
    save_coverage_figure(raw, coverage_figure)
    save_distributions(raw, features, distributions_figure)

    total_cells = int(len(raw) * len(features))
    missing_total = int(raw.loc[:, list(features)].isna().sum().sum())
    exact_zero = int((raw.loc[:, list(features)] == 0.0).sum().sum())
    at_floor = int((raw.loc[:, list(features)] <= FLOOR).sum().sum())
    between_high = variance.loc[variance["between_fraction"] >= 0.5, "feature"].tolist()
    floor_high = floor.loc[floor["floor_fraction"] >= 0.05, "feature"].tolist()

    summary = {
        "phase": 1,
        "step": "audit",
        "versions": package_versions(),
        "inputs": {
            "station_day": str(raw_path.relative_to(ROOT))
            if raw_path.is_relative_to(ROOT)
            else str(raw_path),
            "station_day_sha256": sha256(raw_path),
            "station_day_scaled": str(scaled_path.relative_to(ROOT))
            if scaled_path.is_relative_to(ROOT)
            else str(scaled_path),
            "station_day_scaled_sha256": sha256(scaled_path),
        },
        "shape": {
            "rows": int(len(raw)),
            "station_days": int(len(raw)),
            "days": int(raw["day_index"].nunique()),
            "stations": int(raw["station_id"].nunique()),
            "features": int(len(features)),
        },
        "split": splits.to_dict(orient="records"),
        "station_groups": groups.to_dict(orient="records"),
        "missing_values": missing_total,
        "total_feature_cells": total_cells,
        "exact_zero_values": exact_zero,
        "at_floor_values": at_floor,
        "scaled_check": scaled_check,
        "between_station_ge_0_5": between_high,
        "floor_fraction_ge_0_05": floor_high,
        "top_correlations": top.head(10).to_dict(orient="records"),
    }

    summary_path = args.table_dir / "audit_summary.json"
    write_json(summary_path, summary)

    log = {
        "phase": 1,
        "step": "audit",
        "git_commit": git_commit(),
        "versions": package_versions(),
        "outputs": {
            "variance": str(variance_path),
            "correlations": str(correlation_path),
            "top_correlations": str(top_path),
            "floor_values": str(floor_path),
            "feature_families": str(family_path),
            "station_summary": str(station_path),
            "split_summary": str(split_path),
            "group_summary": str(group_path),
            "summary_json": str(summary_path),
            "correlation_figure": str(correlation_figure),
            "variance_figure": str(variance_figure),
            "coverage_figure": str(coverage_figure),
            "distributions_figure": str(distributions_figure),
        },
        "checksums": {
            "station_day": sha256(raw_path),
            "station_day_scaled": sha256(scaled_path),
        },
    }
    log_path = args.log_dir / "02_audit.json"
    write_json(log_path, log)

    lines: list[str] = []
    lines.append("# Phase 1 — Data audit summary")
    lines.append("")
    lines.append("Generated by `scripts/02_audit.py` from the committed station-day table.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        md_table(
            ["Property", "Value"],
            [
                ("Station-days", f"{len(raw):,}"),
                ("Days", f"{raw['day_index'].nunique():,}"),
                ("Stations", raw["station_id"].nunique()),
                ("Features", len(features)),
                ("Date range", f"{raw['date'].min().date()} to {raw['date'].max().date()}"),
                ("Missing feature values", missing_total),
                ("Feature cells", f"{total_cells:,}"),
            ],
        )
    )
    lines.append("")
    lines.append("## Split")
    lines.append("")
    lines.append(
        md_table(
            ["Split", "Rows", "Days", "Stations", "Start", "End"],
            [
                (
                    row.split,
                    f"{row.rows:,}",
                    row.days,
                    row.stations,
                    row.start,
                    row.end,
                )
                for row in splits.itertuples(index=False)
            ],
        )
    )
    lines.append("")
    lines.append("## Between-station vs within-station variance")
    lines.append("")
    ordered_variance = variance.sort_values("between_fraction", ascending=False)
    lines.append(
        md_table(
            ["Feature", "Family", "Between fraction", "Median within IQR", "Floor fraction"],
            [
                (
                    row.feature,
                    row.family,
                    f"{row.between_fraction:.3f}",
                    f"{row.median_within_iqr:.4f}",
                    f"{row.floor_fraction:.3%}",
                )
                for row in ordered_variance.itertuples(index=False)
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Features with at least half their variance between stations: "
        f"**{', '.join(between_high)}**."
    )
    lines.append("")
    lines.append("## Strongest correlations")
    lines.append("")
    lines.append(
        md_table(
            ["Feature A", "Feature B", "Pearson r"],
            [
                (row.feature_a, row.feature_b, f"{row.r:.3f}")
                for row in top.head(10).itertuples(index=False)
            ],
        )
    )
    lines.append("")
    lines.append("## Station groups")
    lines.append("")
    lines.append(
        md_table(
            ["Group", "Stations", "Rows"],
            [
                (row.station_group, row.stations, f"{row.rows:,}")
                for row in groups.itertuples(index=False)
            ],
        )
    )
    lines.append("")
    lines.append("## Data-quality checks")
    lines.append("")
    lines.append(f"- Missing feature values: **{missing_total}** of {total_cells:,} cells.")
    lines.append(f"- Exact zeros: **{exact_zero:,}**.")
    lines.append(f"- Values at the normalisation floor (`<= {FLOOR:.4f}`): **{at_floor:,}**.")
    if floor_high:
        lines.append(
            f"- Features with at least 5% of values at the floor: "
            f"**{', '.join(floor_high)}**."
        )
    else:
        lines.append("- No feature has at least 5% of values at the floor.")
    lines.append("")
    lines.append("### Scaled-representation check (training days only)")
    lines.append("")
    lines.append(
        md_table(
            ["Check", "Value"],
            [
                ("Max absolute median", f"{scaled_check['max_abs_train_median']:.3e}"),
                ("Median non-zero IQR", f"{scaled_check['median_nonzero_train_iqr']:.4f}"),
                ("Min non-zero IQR", f"{scaled_check['min_nonzero_train_iqr']:.4f}"),
                ("Max IQR", f"{scaled_check['max_train_iqr']:.4f}"),
                ("Cells with zero IQR", scaled_check["cells_with_zero_train_iqr"]),
                ("Cells checked", scaled_check["cells_checked"]),
            ],
        )
    )
    lines.append("")
    lines.append("## Implications for the project")
    lines.append("")
    lines.append(
        "1. **Within-station scaling is mandatory.** "
        f"{len(between_high)} of {len(features)} features have at least half their "
        "variance between stations, so a pooled detector would largely learn station identity."
    )
    lines.append(
        "2. **Attribution must be interpreted by feature family.** "
        f"The strongest pair is `{top.iloc[0]['feature_a']}`–`{top.iloc[0]['feature_b']}` "
        f"at r = {top.iloc[0]['r']:.3f}; within-family substitution will make "
        "single-column attributions unstable."
    )
    lines.append(
        "3. **The split is temporal and must not be shuffled.** "
        "The table uses days 0–422 for training and 423–704 for testing."
    )
    lines.append(
        "4. **No imputation is needed.** "
        f"The feature matrix has {missing_total} missing values."
    )
    lines.append("")

    report_path = ROOT / "results" / "phase1_summary.md"
    report_path.write_text("\n".join(lines) + "\n")

    print()
    print(f"Rows:                 {len(raw):,}")
    print(f"Stations:             {raw['station_id'].nunique()}")
    print(f"Features:             {len(features)}")
    print(f"Missing values:       {missing_total}")
    print(f"Between-station >=0.5: {', '.join(between_high)}")
    print(f"Strongest correlation: {top.iloc[0]['feature_a']}–{top.iloc[0]['feature_b']} "
          f"r={top.iloc[0]['r']:.3f}")
    print()
    print(f"Wrote {variance_path}")
    print(f"Wrote {correlation_path}")
    print(f"Wrote {top_path}")
    print(f"Wrote {floor_path}")
    print(f"Wrote {family_path}")
    print(f"Wrote {station_path}")
    print(f"Wrote {split_path}")
    print(f"Wrote {group_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {correlation_figure}")
    print(f"Wrote {variance_figure}")
    print(f"Wrote {coverage_figure}")
    print(f"Wrote {distributions_figure}")
    print(f"Wrote {report_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

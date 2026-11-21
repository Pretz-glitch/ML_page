#!/usr/bin/env python3
"""Phase 3: detector x fault-type x magnitude benchmark.

For each seed:
  1. fit every detector on clean training data;
  2. inject each fault condition into the held-out test period;
  3. score the corrupted test set and compute detection metrics.

Outputs
-------
results/tables/detection.csv           one row per seed x condition x detector
results/tables/detection_summary.csv   mean, SD and bootstrap CI per condition
results/tables/detection_paired.csv    paired bootstrap comparisons
results/figures/detection_heatmap.png  mean ROC-AUC and recall@k
results/logs/03_run_detection.json     run log
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

from src.detectors import make_detector  # noqa: E402
from src.faults import FaultInjector, build_injection_profile  # noqa: E402
from src.io import TRAIN_DAYS, load_dataset  # noqa: E402
from src.metrics import (  # noqa: E402
    bootstrap_ci,
    detection_metrics,
    paired_bootstrap_diff,
)
from src.plots import save_detection_heatmap  # noqa: E402
from src.preprocess import fit_scaler, transform  # noqa: E402
from src.utils import git_commit, package_versions, sha256, write_json  # noqa: E402

DETECTOR_ROLES: dict[str, str] = {
    "mad": "primary",
    "mahalanobis": "primary",
    "iforest": "primary",
    "pca": "secondary",
    "eif": "secondary",
}

DETECTOR_NAMES: dict[str, str] = {
    "mad": "MAD",
    "mahalanobis": "Mahalanobis",
    "iforest": "Isolation Forest",
    "pca": "PCA reconstruction*",
    "eif": "Extended IF*",
}

FAULT_NAMES: dict[str, str] = {
    "spike": "Spike",
    "drift": "Drift",
    "correlation_break": "Correlation break",
    "stuck": "Stuck",
}

FAULT_ORDER: tuple[str, ...] = ("spike", "drift", "correlation_break", "stuck")
METRICS: tuple[str, ...] = (
    "auc_roc",
    "auc_pr",
    "recall_at_k",
    "fpr",
)

PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mahalanobis", "iforest"),
    ("mahalanobis", "mad"),
    ("iforest", "mad"),
)

SECONDARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("eif", "iforest"),
    ("pca", "mad"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "water_dataset.mat",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--magnitudes", nargs="+", type=float, default=[4.0, 6.0, 8.0]
    )
    parser.add_argument("--prevalence", type=float, default=0.02)
    parser.add_argument("--window-min", type=int, default=5)
    parser.add_argument("--window-max", type=int, default=20)
    parser.add_argument("--fpr-quantile", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(DETECTOR_ROLES),
        choices=list(DETECTOR_ROLES),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "tables" / "detection.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "tables" / "detection_summary.csv",
    )
    parser.add_argument(
        "--paired-out",
        type=Path,
        default=ROOT / "results" / "tables" / "detection_paired.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results" / "figures" / "detection_heatmap.png",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results" / "logs" / "03_run_detection.json",
    )
    return parser.parse_args()


def build_detector(name: str, seed: int):
    """Construct a detector with a fixed seed and single-threaded fitting."""
    if name == "eif":
        return make_detector(name, random_seed=seed, nthreads=1)
    if name == "iforest":
        return make_detector(name, random_state=seed, n_jobs=1)
    if name == "pca":
        return make_detector(name, random_state=seed)
    return make_detector(name)


def magnitude_key(value: str) -> float:
    """Sort numeric magnitudes before ``n/a``."""
    try:
        return float(value)
    except ValueError:
        return float("inf")


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required for paired comparisons")
    if not args.magnitudes:
        raise ValueError("At least one magnitude is required")

    for path in (args.out, args.summary_out, args.paired_out, args.figure, args.log):
        path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.raw)
    scaler = fit_scaler(
        dataset.tensor[:TRAIN_DAYS], dataset.station_ids, dataset.feature_names
    )
    profile = build_injection_profile(
        dataset.tensor[:TRAIN_DAYS], dataset.feature_names
    )

    train_scaled = transform(dataset.tensor[:TRAIN_DAYS], scaler).reshape(
        -1, dataset.n_features
    )
    clean_raw = dataset.tensor[TRAIN_DAYS:]
    clean_scaled = transform(clean_raw, scaler).reshape(-1, dataset.n_features)
    test_dates = dataset.dates[TRAIN_DAYS:]

    conditions: list[tuple[str, float | None]] = []
    for fault in FAULT_ORDER:
        if fault in {"spike", "drift"}:
            conditions.extend((fault, magnitude) for magnitude in args.magnitudes)
        else:
            conditions.append((fault, None))

    detector_order = [name for name in DETECTOR_ROLES if name in args.detectors]
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        detectors = {
            name: build_detector(name, seed).fit(train_scaled)
            for name in detector_order
        }
        train_scores = {
            name: detector.score(train_scaled)
            for name, detector in detectors.items()
        }
        clean_scores = {
            name: detector.score(clean_scaled)
            for name, detector in detectors.items()
        }
        thresholds = {
            name: float(np.quantile(scores, args.fpr_quantile))
            for name, scores in train_scores.items()
        }

        for fault, magnitude in conditions:
            injector = FaultInjector(
                profile=profile,
                station_ids=dataset.station_ids,
                dates=test_dates,
                seed=seed,
            )
            result = injector.inject(
                clean_raw,
                fault,
                prevalence=args.prevalence,
                magnitude=4.0 if magnitude is None else magnitude,
                window_range=(args.window_min, args.window_max),
                day_offset=TRAIN_DAYS,
            )
            corrupted_scaled = transform(result.corrupted, scaler).reshape(
                -1, dataset.n_features
            )
            labels = result.station_day_mask.ravel()
            magnitude_label = (
                "not_applicable" if magnitude is None else f"{magnitude:g}"
            )

            for name in detector_order:
                scores = detectors[name].score(corrupted_scaled)
                metrics = detection_metrics(
                    labels,
                    scores,
                    k=int(labels.sum()),
                    threshold=thresholds[name],
                    clean_scores=clean_scores[name],
                )
                rows.append(
                    {
                        "seed": seed,
                        "fault": fault,
                        "magnitude": magnitude_label,
                        "detector": name,
                        "role": DETECTOR_ROLES[name],
                        "n_events": result.n_events,
                        "n_total": metrics["n_total"],
                        "n_positive": metrics["n_positive"],
                        "prevalence": metrics["prevalence"],
                        "threshold": metrics["threshold"],
                        **{metric: metrics[metric] for metric in METRICS},
                    }
                )

        print(f"seed {seed}: {len(conditions)} conditions scored")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)

    summary_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["fault", "magnitude", "detector", "role"], sort=False
    ):
        fault, magnitude, detector, role = keys
        row: dict[str, object] = {
            "fault": fault,
            "magnitude": magnitude,
            "detector": detector,
            "role": role,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            stats = bootstrap_ci(
                group[metric].to_numpy(),
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["sd"]
            row[f"{metric}_ci_lo"] = stats["ci_lo"]
            row[f"{metric}_ci_hi"] = stats["ci_hi"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    fault_rank = {fault: index for index, fault in enumerate(FAULT_ORDER)}
    role_rank = {"primary": 0, "secondary": 1}
    detector_rank = {name: index for index, name in enumerate(DETECTOR_ROLES)}
    summary["_fault"] = summary["fault"].map(fault_rank)
    summary["_magnitude"] = summary["magnitude"].map(magnitude_key)
    summary["_role"] = summary["role"].map(role_rank)
    summary["_detector"] = summary["detector"].map(detector_rank)
    summary = summary.sort_values(
        ["_fault", "_magnitude", "_role", "_detector"]
    ).drop(columns=["_fault", "_magnitude", "_role", "_detector"])
    summary.to_csv(args.summary_out, index=False)

    paired_rows: list[dict[str, object]] = []
    comparisons = [
        (left, right, "primary") for left, right in PRIMARY_COMPARISONS
    ] + [(left, right, "secondary") for left, right in SECONDARY_COMPARISONS]

    for (fault, magnitude), group in frame.groupby(
        ["fault", "magnitude"], sort=False
    ):
        for left, right, kind in comparisons:
            if left not in group["detector"].values or right not in group["detector"].values:
                continue
            for metric in ("auc_roc", "recall_at_k"):
                pivot = (
                    group.loc[group["detector"].isin([left, right])]
                    .pivot(index="seed", columns="detector", values=metric)
                )
                if left not in pivot.columns or right not in pivot.columns:
                    continue
                stats = paired_bootstrap_diff(
                    pivot[left].to_numpy(),
                    pivot[right].to_numpy(),
                    n_boot=args.bootstrap,
                    seed=args.bootstrap_seed,
                )
                paired_rows.append(
                    {
                        "comparison": f"{left}_vs_{right}",
                        "kind": kind,
                        "fault": fault,
                        "magnitude": magnitude,
                        "metric": metric,
                        **stats,
                    }
                )

    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["_fault"] = paired["fault"].map(fault_rank)
        paired["_magnitude"] = paired["magnitude"].map(magnitude_key)
        paired["_kind"] = paired["kind"].map({"primary": 0, "secondary": 1})
        paired = paired.sort_values(
            ["_fault", "_magnitude", "_kind", "comparison", "metric"]
        ).drop(columns=["_fault", "_magnitude", "_kind"])
    paired.to_csv(args.paired_out, index=False)

    display_order = [name for name in DETECTOR_ROLES if name in detector_order]
    auc_pivot = (
        summary.pivot_table(
            index="detector", columns="fault", values="auc_roc_mean", aggfunc="mean"
        )
        .reindex(index=display_order, columns=list(FAULT_ORDER))
    )
    recall_pivot = (
        summary.pivot_table(
            index="detector",
            columns="fault",
            values="recall_at_k_mean",
            aggfunc="mean",
        )
        .reindex(index=display_order, columns=list(FAULT_ORDER))
    )
    save_detection_heatmap(
        auc_pivot,
        recall_pivot,
        args.figure,
        detector_names=DETECTOR_NAMES,
        fault_names=FAULT_NAMES,
    )

    write_json(
        args.log,
        {
            "phase": 3,
            "step": "detection_benchmark",
            "git_commit": git_commit(),
            "versions": package_versions(),
            "seeds": args.seeds,
            "magnitudes": args.magnitudes,
            "prevalence": args.prevalence,
            "window": [args.window_min, args.window_max],
            "fpr_quantile": args.fpr_quantile,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "detectors": detector_order,
            "outputs": {
                "detection": str(args.out),
                "summary": str(args.summary_out),
                "paired": str(args.paired_out),
                "figure": str(args.figure),
            },
            "checksums": {
                "detection": sha256(args.out),
                "summary": sha256(args.summary_out),
                "paired": sha256(args.paired_out),
                "figure": sha256(args.figure),
            },
            "rows": int(len(frame)),
            "summary_rows": int(len(summary)),
            "paired_rows": int(len(paired)),
        },
    )

    print()
    print(f"Wrote {args.out} ({len(frame)} rows)")
    print(f"Wrote {args.summary_out} ({len(summary)} rows)")
    print(f"Wrote {args.paired_out} ({len(paired)} rows)")
    print(f"Wrote {args.figure}")
    print(f"Wrote {args.log}")
    print()
    print("Mean ROC-AUC by detector and fault (over seeds and magnitudes)")
    print(auc_pivot.round(3).to_string())
    print()
    print("Mean recall@k by detector and fault (over seeds and magnitudes)")
    print(recall_pivot.round(3).to_string())


if __name__ == "__main__":
    main()

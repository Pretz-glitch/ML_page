#!/usr/bin/env python3
"""Phase 4: attribution faithfulness on injected faults.

For each seed and fault type:
  1. fit every detector on clean training data;
  2. inject one fault type into the held-out test period;
  3. explain every injected row with occlusion and, for Mahalanobis, the exact
     additive decomposition;
  4. measure top-1 recovery, top-3, MRR, mean rank and deletion advantage.

A separate stability pass fixes the injected rows and varies the detector seed.

Outputs
-------
results/tables/faithfulness.csv           aggregated metrics with CIs
results/tables/attribution_rows.csv       one row per seed x fault x method x row
results/tables/faithfulness_paired.csv    paired bootstrap comparisons
results/tables/faithfulness_stability.csv top-1 agreement across detector seeds
results/figures/faithfulness.png          top-1 and MRR heatmaps
results/logs/04_run_explanations.json     run log
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
from src.explain import (  # noqa: E402
    attribution_metrics,
    conditional_occlusion_contributions,
    deletion_metrics,
    mahalanobis_contributions,
    majority_agreement,
    occlusion_contributions,
)
from src.faults import FaultInjector, build_injection_profile  # noqa: E402
from src.io import TRAIN_DAYS, load_dataset  # noqa: E402
from src.metrics import bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from src.plots import save_faithfulness_heatmap  # noqa: E402
from src.preprocess import fit_scaler, transform  # noqa: E402
from src.utils import git_commit, package_versions, sha256, write_json  # noqa: E402

METHOD_SPECS: dict[str, dict[str, str]] = {
    "if_simple": {
        "detector": "iforest",
        "attribution": "simple",
        "role": "primary",
        "display": "IF + simple occlusion",
    },
    "if_conditional": {
        "detector": "iforest",
        "attribution": "conditional",
        "role": "primary",
        "display": "IF + conditional occlusion",
    },
    "mahalanobis_simple": {
        "detector": "mahalanobis",
        "attribution": "simple",
        "role": "primary",
        "display": "Mahalanobis + simple occlusion",
    },
    "mahalanobis_conditional": {
        "detector": "mahalanobis",
        "attribution": "conditional",
        "role": "primary",
        "display": "Mahalanobis + conditional occlusion",
    },
    "mahalanobis_exact": {
        "detector": "mahalanobis",
        "attribution": "exact",
        "role": "primary",
        "display": "Mahalanobis exact",
    },
    "mad_simple": {
        "detector": "mad",
        "attribution": "simple",
        "role": "secondary",
        "display": "MAD + simple occlusion*",
    },
    "mad_conditional": {
        "detector": "mad",
        "attribution": "conditional",
        "role": "secondary",
        "display": "MAD + conditional occlusion*",
    },
    "pca_simple": {
        "detector": "pca",
        "attribution": "simple",
        "role": "secondary",
        "display": "PCA + simple occlusion*",
    },
    "pca_conditional": {
        "detector": "pca",
        "attribution": "conditional",
        "role": "secondary",
        "display": "PCA + conditional occlusion*",
    },
    "eif_simple": {
        "detector": "eif",
        "attribution": "simple",
        "role": "secondary",
        "display": "EIF + simple occlusion*",
    },
    "eif_conditional": {
        "detector": "eif",
        "attribution": "conditional",
        "role": "secondary",
        "display": "EIF + conditional occlusion*",
    },
}

METHOD_ORDER: tuple[str, ...] = tuple(METHOD_SPECS)
FAULT_ORDER: tuple[str, ...] = ("spike", "drift", "correlation_break")
FAULT_NAMES: dict[str, str] = {
    "spike": "Spike",
    "drift": "Drift",
    "correlation_break": "Correlation break",
}
METHOD_NAMES: dict[str, str] = {
    key: value["display"] for key, value in METHOD_SPECS.items()
}

METRIC_SOURCES: dict[str, str] = {
    "top1": "top1",
    "top3": "top3",
    "mrr": "mrr",
    "true_rank": "true_rank",
    "deletion_advantage_relative": "deletion_advantage_relative",
    "top1_detected": "top1",
    "mrr_detected": "mrr",
}

PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("mahalanobis_exact", "if_simple"),
    ("mahalanobis_conditional", "if_conditional"),
    ("mahalanobis_exact", "mahalanobis_conditional"),
    ("mahalanobis_conditional", "mahalanobis_simple"),
    ("if_conditional", "if_simple"),
    ("mahalanobis_exact", "if_conditional"),
)

SECONDARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("eif_conditional", "if_conditional"),
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
        "--faults",
        nargs="+",
        default=list(FAULT_ORDER),
        choices=list(FAULT_ORDER),
    )
    parser.add_argument("--magnitude", type=float, default=6.0)
    parser.add_argument("--prevalence", type=float, default=0.02)
    parser.add_argument("--window-min", type=int, default=5)
    parser.add_argument("--window-max", type=int, default=20)
    parser.add_argument("--fpr-quantile", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--stability-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "tables" / "faithfulness.csv",
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=ROOT / "results" / "tables" / "attribution_rows.csv",
    )
    parser.add_argument(
        "--paired-out",
        type=Path,
        default=ROOT / "results" / "tables" / "faithfulness_paired.csv",
    )
    parser.add_argument(
        "--stability-out",
        type=Path,
        default=ROOT / "results" / "tables" / "faithfulness_stability.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results" / "figures" / "faithfulness.png",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results" / "logs" / "04_run_explanations.json",
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


def explain_rows(
    detectors: dict[str, object],
    X: np.ndarray,
    reference_mean: np.ndarray,
    reference_precision: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Return base scores and attributions for every method on ``X``."""
    base_scores: dict[str, np.ndarray] = {}
    simple: dict[str, np.ndarray] = {}
    conditional: dict[str, np.ndarray] = {}
    exact: dict[str, np.ndarray] = {}

    for name, detector in detectors.items():
        base, simple_contributions = occlusion_contributions(detector, X)
        _, conditional_contributions = conditional_occlusion_contributions(
            detector, X, reference_mean, reference_precision
        )
        base_scores[name] = base
        simple[name] = simple_contributions
        conditional[name] = conditional_contributions
        if name == "mahalanobis":
            exact[name] = mahalanobis_contributions(detector, X)

    attributions: dict[str, dict[str, np.ndarray]] = {}
    for method, spec in METHOD_SPECS.items():
        detector_name = spec["detector"]
        kind = spec["attribution"]
        if kind == "simple":
            attribution = simple[detector_name]
        elif kind == "conditional":
            attribution = conditional[detector_name]
        else:
            attribution = exact[detector_name]

        attributions[method] = {
            "base": base_scores[detector_name],
            # The deletion test uses the conditional intervention for every
            # method, so the comparison is not confounded by implausible rows.
            "occlusion": conditional[detector_name],
            "attribution": attribution,
        }
    return attributions


def prepare_injected_rows(
    result,
    n_stations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return local test indices, true feature indices and dropped-row count."""
    labels = result.labels.copy()
    labels["local_index"] = labels["row_id"] - TRAIN_DAYS * n_stations

    counts = labels.groupby("local_index")["feature_index"].nunique()
    multi = counts[counts > 1].index
    excluded = int(len(multi))
    if excluded:
        labels = labels.loc[~labels["local_index"].isin(multi)]

    labels = labels.drop_duplicates("local_index").sort_values("local_index")
    local_indices = labels["local_index"].to_numpy(dtype=int)
    true_idx = labels["feature_index"].to_numpy(dtype=int)
    return local_indices, true_idx, excluded


def summarize(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Bootstrap summary that tolerates an all-NaN conditional metric."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    return bootstrap_ci(array, n_boot=n_boot, seed=seed)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required")
    if not args.faults:
        raise ValueError("At least one fault type is required")

    for path in (
        args.out,
        args.rows_out,
        args.paired_out,
        args.stability_out,
        args.figure,
        args.log,
    ):
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
    test_dates = dataset.dates[TRAIN_DAYS:]
    detector_names = sorted({spec["detector"] for spec in METHOD_SPECS.values()})

    rows: list[dict[str, object]] = []
    multilabel_excluded = 0

    for seed in args.seeds:
        detectors = {
            name: build_detector(name, seed).fit(train_scaled)
            for name in detector_names
        }
        thresholds = {
            name: float(np.quantile(detector.score(train_scaled), args.fpr_quantile))
            for name, detector in detectors.items()
        }

        for fault in args.faults:
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
                magnitude=args.magnitude,
                window_range=(args.window_min, args.window_max),
                day_offset=TRAIN_DAYS,
            )
            local_indices, true_idx, excluded = prepare_injected_rows(
                result, dataset.n_stations
            )
            multilabel_excluded += excluded
            if local_indices.size == 0:
                continue

            corrupted_scaled = transform(result.corrupted, scaler).reshape(
                -1, dataset.n_features
            )
            X_rows = corrupted_scaled[local_indices]
            reference = detectors["mahalanobis"]
            attributions = explain_rows(
                detectors,
                X_rows,
                reference.location_,
                reference.precision_,
            )

            feature_names = np.asarray(dataset.feature_names)
            station_indices = local_indices % dataset.n_stations
            day_indices = TRAIN_DAYS + local_indices // dataset.n_stations
            global_ids = local_indices + TRAIN_DAYS * dataset.n_stations

            for method, payload in attributions.items():
                metrics = attribution_metrics(payload["attribution"], true_idx)
                deletion = deletion_metrics(
                    payload["occlusion"],
                    metrics["top_idx"],
                    true_idx,
                    base_scores=payload["base"],
                )
                detected = payload["base"] >= thresholds[METHOD_SPECS[method]["detector"]]

                for row in range(local_indices.size):
                    rows.append(
                        {
                            "seed": seed,
                            "fault": fault,
                            "method": method,
                            "role": METHOD_SPECS[method]["role"],
                            "row_id": int(global_ids[row]),
                            "day_index": int(day_indices[row]),
                            "station_id": int(dataset.station_ids[station_indices[row]]),
                            "true_feature": str(feature_names[true_idx[row]]),
                            "top_feature": str(
                                feature_names[metrics["top_idx"][row]]
                            ),
                            "true_rank": int(metrics["true_rank"][row]),
                            "top1": float(metrics["top1"][row]),
                            "top3": float(metrics["top3"][row]),
                            "mrr": float(metrics["mrr"][row]),
                            "score_base": float(payload["base"][row]),
                            "threshold": float(
                                thresholds[METHOD_SPECS[method]["detector"]]
                            ),
                            "detected": bool(detected[row]),
                            "drop_top": float(deletion["drop_top"][row]),
                            "drop_true": float(deletion["drop_true"][row]),
                            "drop_random": float(deletion["drop_random"][row]),
                            "deletion_advantage": float(
                                deletion["deletion_advantage"][row]
                            ),
                            "deletion_advantage_relative": float(
                                deletion["deletion_advantage_relative"][row]
                            ),
                            "drop_true_vs_random": float(
                                deletion["drop_true_vs_random"][row]
                            ),
                        }
                    )

            print(
                f"seed {seed} {fault:18s} rows={local_indices.size:4d} "
                f"excluded_multilabel={excluded}"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.rows_out, index=False)

    summary_rows: list[dict[str, object]] = []
    for (method, fault), group in frame.groupby(["method", "fault"], sort=False):
        row: dict[str, object] = {
            "method": method,
            "role": METHOD_SPECS[method]["role"],
            "fault": fault,
            "n_seeds": int(group["seed"].nunique()),
            "n_seeds_detected": int(group.loc[group["detected"], "seed"].nunique()),
            "n_rows": int(len(group)),
            "n_detected": int(group["detected"].sum()),
            "detection_rate": float(group["detected"].mean()),
        }

        for metric, source in METRIC_SOURCES.items():
            per_seed: list[float] = []
            for _, seed_group in group.groupby("seed"):
                if metric in {"top1_detected", "mrr_detected"}:
                    detected_group = seed_group.loc[seed_group["detected"]]
                    per_seed.append(
                        float(detected_group[source].mean())
                        if len(detected_group)
                        else np.nan
                    )
                else:
                    per_seed.append(float(seed_group[source].mean()))

            stats = summarize(np.asarray(per_seed), args.bootstrap, args.bootstrap_seed)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["sd"]
            row[f"{metric}_ci_lo"] = stats["ci_lo"]
            row[f"{metric}_ci_hi"] = stats["ci_hi"]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    fault_rank = {fault: index for index, fault in enumerate(FAULT_ORDER)}
    summary["_method"] = summary["method"].map(method_rank)
    summary["_fault"] = summary["fault"].map(fault_rank)
    summary = summary.sort_values(["_fault", "_method"]).drop(
        columns=["_method", "_fault"]
    )
    summary.to_csv(args.out, index=False)

    paired_rows: list[dict[str, object]] = []
    comparisons = [
        (left, right, "primary") for left, right in PRIMARY_COMPARISONS
    ] + [(left, right, "secondary") for left, right in SECONDARY_COMPARISONS]

    for fault in args.faults:
        fault_rows = frame.loc[frame["fault"] == fault]
        for left, right, kind in comparisons:
            left_rows = fault_rows.loc[
                fault_rows["method"] == left, ["seed", "row_id", "top1", "mrr", "deletion_advantage_relative"]
            ]
            right_rows = fault_rows.loc[
                fault_rows["method"] == right, ["seed", "row_id", "top1", "mrr", "deletion_advantage_relative"]
            ]
            merged = left_rows.merge(
                right_rows, on=["seed", "row_id"], suffixes=("_left", "_right")
            )
            if merged.empty:
                continue
            for metric in ("top1", "mrr", "deletion_advantage_relative"):
                stats = paired_bootstrap_diff(
                    merged[f"{metric}_left"].to_numpy(),
                    merged[f"{metric}_right"].to_numpy(),
                    n_boot=args.bootstrap,
                    seed=args.bootstrap_seed,
                )
                paired_rows.append(
                    {
                        "comparison": f"{left}_vs_{right}",
                        "kind": kind,
                        "fault": fault,
                        "metric": metric,
                        **stats,
                    }
                )

    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["_fault"] = paired["fault"].map(fault_rank)
        paired["_kind"] = paired["kind"].map({"primary": 0, "secondary": 1})
        paired = paired.sort_values(
            ["_fault", "_kind", "comparison", "metric"]
        ).drop(columns=["_fault", "_kind"])
    paired.to_csv(args.paired_out, index=False)

    stability_rows: list[dict[str, object]] = []
    stability_seed = args.seeds[0]
    for fault in args.faults:
        injector = FaultInjector(
            profile=profile,
            station_ids=dataset.station_ids,
            dates=test_dates,
            seed=stability_seed,
        )
        result = injector.inject(
            clean_raw,
            fault,
            prevalence=args.prevalence,
            magnitude=args.magnitude,
            window_range=(args.window_min, args.window_max),
            day_offset=TRAIN_DAYS,
        )
        local_indices, true_idx, _ = prepare_injected_rows(
            result, dataset.n_stations
        )
        if local_indices.size == 0:
            continue
        corrupted_scaled = transform(result.corrupted, scaler).reshape(
            -1, dataset.n_features
        )
        X_rows = corrupted_scaled[local_indices]

        tops: dict[str, list[np.ndarray]] = {method: [] for method in METHOD_ORDER}
        for detector_seed in args.stability_seeds:
            detectors = {
                name: build_detector(name, detector_seed).fit(train_scaled)
                for name in detector_names
            }
            attributions = explain_rows(
                detectors,
                X_rows,
                detectors["mahalanobis"].location_,
                detectors["mahalanobis"].precision_,
            )
            for method, payload in attributions.items():
                metrics = attribution_metrics(payload["attribution"], true_idx)
                tops[method].append(metrics["top_idx"])

        for method in METHOD_ORDER:
            matrix = np.vstack(tops[method])
            agreement = majority_agreement(matrix)
            stats = bootstrap_ci(
                agreement, n_boot=args.bootstrap, seed=args.bootstrap_seed
            )
            stability_rows.append(
                {
                    "method": method,
                    "role": METHOD_SPECS[method]["role"],
                    "fault": fault,
                    "n_rows": int(local_indices.size),
                    "n_detector_seeds": int(matrix.shape[0]),
                    "top1_agreement_mean": stats["mean"],
                    "top1_agreement_sd": stats["sd"],
                    "top1_agreement_ci_lo": stats["ci_lo"],
                    "top1_agreement_ci_hi": stats["ci_hi"],
                }
            )

    stability = pd.DataFrame(stability_rows)
    if not stability.empty:
        stability["_method"] = stability["method"].map(method_rank)
        stability["_fault"] = stability["fault"].map(fault_rank)
        stability = stability.sort_values(["_fault", "_method"]).drop(
            columns=["_method", "_fault"]
        )
    stability.to_csv(args.stability_out, index=False)

    top1_pivot = (
        summary.pivot_table(
            index="method", columns="fault", values="top1_mean", aggfunc="mean"
        )
        .reindex(index=list(METHOD_ORDER), columns=list(FAULT_ORDER))
    )
    mrr_pivot = (
        summary.pivot_table(
            index="method", columns="fault", values="mrr_mean", aggfunc="mean"
        )
        .reindex(index=list(METHOD_ORDER), columns=list(FAULT_ORDER))
    )
    figure_methods = [
        method
        for method in METHOD_ORDER
        if METHOD_SPECS[method]["role"] == "primary"
    ]
    save_faithfulness_heatmap(
        top1_pivot.reindex(index=figure_methods),
        mrr_pivot.reindex(index=figure_methods),
        args.figure,
        method_names=METHOD_NAMES,
        fault_names=FAULT_NAMES,
    )

    write_json(
        args.log,
        {
            "phase": 4,
            "step": "explanation_faithfulness",
            "git_commit": git_commit(),
            "versions": package_versions(),
            "seeds": args.seeds,
            "faults": args.faults,
            "magnitude": args.magnitude,
            "prevalence": args.prevalence,
            "window": [args.window_min, args.window_max],
            "fpr_quantile": args.fpr_quantile,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "stability_seeds": args.stability_seeds,
            "multilabel_excluded": int(multilabel_excluded),
            "rows": int(len(frame)),
            "summary_rows": int(len(summary)),
            "paired_rows": int(len(paired)),
            "stability_rows": int(len(stability)),
            "outputs": {
                "summary": str(args.out),
                "rows": str(args.rows_out),
                "paired": str(args.paired_out),
                "stability": str(args.stability_out),
                "figure": str(args.figure),
            },
            "checksums": {
                "summary": sha256(args.out),
                "rows": sha256(args.rows_out),
                "paired": sha256(args.paired_out),
                "stability": sha256(args.stability_out),
                "figure": sha256(args.figure),
            },
        },
    )

    print()
    print(f"Wrote {args.rows_out} ({len(frame)} rows)")
    print(f"Wrote {args.out} ({len(summary)} rows)")
    print(f"Wrote {args.paired_out} ({len(paired)} rows)")
    print(f"Wrote {args.stability_out} ({len(stability)} rows)")
    print(f"Wrote {args.figure}")
    print(f"Wrote {args.log}")
    print()
    print("Mean top-1 recovery")
    print(top1_pivot.round(3).to_string())
    print()
    print("Mean reciprocal rank")
    print(mrr_pivot.round(3).to_string())


if __name__ == "__main__":
    main()

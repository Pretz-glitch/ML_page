#!/usr/bin/env python3
"""Phase 2 end-to-end smoke test.

Fits every detector on clean training data, injects each fault type into the
held-out raw test period, transforms the corrupted data with the train-fitted
scaler, and reports detection metrics.

This is an integration check, not the final Phase 3 benchmark (which uses
multiple seeds, magnitudes and confidence intervals).

Outputs
-------
results/tables/phase2_smoke.csv
results/logs/phase2_smoke.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors import make_detector  # noqa: E402
from src.faults import FaultInjector, build_injection_profile  # noqa: E402
from src.io import TRAIN_DAYS, load_dataset  # noqa: E402
from src.preprocess import fit_scaler, transform  # noqa: E402
from src.utils import git_commit, package_versions, write_json  # noqa: E402

FAULT_ORDER = ("spike", "drift", "correlation_break", "stuck")
DETECTOR_ORDER = ("mad", "mahalanobis", "pca", "iforest", "eif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "water_dataset.mat",
        help="Path to water_dataset.mat",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "tables" / "phase2_smoke.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results" / "logs" / "phase2_smoke.json",
        help="Output JSON log",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prevalence", type=float, default=0.02)
    parser.add_argument("--magnitude-min", type=float, default=4.0)
    parser.add_argument("--magnitude-max", type=float, default=8.0)
    parser.add_argument("--window-min", type=int, default=5)
    parser.add_argument("--window-max", type=int, default=20)
    return parser.parse_args()


def build_detector(name: str, seed: int):
    if name == "eif":
        return make_detector(name, random_seed=seed)
    if name in {"pca", "iforest"}:
        return make_detector(name, random_state=seed)
    return make_detector(name)


def check_integrity(clean: np.ndarray, result) -> None:
    """Assert the invariants the Phase 2 definition of done requires."""
    assert np.array_equal(
        result.corrupted[~result.cell_mask], clean[~result.cell_mask]
    ), "clean cells were modified"
    assert np.all(
        ~np.isclose(
            result.corrupted[result.cell_mask],
            clean[result.cell_mask],
            atol=1e-12,
            rtol=0.0,
        )
    ), "a marked cell did not change"
    assert result.corrupted.min() >= 0.0 and result.corrupted.max() <= 1.0
    assert int(result.cell_mask.sum()) == len(result.labels)
    assert not result.labels.duplicated(
        ["day_index", "station_id", "feature"]
    ).any()
    assert (
        int(result.station_day_mask.sum())
        == len(result.labels[["day_index", "station_id"]].drop_duplicates())
    )
    expected_row_id = (
        result.labels["day_index"] * result.station_day_mask.shape[1]
        + result.labels["station_index"]
    )
    assert np.array_equal(
        result.labels["row_id"].to_numpy(), expected_row_id.to_numpy()
    )
    assert result.actual_prevalence > 0.0
    assert result.n_events > 0


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

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

    detectors = {
        name: build_detector(name, args.seed).fit(train_scaled)
        for name in DETECTOR_ORDER
    }

    rows: list[dict[str, object]] = []
    for fault in FAULT_ORDER:
        injector = FaultInjector(
            profile=profile,
            station_ids=dataset.station_ids,
            dates=dataset.dates[TRAIN_DAYS:],
            seed=args.seed,
        )
        result = injector.inject(
            clean_raw,
            fault,
            prevalence=args.prevalence,
            magnitude=(args.magnitude_min, args.magnitude_max),
            window_range=(args.window_min, args.window_max),
            day_offset=TRAIN_DAYS,
        )
        check_integrity(clean_raw, result)

        corrupted_scaled = transform(result.corrupted, scaler).reshape(
            -1, dataset.n_features
        )
        labels = result.station_day_mask.ravel()
        positive = int(labels.sum())
        print(
            f"{fault:18s} events={result.n_events:3d} "
            f"labels={len(result.labels):4d} "
            f"prevalence={result.actual_prevalence:.4f}"
        )

        for name, detector in detectors.items():
            scores = detector.score(corrupted_scaled)
            top_k = np.argsort(-scores)[:positive]
            rows.append(
                {
                    "fault": fault,
                    "detector": name,
                    "events": result.n_events,
                    "prevalence": result.actual_prevalence,
                    "auc_roc": roc_auc_score(labels, scores),
                    "auc_pr": average_precision_score(labels, scores),
                    "recall_at_k": float(labels[top_k].sum() / positive),
                    "precision_at_k": float(labels[top_k].sum() / positive),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)

    auc = frame.pivot(index="detector", columns="fault", values="auc_roc").round(3)
    recall = frame.pivot(index="detector", columns="fault", values="recall_at_k").round(3)
    print("\nROC-AUC")
    print(auc.loc[list(DETECTOR_ORDER), list(FAULT_ORDER)].to_string())
    print("\nRecall at k")
    print(recall.loc[list(DETECTOR_ORDER), list(FAULT_ORDER)].to_string())

    write_json(
        args.log,
        {
            "phase": 2,
            "step": "smoke",
            "git_commit": git_commit(),
            "versions": package_versions(),
            "seed": args.seed,
            "prevalence": args.prevalence,
            "magnitude": [args.magnitude_min, args.magnitude_max],
            "window": [args.window_min, args.window_max],
            "output": str(args.out),
            "rows": int(len(frame)),
        },
    )

    print(f"\nWrote {args.out}")
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()

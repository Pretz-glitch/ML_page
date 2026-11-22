#!/usr/bin/env python3
"""Phase 5: descriptive analysis of detector flags in the unlabelled test period.

No ground-truth labels exist here. The script applies the detectors to the
clean test period at a fixed threshold, summarises agreement and concentration,
and selects three case studies. It does not claim that a flag is pollution or a
confirmed sensor failure.

Outputs
-------
results/tables/real_flags.csv
results/tables/real_flag_summary.csv
results/tables/real_overlap.csv
results/tables/real_station_summary.csv
results/tables/real_date_summary.csv
results/tables/real_case_studies.csv
results/figures/case_studies.png
results/phase5_summary.md
results/logs/05_real_record.json
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
from src.faults import build_injection_profile  # noqa: E402
from src.io import TRAIN_DAYS, TEST_DAYS, load_dataset  # noqa: E402
from src.plots import save_case_studies  # noqa: E402
from src.preprocess import fit_scaler, transform  # noqa: E402
from src.realdata import (  # noqa: E402
    concentration_summary,
    flag_summary,
    max_score_ratio,
    pairwise_overlap,
    select_case_studies,
)
from src.utils import git_commit, package_versions, sha256, write_json  # noqa: E402

DETECTORS: tuple[str, ...] = ("mad", "mahalanobis", "iforest", "pca", "eif")
PRIMARY: tuple[str, ...] = ("mad", "mahalanobis", "iforest")
ROLES: dict[str, str] = {
    "mad": "primary",
    "mahalanobis": "primary",
    "iforest": "primary",
    "pca": "secondary",
    "eif": "secondary",
}
DISPLAY: dict[str, str] = {
    "mad": "MAD",
    "mahalanobis": "Mahalanobis",
    "iforest": "Isolation Forest",
    "pca": "PCA reconstruction",
    "eif": "Extended IF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "water_dataset.mat",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fpr-quantile", type=float, default=0.99)
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument(
        "--flags-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_flags.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_flag_summary.csv",
    )
    parser.add_argument(
        "--overlap-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_overlap.csv",
    )
    parser.add_argument(
        "--station-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_station_summary.csv",
    )
    parser.add_argument(
        "--date-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_date_summary.csv",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        default=ROOT / "results" / "tables" / "real_case_studies.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results" / "figures" / "case_studies.png",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "phase5_summary.md",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "results" / "logs" / "05_real_record.json",
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


def md_table(headers: list[str], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for path in (
        args.flags_out,
        args.summary_out,
        args.overlap_out,
        args.station_out,
        args.date_out,
        args.cases_out,
        args.figure,
        args.report,
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
    eligible_features = tuple(
        dataset.feature_names[index] for index in profile.eligible_features
    )

    train_scaled = transform(dataset.tensor[:TRAIN_DAYS], scaler).reshape(
        -1, dataset.n_features
    )
    test_raw = dataset.tensor[TRAIN_DAYS:]
    test_scaled = transform(test_raw, scaler).reshape(-1, dataset.n_features)

    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for detector_name in DETECTORS:
        test_seeds: list[np.ndarray] = []
        threshold_seeds: list[float] = []
        for seed in args.seeds:
            detector = build_detector(detector_name, seed).fit(train_scaled)
            train_scores = detector.score(train_scaled)
            test_seeds.append(detector.score(test_scaled))
            threshold_seeds.append(
                float(np.quantile(train_scores, args.fpr_quantile))
            )
        scores[detector_name] = np.mean(np.vstack(test_seeds), axis=0)
        thresholds[detector_name] = float(np.mean(threshold_seeds))
        print(
            f"{detector_name:12s} threshold={thresholds[detector_name]:.4f} "
            f"flags={(scores[detector_name] >= thresholds[detector_name]).sum()}"
        )

    n_stations = dataset.n_stations
    frame = pd.DataFrame(
        {
            "day_index": np.repeat(
                np.arange(TRAIN_DAYS, TRAIN_DAYS + TEST_DAYS), n_stations
            ),
            "date": np.repeat(dataset.dates[TRAIN_DAYS:], n_stations),
            "station_id": np.tile(dataset.station_ids, TEST_DAYS),
            "station_group": np.tile(dataset.station_groups, TEST_DAYS),
        }
    )

    raw_values = test_raw.reshape(-1, dataset.n_features)
    z_values = test_scaled
    for index, feature in enumerate(dataset.feature_names):
        frame[feature] = raw_values[:, index]
        frame[f"z_{feature}"] = z_values[:, index]

    for detector_name in DETECTORS:
        frame[f"score_{detector_name}"] = scores[detector_name]
        frame[f"threshold_{detector_name}"] = thresholds[detector_name]
        frame[f"flag_{detector_name}"] = (
            scores[detector_name] >= thresholds[detector_name]
        )

    flag_columns = [f"flag_{detector_name}" for detector_name in DETECTORS]
    primary_columns = [f"flag_{detector_name}" for detector_name in PRIMARY]

    frame["n_flags"] = frame[flag_columns].sum(axis=1).astype(int)
    frame["n_primary_flags"] = frame[primary_columns].sum(axis=1).astype(int)
    frame["consensus_primary"] = frame["n_primary_flags"] == len(PRIMARY)
    frame["consensus_any"] = frame["n_flags"] >= 2
    frame["max_ratio"] = max_score_ratio(frame, DETECTORS)

    eligible_z_columns = [f"z_{feature}" for feature in eligible_features]
    z_matrix = frame[eligible_z_columns].to_numpy(dtype=float)
    top_index = np.argmax(np.abs(z_matrix), axis=1)
    frame["top_feature"] = [eligible_features[index] for index in top_index]
    frame["top_abs_z"] = np.max(np.abs(z_matrix), axis=1)
    frame["profile_severity"] = np.mean(np.abs(z_matrix), axis=1)

    frame.to_csv(args.flags_out, index=False)

    summary = flag_summary(frame, DETECTORS, ROLES)
    summary.to_csv(args.summary_out, index=False)

    flags = frame[flag_columns].copy()
    flags.columns = list(DETECTORS)
    overlap = pairwise_overlap(flags, ROLES)
    overlap.to_csv(args.overlap_out, index=False)

    station_summary = concentration_summary(frame, DETECTORS, "station_id")
    station_group_map = (
        frame.drop_duplicates("station_id")
        .set_index("station_id")["station_group"]
    )
    station_summary.insert(
        1,
        "station_group",
        station_summary["station_id"].map(station_group_map),
    )
    station_summary = station_summary.sort_values("n_any", ascending=False)
    station_summary.to_csv(args.station_out, index=False)

    date_summary = concentration_summary(frame, DETECTORS, "date")
    date_summary = date_summary.sort_values("date")
    date_summary.to_csv(args.date_out, index=False)

    cases = select_case_studies(
        frame, n_cases=args.n_cases, severity_column="profile_severity"
    )
    case_rows: list[dict[str, object]] = []
    for _, case in cases.iterrows():
        z_row = case[eligible_z_columns].to_numpy(dtype=float)
        order = np.argsort(-np.abs(z_row))[:3]
        payload: dict[str, object] = {
            "case_id": int(case["case_id"]),
            "date": case["date"],
            "station_id": int(case["station_id"]),
            "station_group": int(case["station_group"]),
            "n_flags": int(case["n_flags"]),
            "n_primary_flags": int(case["n_primary_flags"]),
            "max_ratio": float(case["max_ratio"]),
            "top_feature": case["top_feature"],
            "top_abs_z": float(case["top_abs_z"]),
        }
        for rank, feature_index in enumerate(order, start=1):
            payload[f"top_feature_{rank}"] = eligible_features[feature_index]
            payload[f"z_{rank}"] = float(z_row[feature_index])
        for detector_name in DETECTORS:
            payload[f"score_{detector_name}"] = float(case[f"score_{detector_name}"])
            payload[f"flag_{detector_name}"] = bool(case[f"flag_{detector_name}"])
        case_rows.append(payload)

    case_table = pd.DataFrame(case_rows)
    case_table.to_csv(args.cases_out, index=False)

    save_case_studies(
        frame,
        cases,
        eligible_features,
        args.figure,
        DETECTORS,
    )

    n_any = int((frame["n_flags"] >= 1).sum())
    n_consensus_primary = int(frame["consensus_primary"].sum())
    n_consensus_any = int(frame["consensus_any"].sum())
    top_station = station_summary.iloc[0]
    top_date = date_summary.sort_values("n_any", ascending=False).iloc[0]
    jaccard_min = float(overlap["jaccard"].min())
    jaccard_max = float(overlap["jaccard"].max())
    flag_rate_min = float(summary["flag_rate"].min())
    flag_rate_max = float(summary["flag_rate"].max())

    report: list[str] = []
    report.append("# Phase 5 — Real-record flag analysis")
    report.append("")
    report.append("**Status:** descriptive analysis of the unlabelled test period. No ground-truth labels.")
    report.append("")
    report.append("## Setup")
    report.append("")
    report.append(
        md_table(
            ["Item", "Value"],
            [
                ("Test period", "2017-03-26 to 2018-01-01"),
                ("Station-days", f"{len(frame):,}"),
                ("Detectors", "MAD, Mahalanobis, Isolation Forest (primary); PCA, Extended IF (secondary)"),
                ("Threshold", f"{args.fpr_quantile:.0%} of clean training scores, averaged over {len(args.seeds)} seeds"),
                ("Flag", "score at or above that threshold"),
                ("Case features", "8 eligible features; near-constant SC_max, SC_min and pH_min are excluded"),
                ("Case studies", args.n_cases),
            ],
        )
    )
    report.append("")
    report.append("## Flag counts")
    report.append("")
    report.append(
        md_table(
            ["Detector", "Role", "Flags", "Rate", "Stations", "Dates", "Median flagged score"],
            [
                (
                    DISPLAY[row.detector],
                    row.role,
                    row.n_flags,
                    f"{row.flag_rate:.2%}",
                    row.n_unique_stations,
                    row.n_unique_dates,
                    f"{row.median_score_flagged:.4f}",
                )
                for row in summary.itertuples(index=False)
            ],
        )
    )
    report.append("")
    report.append(
        f"Across detectors, flag rates range from **{flag_rate_min:.2%}** to "
        f"**{flag_rate_max:.2%}**. A total of **{n_any:,}** station-days "
        f"({n_any / len(frame):.2%}) are flagged by at least one detector."
    )
    report.append("")
    report.append("## Pairwise overlap")
    report.append("")
    report.append(
        md_table(
            ["Detector A", "Detector B", "Type", "Intersection", "Union", "Jaccard", "Agreement"],
            [
                (
                    DISPLAY[row.detector_a],
                    DISPLAY[row.detector_b],
                    row.pair_type,
                    row.intersection,
                    row.union,
                    f"{row.jaccard:.3f}",
                    f"{row.agreement:.3f}",
                )
                for row in overlap.sort_values("jaccard", ascending=False).itertuples(index=False)
            ],
        )
    )
    report.append("")
    report.append(
        f"Pairwise Jaccard ranges from **{jaccard_min:.3f}** to "
        f"**{jaccard_max:.3f}**. Only **{n_consensus_primary:,}** station-days "
        f"are flagged by all three primary detectors; **{n_consensus_any:,}** "
        "are flagged by at least two detectors."
    )
    report.append("")
    report.append("## Concentration by station")
    report.append("")
    report.append(
        md_table(
            ["Station", "Group", "Rows", "Any flag", "Rate", "MAD", "Mahalanobis", "IF", "PCA", "EIF"],
            [
                (
                    int(row.station_id),
                    int(row.station_group),
                    row.n_rows,
                    row.n_any,
                    f"{row.rate_any:.2%}",
                    row.n_mad,
                    row.n_mahalanobis,
                    row.n_iforest,
                    row.n_pca,
                    row.n_eif,
                )
                for row in station_summary.head(5).itertuples(index=False)
            ],
        )
    )
    report.append("")
    report.append(
        f"The most-flagged station ({int(top_station.station_id)}) accounts for "
        f"**{int(top_station.n_any):,}** flagged station-days, "
        f"**{top_station.n_any / max(n_any, 1):.1%}** of all flagged rows. "
        f"It is flagged {int(top_station.n_mad)} times by MAD, "
        f"{int(top_station.n_mahalanobis)} by Mahalanobis, "
        f"{int(top_station.n_iforest)} by Isolation Forest, "
        f"{int(top_station.n_pca)} by PCA and {int(top_station.n_eif)} by Extended IF. "
        "A high flag count without cross-detector consensus is a station-level "
        "disagreement, not a confirmed fault."
    )
    report.append("")
    report.append("## Concentration by date")
    report.append("")
    report.append(
        md_table(
            ["Date", "Rows", "Any flag", "Rate", "MAD", "Mahalanobis", "IF", "PCA", "EIF"],
            [
                (
                    pd.Timestamp(row.date).date(),
                    row.n_rows,
                    row.n_any,
                    f"{row.rate_any:.2%}",
                    row.n_mad,
                    row.n_mahalanobis,
                    row.n_iforest,
                    row.n_pca,
                    row.n_eif,
                )
                for row in date_summary.sort_values("n_any", ascending=False).head(5).itertuples(index=False)
            ],
        )
    )
    report.append("")
    report.append(
        f"The most-flagged date ({pd.Timestamp(top_date.date).date()}) has "
        f"**{int(top_date.n_any)}** flagged station-days. Isolation Forest "
        f"contributes **{int(top_date.n_iforest)}** of them. A single date with "
        "many flags can reflect a shared regime shift, a boundary effect, or a "
        "system-wide event rather than independent faults."
    )
    report.append("")
    report.append("## Case studies")
    report.append("")
    report.append(
        md_table(
            ["Case", "Date", "Station", "Flags", "Primary", "Top feature", "Top |z|"],
            [
                (
                    int(row.case_id),
                    pd.Timestamp(row.date).date(),
                    int(row.station_id),
                    f"{int(row.n_flags)}/5",
                    f"{int(row.n_primary_flags)}/3",
                    row.top_feature,
                    f"{row.top_abs_z:.2f}",
                )
                for row in case_table.itertuples(index=False)
            ],
        )
    )
    report.append("")
    report.append(
        "The figure shows, for each case, the station's history of the most "
        "extreme feature and the full within-station z-profile at the flagged "
        "station-day."
    )
    report.append("")
    last_case = case_table.iloc[-1]
    if float(last_case["top_abs_z"]) < 2.0:
        report.append(
            f"Case {int(last_case['case_id'])} is a **multivariate flag**: its "
            f"largest eligible-feature deviation is only "
            f"{float(last_case['top_abs_z']):.2f} IQR, yet all five detectors "
            "flag the row. This is the combination-driven pattern the project "
            "was designed to examine."
        )
        report.append("")
    report.append("## What this analysis can and cannot say")
    report.append("")
    report.append("- These are **candidate data-quality faults**. There is no ground truth, so a flag is not a confirmed sensor failure, pollution event, or unsafe-water observation.")
    report.append("- A flag can reflect a real environmental event, a sensor fault, a seasonal shift, or a model artefact. This analysis cannot distinguish them.")
    report.append("- The values are normalised, so no physical threshold or regulatory classification is applied.")
    report.append("- The threshold is a design choice. The 99th percentile of training scores gives flag rates above 1% on test because the test period drifts slightly from training.")
    report.append("- Cross-detector agreement is modest: Jaccard ranges from "
                  f"{jaccard_min:.3f} to {jaccard_max:.3f}. Consensus flags are a small, conservative subset.")
    report.append("")
    report.append("## Caveats")
    report.append("")
    report.append("- Unlabelled data: no precision, recall or F1 can be computed.")
    report.append("- The detectors are averaged over five seeds for stochastic models; deterministic detectors are repeated identically.")
    report.append("- Station confounding is reduced by within-station scaling but not eliminated.")
    report.append("- Near-constant features (`SC_max`, `SC_min`, `pH_min`) can still influence detector scores. Case studies and profile plots are restricted to the 8 features with meaningful within-station variation.")
    report.append("- The case studies are selected for consensus and eligible-feature severity, not because they are known faults.")
    report.append("")
    args.report.write_text("\n".join(report))

    write_json(
        args.log,
        {
            "phase": 5,
            "step": "real_record",
            "git_commit": git_commit(),
            "versions": package_versions(),
            "seeds": args.seeds,
            "fpr_quantile": args.fpr_quantile,
            "n_cases": args.n_cases,
            "rows": int(len(frame)),
            "n_any": n_any,
            "n_consensus_primary": n_consensus_primary,
            "n_consensus_any": n_consensus_any,
            "outputs": {
                "flags": str(args.flags_out),
                "summary": str(args.summary_out),
                "overlap": str(args.overlap_out),
                "stations": str(args.station_out),
                "dates": str(args.date_out),
                "cases": str(args.cases_out),
                "figure": str(args.figure),
                "report": str(args.report),
            },
            "checksums": {
                "flags": sha256(args.flags_out),
                "summary": sha256(args.summary_out),
                "overlap": sha256(args.overlap_out),
                "stations": sha256(args.station_out),
                "dates": sha256(args.date_out),
                "cases": sha256(args.cases_out),
                "figure": sha256(args.figure),
                "report": sha256(args.report),
            },
        },
    )

    print()
    print(f"Wrote {args.flags_out} ({len(frame)} rows)")
    print(f"Wrote {args.summary_out} ({len(summary)} rows)")
    print(f"Wrote {args.overlap_out} ({len(overlap)} rows)")
    print(f"Wrote {args.station_out} ({len(station_summary)} rows)")
    print(f"Wrote {args.date_out} ({len(date_summary)} rows)")
    print(f"Wrote {args.cases_out} ({len(case_table)} rows)")
    print(f"Wrote {args.figure}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()

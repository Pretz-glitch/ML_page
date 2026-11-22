#!/usr/bin/env python3
"""Phase 6: release consistency check.

Verifies that every expected result artefact exists, has the expected row
count, and contains no missing values. Writes a deterministic manifest of
checksums and row counts.

Output
------
results/release_manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import package_versions, sha256, write_json  # noqa: E402

EXPECTED_ROWS: dict[str, int | None] = {
    "results/tables/audit_variance.csv": 11,
    "results/tables/audit_correlations.csv": 11,
    "results/tables/audit_top_correlations.csv": 55,
    "results/tables/audit_floor_values.csv": 11,
    "results/tables/audit_feature_families.csv": 11,
    "results/tables/audit_station_summary.csv": 37,
    "results/tables/audit_split_summary.csv": 2,
    "results/tables/audit_group_summary.csv": 3,
    "results/tables/audit_summary.json": None,
    "results/tables/phase2_smoke.csv": 20,
    "results/tables/detection.csv": 200,
    "results/tables/detection_summary.csv": 40,
    "results/tables/detection_paired.csv": 80,
    "results/tables/faithfulness.csv": 33,
    "results/tables/attribution_rows.csv": 34_892,
    "results/tables/faithfulness_paired.csv": 63,
    "results/tables/faithfulness_stability.csv": 33,
    "results/tables/real_flags.csv": 10_434,
    "results/tables/real_flag_summary.csv": 5,
    "results/tables/real_overlap.csv": 10,
    "results/tables/real_station_summary.csv": 37,
    "results/tables/real_date_summary.csv": 282,
    "results/tables/real_case_studies.csv": 3,
    "results/phase1_summary.md": None,
    "results/phase2_smoke.md": None,
    "results/phase3_summary.md": None,
    "results/phase4_summary.md": None,
    "results/phase5_summary.md": None,
    "results/figures/audit_correlation.png": None,
    "results/figures/audit_variance.png": None,
    "results/figures/audit_coverage.png": None,
    "results/figures/audit_distributions.png": None,
    "results/figures/detection_heatmap.png": None,
    "results/figures/faithfulness.png": None,
    "results/figures/case_studies.png": None,
    "docs/report.md": None,
    "docs/slides.md": None,
}

NO_NAN_FILES = (
    "results/tables/phase2_smoke.csv",
    "results/tables/detection.csv",
    "results/tables/detection_summary.csv",
    "results/tables/detection_paired.csv",
    "results/tables/faithfulness.csv",
    "results/tables/attribution_rows.csv",
    "results/tables/faithfulness_paired.csv",
    "results/tables/faithfulness_stability.csv",
    "results/tables/real_flags.csv",
    "results/tables/real_flag_summary.csv",
    "results/tables/real_overlap.csv",
    "results/tables/real_station_summary.csv",
    "results/tables/real_date_summary.csv",
    "results/tables/real_case_studies.csv",
)


def main() -> None:
    checks: list[str] = []
    manifest: dict[str, object] = {
        "version": "v1.0",
        "generated_by": "scripts/06_release_check.py",
        "versions": package_versions(),
        "files": {},
    }

    for relative, expected_rows in EXPECTED_ROWS.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing release artefact: {relative}")

        entry: dict[str, object] = {"sha256": sha256(path)}
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            if expected_rows is not None and len(frame) != expected_rows:
                raise AssertionError(
                    f"{relative}: expected {expected_rows} rows, got {len(frame)}"
                )
            entry["rows"] = int(len(frame))
        manifest["files"][relative] = entry  # type: ignore[index]
        checks.append(f"exists: {relative}")

    for relative in NO_NAN_FILES:
        frame = pd.read_csv(ROOT / relative)
        if bool(frame.isna().any().any()):
            raise AssertionError(f"{relative} contains missing values")
        checks.append(f"no-missing: {relative}")

    detection = pd.read_csv(ROOT / "results/tables/detection.csv")
    detection_summary = pd.read_csv(ROOT / "results/tables/detection_summary.csv")
    faithfulness = pd.read_csv(ROOT / "results/tables/faithfulness.csv")
    real_flags = pd.read_csv(ROOT / "results/tables/real_flags.csv")
    overlap = pd.read_csv(ROOT / "results/tables/real_overlap.csv")
    cases = pd.read_csv(ROOT / "results/tables/real_case_studies.csv")

    if not detection["auc_roc"].between(0.0, 1.0).all():
        raise AssertionError("detection AUC outside [0, 1]")
    if not detection["fpr"].between(0.0, 1.0).all():
        raise AssertionError("detection FPR outside [0, 1]")
    checks.append("detection metrics in range")

    if not detection_summary["n_seeds"].eq(5).all():
        raise AssertionError("detection summary does not use five seeds")
    checks.append("detection summary uses five seeds")

    if not faithfulness["n_seeds_detected"].eq(5).all():
        raise AssertionError("faithfulness summary does not use five seeds")
    checks.append("faithfulness summary uses five seeds")

    if not faithfulness["top1_mean"].between(0.0, 1.0).all():
        raise AssertionError("top-1 recovery outside [0, 1]")
    checks.append("faithfulness metrics in range")

    if not real_flags["n_flags"].between(0, 5).all():
        raise AssertionError("n_flags outside 0..5")
    if int(real_flags["n_flags"].max()) != 5:
        raise AssertionError("no row is flagged by all five detectors")
    if int(real_flags["consensus_primary"].sum()) == 0:
        raise AssertionError("no primary consensus flags")
    if int(real_flags["consensus_primary"].sum()) == len(real_flags):
        raise AssertionError("every row is a primary consensus flag")
    checks.append("real-record flags consistent")

    if not overlap["jaccard"].between(0.0, 1.0).all():
        raise AssertionError("Jaccard outside [0, 1]")
    checks.append("overlap metrics in range")

    if cases["station_id"].nunique() != len(cases):
        raise AssertionError("case studies are not from distinct stations")
    checks.append("case studies use distinct stations")

    manifest["checks"] = checks
    manifest["status"] = "passed"
    write_json(ROOT / "results" / "release_manifest.json", manifest)

    print(f"Release check passed: {len(EXPECTED_ROWS)} files, {len(checks)} checks")
    print("Wrote results/release_manifest.json")


if __name__ == "__main__":
    main()

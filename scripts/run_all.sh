#!/usr/bin/env bash
# Reproduce the full project from a fresh clone.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/01_build_dataset.py
"$PYTHON" scripts/02_audit.py
"$PYTHON" scripts/phase2_smoke.py
"$PYTHON" scripts/03_run_detection.py
"$PYTHON" scripts/04_run_explanations.py
"$PYTHON" scripts/05_real_record.py
"$PYTHON" scripts/06_release_check.py
"$PYTHON" -m pytest -q

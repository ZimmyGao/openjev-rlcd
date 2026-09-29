#!/usr/bin/env bash
# Full result tables (all arms, 3 seeds, after TS, paired tests) from results/ (or from RESULTS=<dir>).
# If a paper/ directory with make_assets.py is present, also regenerates its tables/figures and the PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -m openjev_rlcd.final_tables --task gsm8kv --results "${RESULTS:-results}"
$PY -m openjev_rlcd.final_tables --task mmlupro --results "${RESULTS:-results}"
if [ -f paper/make_assets.py ]; then
  OPENJEV_RESULTS="${RESULTS:-results}" $PY paper/make_assets.py
  if command -v tectonic > /dev/null; then (cd paper && tectonic main.tex); fi
fi

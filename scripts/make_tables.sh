#!/usr/bin/env bash
# Tables, figures and in-text numbers of the paper from results/ (or from RESULTS=<dir>), then the PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -m openjev.final_tables --task gsm8kv --results "${RESULTS:-results}"
$PY -m openjev.final_tables --task mmlupro --results "${RESULTS:-results}"
OPENJEV_RESULTS="${RESULTS:-results}" $PY paper/make_assets.py
if command -v tectonic > /dev/null; then (cd paper && tectonic main.tex); fi

#!/usr/bin/env bash
# Keep runs/anchored/REPORT.md current: regenerate it (low priority, CPU only, 2 threads) whenever a queued run has
# finished since the last report. Sections whose run files did not change come from the cache, so a refresh is cheap.
#   REPO=<repo> PY=.venv/bin/python bash scripts/report_loop.sh
cd "${REPO:-$(dirname "$0")/..}" || exit 1
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
R=runs/anchored/REPORT.md
while true; do
  if [ ! -f "$R" ] || [ runs/queue/history.txt -nt "$R" ]; then
    if ! nice -n 19 $PY -m openjev_rlcd.backbone_report > runs/anchored/report.log 2>&1; then
      echo "$(date '+%m-%d %H:%M') report failed (see above); retried on the next pass" >> runs/anchored/report.log
    fi
  fi
  sleep "${REPORT_SLEEP:-600}"
done

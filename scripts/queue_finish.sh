#!/usr/bin/env bash
# Hand the GPUs back when the queue is done: once no job is queued or running, stop the workers (STOP file), write
# the final report and end the report loop.
#   REPO=<repo> PY=.venv/bin/python STOP_FILE=runs/queue/STOP.v2 bash scripts/queue_finish.sh
# WORKER_SESSIONS / REPORT_SESSION: tmux sessions of the workers and of scripts/report_loop.sh.
cd "${REPO:-$(dirname "$0")/..}" || exit 1
QDIR=${QDIR:-runs/queue}
PY=${PY:-python}
jobs_left() { grep -v '^#' "$QDIR/queue.txt" | grep -c .; }
jobs_running() { grep -c . "$QDIR/running.txt"; }
while [ "$(jobs_left)" -gt 0 ] || [ "$(jobs_running)" -gt 0 ]; do sleep 300; done
echo "$(date '+%m-%d %H:%M') queue finished: stopping the workers" >> "$QDIR/status.txt"
touch "${STOP_FILE:-$QDIR/STOP}"
for s in ${WORKER_SESSIONS:-ojq0b ojq1b}; do
  while tmux has-session -t "=$s" 2>/dev/null; do sleep 30; done
done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 nice -n 19 \
  $PY -m openjev_rlcd.backbone_report > runs/anchored/report.log 2>&1
tmux kill-session -t "=${REPORT_SESSION:-ojreport}" 2>/dev/null
echo "$(date '+%m-%d %H:%M') ALL DONE: workers stopped, final report written, GPUs free" >> "$QDIR/status.txt"

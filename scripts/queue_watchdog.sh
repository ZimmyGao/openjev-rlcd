#!/usr/bin/env bash
# Restart a GPU worker whose tmux session disappeared (e.g. its pane killed by systemd-oomd under swap pressure, as on
# 2026-10-11/12); the queue re-queues the dead worker's job by itself. Stops once the STOP file exists (queue done).
#   REPO=<repo> PY=.venv/bin/python STOP_FILE=runs/queue/STOP.v2 bash scripts/queue_watchdog.sh
# WORKERS: "session:gpu:worker script" entries (default: ojq0b/ojq1b with runs/queue/worker_gpu{0,1}_v2.sh)
cd "${REPO:-$(dirname "$0")/..}" || exit 1
R=$PWD
QDIR=${QDIR:-runs/queue}
STOP=${STOP_FILE:-$R/$QDIR/STOP}
WORKERS=${WORKERS:-"ojq0b:0:$QDIR/worker_gpu0_v2.sh ojq1b:1:$QDIR/worker_gpu1_v2.sh"}
while [ ! -f "$STOP" ]; do
  for w in $WORKERS; do
    IFS=: read -r s g script <<< "$w"
    if ! tmux has-session -t "=$s" 2>/dev/null && [ ! -f "$STOP" ]; then
      echo "$(date '+%m-%d %H:%M') watchdog: worker session $s gone, restarting it" >> "$QDIR/status.txt"
      tmux new-session -d -s "$s" "REPO=$R GPU=$g PY=$PY STOP_FILE=$STOP bash $R/$script"
    fi
  done
  sleep "${WATCHDOG_SLEEP:-300}"
done

#!/usr/bin/env bash
# Watch the GPU job queue; exit (so whoever waits on it wakes up) on the first event: a queue job ends / fails / a
# worker goes IDLE, free disk < 80G, a queue tmux session is gone, a running job's log is silent > 90 min, or a GPU
# has had no compute process for ~10 min, or no queue job has run for ~10 min while jobs are queued (safety nets
# against scheduler bugs). Prints a short status report.
#   [IGNORE_IDLE="1"] [SESSIONS="ojq0b ..."] bash scripts/watch_queue.sh   # IGNORE_IDLE: GPUs expected to idle;
#   SESSIONS: tmux sessions that must stay up (default: both workers, report loop, finisher, watchdog)
cd "$(dirname "$0")/.." || exit 1
QS=runs/queue/status.txt
q0=$(wc -l < "$QS")
# GPU busy test shared with the queue workers (works with or without NVML)
eval "$(sed -n '/^gpu_busy_nonvml()/,/^}/p; /^gpu_busy()/,/^}/p' scripts/queue_worker.sh)"
declare -A idle=([0]=0 [1]=0)
stalled=0
event=""
while [ -z "$event" ]; do
  sleep 120
  newq=$(tail -n +$((q0 + 1)) "$QS")
  echo "$newq" | grep -q " end \| FAILED\|IDLE\|ALL DONE\|watchdog" && \
    event="queue: $(echo "$newq" | grep " end \| FAILED\|IDLE\|ALL DONE\|watchdog" | tail -3 | tr '\n' '|')"
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "$avail" -lt 80 ] && event="$event; disk low (${avail}G)"
  if [ ! -f runs/queue/STOP.v2 ]; then
    for s in ${SESSIONS:-ojq0b ojq1b ojreport ojfinish ojwatchdog}; do
      tmux has-session -t "=$s" 2>/dev/null || event="$event; session $s gone"
    done
  fi
  while read -r gpu pid t0 name out rest; do
    [ -n "$name" ] || continue
    f="$out/$name.log"; [ -f "$f" ] || continue
    age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
    [ "$age" -gt 90 ] && event="$event; $f silent ${age} min"
  done < runs/queue/running.txt
  for g in 0 1; do
    if GPU=$g gpu_busy; then idle[$g]=0; else idle[$g]=$((idle[$g] + 1)); fi
    case " ${IGNORE_IDLE:-} " in *" $g "*) continue ;; esac  # GPUs expected to idle (e.g. only offload jobs left)
    [ "${idle[$g]}" -ge 5 ] && [ ! -f runs/queue/STOP.v2 ] && event="$event; GPU$g idle ~$((idle[$g] * 2)) min"
  done
  # scheduler-level net, independent of GPU use by other projects: nothing running while jobs are queued
  if [ -z "$(awk 'NF' runs/queue/running.txt)" ] && grep -q '^[^#[:space:]]' runs/queue/queue.txt; then
    stalled=$((stalled + 1)); else stalled=0; fi
  [ "$stalled" -ge 5 ] && [ ! -f runs/queue/STOP.v2 ] && event="$event; queue stalled: nothing running for ~$((stalled * 2)) min"
done
date
echo "EVENT: ${event#; }"
df -h / | tail -1
echo "--- new queue status lines"; tail -n +$((q0 + 1)) "$QS"
echo "--- running"; awk '{print $1, $5, $4}' runs/queue/running.txt
echo "--- queued: $(grep -vc '^#' runs/queue/queue.txt)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
tmux ls 2>&1 | grep oj

#!/usr/bin/env bash
# One worker per GPU: waits until the GPU is free, then runs jobs from the shared queue (scripts/jobqueue.py) one
# after another, forever (it polls while the queue is empty or every queued job is blocked).
#   GPU=0 PY=.venv/bin/python [WAIT_SESSIONS="tmux session names"] bash scripts/queue_worker.sh
# WAIT_SESSIONS: tmux sessions (exact names) that must end before the worker starts, e.g. a run already on the GPU.
# Before every job the GPU must show no compute process for FREE_CHECKS checks FREE_SLEEP s apart (also covers
# jobs started by hand). Stop all workers after their current job: touch $QDIR/STOP (or $STOP_FILE; remove to restart).
cd "${REPO:-$(dirname "$0")/..}" || exit 1
export QDIR=${QDIR:-runs/queue}
GPU=${GPU:?set GPU}
PY=${PY:-python}
RUNNER=${RUNNER:-"$PY -m openjev_rlcd."}
mkdir -p "$QDIR"
say() { echo "$(date '+%m-%d %H:%M') gpu$GPU $*" >> "$QDIR/status.txt"; }
gpu_busy_nonvml() {  # NVML unusable (e.g. driver libraries upgraded under a loaded kernel module): a python process
  local p n          # with CUDA_VISIBLE_DEVICES=<this gpu>, or holding many handles on its device file (a CUDA context)
  for p in /proc/[0-9]*; do
    case "$(cat "$p/comm" 2>/dev/null)" in python*) ;; *) continue ;; esac
    tr '\0' '\n' 2>/dev/null < "$p/environ" | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && return 0
    n=$(ls -l "$p/fd" 2>/dev/null | grep -c "/dev/nvidia$GPU$")
    [ "$n" -ge 8 ] && return 0
  done
  return 1
}
gpu_busy() {
  [ "${GPU_CHECK:-1}" = 0 ] && return 1
  local uuid
  if uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null) && [ -n "$uuid" ]; then
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
  else
    gpu_busy_nonvml
  fi
}
for s in ${WAIT_SESSIONS:-}; do
  while tmux has-session -t "=$s" 2>/dev/null; do sleep 60; done
done
say "worker up (pid $$)"
idle=""
while true; do
  [ -f "${STOP_FILE:-$QDIR/STOP}" ] && { say "STOP file present: worker exits"; exit 0; }
  free=0
  while [ "$free" -lt "${FREE_CHECKS:-3}" ]; do
    if gpu_busy; then free=0; else free=$((free + 1)); fi
    sleep "${FREE_SLEEP:-20}"
  done
  job=$(python3 scripts/jobqueue.py pop --gpu "$GPU" --worker $$)
  rc=$?
  if [ "$rc" -ne 0 ]; then
    if [ "$rc" = 4 ]; then reason="queue empty"; elif [ "$rc" = 3 ]; then reason="every queued job is blocked"; else reason="queue.py error rc=$rc"; fi
    [ "$idle" = "$reason" ] || say "IDLE: $reason"
    idle=$reason
    sleep "${IDLE_SLEEP:-120}"
    continue
  fi
  idle=""
  read -r name out res needs module args <<< "$job"
  mkdir -p "$out"
  say "start $out/$name"
  # shellcheck disable=SC2086  # $RUNNER and $args are meant to split into words
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $RUNNER$module $args --output "$out/$name.json" > "$out/$name.log" 2>&1
  rc=$?
  ok=0
  [ -f "$out/$name.json" ] && grep -q '"test": {' "$out/$name.json" && ok=1
  python3 scripts/jobqueue.py finish --gpu "$GPU" --name "$name" --out "$out" --ok "$ok" --rc "$rc"
  say "end   $out/$name rc=$rc complete=$ok"
done

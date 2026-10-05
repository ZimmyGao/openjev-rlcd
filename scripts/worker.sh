#!/usr/bin/env bash
# Run a list of backbone jobs one after another on one GPU. Each line of the job file:
#   <model> <offload 0|1> <task> <seed> <out dir>
# e.g. "HuggingFaceTB/SmolLM3-3B 1 gsm8kv 17 runs/backbones/smollm3-3b". Every job runs
# scripts/backbone.sh for one seed; finished runs are skipped, so a worker can be restarted.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/worker.sh jobs_gpu0.txt
cd "$(dirname "$0")/.."
STATUS=${STATUS:-runs/backbones/status.txt}
mkdir -p "$(dirname "$STATUS")"
while read -r model offload task seed out; do
  [ -z "$model" ] || [ "${model:0:1}" = "#" ] && continue
  echo "$(date '+%m-%d %H:%M') gpu${CUDA_VISIBLE_DEVICES:-?} start $model $task s$seed" >> "$STATUS"
  if MODEL=$model OFFLOAD=$offload SEEDS=$seed OUT=$out bash scripts/backbone.sh "$task"; then
    echo "$(date '+%m-%d %H:%M') gpu${CUDA_VISIBLE_DEVICES:-?} done  $model $task s$seed" >> "$STATUS"
  else
    echo "$(date '+%m-%d %H:%M') gpu${CUDA_VISIBLE_DEVICES:-?} FAILED $model $task s$seed" >> "$STATUS"
  fi
done < "$1"
echo "$(date '+%m-%d %H:%M') gpu${CUDA_VISIBLE_DEVICES:-?} ALL DONE $1" >> "$STATUS"

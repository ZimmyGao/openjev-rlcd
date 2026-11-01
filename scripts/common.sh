#!/usr/bin/env bash
# Shared settings for every run in the paper. Sourced by the other scripts; not run directly.
#   OUT=<dir>   where run files go (default: runs/; the released files are in results/)
#   CKPT=<dir>  where checkpoints go (default: $OUT); SAVE_FINAL=1 also saves the final weights of every train run
#   SEEDS="17 23 29"  PY=python  CUDA_VISIBLE_DEVICES=<gpu>
#   MODEL=<hf id> (default Qwen/Qwen3-1.7B)   OFFLOAD=1: fp32 master weights + AdamW on the CPU (3-4B models)
# One run = one 48 GB GPU (Qwen3-1.7B: fp32 trained weights + bf16 sampling and reference copies). With
# OFFLOAD=1 a run also needs 16 bytes of host memory per parameter (~50-60 GB for 3-4B models).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=${OUT:-runs}
SEEDS=${SEEDS:-"17 23 29"}
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
MFLAGS="--model $MODEL"
[ "${OFFLOAD:-0}" = 1 ] && MFLAGS="$MFLAGS --offload-optimizer --gpu-headroom-gb ${GPU_HEADROOM_GB:-16} --gen-batch 32"
CKPT=${CKPT:-$OUT}
mkdir -p "$OUT" "$CKPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8 questions x M=4 rationales per step, 400 steps, per-rationale objective (lambda = 0)
GV="--task gsm8kv --m 4 --m-eval 4 --max-new-tokens 320 --steps 400 --dev-cap 300 --test-cap 1260 --diversity-weight 0 $MFLAGS"
MP="--task mmlupro --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 --dev-cap 300 --test-cap 1000 --diversity-weight 0 $MFLAGS"

task_args() { case $1 in gsm8kv) echo "$GV" ;; mmlupro) echo "$MP" ;; *) echo "unknown task $1" >&2; exit 1 ;; esac; }
prefix() { case $1 in gsm8kv) echo gv ;; mmlupro) echo mp ;; esac; }
# test-set size for the direct-answer baseline (0 = whole split)
direct_test_cap() { case $1 in gsm8kv) echo 0 ;; mmlupro) echo 1000 ;; esac; }

# retire <file>...: move files aside (<file>.superseded-<time>) instead of deleting them
retire() { local f; for f in "$@"; do [ -e "$f" ] && mv -n "$f" "$f.superseded-$(date +%Y%m%d-%H%M%S)"; done; return 0; }

# run <name> <module> <args...>  ->  $OUT/<name>.json (+ .log); skips runs that already finished; the files of an
# unfinished earlier attempt are kept (retired), never overwritten
run() {
  local name=$1 module=$2; shift 2
  if [ -f "$OUT/$name.json" ] && grep -q '"test": {' "$OUT/$name.json"; then echo "skip $name (done)"; return; fi
  echo "[$(date '+%m-%d %H:%M')] $name"
  retire "$OUT/$name.json" "$OUT/$name.log"
  local save=()
  if [ "${SAVE_FINAL:-0}" = 1 ] && [ "$module" = train ] && [[ " $* " != *" --save-checkpoint "* ]]; then
    save=(--save-checkpoint "$CKPT/$name.pt")
  fi
  $PY -m "openjev_rlcd.$module" "$@" "${save[@]}" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
}

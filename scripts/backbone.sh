#!/usr/bin/env bash
# Core comparison on another backbone: base, SFT + TS, GRPO + TS, RFT/STaR + KL + TS, RLCD stage 1 and
# two-stage RLCD (stage 1 + RLCD-RL), with the paper's hyperparameters and seeds.
#   MODEL=Qwen/Qwen3-0.6B OUT=runs/qwen3-0.6b bash scripts/backbone.sh gsm8kv|mmlupro
#   MODEL=HuggingFaceTB/SmolLM3-3B OFFLOAD=1 OUT=runs/smollm3-3b bash scripts/backbone.sh gsm8kv
# File names match the main runs, so `python -m openjev_rlcd.final_tables --results $OUT` builds the tables.
source "$(dirname "$0")/common.sh"
TASK=$1; A=$(task_args "$TASK"); P=$(prefix "$TASK")
base=${P}_base_eval; [ "$TASK" = gsm8kv ] && base=gv_base_eval_full
run $base train $A --eval-only --seed 17
for s in $SEEDS; do  # baselines first, then RLCD stage 1 and stage 2
  run ${P}_grpokl_seed$s  train $A --objective grpo --kl-beta 0.04 --seed $s --eval-every 200
  run ${P}_rftkl_seed$s   train $A --objective rft --kl-beta 0.04 --seed $s --eval-every 400
  run ${P}_direct_lr2e-6_seed$s train_direct --task "$TASK" $MFLAGS --labels fresh --lr 2e-6 --seed $s --steps 400 \
    --dev-cap 300 --test-cap $(direct_test_cap "$TASK")
  ck=$CKPT/${P}_warm400_s$s.pt
  fork=$OUT/${P}_fork_rl_seed$s.json
  if [ -f "$fork" ] && grep -q '"test": {' "$fork"; then continue; fi  # stage 2 done: stage 1 need not be rerun
  [ -f "$ck" ] || retire "$OUT/${P}_warm400_seed$s.json"  # stage 1 must be rerun to recreate the checkpoint
  run ${P}_warm400_seed$s train $A --kl-beta 0.04 --seed $s --eval-every 400 --no-score-function --save-checkpoint "$ck"
  run ${P}_fork_rl_seed$s train $A --kl-beta 0.04 --seed $((1000 + s)) --eval-every 400 --init-checkpoint "$ck" --sf-coef 0.3
done

#!/usr/bin/env bash
# Baselines of Table 1 (every one is temperature-scaled on dev at analysis time).
#   bash scripts/baselines.sh gsm8kv|mmlupro
source "$(dirname "$0")/common.sh"
TASK=$1; A=$(task_args "$TASK"); P=$(prefix "$TASK")
# zero-shot reasoning model
base=${P}_base_eval; [ "$TASK" = gsm8kv ] && base=gv_base_eval_full
run $base train $A --eval-only --seed 17
for s in $SEEDS; do
  # SFT + TS: direct answer, cross-entropy on the option token (400 and 800 steps)
  run ${P}_direct_lr2e-6_seed$s    train_direct --task "$TASK" --labels fresh --lr 2e-6 --seed $s --steps 400 --dev-cap 300 --test-cap $(direct_test_cap "$TASK")
  run ${P}_direct800_lr2e-6_seed$s train_direct --task "$TASK" --labels fresh --lr 2e-6 --seed $s --steps 800 --dev-cap 300 --test-cap $(direct_test_cap "$TASK")
  # GRPO + KL from the base model
  run ${P}_grpokl_seed$s train $A --objective grpo --kl-beta 0.04 --seed $s --eval-every 200
  # RFT / STaR, without and with the KL anchor
  run ${P}_rft_seed$s   train $A --objective rft --seed $s --eval-every 400
  run ${P}_rftkl_seed$s train $A --objective rft --kl-beta 0.04 --seed $s --eval-every 400
  # stage 1 with the log score instead of the Brier score
  run ${P}_ce_seed$s train $A --kl-beta 0.04 --no-score-function --readout-score log --seed $s --eval-every 400
done

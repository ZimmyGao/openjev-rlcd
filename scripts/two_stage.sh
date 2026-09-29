#!/usr/bin/env bash
# Two-stage RLCD and the forked stage-2 controls (Tables 1-2, Figures 2-3).
#   bash scripts/two_stage.sh gsm8kv|mmlupro
# Per seed s:
#   stage 1  <px>_warm400_seed<s> : 400 steps readout-only (pathwise proper score on the model's own
#                                   rationales), saved as a checkpoint
#   stage 2, four 400-step continuations forked from that checkpoint with the same data stream
#   (seed 1000+s), a fresh optimizer and the KL reference reset to the checkpoint:
#     <px>_fork_rl     RLCD-RL: REINFORCE on the rationales scored by the own readout (c = 0.3)  <- two-stage RLCD
#     <px>_fork_ro     readout-only continuation
#     <px>_fork_anchor same REINFORCE noise and KL anchor, no outcome reward
#     <px>_fork_grpo   GRPO (correctness reward) continuation
source "$(dirname "$0")/common.sh"
TASK=$1; A=$(task_args "$TASK"); P=$(prefix "$TASK")
for s in $SEEDS; do
  ck=$OUT/${P}_warm400_s$s.pt
  [ -f "$ck" ] || rm -f "$OUT/${P}_warm400_seed$s.json"
  run ${P}_warm400_seed$s train $A --kl-beta 0.04 --seed $s --eval-every 400 --no-score-function --save-checkpoint "$ck"
  F="$A --kl-beta 0.04 --seed $((1000 + s)) --eval-every 400 --init-checkpoint $ck"
  run ${P}_fork_rl_seed$s     train $F --sf-coef 0.3
  run ${P}_fork_ro_seed$s     train $F --no-score-function
  run ${P}_fork_anchor_seed$s train $F --sf-coef 0.3 --sf-reader none
  run ${P}_fork_grpo_seed$s   train $F --objective grpo
done

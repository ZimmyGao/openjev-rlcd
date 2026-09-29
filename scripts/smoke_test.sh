#!/usr/bin/env bash
# End-to-end check of every training mode on a few questions (one GPU, ~5 min). Writes to runs/smoke/.
source "$(dirname "$0")/common.sh"
OUT=runs/smoke; mkdir -p $OUT
S="--task gsm8kv --m 4 --m-eval 4 --max-new-tokens 64 --steps 2 --eval-every 2 --dev-cap 4 --test-cap 4 --diversity-weight 0 --seed 1"
run stage1 train $S --kl-beta 0.04 --no-score-function --save-checkpoint $OUT/stage1.pt
run stage2 train $S --kl-beta 0.04 --init-checkpoint $OUT/stage1.pt --sf-coef 0.3
run anchor train $S --kl-beta 0.04 --init-checkpoint $OUT/stage1.pt --sf-coef 0.3 --sf-reader none
run grpo   train $S --kl-beta 0.04 --objective grpo
run rftkl  train $S --kl-beta 0.04 --objective rft
run logscore train $S --kl-beta 0.04 --no-score-function --readout-score log
run mixture train $S --diversity-weight 1 --grad-diag 1
run base   train $S --eval-only
run sft    train_direct --task mmlupro --labels fresh --steps 2 --eval-every 2 --dev-cap 4 --test-cap 4 --seed 1
rm -f $OUT/stage1.pt
echo "smoke test OK"

#!/usr/bin/env bash
# Figure 1, Table 3 (ChaosNLI) and Table 4 (score-function weight).
#   bash scripts/ablations.sh
source "$(dirname "$0")/common.sh"
MP1="$MP --seed 17 --eval-every 200"
# Figure 1: per-rationale objective trained end to end on MMLU-Pro, without and with the KL anchor
# (the paper's no-KL run was stopped at step 150, after the collapse)
run mp_rlcd0_seed17   train $MP1
# Table 4: weight c of the score-function term when training from scratch (c = 1 is the run above with KL)
run mp_rlcd0kl_seed17        train $MP1 --kl-beta 0.04
run mp_rlcd0kl_sf0.3_seed17  train $MP1 --kl-beta 0.04 --sf-coef 0.3 --grad-diag 10
run mp_rlcd0kl_sf0.1_seed17  train $MP1 --kl-beta 0.04 --sf-coef 0.1 --grad-diag 10
run mp_rlcd0klnosf_seed17    train $MP1 --kl-beta 0.04 --no-score-function
# Table 3: ChaosNLI (8 rationales per question, 800 steps, 16-rationale mixture at evaluation)
C="--task chaosnli --m 8 --m-eval 16 --max-new-tokens 128 --steps 800"
for s in $SEEDS; do
  run teacher2_rb_seed$s train $C --seed $s --diversity-weight 1            # mixture objective
  run direct_fresh_lr2e-6_seed$s train_direct --task chaosnli --labels fresh --lr 2e-6 --seed $s --steps 800
done
run teacher2_rbkl_seed17    train $C --seed 17 --diversity-weight 1 --kl-beta 0.01 --length-penalty 0.5
run teacher3_single_seed17  train $C --seed 17 --diversity-weight 0       # per-rationale objective, no KL

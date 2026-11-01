#!/usr/bin/env bash
# GPU check of every option added for the npj protocol (paper/npj/PROTOCOL.md) on a few questions (one GPU; MODEL as in
# common.sh, default Qwen3-0.6B here). SMOKE_SET=offload runs the subset that touches the offload paths (use with
#   MODEL=HuggingFaceTB/SmolLM3-3B OFFLOAD=1). Writes to runs/smoke_npj/<model>/ and its checkpoints to $CKPT_SMOKE
# (default ${OPENJEV_CKPT_DIR:-checkpoints}/smoke_npj/<model>); earlier smoke files are retired, never deleted.
# Exits non-zero unless every run file has the expected split, log-probabilities, calib block and snapshots.
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
source "$(dirname "$0")/common.sh"
# offload: the placement the npj 3-4B MMLU-Pro jobs use (plan_anchored.margs), and only the offload subset (snapshots
# are refused with --offload-optimizer)
[ "${OFFLOAD:-0}" = 1 ] && MFLAGS="--model $MODEL --offload-optimizer --gpu-headroom-gb 24 --gen-batch 16"
SET=${SMOKE_SET:-$([ "${OFFLOAD:-0}" = 1 ] && echo offload || echo all)}
OUT=runs/smoke_npj/$(basename "$MODEL"); mkdir -p "$OUT"
CK=${CKPT_SMOKE:-${OPENJEV_CKPT_DIR:-checkpoints}/smoke_npj/$(basename "$MODEL")}; mkdir -p "$CK"
S="--task mmlupro --m 4 --m-eval 4 --max-new-tokens 64 --steps 2 --eval-every 2 --dev-cap 4 --test-cap 4 --diversity-weight 0 --seed 1 $MFLAGS"
retire "$OUT"/*.json "$OUT"/*.log
retire "$CK"/*.pt
A="--kl-beta 0.04 --kl-tokens 0.01 --no-score-function"
run s1      train $S $A --test-split fresh --extra-splits calib --save-checkpoint "$CK/s1.pt"
run grpo    train $S --kl-beta 0.04 --objective grpo --test-split fresh --extra-splits calib --save-checkpoint "$CK/grpo.pt"
run sft     train_direct --task gsm8kv $MFLAGS --labels fresh --steps 2 --eval-every 2 --dev-cap 4 --test-cap 4 --seed 1 --test-split fresh --extra-splits calib --save-checkpoint "$CK/sft.pt"
run think   train $S $A --thinking --gen-batch 16 --test-split fresh
if [ "$SET" = all ]; then
  run snap    train $S $A --snapshot-steps 1 --save-checkpoint "$CK/snap.pt"
  run ce      train $S $A --readout-score log --test-split fresh
  run lin     train $S $A --readout-score linear --test-split fresh
  run refrat  train $S $A --rationale-source ref --test-split fresh
  run mix     train $S $A --diversity-weight 1 --test-split fresh
  run correct train $S --kl-beta 0.04 --kl-tokens 0.01 --init-checkpoint "$CK/s1.pt" --sf-coef 0.3 --sf-reward correct --test-split fresh
  run rlcr    train $S --kl-beta 0.04 --objective rlcr --test-split fresh
  run evcalib train $S --eval-only --checkpoint "$CK/grpo.pt" --test-split calib --extra-splits fresh
  run probe   think_probe --task mmlupro --model "$MODEL" --n 4 --cap 128 --gen-batch 4
fi
python3 - "$OUT" "$CK" "$SET" <<'PYEOF'
import json, sys
from pathlib import Path
out, ck, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
# name: (split, has logps, has calib, snapshots, checkpoint)
want = {"s1": ("fresh", True, True, [], "s1.pt"), "grpo": ("fresh", True, True, [], "grpo.pt"),
        "sft": ("fresh", True, True, [], "sft.pt"), "think": ("fresh", True, False, [], None)}
if which == "all":
    want.update({"snap": ("test", True, False, ["1"], "snap.pt"), "ce": ("fresh", True, False, [], None),
                 "lin": ("fresh", True, False, [], None), "refrat": ("fresh", True, False, [], None),
                 "mix": ("fresh", True, False, [], None), "correct": ("fresh", True, False, [], None),
                 "rlcr": ("fresh", False, False, [], None), "evcalib": ("calib", True, False, [], None)})
bad = []
for name, (split, logps, calib, snaps, cpt) in want.items():
    f = out / f"{name}.json"
    if not f.exists() or '"test": {' not in f.read_text():
        bad.append(f"{name}: no finished run file")
        continue
    d = json.loads(f.read_text())
    got = (d["config"].get("test_split"), "logps" in d.get("test", {}) or "test_logps" in d,
           "eval_calib" in d or "eval_fresh" in d, sorted(d.get("snapshots", {})))
    if got != (split, logps, calib or name == "evcalib", snaps):
        bad.append(f"{name}: got {got}")
    if cpt and not (ck / cpt).exists():
        bad.append(f"{name}: checkpoint {cpt} missing")
    print(f"{name:8s} split={got[0]} logps={got[1]} extra-split={got[2]} snapshots={got[3]}")
if which == "all":
    p = out / "probe.json"
    if not p.exists() or "chosen_cap" not in json.loads(p.read_text()).get("test", {}):
        bad.append("probe: no result")
if bad:
    print("SMOKE TEST FAILED:", *bad, sep="\n  ")
    sys.exit(1)
PYEOF
echo "smoke test OK ($SET)"

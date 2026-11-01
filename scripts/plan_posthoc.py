#!/usr/bin/env python3
"""Job plan of the post-hoc exploratory experiments X1-X3 (paper/npj/POSTHOC_20261029.md), designed after the
registered final results; every job line is validated with train.check_args (scripts/plan_npj.parse) before it is
written. Outputs go to runs/npj_posthoc/<model>/, so the registered analysis (npj_decide) never reads them.

    python3 scripts/plan_posthoc.py > runs/npj_posthoc/plan_posthoc.txt      # then: jobqueue.py push < that file
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from plan_npj import parse  # noqa: E402

CK = "/data/Projects/openjev-rlcd/checkpoints/npj"
SEEDS = (17, 23, 29)
MODEL = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B"}
TASK = {  # the registered fresh-protocol settings of each task (runs/npj/plan_tier1.txt)
    "mp": "--task mmlupro --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 --dev-cap 300 --test-cap 1000 "
          "--diversity-weight 0 --kl-beta 0.04 --eval-every 400 --test-split fresh",
    "gv": "--task gsm8kv --m 4 --m-eval 4 --max-new-tokens 320 --steps 400 --dev-cap 300 --test-cap 1260 "
          "--diversity-weight 0 --kl-beta 0.04 --eval-every 400 --test-split fresh",
}
CHAOS = ("--task chaosnli --m 8 --m-eval 16 --max-new-tokens 128 --steps 400 --dev-cap 199 --test-cap 400 "
         "--model Qwen/Qwen3-1.7B --kl-beta 0.04 --kl-tokens 0.01 --diversity-weight 0 --eval-every 400")


def job(name, model, args):
    parse("train", args)  # raises SystemExit on an invalid line
    return f"{name} runs/npj_posthoc/{model} std - train {args}"


def x1():
    """ChaosNLI stage 2 from the per-rationale stage-1 checkpoints: proper vs sampled-answer reward vs continuation"""
    out = []
    for s in SEEDS:
        init = f"--seed 10{s} --init-checkpoint {CK}/qwen3-1.7b/cn_s1_lam0_seed{s}.pt"
        out.append(job(f"cn_s2_rl_seed{s}", "qwen3-1.7b", f"{CHAOS} {init} --sf-coef 0.3"))
        out.append(job(f"cn_s2_correct_seed{s}", "qwen3-1.7b", f"{CHAOS} {init} --sf-coef 0.3 --sf-reward correct"))
        out.append(job(f"cn_s2_cont_seed{s}", "qwen3-1.7b", f"{CHAOS} {init} --no-score-function"))
    return out


def x2(cells, arms):
    """400 more steps from the stage-1 (Brier) checkpoints: stage-1 continuation (budget control), GRPO continuation"""
    out = []
    for m, px in cells:
        for s in SEEDS:
            base = (f"{TASK[px]} --model {MODEL[m]} --seed 10{s} "
                    f"--init-checkpoint {CK}/{m}/{px}_warm400_seed{s}.pt --kl-tokens 0.01")
            if "s1cont" in arms:
                out.append(job(f"{px}_s1cont_seed{s}", m, f"{base} --no-score-function"))
            if "fork_grpo" in arms:
                out.append(job(f"{px}_fork_grpo_seed{s}", m, f"{base} --objective grpo"))
    return out


def x3(cells):
    """stage 1 without the token anchor: the model's own rationales vs offline rationales from the initial model"""
    out = []
    for s in SEEDS:
        for m, px in cells:
            base = f"{TASK[px]} --model {MODEL[m]} --seed {s} --no-score-function --kl-tokens 0"
            out.append(job(f"{px}_noanchor_seed{s}", m, base))
            out.append(job(f"{px}_noanchor_refrat_seed{s}", m, f"{base} --rationale-source ref"))
    return out


CKP = "/data/Projects/openjev-rlcd/checkpoints/npj_posthoc"
SEEDS_X4 = (41, 53)


def x4():
    """protocol entry P2: two more seeds of post-rationale supervision (warm400) and GRPO+TS on Qwen3-0.6B
    GSM8K-Verify, with the registered settings (runs/npj/plan_tier1.txt, plan_grpo.txt) and the registered
    SVAMP-Verify evaluation of the saved weights (runs/npj/plan_transfer_svamp.txt)"""
    gv = TASK["gv"] + " --model Qwen/Qwen3-0.6B"
    out = []
    for s in SEEDS_X4:
        arms = {"warm400": f"--seed {s} --no-score-function --kl-tokens 0.01",
                "grpo": f"--objective grpo --kl-tokens 0.01 --seed {s}"}
        for arm, extra in arms.items():
            base = gv.replace("--eval-every 400", "--eval-every 200") if arm == "grpo" else gv  # as registered
            args = f"{base} {extra} --extra-splits calib --save-checkpoint {CKP}/qwen3-0.6b/gv_{arm}_seed{s}.pt"
            out.append(job(f"gv_{arm}_seed{s}", "qwen3-0.6b", args))
        for arm in arms:
            args = (f"--task svampv --m 4 --m-eval 4 --steps 400 --diversity-weight 0 --max-new-tokens 320 --dev-cap 200 "
                    f"--test-cap 100000 --model Qwen/Qwen3-0.6B --eval-only --seed {s} "
                    f"--checkpoint {CKP}/qwen3-0.6b/gv_{arm}_seed{s}.pt")
            parse("train", args)
            out.append(f"svampv_{arm}_seed{s} runs/npj_posthoc/qwen3-0.6b std "
                       f"runs/npj_posthoc/qwen3-0.6b/gv_{arm}_seed{s}.json train {args}")
    return out


def main():
    if sys.argv[1:] == ["p2"]:  # entry P2 only: python3 scripts/plan_posthoc.py p2 > runs/npj_posthoc/plan_posthoc_p2.txt
        lines = x4()
        print("\n".join(lines))
        print(f"{len(lines)} post-hoc (P2) jobs validated", file=sys.stderr)
        return
    qwen = [("qwen3-0.6b", "mp"), ("qwen3-0.6b", "gv"), ("qwen3-1.7b", "mp"), ("qwen3-1.7b", "gv")]
    lines = (x1()
             + x2([("qwen3-0.6b", "gv")], ("s1cont", "fork_grpo"))
             + x3(qwen)
             + x2([("qwen3-0.6b", "mp")], ("s1cont", "fork_grpo"))
             + x2([("qwen3-1.7b", "mp"), ("qwen3-1.7b", "gv")], ("s1cont",)))
    names = [(ln.split()[1], ln.split()[0]) for ln in lines]
    assert len(set(names)) == len(names), "duplicate job"
    print("\n".join(lines))
    print(f"{len(lines)} post-hoc jobs validated", file=sys.stderr)


if __name__ == "__main__":
    main()

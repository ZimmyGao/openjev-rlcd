#!/usr/bin/env python3
"""Job plan for the anchored rerun (pathwise KL anchor --kl-tokens 0.04 on the rationales in BOTH stages) and the
remaining multi-backbone baselines, in priority order, for scripts/jobqueue.py.

    python3 scripts/plan_anchored.py              # print the job lines
    python3 scripts/plan_anchored.py --link       # also link the finished validation runs (runs/klval) in place

Layout (file names as in scripts/two_stage.sh, so `final_tables --results runs/anchored/<model>` works once the
recipe-independent baselines are linked in):
  runs/anchored/<model>/  stage 1 (<px>_warm400) and its forked continuations, all with the anchor:
      fork_rl      RLCD-RL continuation = two-stage RLCD
      fork_ro      readout-only continuation          fork_anchor  anchor-only (no outcome reward)
      fork_grpo    GRPO continuation                   fork_rlnoanc RLCD-RL without the anchor in stage 2
      ce           stage 1 with the log score
  runs/backbones/<model>/ recipe-independent baselines (grpokl, rftkl, direct_lr2e-6), unchanged recipe.
Every flag mirrors scripts/common.sh; offload runs on MMLU-Pro use more GPU headroom and smaller generation
batches (the 16 GB / 32 setting runs out of GPU memory there) -- memory placement only, same math.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHOR = "--kl-tokens 0.04"
MODELS = {  # dir: (hf id, resource class, offload flags for gsm8kv, for mmlupro)
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "std", "", ""),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "std", "", ""),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B", "smol",
                   "--offload-optimizer --gpu-headroom-gb 16 --gen-batch 32",
                   "--offload-optimizer --gpu-headroom-gb 24 --gen-batch 16"),
    "phi-4-mini": ("microsoft/Phi-4-mini-instruct", "phi",
                   "--offload-optimizer --gpu-headroom-gb 16 --gen-batch 32",
                   "--offload-optimizer --gpu-headroom-gb 24 --gen-batch 16"),
}
TASKS = {  # task: (prefix, task args as in common.sh, direct-baseline test cap)
    "gsm8kv": ("gv", "--task gsm8kv --m 4 --m-eval 4 --max-new-tokens 320 --steps 400 --dev-cap 300 --test-cap 1260 "
                     "--diversity-weight 0", 0),
    "mmlupro": ("mp", "--task mmlupro --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 --dev-cap 300 --test-cap 1000 "
                      "--diversity-weight 0", 1000),
}
FORKS = {  # stage-2 continuations from the anchored stage-1 checkpoint
    "rl": f"--sf-coef 0.3 {ANCHOR}",
    "ro": f"--no-score-function {ANCHOR}",
    "anchor": f"--sf-coef 0.3 --sf-reader none {ANCHOR}",
    "grpo": f"--objective grpo {ANCHOR}",
    "rlnoanc": "--sf-coef 0.3",
}


def margs(model, task):
    hf, _, off_gv, off_mp = MODELS[model]
    off = off_gv if task == "gsm8kv" else off_mp
    return f"--model {hf} {off}".strip()


def line(name, out, model, needs, module, args):
    return f"{name} {out} {MODELS[model][1]} {needs} {module} {' '.join(args.split())}"


def stage1(model, task, s):
    px, A, _ = TASKS[task]
    out = f"runs/anchored/{model}"
    ck = f"{out}/{px}_warm400_s{s}.pt"
    return line(f"{px}_warm400_seed{s}", out, model, "-", "train",
                f"{A} {margs(model, task)} --kl-beta 0.04 --seed {s} --eval-every 400 --no-score-function {ANCHOR} "
                f"--save-checkpoint {ck}")


def fork(model, task, s, arm):
    px, A, _ = TASKS[task]
    out = f"runs/anchored/{model}"
    ck = f"{out}/{px}_warm400_s{s}.pt"
    return line(f"{px}_fork_{arm}_seed{s}", out, model, f"{out}/{px}_warm400_seed{s}.json", "train",
                f"{A} {margs(model, task)} --kl-beta 0.04 --seed {1000 + s} --eval-every 400 --init-checkpoint {ck} "
                f"{FORKS[arm]}")


def ce(model, task, s):
    px, A, _ = TASKS[task]
    return line(f"{px}_ce_seed{s}", f"runs/anchored/{model}", model, "-", "train",
                f"{A} {margs(model, task)} --kl-beta 0.04 --no-score-function --readout-score log --seed {s} "
                f"--eval-every 400 {ANCHOR}")


def baselines(model, task, s):
    px, A, cap = TASKS[task]
    out = f"runs/backbones/{model}"
    return [
        line(f"{px}_grpokl_seed{s}", out, model, "-", "train",
             f"{A} {margs(model, task)} --objective grpo --kl-beta 0.04 --seed {s} --eval-every 200"),
        line(f"{px}_rftkl_seed{s}", out, model, "-", "train",
             f"{A} {margs(model, task)} --objective rft --kl-beta 0.04 --seed {s} --eval-every 400"),
        line(f"{px}_direct_lr2e-6_seed{s}", out, model, "-", "train_direct",
             f"--task {task} {margs(model, task)} --labels fresh --lr 2e-6 --seed {s} --steps 400 --dev-cap 300 "
             f"--test-cap {cap}"),
    ]


def two_stage(model, task, s, arms=("rl",)):
    return [stage1(model, task, s)] + [fork(model, task, s, a) for a in arms]


TABLE2 = ("rl", "ro", "anchor", "grpo", "rlnoanc")


def plan():
    J = []
    # 1. stage-2 controls on the 1.7B seed-17 checkpoints from the validation runs (Table 2 under the anchor)
    J += [fork("qwen3-1.7b", "gsm8kv", 17, a) for a in ("ro", "anchor", "grpo")]
    J += two_stage("smollm3-3b", "gsm8kv", 17)
    J += [fork("qwen3-1.7b", "mmlupro", 17, a) for a in ("ro", "anchor", "grpo")]
    J += two_stage("smollm3-3b", "mmlupro", 17)
    # 2. seed-17 baselines still missing on the 3-4B backbones
    J += baselines("phi-4-mini", "mmlupro", 17) + baselines("smollm3-3b", "mmlupro", 17)
    # 3. the 1.7B main model, seed 23 (all Table-2 arms), then the 0.6B backbone (cheap)
    J += two_stage("qwen3-1.7b", "gsm8kv", 23, TABLE2) + two_stage("qwen3-1.7b", "mmlupro", 23, TABLE2)
    J += [x for t in TASKS for s in (17, 23, 29) for x in two_stage("qwen3-0.6b", t, s)]
    # 4. seeds 23 and 29 on the 3-4B backbones (baselines + anchored two-stage), interleaved with 1.7B seed 29
    J += baselines("phi-4-mini", "gsm8kv", 23) + two_stage("phi-4-mini", "gsm8kv", 23)
    J += baselines("smollm3-3b", "gsm8kv", 23) + two_stage("smollm3-3b", "gsm8kv", 23)
    J += two_stage("qwen3-1.7b", "gsm8kv", 29, TABLE2) + two_stage("qwen3-1.7b", "mmlupro", 29, TABLE2)
    for task, s in (("mmlupro", 23), ("gsm8kv", 29), ("mmlupro", 29)):
        J += baselines("phi-4-mini", task, s) + two_stage("phi-4-mini", task, s)
        J += baselines("smollm3-3b", task, s) + two_stage("smollm3-3b", task, s)
    # 5. stage 1 with the log score (1.7B), then the stage-2-without-anchor arm on the other backbones
    J += [ce("qwen3-1.7b", t, s) for t in TASKS for s in (17, 23, 29)]
    J += [fork(m, t, s, "rlnoanc") for m in ("qwen3-0.6b", "smollm3-3b", "phi-4-mini") for t in TASKS
          for s in (17, 23, 29)]
    return J


# validation runs (runs/klval, seed 17) that ARE anchored-recipe runs: stage 1, stage 2 with the anchor (rlkl ->
# fork_rl) and without it (rl -> fork_rlnoanc); the checkpoint only where it was kept
LINKS = {"warm400": "warm400", "fork_rlkl": "fork_rl", "fork_rl": "fork_rlnoanc"}


def link():
    for model in ("qwen3-1.7b", "phi-4-mini"):
        out = ROOT / "runs/anchored" / model
        out.mkdir(parents=True, exist_ok=True)
        for px in ("gv", "mp"):
            pairs = [(f"{px}_{a}_seed17", f"{px}_{b}_seed17") for a, b in LINKS.items()]
            files = [(f"{a}{ext}", f"{b}{ext}") for a, b in pairs for ext in (".json", ".log")]
            files.append((f"{px}_warm400_s17.pt", f"{px}_warm400_s17.pt"))
            for src, dst in files:
                d = out / dst
                if d.is_symlink() or d.exists():
                    continue
                if dst.endswith(".pt") and model == "phi-4-mini" and px == "gv":
                    continue  # deleted after its forks finished
                os.symlink(f"../../klval/{model}/{src}", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    if a.link:
        link()
    jobs = plan()
    names = [tuple(j.split()[:2]) for j in jobs]
    assert len(names) == len(set(names)), "duplicate job"
    for j in jobs:
        name, out = j.split()[:2]
        if (ROOT / out / f"{name}.json").is_symlink():
            continue  # produced by a validation run (runs/klval)
        print(j)


if __name__ == "__main__":
    main()

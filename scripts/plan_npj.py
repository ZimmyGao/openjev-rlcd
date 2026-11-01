#!/usr/bin/env python3
"""Job plans of the npj protocol (paper/npj/PROTOCOL.md) for scripts/jobqueue.py, in priority order.

    python3 scripts/plan_npj.py tune                 # tier 0: GRPO configurations on the validation split (seed 17)
    python3 scripts/plan_npj.py tier1                # Qwen3 arms that need no tuning result, fresh split
    python3 scripts/plan_npj.py tier2                # SmolLM3-3B / Phi-4-mini MMLU-Pro, stage-2 arms, fresh split
    python3 scripts/plan_npj.py grpo                 # GRPO+TS with the setting selected on val (selection_grpo.json)
    python3 scripts/plan_npj.py rlcr                 # RLCR+TS with the rule selected on val (selection_rlcr.json)
    python3 scripts/plan_npj.py credo_probe          # amendment A4: analysis-length probe of the zero-shot Qwen3 (dev)
    python3 scripts/plan_npj.py credo                # amendment A4: CREDO+TS, Qwen3 x both tasks (after credo_probe)
    python3 scripts/plan_npj.py probe                # thinking-mode length probe of the zero-shot Qwen3 models (dev)
    python3 scripts/plan_npj.py thinking             # amendment A3: thinking mode (B: training on Qwen3-1.7B GSM8K-Verify;
                                                     #   E: thinking-mode evaluation of the zero-shot and stage-1 weights)
    python3 scripts/plan_npj.py thinking_grpo        # the same for the selected GRPO+TS (after selection_grpo.json)
    python3 scripts/plan_npj.py transfer             # Q3b (amendment A1): the SVAMP-Verify build + eval-only on ARC
    python3 scripts/plan_npj.py transfer_svamp       # Q3b: eval-only on SVAMP-Verify (only after the build completed)
    python3 scripts/plan_npj.py chaos                # Q2, aleatoric (amendment A2): per-rationale vs mixture, ChaosNLI
Every job line is parsed with the current train.py / train_direct.py parser (and train.check_args) and compared with
the saved config of the paper run it mirrors: the only differences allowed are the run's split, its checkpoint path, its
output and the options that define the arm (ALLOWED below); anything else makes the script exit 1. The parity report
goes to stderr, the job lines to stdout:
    python3 scripts/plan_npj.py tier1 > lines && python3 scripts/jobqueue.py push < lines
Run files: runs/npj/<model>/ (tuning: runs/npj_tune/<model>/); final weights (all runs, kept): CKPT/<model>/<name>.pt.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFROOT = Path(os.environ.get("OPENJEV_REF_ROOT", ROOT))  # where the paper's run files are (runs/, results/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import plan_anchored as P  # noqa: E402  (MODELS, TASKS, margs: the paper's flags)

CKPT = "/data/Projects/openjev-rlcd/checkpoints"
QWEN = ("qwen3-0.6b", "qwen3-1.7b")
BIG = ("smollm3-3b", "phi-4-mini")
SEEDS = (17, 23, 29)
ANCHOR = "--kl-tokens 0.01"  # the recipe's token anchor (lambda = 0.01)
# the paper run each arm mirrors (format: model, task prefix, seed); None = no saved run of this kind for the model
REF = {
    "s1": "runs/anchored01/{m}/{px}_warm400_seed{s}.json",
    "fork": "runs/anchored01/{m}/{px}_fork_rl_seed{s}.json",
    "fork_grpo": "runs/anchored01/{m}/{px}_fork_grpo_seed{s}.json",
    "grpo": "results/{px}_grpokl_seed{s}.json",  # Qwen3-1.7B; the other backbones: runs/backbones/<m>/
    "direct": "results/{px}_direct_lr2e-6_seed{s}.json",
    "base": "results/{base}.json",
    "rlcr": "runs/extra_baselines/{m}/{px}_rlcr_seed{s}.json",
    "credo": "runs/extra_baselines/{m}/{px}_rlcr_seed{s}.json",  # no earlier CREDO run: the RLCR run it parallels
}
COMMON = {"output", "test_split", "save_checkpoint", "init_checkpoint", "label", "extra_splits",  # may differ
          "gen_batch", "gpu_headroom_gb"}  # memory placement only (generation batch size, optimizer state on the GPU)
# options absent from older run files take their defaults; they must then equal the default (checked below)
ALLOWED = {  # arm: options that define it (relative to the run it mirrors)
    "s1": set(), "s1_ce": {"readout_score"}, "s1_lin": {"readout_score"}, "s1_ref": {"rationale_source"},
    "s1_mix": {"diversity_weight"}, "fork_rl": set(), "fork_correct": {"sf_reward"}, "fork_grpo": set(),
    "grpo": {"kl_beta", "kl_tokens", "diversity_weight"},
    "direct": {"test_cap"},  # GSM8K-Verify: 1,260 (the whole old test split, and fresh) instead of 0 (= all), so that
                             # calib uses the same 1,260 items as the train.py arms
    "base": {"m", "steps", "diversity_weight"},  # training-only options, unused by --eval-only (v1 base runs: 8/800/1)
    "rlcr": {"rlcr_advantage", "diversity_weight"},
    "credo": {"objective", "kl_beta", "diversity_weight", "credo_analysis_tokens"},  # A4: no KL; settings train.CREDO
    "think_base": {"m", "steps", "diversity_weight", "thinking", "max_new_tokens", "m_eval"},
    "think_s1": {"thinking", "max_new_tokens", "m_eval"},
    "think_grpo": {"kl_beta", "kl_tokens", "diversity_weight", "thinking", "max_new_tokens", "m_eval"},
}
# amendment A3 (2026-10-19): thinking-mode training only on this cell (option B); every Qwen3 cell gets thinking-mode
# evaluations of its zero-shot and non-thinking weights (option E); single query (--m-eval 1); generation batches are
# memory placement only (32 rationales; training 16 if 32 does not fit)
THINK_TRAIN = ("qwen3-1.7b", "gsm8kv")
THINK_GEN_BATCH = 32


def ref_path(kind, m, px, s):
    if kind in ("grpo", "direct", "base") and m != "qwen3-1.7b":
        tmpl = {"grpo": "runs/backbones/{m}/{px}_grpokl_seed{s}.json",
                "direct": "runs/backbones/{m}/{px}_direct_lr2e-6_seed{s}.json",
                "base": "runs/backbones/{m}/{base}.json"}[kind]
    else:
        tmpl = REF[kind]
    base = "gv_base_eval_full" if px == "gv" else "mp_base_eval"
    return REFROOT / tmpl.format(m=m, px=px, s=s, base=base)


def ck(root, m, name):
    return f"{CKPT}/{root}/{m}/{name}.pt"


def line(name, out, m, needs, module, args):
    return f"{name} {out} {P.MODELS[m][1]} {needs} {module} {' '.join(args.split())}"


class Job:
    def __init__(self, arm, m, task, s, name, out, needs, module, args, ref_kind, ref_seed=None):
        self.arm, self.m, self.task, self.s, self.name = arm, m, task, s, name
        self.out, self.needs, self.module, self.args = out, needs, module, args
        self.ref_kind, self.ref_seed = ref_kind, s if ref_seed is None else ref_seed

    def text(self):
        return line(self.name, self.out, self.m, self.needs, self.module, self.args)


CALIB_ARMS = ("s1", "fork_rl", "grpo", "direct", "base")  # deployment analysis (section 6): calib in the same pass


def fresh_jobs(arm, m, task, s, extra=""):
    """one job of the fresh split; arm in ALLOWED"""
    px, A, cap = P.TASKS[task]
    out, root = f"runs/npj/{m}", "npj"
    ma = P.margs(m, task)
    fresh = "--test-split fresh" + (" --extra-splits calib" if arm in CALIB_ARMS else "")
    if arm in ("s1", "s1_ce", "s1_lin", "s1_ref", "s1_mix"):
        name = {"s1": "warm400", "s1_ce": "ce", "s1_lin": "lin", "s1_ref": "refrat", "s1_mix": "mix"}[arm]
        name = f"{px}_{name}_seed{s}"
        opt = {"s1": "", "s1_ce": "--readout-score log", "s1_lin": "--readout-score linear",
               "s1_ref": "--rationale-source ref", "s1_mix": "--diversity-weight 1"}[arm]
        args = (f"{A} {ma} --kl-beta 0.04 --seed {s} --eval-every 400 --no-score-function {ANCHOR} {opt} {fresh} "
                f"--save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, "-", "train", args, "s1")
    if arm in ("fork_rl", "fork_correct", "fork_grpo"):
        name = f"{px}_{arm}_seed{s}"
        s1 = f"{px}_warm400_seed{s}"
        opt = {"fork_rl": f"--sf-coef 0.3 {ANCHOR}", "fork_correct": f"--sf-coef 0.3 {ANCHOR} --sf-reward correct",
               "fork_grpo": f"--objective grpo {ANCHOR}"}[arm]
        args = (f"{A} {ma} --kl-beta 0.04 --seed {1000 + s} --eval-every 400 --init-checkpoint {ck(root, m, s1)} {opt} "
                f"{fresh} --save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, f"{out}/{s1}.json", "train", args,
                   "fork_grpo" if arm == "fork_grpo" else "fork")
    if arm == "grpo":
        name = f"{px}_grpo_seed{s}"
        args = (f"{A} {ma} --objective grpo {extra} --seed {s} --eval-every 200 {fresh} "
                f"--save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, "-", "train", args, "grpo")
    if arm == "rlcr":
        name = f"{px}_rlcr_seed{s}"
        args = (f"{A} {ma} --objective rlcr {extra} --kl-beta 0.04 --seed {s} {fresh} "
                f"--save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, "-", "train", args, "rlcr")
    if arm == "credo":
        name = f"{px}_credo_seed{s}"
        args = (f"{A} {ma} --objective credo --credo-analysis-tokens {credo_cap()} --seed {s} {fresh} "
                f"--save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, "-", "train", args, "credo")
    if arm == "direct":
        name = f"{px}_direct_lr2e-6_seed{s}"
        args = (f"--task {task} {ma} --labels fresh --lr 2e-6 --seed {s} --steps 400 --dev-cap 300 "
                f"--test-cap {cap or 1260} {fresh} --save-checkpoint {ck(root, m, name)}")
        return Job(arm, m, task, s, name, out, "-", "train_direct", args, "direct")
    if arm == "base":
        name = f"{px}_base_eval"
        return Job(arm, m, task, 17, name, out, "-", "train", f"{A} {ma} --eval-only --seed 17 {fresh}", "base")
    raise ValueError(arm)


CREDO_PROBE = "runs/npj/credo_probe"


def credo_probe_jobs():
    """amendment A4: the analysis-length probe of the zero-shot Qwen3 models on dev, one job per cell"""
    J = []
    for m in QWEN:
        for task in ("gsm8kv", "mmlupro"):
            px, A, _ = P.TASKS[task]
            cap = 320 if task == "gsm8kv" else 256  # the task's rationale cap
            assert f"--max-new-tokens {cap}" in A
            J.append(Job("credo_probe", m, task, 17, f"{px}_{m}", CREDO_PROBE, "-", "credo_probe",
                         f"--task {task} --model {P.MODELS[m][0]} --n 100 --m 4 --max-new-tokens {cap} --seed 17 "
                         f"--gen-batch 32", None))
    return J


def credo_cap():
    """A4's rule: the largest over the four Qwen3 cells of the cap the probe picked (the largest candidate for a cell
    that picked none)"""
    from openjev_rlcd.credo_probe import CAPS
    caps = []
    for m in QWEN:
        for task in ("gsm8kv", "mmlupro"):
            f = ROOT / CREDO_PROBE / f"{P.TASKS[task][0]}_{m}.json"
            if not f.exists() or '"test": {' not in f.read_text():
                raise SystemExit(f"{f} is missing: push plan credo_probe first and wait for it")
            caps.append(json.loads(f.read_text())["chosen_cap"] or max(CAPS))
    return max(caps)


def probe_jobs():
    J = []
    for m in QWEN:
        for task in ("gsm8kv", "mmlupro"):
            px = P.TASKS[task][0]
            J.append(Job("probe", m, task, 17, f"{px}_{m}", "runs/npj/think_probe", "-", "think_probe",
                         f"--task {task} --model {P.MODELS[m][0]} --n 300 --cap 2048 --seed 17 --gen-batch 32", None))
    return J


def think_cap(m, task):
    """the cap the protocol's rule picked in the probe (2,048 when fewer than 90% finish within 2,048)"""
    px = P.TASKS[task][0]
    d = json.loads((ROOT / f"runs/npj/think_probe/{px}_{m}.json").read_text())
    return d["chosen_cap"] or 2048


def think_args(m, task):
    """the task arguments with the probe's cap and the single query (amendment A3)"""
    px, A, _ = P.TASKS[task]
    cap = think_cap(m, task)
    A2 = A.replace("--max-new-tokens 256", f"--max-new-tokens {cap}").replace(
        "--max-new-tokens 320", f"--max-new-tokens {cap}").replace("--m-eval 4", "--m-eval 1")
    assert f"--max-new-tokens {cap}" in A2 and "--m-eval 1" in A2
    return px, A2


def think_jobs(grpo_extra=None):
    """amendment A3. Without grpo_extra: the zero-shot model in thinking mode on every Qwen3 cell (seed 17), stage 1
    trained in thinking mode on THINK_TRAIN (option B), and the non-thinking stage-1 weights of every Qwen3 cell
    evaluated in thinking mode (option E; each job waits for its source run). With grpo_extra (the selected flags): the
    same for GRPO+TS. Seeds 17/23/29; single query; final test split, once."""
    J, arm = [], "grpo" if grpo_extra is not None else "warm400"
    T = f"--thinking --gen-batch {THINK_GEN_BATCH} --test-split fresh"
    if grpo_extra is None:
        for m in QWEN[::-1]:
            for task in ("gsm8kv", "mmlupro"):
                px, A2 = think_args(m, task)
                J.append(Job("think_base", m, task, 17, f"{px}_think_base_eval", f"runs/npj/{m}", "-", "train",
                             f"{A2} {P.margs(m, task)} --eval-only --seed 17 {T}", "base"))
    m, task = THINK_TRAIN
    px, A2 = think_args(m, task)
    for s in SEEDS:  # option B: training in thinking mode
        name = f"{px}_think_{arm}_seed{s}"
        if grpo_extra is None:
            args = (f"{A2} {P.margs(m, task)} --kl-beta 0.04 --seed {s} --eval-every 400 --no-score-function {ANCHOR} {T} "
                    f"--save-checkpoint {ck('npj', m, name)}")
            J.append(Job("think_s1", m, task, s, name, f"runs/npj/{m}", "-", "train", args, "s1"))
        else:
            args = (f"{A2} {P.margs(m, task)} --objective grpo {grpo_extra} --seed {s} --eval-every 200 {T} "
                    f"--save-checkpoint {ck('npj', m, name)}")
            J.append(Job("think_grpo", m, task, s, name, f"runs/npj/{m}", "-", "train", args, "grpo"))
    for s in SEEDS:  # option E: the non-thinking weights evaluated in thinking mode
        for m in QWEN[::-1]:
            for task in ("gsm8kv", "mmlupro"):
                px, A2 = think_args(m, task)
                src = f"{px}_{arm}_seed{s}"
                J.append(Job("think_eval", m, task, s, f"{px}_thinkeval_{arm}_seed{s}", f"runs/npj/{m}",
                             f"runs/npj/{m}/{src}.json", "train",
                             f"{A2} {P.margs(m, task)} --eval-only --seed {s} {T} --checkpoint {ck('npj', m, src)}", None))
    return J


TRANSFER = {"mmlupro": ("arc", "--max-new-tokens 256 --dev-cap 299 --test-cap 1172"),
            "gsm8kv": ("svampv", "--max-new-tokens 320 --dev-cap 200 --test-cap 100000")}
SVAMP_BUILD = "runs/npj_transfer/svamp_build"


TRANSFER_SOURCES = (("warm400", "warm400"), ("ce", "ce"), ("grpo", "grpo"), ("fork_rl", "fork_rl"))


def svamp_built():
    """the SVAMP-Verify build finished (its summary is complete) and the data file is the one it summarized"""
    from openjev_rlcd.tasks import svamp_verify
    return svamp_verify.built(ROOT / f"{SVAMP_BUILD}.json")


def transfer_jobs(task):
    """section 4, transfer (amendment A1): the saved weights of base, SFT+TS, stage 1 (Brier, log), the selected GRPO+TS
    and two-stage (Qwen3, seeds 17/23/29) evaluated once on the new task of `task`'s models; each job waits for its
    source run. The temperature of the analysis is the source run's (in-domain dev); the target dev gives a secondary
    one. MMLU-Pro -> ARC-Challenge (plan transfer, which also builds SVAMP-Verify); GSM8K-Verify -> SVAMP-Verify (plan
    transfer_svamp): a queue job waits for one run file only, so these jobs are planned only once the build is complete
    and none of them can start without its data."""
    J = []
    if task == "mmlupro":
        J.append(Job("build", "qwen3-0.6b", "gsm8kv", 0, SVAMP_BUILD.split("/")[-1], "runs/npj_transfer", "-",
                     "tasks.svamp_verify", "--build", None))
    elif not svamp_built():
        raise SystemExit(f"SVAMP-Verify is not built yet ({SVAMP_BUILD}.json incomplete or data/svamp_verify.json "
                         "missing): push plan transfer first and wait for svamp_build")
    for m in QWEN:
        px, A, _ = P.TASKS[task]
        target, opts = TRANSFER[task]
        out = f"runs/npj_transfer/{m}"
        base = f"--task {target} --m 4 --m-eval 4 --steps 400 --diversity-weight 0 {opts} {P.margs(m, task)}"
        J.append(Job("transfer", m, task, 17, f"{target}_base_eval", out, "-", "train",
                     f"{base} --eval-only --seed 17", None))
        for s in SEEDS:
            for arm, stem in TRANSFER_SOURCES:
                src = f"{px}_{stem}_seed{s}"
                J.append(Job("transfer", m, task, s, f"{target}_{arm}_seed{s}", out, f"runs/npj/{m}/{src}.json",
                             "train", f"{base} --eval-only --seed {s} --checkpoint {ck('npj', m, src)}", None))
            src = f"{px}_direct_lr2e-6_seed{s}"
            dcap = "--dev-cap 299 --test-cap 0" if target == "arc" else "--dev-cap 200 --test-cap 0"
            J.append(Job("transfer", m, task, s, f"{target}_direct_seed{s}", out, f"runs/npj/{m}/{src}.json",
                         "train_direct", f"--task {target} {P.margs(m, task)} --labels fresh --eval-only --seed {s} "
                         f"{dcap} --checkpoint {ck('npj', m, src)}", None))
    return J


def chaos_jobs():
    """section 4, ChaosNLI (Q2, aleatoric labels): stage 1 with the token anchor, per-rationale (lambda = 0) vs mixture
    (lambda = 1), Qwen3-1.7B, seeds 17/23/29, with the paper's ChaosNLI settings (M = 8, 16 evaluation rationales, 128
    tokens, 800 steps). Its test split (400 items) is the one of the earlier paper's ChaosNLI table, which already
    compared the two objectives without the anchor: these results are a replication on previously read items (A2)."""
    J = []
    C = "--task chaosnli --m 8 --m-eval 16 --max-new-tokens 128 --steps 800 --dev-cap 199 --test-cap 400"
    m = "qwen3-1.7b"
    for s in SEEDS:
        for lam in ("0", "1"):
            name = f"cn_s1_lam{lam}_seed{s}"
            J.append(Job("chaos", m, "chaosnli", s, name, f"runs/npj/{m}", "-", "train",
                         f"{C} --model {P.MODELS[m][0]} --kl-beta 0.04 --kl-tokens 0.01 --no-score-function "
                         f"--diversity-weight {lam} --seed {s} --eval-every 400 --save-checkpoint {ck('npj', m, name)}",
                         None))
    return J


def tune_jobs():
    """GRPO configurations on the validation split (the old test split), seed 17, Qwen3 x both tasks"""
    J = []
    for m in QWEN:
        for task in ("gsm8kv", "mmlupro"):
            px, A, _ = P.TASKS[task]
            for tag, opt in (("b0", "--kl-beta 0"), ("b0.01", "--kl-beta 0.01"), ("b0.04", "--kl-beta 0.04"),
                             ("b0.1", "--kl-beta 0.1"), ("b0.04_kt0.01", f"--kl-beta 0.04 {ANCHOR}")):
                name = f"{px}_grpo_{tag}_seed17"
                args = (f"{A} {P.margs(m, task)} --objective grpo {opt} --seed 17 --eval-every 200 "
                        f"--save-checkpoint {ck('npj_tune', m, name)}")
                J.append(Job("grpo", m, task, 17, name, f"runs/npj_tune/{m}", "-", "train", args, "grpo"))
    return J


def tier1():
    """Qwen3, both tasks, seeds 17/23/29, fresh split: base, SFT, stage 1 (Brier, log, linear, offline rationales,
    mixture), interleaved so that every arm gets results early; the stage-2 forks of tier 2 need the stage-1 runs"""
    J = [fresh_jobs("base", m, t, 17) for m in QWEN for t in ("gsm8kv", "mmlupro")]
    for s in SEEDS:
        for m in QWEN[::-1]:
            for t in ("mmlupro", "gsm8kv"):
                J += [fresh_jobs(a, m, t, s) for a in ("s1", "s1_ce", "s1_ref", "s1_mix", "s1_lin", "direct")]
    return J


def tier2():
    """stage-2 continuations (Qwen3; the two controls on Qwen3-1.7B), then SmolLM3-3B / Phi-4-mini on MMLU-Pro"""
    J = []
    for s in SEEDS:
        for m in QWEN[::-1]:
            for t in ("mmlupro", "gsm8kv"):
                J.append(fresh_jobs("fork_rl", m, t, s))
                if m == "qwen3-1.7b":
                    J += [fresh_jobs("fork_correct", m, t, s), fresh_jobs("fork_grpo", m, t, s)]
    J += [fresh_jobs("base", m, "mmlupro", 17) for m in BIG]
    for s in SEEDS:
        for m in BIG:
            J += [fresh_jobs(a, m, "mmlupro", s) for a in ("s1", "s1_ce", "direct")]
    return J


def parse(module, cli):
    """the parsed namespace of a job's arguments, validated by train.check_args (train) or argparse (train_direct)"""
    from openjev_rlcd import train, train_direct
    mod = {"train": train, "train_direct": train_direct}[module]
    grabbed = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        ns = orig(self, args, namespace)
        grabbed.update(parser=self, ns=ns)
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    sys.argv = [module] + cli.split() + ["--output", "x.json"]
    try:
        mod.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    if "ns" not in grabbed:
        raise SystemExit(f"argument error in: {cli}")
    if module == "train":
        train.check_args(grabbed["parser"], grabbed["ns"])
    return vars(grabbed["ns"]), grabbed["parser"]


def selected(kind):
    """the flags selected on val (section 5), written by `python -m openjev_rlcd.npj_report select-<kind>`"""
    f = ROOT / "runs/npj_tune" / f"selection_{kind}.json"
    if not f.exists():
        raise SystemExit(f"{f} is missing: run the selection on val first (python -m openjev_rlcd.npj_report "
                         f"select-{kind})")
    d = json.loads(f.read_text())
    print(f"selected {kind} setting on val: {d['selected']} ({d['flags'] or 'defaults'})", file=sys.stderr)
    return d["flags"]


def check_transfer(j):
    """a transfer job evaluates its source run's final weights (or the zero-shot model) with the evaluation settings of
    the source task: model, seed, generation length, number of evaluation rationales, non-thinking mode, test split"""
    new, _ = parse(j.module, j.args)
    px, A, _ = P.TASKS[j.task]
    target = TRANSFER[j.task][0]
    src = j.needs.split("/")[-1].removesuffix(".json") if j.needs != "-" else None
    want = dict(task=target, model=P.MODELS[j.m][0], seed=j.s, eval_only=True, test_split="test", extra_splits="",
                checkpoint=ck("npj", j.m, src) if src else "")
    caps = dict(arc=(299, 1172), svampv=(200, 100000))[target]  # dev 299 / test 1,172; dev 200 / the rest
    if j.module == "train":
        src_args, _ = parse("train", f"{A} {P.margs(j.m, j.task)} --seed {j.s}")
        want.update(m_eval=src_args["m_eval"], max_new_tokens=src_args["max_new_tokens"], thinking=False,
                    gen_batch=src_args["gen_batch"], objective="rlcd", dev_cap=caps[0], test_cap=caps[1])
    else:  # train_direct: --test-cap 0 = the whole split
        want.update(dev_cap=caps[0], test_cap=0)
    arm = j.name.removeprefix(f"{target}_").rsplit("_seed", 1)[0]
    ok_src = (src is None and j.name == f"{target}_base_eval") or (
        src == f"{px}_{dict(direct='direct_lr2e-6').get(arm, arm)}_seed{j.s}"
        and arm in [a for a, _ in TRANSFER_SOURCES] + ["direct"] and j.needs == f"runs/npj/{j.m}/{src}.json"
        and j.module == ("train_direct" if arm == "direct" else "train"))
    wrong = {k: (new.get(k), v) for k, v in want.items() if new.get(k) != v}
    print(f"{j.m:11s} {j.name:28s} (transfer of {src or 'the zero-shot model'}; {j.module})"
          f"{'  <-- WRONG ' + str(wrong) if wrong or not ok_src else ''}", file=sys.stderr)
    return not wrong and ok_src


CHAOS_REF = {"0": "results/teacher3_single_seed17.json",  # the earlier ChaosNLI runs (lambda 0: seed 17 only)
             "1": "results/teacher2_rb_seed{s}.json"}
CHAOS_ALLOWED = {"kl_beta", "kl_tokens", "no_score_function", "eval_every", "seed"}  # stage 1 with the recipe's anchor;
# dev passes every 400 steps (the earlier runs: 200); the seed where the earlier lambda-0 run has seed 17 only


def check_chaos(j):
    """a ChaosNLI job has the earlier ChaosNLI runs' settings except the options of A2's arm (CHAOS_ALLOWED)"""
    new, parser = parse(j.module, j.args)
    lam = str(int(new["diversity_weight"]))
    rp = REFROOT / CHAOS_REF[lam].format(s=j.s)
    base = json.loads(rp.read_text())["config"]
    defaults = {a.dest: a.default for a in parser._actions}
    diff = {k: (base.get(k, defaults.get(k, "<absent>")), new.get(k)) for k in sorted(set(base) | set(new))
            if k not in COMMON and base.get(k, defaults.get(k, "<absent>")) != new.get(k)}
    unexpected = {k: v for k, v in diff.items() if k not in CHAOS_ALLOWED}
    want = dict(kl_beta=0.04, kl_tokens=0.01, no_score_function=True, seed=j.s, test_split="test", extra_splits="")
    wrong = {k: (new.get(k), v) for k, v in want.items() if new.get(k) != v}
    print(f"{j.m:11s} {j.name:28s} vs {rp.relative_to(REFROOT)}: {diff or 'identical'}"
          f"{'  <-- UNEXPECTED ' + str(list(unexpected)) if unexpected else ''}{'  <-- WRONG ' + str(wrong) if wrong else ''}",
          file=sys.stderr)
    return not unexpected and not wrong


def check(jobs):
    bad = False
    for j in jobs:
        if j.module == "think_probe":  # no paper run to mirror; the arguments are checked when the module starts
            print(f"{j.m:11s} {j.name:28s} (probe)", file=sys.stderr)
            continue
        if j.module == "credo_probe":  # no paper run to mirror; dev items only
            print(f"{j.m:11s} {j.name:28s} (credo probe: {j.args})", file=sys.stderr)
            continue
        if j.module == "tasks.svamp_verify":  # the SVAMP-Verify build: no run to mirror
            print(f"{j.m:11s} {j.name:28s} ({j.module} {j.args})", file=sys.stderr)
            continue
        if j.arm == "chaos":  # compared with the earlier ChaosNLI runs
            bad |= not check_chaos(j)
            continue
        if j.arm == "transfer":  # eval-only of a source run's weights: the source's evaluation settings and checkpoint
            bad |= not check_transfer(j)
            continue
        if j.arm == "think_eval":  # eval-only of saved weights in thinking mode: parsed and checked, nothing to mirror
            new, _ = parse(j.module, j.args)
            ok = (new["eval_only"] and new["thinking"] and new["test_split"] == "fresh" and new["m_eval"] == 1
                  and new["checkpoint"] == ck("npj", j.m, j.needs.split("/")[-1].removesuffix(".json")))
            print(f"{j.m:11s} {j.name:28s} (thinking eval of {new['checkpoint'].split('/')[-1]}){'' if ok else '  <-- WRONG'}",
                  file=sys.stderr)
            bad |= not ok
            continue
        new, parser = parse(j.module, j.args)
        px = P.TASKS[j.task][0]
        rp = ref_path(j.ref_kind, j.m, px, j.ref_seed)
        if not rp.exists():  # e.g. no fork_grpo run on this backbone: compare with the closest arm
            alt = {"fork_grpo": "fork"}.get(j.ref_kind)
            rp = ref_path(alt, j.m, px, j.ref_seed) if alt else rp
        if not rp.exists():
            print(f"{j.m} {j.name}: NO REFERENCE RUN {rp}", file=sys.stderr)
            bad = True
            continue
        base = json.loads(rp.read_text())["config"]
        defaults = {a.dest: a.default for a in parser._actions}
        diff = {}
        for k in sorted(set(base) | set(new)):
            if k in COMMON:
                continue
            b = base.get(k, defaults.get(k, "<absent>"))
            if b != new.get(k):
                diff[k] = (b, new.get(k))
        if j.ref_kind == "fork_grpo" and rp.name.startswith(f"{px}_fork_rl"):
            diff.pop("objective", None)
        if j.arm == "grpo" and j.m == "qwen3-1.7b" and "diversity_weight" in diff:  # unused by grpo (v1 runs: 1)
            diff.pop("diversity_weight")
        unexpected = {k: v for k, v in diff.items() if k not in ALLOWED[j.arm]}
        print(f"{j.m:11s} {j.name:28s} vs {rp.relative_to(REFROOT)}: {diff or 'identical'}"
              f"{'  <-- UNEXPECTED ' + str(list(unexpected)) if unexpected else ''}", file=sys.stderr)
        bad |= bool(unexpected)
        want_split = "test" if "npj_tune" in j.out else "fresh"
        if new.get("test_split") != want_split:
            print(f"  wrong split {new.get('test_split')}", file=sys.stderr)
            bad = True
        want_extra = "calib" if (j.arm in CALIB_ARMS and want_split == "fresh") else ""
        if new.get("extra_splits", "") != want_extra:
            print(f"  wrong extra splits {new.get('extra_splits')!r}, want {want_extra!r}", file=sys.stderr)
            bad = True
        if j.arm.startswith("fork"):  # a fork starts from its own stage-1 run's checkpoint
            px = P.TASKS[j.task][0]
            if new.get("init_checkpoint") != ck("npj", j.m, f"{px}_warm400_seed{j.s}") \
                    or j.needs != f"runs/npj/{j.m}/{px}_warm400_seed{j.s}.json":
                print(f"  wrong stage-1 checkpoint or dependency: {new.get('init_checkpoint')} / {j.needs}",
                      file=sys.stderr)
                bad = True
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", choices=["tune", "tier1", "tier2", "grpo", "rlcr", "credo_probe", "credo", "probe",
                                     "thinking", "thinking_grpo", "transfer", "transfer_svamp", "chaos"])
    a = ap.parse_args()
    if a.plan == "tune":
        jobs = tune_jobs()
    elif a.plan == "tier1":
        jobs = tier1()
    elif a.plan == "tier2":
        jobs = tier2()
    elif a.plan == "probe":
        jobs = probe_jobs()
    elif a.plan == "transfer":
        jobs = transfer_jobs("mmlupro")
    elif a.plan == "transfer_svamp":
        jobs = transfer_jobs("gsm8kv")
    elif a.plan == "chaos":
        jobs = chaos_jobs()
    elif a.plan == "thinking":
        jobs = think_jobs()
    elif a.plan == "thinking_grpo":
        jobs = think_jobs(selected("grpo"))
    elif a.plan == "credo_probe":
        jobs = credo_probe_jobs()
    elif a.plan == "credo":  # amendment A4: Qwen3 x both tasks, seeds 17/23/29, fresh split
        jobs = [fresh_jobs("credo", m, t, s) for s in SEEDS for m in QWEN for t in ("gsm8kv", "mmlupro")]
    elif a.plan == "grpo":
        extra = selected("grpo")
        jobs = [fresh_jobs("grpo", m, t, s, extra) for s in SEEDS for m in QWEN for t in ("gsm8kv", "mmlupro")]
        jobs += [fresh_jobs("grpo", m, "mmlupro", s, extra) for s in SEEDS for m in BIG]
    else:
        extra = selected("rlcr")
        jobs = [fresh_jobs("rlcr", m, t, s, extra) for s in SEEDS for m in QWEN for t in ("gsm8kv", "mmlupro")]
    names = [(j.out, j.name) for j in jobs]
    assert len(names) == len(set(names)), "duplicate job names"
    cks = [a.split("--save-checkpoint ")[1].split()[0] for a in (j.args for j in jobs) if "--save-checkpoint " in a]
    assert len(cks) == len(set(cks)), "two jobs would write the same checkpoint"
    bad = check(jobs)
    if bad:  # no job line leaves a plan that failed its checks
        print(f"{len(jobs)} jobs NOT printed: the plan failed its checks (see above)", file=sys.stderr)
        sys.exit(1)
    for j in jobs:
        print(j.text())
    print(f"{len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()

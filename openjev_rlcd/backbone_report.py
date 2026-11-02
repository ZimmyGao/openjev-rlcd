#!/usr/bin/env python3
"""Results report for the multi-backbone study and the anchored rerun: every arm on every backbone and task, after
temperature scaling on dev (same scoring as final_tables), paired tests on the test items, drift statistics of the
rationales, and the progress of the GPU job queue.

  python -m openjev_rlcd.backbone_report [--out runs/anchored/REPORT.md] [--no-cache]

Recipes (file names as in scripts/two_stage.sh / scripts/backbone.sh):
  old   the paper's stage 1 / two-stage: the KL term (beta 0.04) enters only as a score-function reward, so it is
        inactive in stage 1 (no score-function term) -- results/ for Qwen3-1.7B, runs/backbones/<model>/ otherwise
  0.01  main recipe: + the pathwise token-level KL anchor on the rationales (--kl-tokens 0.01) in both stages,
        runs/anchored01/<model>/ (scripts/plan_anchored.py --recipe a01)
  0.04  the same with anchor weight 0.04 (the first anchored rerun, now the anchor-weight ablation), runs/anchored/<model>/;
        <px>_warm400_kt0.0025_seed<s> there = stage 1 with anchor weight 0.0025
The recipe-independent baselines (base, SFT+TS, GRPO+TS, RFT+KL+TS) come from the old directories. Runs listed in
EXCLUDED (collapsed runs, see the rule there; decided 2026-10-11 / 2026-10-15) are left out of every mean and test and
named in the section.
A paired test uses only the seeds both arms have; per-item values are averaged over those seeds first (as in
final_tables), so the p-values reflect item sampling, not seed-to-seed variation.
"""
import argparse
import json
import os
import re
import statistics as st
import time
from pathlib import Path

import numpy as np

from . import evaluation as S
from .tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 23, 29)
MODELS = (("qwen3-0.6b", "Qwen3-0.6B"), ("qwen3-1.7b", "Qwen3-1.7B"), ("smollm3-3b", "SmolLM3-3B"),
          ("phi-4-mini", "Phi-4-mini"))
TASK_PX = (("gsm8kv", "gv", "GSM8K-Verify"), ("mmlupro", "mp", "MMLU-Pro"))
RO_NAME = {"single": "one", "mixture": "mix"}


def old_dir(model):
    return ROOT / "results" if model == "qwen3-1.7b" else ROOT / "runs/backbones" / model


# runs left out of every mean and paired test, with the reason (decided by the authors, 2026-10-11 and 2026-10-15).
# Rule (evaluation.collapsed, also applied by final_tables and to any run missing here): the policy's test rationales
# have collapsed -- mean <= 5 tokens or every rationale at the length limit with repetition >= 0.9. A scan of all 228
# rationale runs (2026-10-16) finds exactly these five, all RFT+KL+TS.
_SMOL_MP = "SmolLM3 MMLU-Pro RFT+KL: all three seeds collapsed (rationales of 2 tokens); excluded, so this cell has no RFT+KL+TS"
EXCLUDED = {"runs/backbones/smollm3-3b/gv_rftkl_seed17.json":
            "SmolLM3 GSM8K-V RFT+KL seed 17 collapsed (rationales of 2 tokens, accuracy 50%); excluded; seeds 23/29 are normal",
            "runs/backbones/smollm3-3b/mp_rftkl_seed17.json": _SMOL_MP,
            "runs/backbones/smollm3-3b/mp_rftkl_seed23.json": _SMOL_MP,
            "runs/backbones/smollm3-3b/mp_rftkl_seed29.json": _SMOL_MP,
            "runs/backbones/phi-4-mini/gv_rftkl_seed29.json":
            "Phi-4-mini GSM8K-V RFT+KL seed 29 collapsed (every rationale at the 320-token limit, repetition 0.99); excluded; seeds 17/23 are normal"}


def arms(model, px):
    """(key, label, directory, file stem -- '{s}' marks the seed; the base run has none --, kind)"""
    old, new, a04 = old_dir(model), ROOT / "runs/anchored01" / model, ROOT / "runs/anchored" / model
    base = "gv_base_eval_full" if px == "gv" else "mp_base_eval"
    return [
        ("base", "base (zero-shot, with reasoning)", old, base, "teacher"),
        ("sft", "SFT+TS", old, f"{px}_direct_lr2e-6_seed{{s}}", "direct"),
        ("grpo", "GRPO+TS", old, f"{px}_grpokl_seed{{s}}", "teacher"),
        ("rftkl", "RFT+KL+TS", old, f"{px}_rftkl_seed{{s}}", "teacher"),
        ("s1_old", "old stage 1", old, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl_old", "old two-stage", old, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("ro_old", "old s1 + readout-only training", old, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc_old", "old s1 + noise only, no reward", old, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo_old", "old s1 + GRPO", old, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce_old", "old stage 1 (log score)", old, f"{px}_ce_seed{{s}}", "teacher"),
        ("s1_04", "anchor 0.04 stage 1", a04, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl_04", "anchor 0.04 two-stage", a04, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("rlnoanc_04", "anchor 0.04 s1 + RLCD-RL (no anchor in stage 2)", a04, f"{px}_fork_rlnoanc_seed{{s}}", "teacher"),
        ("ro_04", "anchor 0.04 s1 + readout-only training", a04, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc_04", "anchor 0.04 s1 + noise only, no reward", a04, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo_04", "anchor 0.04 s1 + GRPO", a04, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce_04", "anchor 0.04 stage 1 (log score)", a04, f"{px}_ce_seed{{s}}", "teacher"),
        ("s1_0025", "anchor 0.0025 stage 1", a04, f"{px}_warm400_kt0.0025_seed{{s}}", "teacher"),
        ("s1", "stage 1", new, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl", "two-stage (s1 + RLCD-RL)", new, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("rlnoanc", "s1 + RLCD-RL (no anchor in stage 2)", new, f"{px}_fork_rlnoanc_seed{{s}}", "teacher"),
        ("ro", "s1 + readout-only training", new, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc", "s1 + noise only, no reward", new, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo", "s1 + GRPO", new, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce", "stage 1 (log score)", new, f"{px}_ce_seed{{s}}", "teacher"),
    ]


# (first, second, question): delta = first - second
PAIRS = [
    ("s1", "s1_old", "stage 1: new recipe (0.01) vs old recipe (no anchor)"),
    ("rl", "rl_old", "two-stage: new recipe (0.01) vs old recipe (no anchor)"),
    ("s1", "s1_04", "stage 1: anchor 0.01 vs 0.04"),
    ("rl", "rl_04", "two-stage: anchor 0.01 vs 0.04"),
    ("s1_0025", "s1", "stage 1: anchor 0.0025 vs 0.01"),
    ("s1_04", "s1_old", "stage 1: anchor 0.04 vs no anchor (old)"),
    ("rl", "s1", "does stage 2 (RLCD-RL) add anything"),
    ("rl", "rlnoanc", "the anchor's effect in stage 2"),
    ("rl", "ro", "RLCD-RL vs readout-only continuation"),
    ("rl", "anc", "RLCD-RL vs noise only, no reward"),
    ("rl", "fgrpo", "RLCD-RL vs GRPO continuation"),
    ("fgrpo", "s1", "does the GRPO continuation add anything"),
    ("s1", "ce", "stage 1: Brier vs log score"),
    ("rl", "sft", "two-stage vs SFT+TS"),
    ("rl", "grpo", "two-stage vs GRPO+TS"),
    ("rl", "rftkl", "two-stage vs RFT+KL+TS"),
    ("rl", "base", "two-stage vs base"),
    ("s1", "sft", "stage 1 vs SFT+TS"),
    ("s1", "grpo", "stage 1 vs GRPO+TS"),
    ("s1", "rftkl", "stage 1 vs RFT+KL+TS"),
    ("s1", "base", "stage 1 vs base"),
    ("rl_04", "s1_04", "anchor 0.04: stage-2 gain"),
    ("rl_04", "ro_04", "anchor 0.04: RLCD-RL vs readout-only continuation"),
    ("rl_04", "anc_04", "anchor 0.04: RLCD-RL vs noise only, no reward"),
    ("rl_04", "fgrpo_04", "anchor 0.04: RLCD-RL vs GRPO continuation"),
]


def run_files(directory, stem):
    seeds = SEEDS if "{s}" in stem else (None,)
    return [(s, directory / f"{stem.format(s=s)}.json") for s in seeds]


def load(path, kind):
    """(readout -> (test dists, dev dists), config seed, drift stats) of a complete run, else None."""
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if d.get("config", {}).get("test_split", "test") != "test":  # npj-protocol runs: openjev_rlcd.npj_report
        raise ValueError(f"{path} was evaluated on split {d['config']['test_split']!r}; this report scores 'test' only")
    if kind == "direct":
        if "test" not in d or "test_dists" not in d:
            return None
        return {"single": (d["test_dists"], d["dev_dists"])}, d["config"]["seed"], None
    t = d.get("test", {})
    if "dists" not in t or not isinstance(d.get("dev_dists"), dict):
        return None
    drift = dict(forced=t.get("forced_marker_rate"), **t.get("rationale", {}))
    if kind == "verbal":  # RLCR: the stated answer stays the decision (S.verbal_decision); single query only
        return ({"single": ([S.verbal_decision(p) for p in t["dists"]["single"]],
                            [S.verbal_decision(p) for p in d["dev_dists"]["single"]])}, d["config"]["seed"], drift)
    return {r: (t["dists"][r], d["dev_dists"][r]) for r in ("single", "mixture")}, d["config"]["seed"], drift


def score_arm(directory, stem, kind, n_test):
    """{readout: [(seed, raw, ts, T)]}, drift stats per seed, the files used"""
    out, drift, used = {}, [], []
    for s, p in run_files(directory, stem):
        if os.path.relpath(os.path.abspath(p), ROOT) in EXCLUDED:  # abspath: links are not followed
            continue
        r = load(p, kind) if p.exists() else None
        if r is None:
            continue
        ros, cseed, dr = r
        if dr and S.collapsed(dr):  # the same rule for runs not (yet) named in EXCLUDED
            print(f"warning: {p} meets the collapse rule but is not in EXCLUDED; left out")
            continue
        used.append(p)
        for ro, (test, dev) in ros.items():
            raw, ts, temp = S.scored(test[:n_test], dev, cseed)
            out.setdefault(ro, []).append((s, raw, ts, temp))
        if dr:
            drift.append(dr)
    return out, drift, used


def pack(runs):
    ts = [r[2] for r in runs]
    return dict(brier=np.mean([t["per_item_brier"] for t in ts], 0), correct=np.mean([t["per_item_correct"] for t in ts], 0),
                seeds=[(np.array(t["per_item_rank"]), 1 - np.array(t["per_item_correct"], float)) for t in ts])


def common(ra, rb):
    sa, sb = {r[0] for r in ra}, {r[0] for r in rb}
    if None in sa:
        return ra, rb, sorted(sb)
    if None in sb:
        return ra, rb, sorted(sa)
    keep = sa & sb
    return [r for r in ra if r[0] in keep], [r for r in rb if r[0] in keep], sorted(keep)


def compare(A, B, ro):
    """paired deltas (first - second) on the test items for one readout; None if the arms share no seed"""
    rb_name = "single" if "mixture" not in B else ro
    if ro not in A or rb_name not in B:
        return None
    ra, rb, seeds = common(A[ro], B[rb_name])
    if not ra or not rb:
        return None
    x, y = pack(ra), pack(rb)
    db, da, du = S.paired(x["brier"], y["brier"]), S.paired(x["correct"], y["correct"]), S.paired_aurc(x, y)
    return dict(seeds=seeds, dbrier=list(map(float, db)), dacc=list(map(float, da)), daurc=list(map(float, du)))


def mean_sd(vals, fmt, scale=1.0):
    vals = [v * scale for v in vals]
    return fmt.format(st.mean(vals)) + (("±" + fmt.format(st.pstdev(vals))) if len(vals) > 1 else "")


def seeds_str(seeds):
    return ",".join(str(s) for s in seeds if s is not None) or "—"


def pval(p):
    return "<1e-4" if p < 1e-4 else f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def star(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def delta_cell(c, key, fmt, scale=1.0):
    v, lo, hi, p = c[key]
    return f"{fmt.format(v * scale)}{star(p)} (p={pval(p)})"


def section(model, label, task, px, tname, n_test):
    scored, drift, files = {}, {}, []
    meta = {k: (lab, kind) for k, lab, _, _, kind in arms(model, px)}
    for key, lab, d, stem, kind in arms(model, px):
        res, dr, used = score_arm(d, stem, kind, n_test)
        if res:
            scored[key], drift[key] = res, dr
            files += used
    lines = [f"### {label} · {tname}", ""]
    notes = list(dict.fromkeys(why for f, why in EXCLUDED.items() if f.startswith(f"runs/backbones/{model}/{px}_")))
    lines += [f"Note: {x}." for x in notes] + ([""] if notes else [])
    if not scored:
        return lines + ["(no results yet)", ""], {}, {}, []
    lines += ["| arm | seeds | readout | accuracy % | Brier | ECE | AURC | coverage@5% | oracle coverage@5% | raw T |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    metrics = {}
    for key, res in scored.items():
        lab = meta[key][0]
        for ro in ("single", "mixture"):
            if ro not in res:
                continue
            runs = res[ro]
            ts = [r[2] for r in runs]
            m = {k: st.mean(t[k] for t in ts) for k in ("acc", "brier", "nll", "ece", "aurc", "cov@0.05", "oracle_cov@0.05")}
            m["T"] = st.mean(r[3] for r in runs)
            m["seeds"] = [r[0] for r in runs]
            metrics[f"{key}/{RO_NAME[ro]}"] = m
            lines.append(f"| {lab} | {seeds_str(m['seeds'])} | {RO_NAME[ro]} | {mean_sd([t['acc'] for t in ts], '{:.1f}', 100)} | "
                         f"{mean_sd([t['brier'] for t in ts], '{:.4f}')} | {mean_sd([t['ece'] for t in ts], '{:.3f}')} | "
                         f"{mean_sd([t['aurc'] for t in ts], '{:.4f}')} | {mean_sd([t['cov@0.05'] for t in ts], '{:.0f}', 100)} | "
                         f"{mean_sd([t['oracle_cov@0.05'] for t in ts], '{:.0f}', 100)} | {m['T']:.2f} |")
    rows = [(k, v) for k, v in drift.items() if v]
    if rows:
        lines += ["", "Rationales (test set, mean over seeds):", "",
                  "| arm | no \"Answer:\" | mean tokens | at the length limit | repetition |", "|---|---|---|---|---|"]
        for key, dr in rows:
            g = lambda f: st.mean(x[f] for x in dr) if all(x.get(f) is not None for x in dr) else None
            vals = [g("forced"), g("mean_rationale_tokens"), g("length_limit_rate"), g("repetition_ratio")]
            fmt = ["{:.1%}", "{:.0f}", "{:.1%}", "{:.2f}"]
            lines.append(f"| {meta[key][0]} | " + " | ".join("—" if v is None else f.format(v) for v, f in zip(vals, fmt)) + " |")
    pairs = {}
    plines = []
    for a, b, q in PAIRS:
        if a not in scored or b not in scored:
            continue
        for ro in ("single", "mixture"):
            if meta[b][1] == "direct" and ro == "mixture":
                continue
            c = compare(scored[a], scored[b], ro)
            if c is None:
                continue
            pairs[f"{a}-{b}/{RO_NAME[ro]}"] = c
            plines.append(f"| {q} | {RO_NAME[ro]} | {seeds_str(c['seeds'])} | {delta_cell(c, 'dacc', '{:+.1f}', 100)} | "
                          f"{delta_cell(c, 'dbrier', '{:+.4f}')} | {delta_cell(c, 'daurc', '{:+.4f}')} |")
    if plines:
        lines += ["", "Paired tests (Δ = first − second; negative Brier/AURC = first is better; * p<0.05, ** p<0.01):", "",
                  "| comparison | readout | seeds | Δaccuracy (pp) | ΔBrier | ΔAURC |", "|---|---|---|---|---|---|"] + plines
    return lines + [""], metrics, pairs, files


def summary_rows(results):
    head = ["| model · task | seeds (s1 / two-stage) | s1: 0.01 − 0.04 ΔAURC | s1: new − old ΔAURC | "
            "stage-2 gain (two-stage − s1) ΔAURC | two-stage vs SFT+TS ΔBrier / ΔAURC | two-stage vs GRPO+TS ΔBrier / ΔAURC | "
            "two-stage vs RFT+KL+TS ΔBrier / ΔAURC |",
            "|---|---|---|---|---|---|---|---|"]
    rows = []
    for (label, tname), (metrics, pairs) in results.items():
        def one(a, b, both=False):
            c = pairs.get(f"{a}-{b}/one")
            if c is None:
                return "—"
            u = f"{c['daurc'][0]:+.4f}{star(c['daurc'][3])}"
            return f"{c['dbrier'][0]:+.4f}{star(c['dbrier'][3])} / {u}" if both else u
        s1 = metrics.get("s1/one", {}).get("seeds", [])
        rl = metrics.get("rl/one", {}).get("seeds", [])
        rows.append(f"| {label} · {tname} | {seeds_str(s1)} / {seeds_str(rl)} | {one('s1', 's1_04')} | {one('s1', 's1_old')} | "
                    f"{one('rl', 's1')} | "
                    f"{one('rl', 'sft', True)} | {one('rl', 'grpo', True)} | {one('rl', 'rftkl', True)} |")
    return head + rows


STATUS = re.compile(r"^(\d\d-\d\d \d\d:\d\d) gpu(\d) (start|end)\s+(\S+)")
DEFAULT_MIN = {"qwen3-0.6b": 50, "qwen3-1.7b": 90, "smollm3-3b": 185, "phi-4-mini": 240}


def progress():
    q = ROOT / "runs/queue"
    read = lambda p: p.read_text().splitlines() if p.exists() else []
    queued = [x.split() for x in read(q / "queue.txt") if x.strip() and not x.startswith("#")]
    running = [x.split() for x in read(q / "running.txt") if x.strip()]
    hist = read(q / "history.txt")
    done = sum(x.endswith("ok=1") for x in hist)
    failed = sum(x.endswith("ok=0") for x in hist)
    year = time.localtime().tm_year
    t = lambda s: time.mktime(time.strptime(f"{year}-{s}", "%Y-%m-%d %H:%M"))
    started, dur = {}, {}
    for x in read(q / "status.txt"):
        m = STATUS.match(x)
        if not m:
            continue
        if m[3] == "start":
            started[m[4]] = t(m[1])
        elif m[4] in started:
            dur.setdefault(job_class(m[4]), []).append((t(m[1]) - started.pop(m[4])) / 60)

    def est(path):
        c = job_class(path)
        if c in dur:
            return st.mean(dur[c])
        model = path.split("/")[2] if path.count("/") >= 3 else ""
        return DEFAULT_MIN.get(model, 120) * (0.5 if c[1] == "direct" else 1.0)

    remaining = sum(est(f"{j[1]}/{j[0]}") for j in queued)
    now = time.time()
    for r in running:  # <gpu> <pid> <start epoch> <name> <out> ...
        remaining += max(0.0, est(f"{r[4]}/{r[3]}") - (now - int(r[2])) / 60)
    eta = time.strftime("%m-%d %H:%M", time.localtime(now + remaining / 2 * 60))
    lines = ["## Queue progress", "",
             f"{done} done, {failed} failed, {len(running)} running, {len(queued)} queued. "
             f"At the mean duration of the finished runs, about {remaining / 60 / 2:.0f} h remain (two GPUs); all done around {eta}.", ""]
    for r in running:
        lines.append(f"- GPU{r[0]}: {r[4]}/{r[3]} (running for {(now - int(r[2])) / 60:.0f} min)")
    by_model = {}
    for j in queued:
        m = j[1].split("/")[2] if j[1].count("/") >= 2 else j[1]
        by_model[m] = by_model.get(m, 0) + 1
    if by_model:
        lines.append("- queued, by model: " + ", ".join(f"{m} {n}" for m, n in by_model.items()))
    return lines + [""]


def job_class(path):
    """(model directory, kind) used to estimate a job's duration"""
    parts = path.split("/")
    model = parts[2] if len(parts) >= 4 else ""
    name = parts[-1]
    kind = "direct" if "_direct" in name else "stage1" if ("warm400" in name or "_ce_" in name) else \
        "fork" if "_fork_" in name else "baseline"
    task = name.split("_")[0]
    return model, kind, task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/anchored/REPORT.md"))
    ap.add_argument("--json", default=str(ROOT / "runs/anchored/report.json"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    cache_path = ROOT / "runs/anchored/.report_cache.json"
    try:
        cache = {} if args.no_cache else json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    # cached sections are reused only while this file and the scoring code (evaluation.py, metrics.py) are unchanged
    code = [Path(f).stat().st_mtime_ns for f in (__file__, S.__file__, Path(S.__file__).with_name("metrics.py"))]
    body, summary, out_json = [], {}, {}
    for task, px, tname in TASK_PX:
        S.SPLITS = TASKS[task]["load"]()
        n_test = len(S.SPLITS["test"])
        for model, label in MODELS:
            cand = sorted({str(p) for _, _, d, stem, _ in arms(model, px) for _, p in run_files(d, stem) if p.exists()})
            sig = [code, n_test] + [[p, os.stat(p).st_mtime_ns, os.stat(p).st_size] for p in cand]
            key = f"{model}/{task}"
            if cache.get(key, {}).get("sig") != sig:
                lines, metrics, pairs, _ = section(model, label, task, px, tname, n_test)
                cache[key] = dict(sig=sig, lines=lines, metrics=metrics, pairs=pairs)
            c = cache[key]
            body += c["lines"]
            if c["metrics"]:
                summary[(label, tname)] = (c["metrics"], c["pairs"])
            out_json[key] = dict(metrics=c["metrics"], pairs=c["pairs"])
    head = ["# Multi-backbone results (anchored rerun)", "",
            f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Every number is after temperature scaling (TS) on dev; arms with several seeds give "
            "mean ± s.d. \"one\" = the readout after one rationale, \"mix\" = the mixture of 4 readouts. Stage 1, two-stage and controls without a prefix "
            "are the main recipe: token-level KL anchor 0.01 (runs/anchored01, fixed 2026-10-11); \"anchor 0.04\" is the first anchored rerun "
            "(runs/anchored), now an ablation of the anchor weight; \"old\" = the paper's original recipe (no effective KL term in stage 1). Paired tests use only the seeds both arms have; "
            "p values reflect the sampling of the test items, not variation between seeds; read single-seed results with care.", ""]
    head += progress()
    head += ["## Overview (single-rationale readout; negative Δ = first is better; * p<0.05, ** p<0.01)", ""] + summary_rows(summary) + [""]
    text = "\n".join(head + ["## By model and task", ""] + body) + "\n"
    for path, content in ((Path(args.out), text), (Path(args.json), json.dumps(out_json, indent=1)),
                          (cache_path, json.dumps(cache))):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, path)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

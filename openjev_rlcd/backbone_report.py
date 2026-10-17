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


# runs left out of every mean and paired test, with the reason (decided with the user, 2026-10-11 and 2026-10-15).
# Rule (evaluation.collapsed, also applied by final_tables and to any run missing here): the policy's test rationales
# have collapsed -- mean <= 5 tokens or every rationale at the length limit with repetition >= 0.9. A scan of all 228
# rationale runs (2026-10-16) finds exactly these five, all RFT+KL+TS.
_SMOL_MP = "SmolLM3 MMLU-Pro RFT+KL 三个 seed 全部塌缩（推理只剩 2 个 token），按决定排除，这一格没有 RFT+KL+TS"
EXCLUDED = {"runs/backbones/smollm3-3b/gv_rftkl_seed17.json":
            "SmolLM3 GSM8K-V RFT+KL seed 17 塌缩（推理只剩 2 个 token，准确率 50%），按决定排除；seed 23/29 正常",
            "runs/backbones/smollm3-3b/mp_rftkl_seed17.json": _SMOL_MP,
            "runs/backbones/smollm3-3b/mp_rftkl_seed23.json": _SMOL_MP,
            "runs/backbones/smollm3-3b/mp_rftkl_seed29.json": _SMOL_MP,
            "runs/backbones/phi-4-mini/gv_rftkl_seed29.json":
            "Phi-4-mini GSM8K-V RFT+KL seed 29 塌缩（每条推理都写到 320 token 上限、重复率 0.99），按决定排除；seed 17/23 正常"}


def arms(model, px):
    """(key, label, directory, file stem -- '{s}' marks the seed; the base run has none --, kind)"""
    old, new, a04 = old_dir(model), ROOT / "runs/anchored01" / model, ROOT / "runs/anchored" / model
    base = "gv_base_eval_full" if px == "gv" else "mp_base_eval"
    return [
        ("base", "base（零样本推理）", old, base, "teacher"),
        ("sft", "SFT+TS", old, f"{px}_direct_lr2e-6_seed{{s}}", "direct"),
        ("grpo", "GRPO+TS", old, f"{px}_grpokl_seed{{s}}", "teacher"),
        ("rftkl", "RFT+KL+TS", old, f"{px}_rftkl_seed{{s}}", "teacher"),
        ("s1_old", "旧 stage 1", old, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl_old", "旧 两阶段", old, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("ro_old", "旧 s1 + 只训读出", old, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc_old", "旧 s1 + 只有噪声无奖励", old, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo_old", "旧 s1 + GRPO", old, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce_old", "旧 stage 1（log score）", old, f"{px}_ce_seed{{s}}", "teacher"),
        ("s1_04", "锚点 0.04 stage 1", a04, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl_04", "锚点 0.04 两阶段", a04, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("rlnoanc_04", "锚点 0.04 s1 + RLCD-RL（stage 2 无锚点）", a04, f"{px}_fork_rlnoanc_seed{{s}}", "teacher"),
        ("ro_04", "锚点 0.04 s1 + 只训读出", a04, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc_04", "锚点 0.04 s1 + 只有噪声无奖励", a04, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo_04", "锚点 0.04 s1 + GRPO", a04, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce_04", "锚点 0.04 stage 1（log score）", a04, f"{px}_ce_seed{{s}}", "teacher"),
        ("s1_0025", "锚点 0.0025 stage 1", a04, f"{px}_warm400_kt0.0025_seed{{s}}", "teacher"),
        ("s1", "stage 1", new, f"{px}_warm400_seed{{s}}", "teacher"),
        ("rl", "两阶段（s1 + RLCD-RL）", new, f"{px}_fork_rl_seed{{s}}", "teacher"),
        ("rlnoanc", "s1 + RLCD-RL（stage 2 无锚点）", new, f"{px}_fork_rlnoanc_seed{{s}}", "teacher"),
        ("ro", "s1 + 只训读出", new, f"{px}_fork_ro_seed{{s}}", "teacher"),
        ("anc", "s1 + 只有噪声无奖励", new, f"{px}_fork_anchor_seed{{s}}", "teacher"),
        ("fgrpo", "s1 + GRPO", new, f"{px}_fork_grpo_seed{{s}}", "teacher"),
        ("ce", "stage 1（log score）", new, f"{px}_ce_seed{{s}}", "teacher"),
    ]


# (first, second, question): delta = first - second
PAIRS = [
    ("s1", "s1_old", "stage 1：新配方（0.01）vs 旧配方（无锚点）"),
    ("rl", "rl_old", "两阶段：新配方（0.01）vs 旧配方（无锚点）"),
    ("s1", "s1_04", "stage 1：锚点 0.01 vs 0.04"),
    ("rl", "rl_04", "两阶段：锚点 0.01 vs 0.04"),
    ("s1_0025", "s1", "stage 1：锚点 0.0025 vs 0.01"),
    ("s1_04", "s1_old", "stage 1：锚点 0.04 vs 无锚点（旧）"),
    ("rl", "s1", "stage 2（RLCD-RL）有没有增益"),
    ("rl", "rlnoanc", "stage 2 里锚点的作用"),
    ("rl", "ro", "RLCD-RL vs 只训读出续训"),
    ("rl", "anc", "RLCD-RL vs 只有噪声无奖励"),
    ("rl", "fgrpo", "RLCD-RL vs GRPO 续训"),
    ("fgrpo", "s1", "GRPO 续训有没有增益"),
    ("s1", "ce", "stage 1：Brier vs log score"),
    ("rl", "sft", "两阶段 vs SFT+TS"),
    ("rl", "grpo", "两阶段 vs GRPO+TS"),
    ("rl", "rftkl", "两阶段 vs RFT+KL+TS"),
    ("rl", "base", "两阶段 vs base"),
    ("s1", "sft", "stage 1 vs SFT+TS"),
    ("s1", "grpo", "stage 1 vs GRPO+TS"),
    ("s1", "rftkl", "stage 1 vs RFT+KL+TS"),
    ("s1", "base", "stage 1 vs base"),
    ("rl_04", "s1_04", "锚点 0.04：stage 2 增益"),
    ("rl_04", "ro_04", "锚点 0.04：RLCD-RL vs 只训读出续训"),
    ("rl_04", "anc_04", "锚点 0.04：RLCD-RL vs 只有噪声无奖励"),
    ("rl_04", "fgrpo_04", "锚点 0.04：RLCD-RL vs GRPO 续训"),
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
    lines += [f"注：{x}。" for x in notes] + ([""] if notes else [])
    if not scored:
        return lines + ["（还没有结果）", ""], {}, {}, []
    lines += ["| 组 | seeds | 读出 | 准确率 % | Brier | ECE | AURC | 自动化覆盖@5% | oracle 覆盖@5% | 原始 T |",
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
        lines += ["", "推理文本（测试集，seed 平均）：", "",
                  "| 组 | 不写 \"Answer:\" | 平均 token 数 | 触到长度上限 | 重复率 |", "|---|---|---|---|---|"]
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
        lines += ["", "配对检验（Δ = 前者 − 后者；Brier/AURC 为负 = 前者更好；* p<0.05，** p<0.01）：", "",
                  "| 比较 | 读出 | seeds | Δ准确率 (pp) | ΔBrier | ΔAURC |", "|---|---|---|---|---|---|"] + plines
    return lines + [""], metrics, pairs, files


def summary_rows(results):
    head = ["| 模型 · 任务 | seeds（s1 / 两阶段） | s1：0.01 − 0.04 ΔAURC | s1：新 − 旧 ΔAURC | "
            "stage 2 增益（两阶段 − s1）ΔAURC | 两阶段 vs SFT+TS ΔBrier / ΔAURC | 两阶段 vs GRPO+TS ΔBrier / ΔAURC | "
            "两阶段 vs RFT+KL+TS ΔBrier / ΔAURC |",
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
    lines = ["## 队列进度", "",
             f"已完成 {done} 个，失败 {failed} 个，正在跑 {len(running)} 个，排队 {len(queued)} 个。"
             f"按已完成实验的平均耗时粗估，剩余约 {remaining / 60 / 2:.0f} 小时（两张卡），预计 {eta} 左右全部跑完。", ""]
    for r in running:
        lines.append(f"- GPU{r[0]}：{r[4]}/{r[3]}（已跑 {(now - int(r[2])) / 60:.0f} 分钟）")
    by_model = {}
    for j in queued:
        m = j[1].split("/")[2] if j[1].count("/") >= 2 else j[1]
        by_model[m] = by_model.get(m, 0) + 1
    if by_model:
        lines.append("- 排队中按模型：" + "，".join(f"{m} {n} 个" for m, n in by_model.items()))
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
    head = ["# 多 backbone 结果汇总（锚点重跑）", "",
            f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}。所有数字都在 dev 上做过温度缩放（TS）；每组多个 seed 时给出 "
            "均值±标准差。\"one\" = 单条推理的读出，\"mix\" = 4 条推理混合的读出。不带前缀的 stage 1 / 两阶段 / 对照"
            "都是主配方：token 级 KL 锚点 0.01（runs/anchored01，2026-10-11 定）；\"锚点 0.04\" 是第一次锚点重跑"
            "（runs/anchored），现作锚点强度消融；\"旧\" = 论文原配方（stage 1 里 KL 项不起作用）。配对检验只用两组共有的 seed，"
            "p 值反映的是测试题目的抽样误差，不包含 seed 间的差异；只有一个 seed 时要谨慎解读。", ""]
    head += progress()
    head += ["## 总览（单条推理读出；Δ 为负 = 前者更好；* p<0.05，** p<0.01）", ""] + summary_rows(summary) + [""]
    text = "\n".join(head + ["## 分模型、分任务明细", ""] + body) + "\n"
    for path, content in ((Path(args.out), text), (Path(args.json), json.dumps(out_json, indent=1)),
                          (cache_path, json.dumps(cache))):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, path)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

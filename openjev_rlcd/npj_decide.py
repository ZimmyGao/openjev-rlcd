"""The decision rules of the npj protocol (paper/npj/PROTOCOL.md, section 8, with amendments A1, A2 and A5) on the
fresh-split runs, with the statistics of section 7: paired tests (npj_report.paired_diff) with Holm within each family,
the per-seed sign agreement of every difference and the seed-and-item hierarchical bootstrap (robustness check).

Run files, arm names as in scripts/plan_npj.py:
  runs/npj/<model>/<px>_<arm>_seed<s>.json
    warm400 (stage 1, Brier)  ce (log score)  lin (linear)  refrat (offline rationales)  mix (mixture objective)
    grpo (selected GRPO+TS)   direct_lr2e-6 (SFT+TS)       fork_rl (two-stage)  fork_correct  fork_grpo
    rlcr (RLCR+TS)            credo (CREDO+TS, A4)         closest methods: reported, no claim rests on them
  runs/npj_transfer/<model>/<target>_<arm>_seed<s>.json      transfer (A1): target arc (from mp) or svampv (from gv)
  runs/npj/qwen3-1.7b/cn_s1_lam<0|1>_seed<s>.json           ChaosNLI (A2)
    python -m openjev_rlcd.npj_decide [--root runs/npj] [--out paper/npj/decisions.json]
A family is decided only when every run it needs is present and complete; otherwise it is reported as pending.
Exclusion (section 6): a seed whose test rationales collapsed (evaluation.collapsed) is left out of its arm, and in a
transfer comparison also a seed whose source run collapsed. A cell where every seed of an arm is left out has no
comparison: it is outside the Holm family and never a win; a condition that must hold in every cell fails there; for
claim C see A5 and npj_report.decide_central. ChaosNLI has no exclusion (A2). Next to each deciding family, accuracy,
NLL and E-AURC are reported, each in a family of its own (A5). Datasets are read offline (the cached revisions).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import npj_report as R

QWEN = ("qwen3-0.6b", "qwen3-1.7b")
BIG = ("smollm3-3b", "phi-4-mini")
SEEDS = (17, 23, 29)
MP_CELLS = [(m, "mp") for m in QWEN + BIG]
QWEN_CELLS = [(m, px) for m in QWEN for px in ("gv", "mp")]
CELLS_C = MP_CELLS + [(m, "gv") for m in QWEN]
SOURCE_PX = {"arc": "mp", "svampv": "gv"}  # transfer target -> the task its models were trained on
TRANSFER_CELLS = [(m, t) for m in QWEN for t in ("arc", "svampv")]
SOURCE_NAME = {"direct": "direct_lr2e-6"}  # transfer arm -> the name of its source run's arm
SECONDARY = ("acc", "nll", "eaurc")  # reported next to every deciding family, each in a family of its own (A5)
HIGHER_IS_BETTER = ("acc", "majority_acc")
NAN = (float("nan"),) * 3 + (1.0,)

_CACHE = {}


def complete(p):
    return p.exists() and '"test": {' in p.read_text()


def cached(fn, *paths):
    key = (fn.__name__,) + tuple(str(p) for p in paths)
    if key not in _CACHE:
        _CACHE[key] = fn(*paths)
    return _CACHE[key]


def runs(root, model, px, arm):
    """{seed: score_run output} of the seeds that did not collapse (section 6: the only exclusion); None if a seed is
    missing or unfinished; {} when every seed collapsed"""
    out = {}
    for s in SEEDS:
        p = Path(root) / model / f"{px}_{arm}_seed{s}.json"
        if not complete(p):
            return None
        x = cached(R.score_run, p)
        if not x["single"]["collapsed"]:
            out[s] = x
    return out


def transfer_runs(root, model, target, arm):
    """{seed: score_transfer output} of one arm on a transfer task (A1; base: seed 17 only): a seed is left out when its
    transfer run or its source run collapsed; None if a run is missing or unfinished"""
    troot, px = Path(root).parent / "npj_transfer", SOURCE_PX[target]
    pairs = ([(17, f"{target}_base_eval", f"{px}_base_eval")] if arm == "base" else
             [(s, f"{target}_{arm}_seed{s}", f"{px}_{SOURCE_NAME.get(arm, arm)}_seed{s}") for s in SEEDS])
    out = {}
    for s, t, src in pairs:
        tp, sp = troot / model / f"{t}.json", Path(root) / model / f"{src}.json"
        if not (complete(tp) and complete(sp)):
            return None
        x = cached(R.score_transfer, tp, sp)
        if not (x["single"]["collapsed"] or x["single"]["source_collapsed"]):
            out[s] = x
    return out


def chaos_runs(root, arm):
    """{seed: score_soft output} of a ChaosNLI arm (A2: no exclusion); None if a run is missing or unfinished"""
    out = {}
    for s in SEEDS:
        p = Path(root) / "qwen3-1.7b" / f"{arm}_seed{s}.json"
        if not complete(p):
            return None
        out[s] = cached(R.score_soft, p)
    return out


def dispersion_of(path):
    """mean over the test items of ||u_1 - mean u||^2 from a run's saved single and mixture readouts (P2b)"""
    d = json.loads(Path(path).read_text())["test"]["dists"]
    return float(np.mean([sum((a - b) ** 2 for a, b in zip(u, v)) for u, v in zip(d["single"], d["mixture"])]))


def lower_is_better(metric):
    return not (metric in HIGHER_IS_BETTER or metric.startswith(("oracle_cov@", "cov@")))


def seed_signs(sa, sb, metric, readout="single", scale="ts"):
    """section 7, per-seed sign agreement: {seed: sign of A_s - B_s} over the seeds kept in both arms"""
    return {s: int(np.sign(sa[s][readout][scale][metric] - sb[s][readout][scale][metric]))
            for s in sorted(set(sa) & set(sb))}


def family(cells, get, a, b, metrics, readout="single", scale="ts", hier=True):
    """arm a minus arm b over cells x metrics, Holm within the family; get(cell, arm) -> {seed: scores} or None (a run is
    missing). Returns None if a run is missing, else dict(res={(cell, metric): (d, lo, hi, p)}, adj={(cell, metric):
    Holm p}, signs={(cell, metric): {seed: sign}}, hier={(cell, metric): (mean, lo, hi)}, left_out={cell: a, b or
    "both", the arm(s) whose every seed was left out}). A left-out cell has (nan, nan, nan, 1.0) and is outside the
    Holm family."""
    res, signs, hb, left = {}, {}, {}, {}
    for c in cells:
        ra, rb = get(c, a), get(c, b)
        if ra is None or rb is None:
            return None
        if not ra or not rb:
            left[c] = "both" if not ra and not rb else a if not ra else b
            print(f"note: {c[0]}/{c[1]}: every seed of {left[c]} left out (collapsed): no comparison", file=sys.stderr)
            for met in metrics:
                res[(c, met)] = NAN
            continue
        for met in metrics:
            res[(c, met)] = R.paired_diff(list(ra.values()), list(rb.values()), met, readout, scale=scale)
            signs[(c, met)] = seed_signs(ra, rb, met, readout, scale)
            if hier and (met in R.PER_ITEM or met == "aurc"):
                hb[(c, met)] = R.hier_boot(list(ra.values()), list(rb.values()), met, readout, scale=scale)
    tested = sorted(k for k in res if k[0] not in left)
    adj = {k: 1.0 for k in res} | dict(zip(tested, R.holm([res[k][3] for k in tested])))
    return dict(res=res, adj=adj, signs=signs, hier=hb, left_out=left)


def outcome(F, c, met):
    """+1: A significantly better than B (Holm), -1: significantly worse, 0: neither or no comparison"""
    d = F["res"][(c, met)][0]
    return 0 if math.isnan(d) else R.better(d, F["adj"][(c, met)], lower_is_better(met))


def wins(F, cells, met):
    """cells where A is significantly better / worse than B in met (Holm)"""
    return [c for c in cells if outcome(F, c, met) == 1], [c for c in cells if outcome(F, c, met) == -1]


def name(c):
    return f"{c[0]}/{c[1]}"


def report(F):
    """the reported form of a family: per 'model/cell metric' the difference A - B, 95% CI, p, Holm p, outcome (+1: A
    better), per-seed signs, hierarchical bootstrap and, for a cell without a comparison, the arm(s) left out"""
    if F is None:
        return "pending"
    out = {}
    for (c, met), (d, lo, hi, p) in sorted(F["res"].items()):
        out[f"{name(c)} {met}"] = dict(diff=d, ci=[lo, hi], p=p, p_holm=F["adj"][(c, met)], outcome=outcome(F, c, met),
                                       seed_signs=F["signs"].get((c, met)), hier_boot=F["hier"].get((c, met)),
                                       left_out=F["left_out"].get(c))
    return out


def secondary(cells, get, a, b, readout="single", scale="ts"):
    """the reported-only families next to a deciding one (A5): accuracy, NLL and E-AURC, each with its own Holm"""
    return {met: report(family(cells, get, a, b, (met,), readout, scale)) for met in SECONDARY}


def without_comparison(F):
    return {name(c): arm for c, arm in F["left_out"].items()}


def decide(root):
    root = Path(root)
    G = lambda c, arm: runs(root, c[0], c[1], arm)  # noqa: E731
    D = {}
    # C: stage 1 (Brier) vs the selected GRPO+TS
    F = family(CELLS_C, G, "warm400", "grpo", ("brier", "aurc"))
    if F is None:
        D["C"] = "pending"
    else:
        flat = {(name(c), met): v for (c, met), v in F["res"].items()}
        # A5: a cell where only the comparator was left out counts for neither verdict; where stage 1 was left out
        # (alone or with the comparator) it counts as 'better in neither'
        excused = [name(c) for c, arm in F["left_out"].items() if arm == "grpo"]
        verdict, _ = R.decide_central(flat, [f"{m}/mp" for m in QWEN + BIG], left_out=excused)
        D["C"] = dict(verdict=verdict, cells_without_comparison=without_comparison(F),
                      cells_counting_for_neither_verdict=excused, family=report(F),
                      secondary=secondary(CELLS_C, G, "warm400", "grpo"))
    # Q1a: Brier vs log score (two-sided, Qwen3 cells); a score is claimed better only if it wins one metric in at least
    # 3 of the 4 cells (per metric, A5)
    F = family(QWEN_CELLS, G, "warm400", "ce", ("brier", "aurc"))
    if F is None:
        D["Q1a"] = "pending"
    else:
        claims = []
        for met in ("brier", "aurc"):
            b, w = wins(F, QWEN_CELLS, met)
            if len(b) >= 3:
                claims.append(f"Brier stage 1 better in {met} ({len(b)}/4)")
            if len(w) >= 3:
                claims.append(f"log-score stage 1 better in {met} ({len(w)}/4)")
        D["Q1a"] = dict(verdict="; ".join(claims) or "no consistent difference",
                        cells_without_comparison=without_comparison(F), family=report(F),
                        secondary=secondary(QWEN_CELLS, G, "warm400", "ce"))
    # Q1b: on-policy vs offline rationales; a cell counts if on-policy is better in Brier or AURC (per cell, A5)
    F = family(QWEN_CELLS, G, "warm400", "refrat", ("brier", "aurc"))
    if F is None:
        D["Q1b"] = "pending"
    else:
        better_any = [c for c in QWEN_CELLS if any(outcome(F, c, m) == 1 for m in ("brier", "aurc"))]
        worse_any = [c for c in QWEN_CELLS if any(outcome(F, c, m) == -1 for m in ("brier", "aurc"))]
        verdict = ("supported" if len(better_any) >= 3 and not worse_any else
                   "refuted" if len(worse_any) >= 2 else "inconclusive")
        D["Q1b"] = dict(verdict=verdict, cells_without_comparison=without_comparison(F), family=report(F),
                        secondary=secondary(QWEN_CELLS, G, "warm400", "refrat"))
    # Q1c: Brier vs linear (properness vs a dense signal): decided on AURC (its own family) and the fitted temperatures
    # (a cell without a comparison fails the every-cell temperature condition, A5); accuracy is reported separately
    F = family(QWEN_CELLS, G, "warm400", "lin", ("aurc",))
    if F is None:
        D["Q1c"] = "pending"
    else:
        b, _ = wins(F, QWEN_CELLS, "aurc")
        temps = {}
        for c in QWEN_CELLS:
            tb, tl = ([r["single"]["T"] for r in G(c, arm).values()] for arm in ("warm400", "lin"))
            temps[name(c)] = (float(np.mean(tb)) if tb else float("nan"), float(np.mean(tl)) if tl else float("nan"))
        verdict = "supported" if len(b) >= 3 and all(tl > tb for tb, tl in temps.values()) else "not supported"
        D["Q1c"] = dict(verdict=verdict, temperatures_brier_linear=temps, cells_without_comparison=without_comparison(F),
                        family=report(F), secondary=secondary(QWEN_CELLS, G, "warm400", "lin"))
    # Q2: per-rationale (warm400) vs mixture objective (mix); P2b must hold in every cell (a cell without a comparison
    # fails it, A5)
    F = family(QWEN_CELLS, G, "warm400", "mix", ("brier",))
    Fm = family(QWEN_CELLS, G, "warm400", "mix", ("brier",), readout="mixture")
    if F is None or Fm is None:
        D["Q2"] = "pending"
    else:
        p2a, mix_better = wins(F, QWEN_CELLS, "brier")
        disp = {}
        for c in QWEN_CELLS:  # over the seeds kept in each arm (section 6)
            ds, dm = ([dispersion_of(root / c[0] / f"{c[1]}_{arm}_seed{s}.json") for s in G(c, arm)]
                      for arm in ("warm400", "mix"))
            disp[name(c)] = (float(np.mean(ds)) if ds else float("nan"), float(np.mean(dm)) if dm else float("nan"))
        p2b = all(dm > ds for ds, dm in disp.values())
        mix_sig_worse = [name(c) for c in QWEN_CELLS if outcome(Fm, c, "brier") == 1]
        verdict = ("supported" if len(p2a) >= 3 and p2b else "refuted" if len(mix_better) >= 2 else "inconclusive")
        D["Q2"] = dict(verdict=verdict, P2a_cells=len(p2a), P2b=p2b, dispersion_per_rationale_mixture=disp,
                       P2c_mixture_objective_significantly_worse_on_its_readout=mix_sig_worse,
                       cells_without_comparison=without_comparison(F), family_single=report(F),
                       family_mixture=report(Fm), secondary=secondary(QWEN_CELLS, G, "warm400", "mix"))
    # Q3a: oracle coverage at realized error 0.2 on MMLU-Pro, per comparator (A5: Holm over the 4 backbones; 0.1 is
    # reported in a family of its own); the risk-controlled thresholds (calib -> fresh) are reported
    for other in ("grpo", "direct_lr2e-6"):
        key = f"Q3a_vs_{other}"
        F = family(MP_CELLS, G, "warm400", other, ("oracle_cov@0.2",))
        if F is None:
            D[key] = "pending"
            continue
        higher, _ = wins(F, MP_CELLS, "oracle_cov@0.2")
        rc = {}
        for c in MP_CELLS:
            for arm in ("warm400", other):
                kept = G(c, arm)
                rc[f"{name(c)} {arm}"] = {f"risk {eps}": dict(zip(kept, R.risk_controlled(list(kept.values()), eps=eps)))
                                          for eps in (0.1, 0.2)}
        D[key] = dict(verdict="supported" if len(higher) >= 3 else "not supported",
                      cells_without_comparison=without_comparison(F), family=report(F),
                      coverage_at_error_01_reported=report(family(MP_CELLS, G, "warm400", other, ("oracle_cov@0.1",))),
                      risk_controlled_reported=rc)
    # Q3b: transfer (A1), stage 1 vs GRPO+TS in AURC with the source run's temperature; reported: the target-dev
    # temperature, the secondary metrics, and every arm's metrics
    GT = lambda c, arm: transfer_runs(root, c[0], c[1], arm)  # noqa: E731
    F = family(TRANSFER_CELLS, GT, "warm400", "grpo", ("aurc",))
    if F is None:
        D["Q3b"] = "pending"
    else:
        b, _ = wins(F, TRANSFER_CELLS, "aurc")
        arms = {}
        for c in TRANSFER_CELLS:
            for arm in ("base", "direct", "warm400", "ce", "grpo", "fork_rl"):
                xs = GT(c, arm)
                arms[f"{name(c)} {arm}"] = "pending" if xs is None else {
                    scale: {met: float(np.mean([x["single"][scale][met] for x in xs.values()])) if xs else None
                            for met in ("acc", "brier", "nll", "aurc", "oracle_cov@0.2")}
                    for scale in ("ts", "ts_target", "raw")}
        D["Q3b"] = dict(verdict="supported" if len(b) >= 3 else "not supported",
                        cells_without_comparison=without_comparison(F), family=report(F),
                        target_temperature_reported=report(family(TRANSFER_CELLS, GT, "warm400", "grpo", ("aurc",),
                                                                  scale="ts_target")),
                        secondary=secondary(TRANSFER_CELLS, GT, "warm400", "grpo"), arms_reported=arms)
    # ChaosNLI (A2): per-rationale (lambda = 0) vs mixture objective (lambda = 1), raw readouts against q (primary;
    # Holm over 2 readouts x 2 metrics), TS secondary; majority accuracy (raw; Holm over the 2 readouts); dispersion and
    # rationale statistics per seed; descriptive
    ra, rb = chaos_runs(root, "cn_s1_lam0"), chaos_runs(root, "cn_s1_lam1")
    if ra is None or rb is None:
        D["ChaosNLI"] = "pending"
    else:
        def tests(pairs, scale):
            res = {(ro, met): R.paired_diff(list(ra.values()), list(rb.values()), met, ro, scale=scale) for ro, met in pairs}
            keys = sorted(res)
            adj = dict(zip(keys, R.holm([res[k][3] for k in keys])))
            return {f"{ro} {met}": dict(diff=res[(ro, met)][0], ci=list(res[(ro, met)][1:3]), p=res[(ro, met)][3],
                                        p_holm=adj[(ro, met)],
                                        outcome=R.better(res[(ro, met)][0], adj[(ro, met)], lower_is_better(met)),
                                        seed_signs=seed_signs(ra, rb, met, ro, scale),
                                        hier_boot=R.hier_boot(list(ra.values()), list(rb.values()), met, ro, scale=scale))
                    for ro, met in keys}
        readouts = ("single", "mixture")
        out = {scale: tests([(ro, met) for ro in readouts for met in ("jsd", "sqdist")], scale) for scale in ("raw", "ts")}
        per_seed = {arm: {s: dict(dispersion=x["single"]["dispersion"], T=x["single"]["T"],
                                  rationale=x["single"]["rationale"],
                                  **{f"{ro} {met}": x[ro]["raw"][met] for ro in readouts
                                     for met in ("jsd", "sqdist", "majority_acc")})
                          for s, x in xs.items()} for arm, xs in (("lambda0", ra), ("lambda1", rb))}
        disp = {arm: float(np.mean([v["dispersion"] for v in xs.values()])) for arm, xs in per_seed.items()}
        D["ChaosNLI"] = dict(prediction_single_query_per_rationale_better=all(
                                 out["raw"][f"single {met}"]["outcome"] == 1 for met in ("jsd", "sqdist")),
                             prediction_mixture_spreads_rationales=disp["lambda1"] > disp["lambda0"],
                             mean_dispersion=disp, tests=out,
                             majority_accuracy_reported=tests([(ro, "majority_acc") for ro in readouts], "raw"),
                             per_seed=per_seed)
    # stage 2 (optional extension): two-stage vs stage 1 in every Qwen3 cell; the reasoning-reward control on 1.7B
    D["stage2_vs_stage1"] = report(family(QWEN_CELLS, G, "fork_rl", "warm400", ("acc", "brier", "aurc")))
    cells17 = [("qwen3-1.7b", "gv"), ("qwen3-1.7b", "mp")]
    F = family(cells17, G, "fork_rl", "fork_correct", ("brier", "aurc"))
    if F is None:
        D["stage2_reward"] = "pending"
    else:
        ok = all(any(outcome(F, c, m) == 1 for m in ("brier", "aurc")) for c in cells17)
        D["stage2_reward"] = dict(verdict="proper reward better on both tasks" if ok else "not established",
                                  cells_without_comparison=without_comparison(F), family=report(F),
                                  secondary=secondary(cells17, G, "fork_rl", "fork_correct"))
    # closest methods (A4, reported): stage 1 vs RLCR+TS and vs CREDO+TS in every Qwen3 cell, Brier and AURC with Holm
    # within each method; accuracy, NLL and E-AURC each in a family of its own
    for arm in ("rlcr", "credo"):
        F = family(QWEN_CELLS, G, "warm400", arm, ("brier", "aurc"))
        D[f"closest_{arm}"] = "pending" if F is None else dict(
            cells_without_comparison=without_comparison(F), family=report(F),
            secondary=secondary(QWEN_CELLS, G, "warm400", arm))
    return D


def finite(x):
    """the decisions with NaN (cells without a comparison) written as null, so that the file is strict JSON"""
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, dict):
        return {str(k): finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # score against the cached dataset revisions (labels), as the runs
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(R.ROOT / "runs/npj"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = decide(a.root)
    for k, v in D.items():
        print(k, ":", v if isinstance(v, str) else v.get("verdict", "reported"))
    if a.out:
        tmp = Path(a.out).with_suffix(".tmp")
        tmp.write_text(json.dumps(finite(D), indent=1, allow_nan=False))
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()

"""Scoring, selection and decision rules of the npj protocol (paper/npj/PROTOCOL.md, sections 5-8).

A run is scored on the split it was evaluated on (config.test_split; train_direct runs record it too), from its saved
log-probabilities (logps; no floor) or, for runs that only saved probabilities, from those floored at 1e-45 (which
changes exact zeros only). A stated confidence (RLCR, --objective rlcr) is clipped to [0.001, 0.999]; with K > 2
options a C below 1/K is raised to just above 1/K, so the stated answer stays the decision (evaluation.verbal_decision);
with K = 2 the readout is exact and C < 0.5 decides the other answer. CREDO's confidence (--objective credo, amendment
A4) is read from logits: it is scored from the saved logit gap of its slot without a clip (from_slot), with the same
two decision rules. Temperature scaling (TS) is fitted on the run's own dev readouts with metrics.fit_temperature
(the same grid and one-label-per-item draw as before).

  python -m openjev_rlcd.npj_report select-grpo      # section 5: the GRPO+TS setting from the tuning runs on val
  python -m openjev_rlcd.npj_report select-rlcr      # section 5: RLCR's advantage rule from the seed-17 runs on val
"""
import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .evaluation import KMAX, aurc_boot, collapsed, paired, rank_key, running_risk, verbal_decision
from .metrics import fit_temperature, jsd, selective_metrics
from .tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
# the analysis runs thousands of tiny tensor ops (one temperature fit = 311 small cross-entropies): many intra-op
# threads only add overhead, and on a loaded machine they slow a fit from ~10 ms to ~2 s
torch.set_num_threads(min(4, torch.get_num_threads()))
FLOOR = 1e-45  # for runs without logps: only exact zeros are floored
VERBAL_CLIP = 1e-3  # a stated confidence C is used as min(max(C, 0.001), 0.999)
BUDGETS = (0.05, 0.1, 0.2, 0.3)
CELLS_TUNE = [(m, t) for m in ("qwen3-0.6b", "qwen3-1.7b") for t in ("gsm8kv", "mmlupro")]
PX = {"gsm8kv": "gv", "mmlupro": "mp"}


@lru_cache(maxsize=None)
def splits(task):
    return TASKS[task]["load"]()


def items(task, split, n):
    out = splits(task)[split][:n]
    assert len(out) == n, (task, split, n, len(out))
    return out


def pad(rows, ks):
    """[N, KMAX] float64 log-probabilities, -inf beyond each item's option count k"""
    z = torch.full((len(rows), KMAX), float("-inf"), dtype=torch.float64)
    for i, (r, k) in enumerate(zip(rows, ks)):
        assert len(r) == k, (len(r), k)
        z[i, :k] = torch.tensor(r, dtype=torch.float64)
    return z


def from_probs(dists, verbal=False):
    """log-probabilities of saved probabilities: floored at 1e-45 (exact zeros only), or for a stated confidence its
    decision-preserving form with C clipped to [0.001, 0.999]"""
    out = []
    for p in dists:
        if verbal:
            q = verbal_decision(p)
            k = len(q)
            if k == 2:
                c = min(max(max(q), VERBAL_CLIP), 1 - VERBAL_CLIP)
                a = q.index(max(q))
                q = [c if j == a else 1 - c for j in range(2)]
            elif len(set(q)) > 1:
                cnt = {}
                for x in q:
                    cnt[x] = cnt.get(x, 0) + 1
                a = next(i for i, x in enumerate(q) if cnt[x] == 1)
                c = min(max(q[a], VERBAL_CLIP), 1 - VERBAL_CLIP)
                q = [c if j == a else (1 - c) / (k - 1) for j in range(k)]
            p = q
        out.append([math.log(max(x, FLOOR)) for x in p])
    return out


def logsigmoid(x):
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))


def from_slot(slot, ks):
    """CREDO (amendment A4): log-probabilities of the single query from its saved decision a and slot logit gap
    g = z_high - z_low (the first rationale of each item): log c = logsigmoid(g) on a and logsigmoid(-g) - log(K - 1) on
    every other option, without a clip. The decision rules of a confidence on a stated answer (section 6): with K > 2,
    a c below 1/K is raised to just above 1/K, so a stays the decision; with K = 2 the readout is exact and c < 0.5
    decides the other answer (c = 0.5 exactly, a gap of 0, ties and is decided as the first option, as every tied
    readout). ks: the option count of each item."""
    out = []
    for a_row, g_row, k in zip(slot["answer"], slot["gap"], ks, strict=True):
        a, g = a_row[0], g_row[0]
        lc, lo = logsigmoid(g), logsigmoid(-g)
        if k > 2 and lc < -math.log(k):
            c = 1.0 / k + 1e-6
            lc, lo = math.log(c), math.log1p(-c)
        out.append([lc if j == a else lo - math.log(k - 1) for j in range(k)])
    return out


def readouts(run, verbal=False):
    """{readout: (test log-probs, dev log-probs)} of a run file (train.py, train.py --eval-only, train_direct.py)"""
    if run.get("config", {}).get("objective") == "credo":  # A4: the slot gap; single query only
        if "slot" not in run.get("test", {}) or "dev_slot" not in run:
            raise ValueError("a CREDO run file without its saved slot gaps (test.slot, dev_slot)")
        k_t = [len(p) for p in run["test"]["dists"]["single"]]
        k_d = [len(p) for p in run["dev_dists"]["single"]]
        return {"single": (from_slot(run["test"]["slot"], k_t), from_slot(run["dev_slot"], k_d))}
    if "test_dists" in run:  # train_direct: one readout
        t = run.get("test_logps") or from_probs(run["test_dists"])
        d = run.get("dev_logps") or from_probs(run["dev_dists"])
        return {"single": (t, d)}
    test, out = run["test"], {}
    for ro in (("single",) if verbal else ("single", "mixture")):
        if verbal or "logps" not in test or test["logps"].get(ro) is None or "dev_logps" not in run:
            out[ro] = (from_probs(test["dists"][ro], verbal), from_probs(run["dev_dists"][ro], verbal))
        else:
            out[ro] = (test["logps"][ro], run["dev_logps"][ro])
    return out


def margins_of(z):
    """log-margin between the two largest log-probabilities of each row (rows with one option: +inf)"""
    top = torch.topk(z, 2, dim=1).values
    return (top[:, 0] - top[:, 1]).tolist()


def eaurc(aurc, n_err, n):
    """AURC minus the AURC of the same errors ranked last (a perfect ranking)"""
    best = sum((i - (n - n_err)) / i for i in range(n - n_err + 1, n + 1)) / n
    return aurc - best


def toplabel_decomposition(conf, correct, bins=10):
    """Murphy decomposition of the top-label Brier score: reliability, resolution, uncertainty (10 equal-width bins)"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n, acc = len(conf), correct.mean()
    b = np.minimum((conf * bins).astype(int), bins - 1)
    rel = res = 0.0
    for j in range(bins):
        m = b == j
        if m.any():
            w = m.mean()
            rel += w * (conf[m].mean() - correct[m].mean()) ** 2
            res += w * (correct[m].mean() - acc) ** 2
    return dict(reliability=rel, resolution=res, uncertainty=acc * (1 - acc))


def score_logits(z, its, tiebreak):
    """metrics of the readout log-probabilities z [N, KMAX] on items its; ranking by max p, ties broken by tiebreak"""
    lp = torch.log_softmax(z, dim=1)
    probs = [row[:len(it["q"])] for row, it in zip(lp.exp().tolist(), its)]
    ys = [it["q"].index(1.0) for it in its]
    correct = [int(p.index(max(p)) == y) for p, y in zip(probs, ys)]
    brier = [sum((a - (i == y)) ** 2 for i, a in enumerate(p)) for p, y in zip(probs, ys)]
    nll = [-lp[i, y].item() for i, y in enumerate(ys)]
    conf = [max(p) for p in probs]
    rank = rank_key(conf, tiebreak)
    n = len(its)
    sel = selective_metrics(its, probs, BUDGETS, rank=rank)
    out = dict(acc=sum(correct) / n, brier=sum(brier) / n, nll=sum(nll) / n, maxp=sum(conf) / n, **sel,
               eaurc=eaurc(sel["aurc"], n - sum(correct), n), **toplabel_decomposition(conf, correct),
               per_item_correct=correct, per_item_brier=brier, per_item_nll=nll, per_item_conf=conf,
               per_item_rank=rank)
    return out


def score_run(path, verbal=None):
    """{readout: dict(raw=metrics, ts=metrics, T=temperature)} of a run file, on its own test split"""
    run = json.loads(Path(path).read_text())
    cfg = run["config"]
    verbal = cfg.get("objective") in ("rlcr", "credo") if verbal is None else verbal
    split = cfg.get("test_split", "test")
    test = run.get("test") if isinstance(run.get("test"), dict) else {}
    flag = bool(collapsed(test.get("rationale")))  # the exclusion rule of every arm (evaluation.collapsed)
    out = {}
    for ro, (t, d) in readouts(run, verbal).items():
        ti, di = items(cfg["task"], split, len(t)), items(cfg["task"], "dev", len(d))
        check_uids(run, ti, di)
        zt = pad(t, [len(it["q"]) for it in ti])
        zd = pad(d, [len(it["q"]) for it in di])
        temp = fit_temperature(list(zd.float()), di, cfg["seed"])
        m = margins_of(zt)
        out[ro] = dict(raw=score_logits(zt, ti, m), ts=score_logits(zt / temp, ti, m), T=temp, split=split,
                       uids=[it["uid"] for it in ti], collapsed=flag)
        c = calib_block(run, ro, verbal)
        if c is not None:  # held-out items of the same pass, scaled with the same temperature
            ci = items(cfg["task"], "calib", len(c[0]))
            assert c[1] is None or c[1] == [it["uid"] for it in ci], "calib items differ from the split"
            zc = pad(c[0], [len(it["q"]) for it in ci]) / temp
            pc = torch.softmax(zc, dim=1)
            out[ro]["calib"] = dict(conf=pc.max(1).values.tolist(),
                                    wrong=[int(int(pc[i].argmax()) != it["q"].index(1.0)) for i, it in enumerate(ci)])
    return out


def score_transfer(path, source_path, verbal=False):
    """a transfer run (eval-only on a new task) scored with the temperature of its source run, i.e. fitted on the
    training task's dev readouts (the deployed calibration); the transfer run's own dev (target domain) gives a second
    temperature, reported as 'ts_target'"""
    src = score_run(source_path, verbal)
    out = score_run(path, verbal)
    run = json.loads(Path(path).read_text())
    task = run["config"]["task"]
    for ro, x in out.items():
        t_src = src[ro]["T"]
        logps, _ = readouts(run, verbal)[ro]
        ti = items(task, x["split"], len(logps))
        zt = pad(logps, [len(it["q"]) for it in ti])
        m = margins_of(zt)
        x["ts_target"], x["T_target"] = x["ts"], x["T"]
        x["ts"], x["T"] = score_logits(zt / t_src, ti, m), t_src
        x["source_collapsed"] = src[ro]["collapsed"]  # the exclusion rule applies to the source run as well
    return out


def soft_metrics(lp, its):
    """against the label distributions q of items its (ChaosNLI, amendment A2): per item the JSD (bits, as
    metrics.readout_metrics) and the squared distance sum_k (u_k - q_k)^2, i.e. the expected Brier score against one
    random annotator minus the constant 1 - sum_k q_k^2; lp [N, KMAX] readout log-probabilities"""
    probs = [row[:len(it["q"])] for row, it in zip(lp.exp().tolist(), its)]
    js = [jsd(u, it["q"]) for u, it in zip(probs, its)]
    sq = [sum((a - b) ** 2 for a, b in zip(u, it["q"])) for u, it in zip(probs, its)]
    maj = [int(u.index(max(u)) == it["q"].index(max(it["q"]))) for u, it in zip(probs, its)]
    n = len(its)
    return dict(jsd=sum(js) / n, sqdist=sum(sq) / n, majority_acc=sum(maj) / n, per_item_jsd=js, per_item_sqdist=sq,
                per_item_majority=maj)


def score_soft(path):
    """a ChaosNLI run (amendment A2): {readout: dict(raw=soft_metrics, ts=soft_metrics, T, split, uids, collapsed,
    rationale, dispersion, per_item_dispersion)}. raw (the trained readout, as the earlier paper's ChaosNLI table) is
    primary; ts uses the temperature fitted on dev with one annotator label per item (fit_temperature). The dispersion
    across rationales is, per test item, ||u_1 - mean u||^2 from the saved single and mixture readouts (raw)."""
    run = json.loads(Path(path).read_text())
    cfg = run["config"]
    split = cfg.get("test_split", "test")
    test = run["test"]
    single, mix = test["dists"]["single"], test["dists"]["mixture"]
    disp = [sum((a - b) ** 2 for a, b in zip(u, v)) for u, v in zip(single, mix)]
    out = {}
    for ro, (t, d) in readouts(run).items():
        ti, di = items(cfg["task"], split, len(t)), items(cfg["task"], "dev", len(d))
        check_uids(run, ti, di)
        zt = pad(t, [len(it["q"]) for it in ti])
        zd = pad(d, [len(it["q"]) for it in di])
        temp = fit_temperature(list(zd.float()), di, cfg["seed"])
        out[ro] = dict(raw=soft_metrics(torch.log_softmax(zt, 1), ti), ts=soft_metrics(torch.log_softmax(zt / temp, 1), ti),
                       T=temp, split=split, uids=[it["uid"] for it in ti], collapsed=bool(collapsed(test.get("rationale"))),
                       rationale=test.get("rationale"), dispersion=sum(disp) / len(disp), per_item_dispersion=disp)
    return out


def check_uids(run, test_items, dev_items):
    """the uids a run file recorded (npj runs) must be the items it is scored against"""
    t = run.get("test", {}).get("uids") if isinstance(run.get("test"), dict) else None
    t = run.get("test_uids", t)
    if t is not None:
        assert t == [it["uid"] for it in test_items], "test items differ from the split"
    if run.get("dev_uids") is not None:
        assert run["dev_uids"] == [it["uid"] for it in dev_items], "dev items differ from the split"


def calib_block(run, readout, verbal):
    """(log-probabilities, uids or None) of the run's calib pass for one readout, or None"""
    c = run.get("eval_calib")
    if c is None:
        return None
    if "dists" in c and isinstance(c["dists"], list):  # train_direct: one readout
        return (c.get("logps") or from_probs(c["dists"])), c.get("uids")
    if run.get("config", {}).get("objective") == "credo":  # A4: the slot gap (CREDO runs have no calib pass)
        return from_slot(c["slot"], [len(p) for p in c["dists"][readout]]), c.get("uids")
    if verbal or not c.get("logps") or c["logps"].get(readout) is None:
        return from_probs(c["dists"][readout], verbal), c.get("uids")
    return c["logps"][readout], c.get("uids")


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)"""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def cp_upper(k, n, delta):
    """one-sided (1 - delta) Clopper-Pearson upper confidence bound of a binomial proportion (k events in n trials)"""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(14)) + (0.97, 0.98, 0.99, 0.995, 0.999)  # fixed in advance


def risk_threshold(conf, wrong, eps, delta=0.1, grid=GRID):
    """section 6: the lowest threshold of a grid fixed in advance whose selected calib items have a one-sided
    (1 - delta / |grid|) Clopper-Pearson upper bound on the selective risk of at most eps (Bonferroni over the grid, so
    with probability >= 1 - delta every accepted threshold has selective risk <= eps); None if no threshold passes.
    Given the number of selected items, their errors are binomial with the selective risk as parameter."""
    conf, wrong = np.asarray(conf, float), np.asarray(wrong, int)
    ok = []
    for lam in grid:
        sel = conf >= lam
        n = int(sel.sum())
        if n and cp_upper(int(wrong[sel].sum()), n, delta / len(grid)) <= eps:
            ok.append(lam)
    return min(ok) if ok else None


def apply_threshold(conf, wrong, lam):
    """(coverage, realized selective risk) of the items with confidence >= lam (lam None: nothing decided)"""
    if lam is None:
        return 0.0, None
    sel = [w for c, w in zip(conf, wrong) if c >= lam]
    return len(sel) / len(conf), (sum(sel) / len(sel) if sel else None)


def risk_controlled(run_scores, readout="single", eps=0.2, delta=0.1):
    """per seed: the threshold chosen on calib and its coverage / realized risk on the run's test split (fresh)"""
    out = []
    for r in run_scores:
        x = r[readout]
        if "calib" not in x:
            out.append(None)
            continue
        lam = risk_threshold(x["calib"]["conf"], x["calib"]["wrong"], eps, delta)
        ts = x["ts"]
        cov, risk = apply_threshold(ts["per_item_conf"], [1 - c for c in ts["per_item_correct"]], lam)
        out.append(dict(threshold=lam, coverage=cov, risk=risk))
    return out


def holm(pvals):
    """Holm-adjusted p-values (same order as the input)"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - r) * pvals[i]))
        adj[i] = running
    return adj


PER_ITEM = {"acc": "per_item_correct", "brier": "per_item_brier", "nll": "per_item_nll",  # sign-flip tests
            "jsd": "per_item_jsd", "sqdist": "per_item_sqdist", "majority_acc": "per_item_majority"}  # score_soft


def paired_diff(a_runs, b_runs, metric, readout="single", n_boot=4000, seed=0, scale="ts"):
    """A minus B on the same test items, per-item values averaged over each arm's seeds: (mean difference, 95% CI, p).
    metric in acc / brier / nll / jsd / sqdist (sign-flip test) or aurc / eaurc / oracle_cov@<e> (item bootstrap).
    a_runs, b_runs: lists of score_run (or score_soft) outputs, one per seed; scale: "ts" (after temperature scaling,
    the protocol's default) or "raw"."""
    A = [r[readout][scale] for r in a_runs]
    B = [r[readout][scale] for r in b_runs]
    ua, ub = a_runs[0][readout]["uids"], b_runs[0][readout]["uids"]
    assert ua == ub and all(r[readout]["uids"] == ua for r in a_runs + b_runs), "different test items"
    if metric in PER_ITEM:
        key = PER_ITEM[metric]
        xa = np.mean([x[key] for x in A], 0)
        xb = np.mean([x[key] for x in B], 0)
        d, lo, hi, p = paired(xa, xb)
        return float(d), float(lo), float(hi), max(float(p), 1 / 20001)  # a permutation p-value is never 0
    if metric not in ("aurc", "eaurc") and not metric.startswith("oracle_cov@"):
        raise ValueError(metric)
    seeds_a = [(np.asarray(x["per_item_rank"]), 1 - np.asarray(x["per_item_correct"], float)) for x in A]
    seeds_b = [(np.asarray(x["per_item_rank"]), 1 - np.asarray(x["per_item_correct"], float)) for x in B]
    m = len(ua)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, (n_boot, m))
    full = np.arange(m)[None]

    def stat(seeds, ix):
        if metric in ("aurc", "eaurc"):
            a = aurc_boot(seeds, ix)
            if metric == "aurc":
                return a
            n = ix.shape[1]  # E-AURC: minus the AURC of the same errors ranked last, per seed and resample
            best = []
            for _, risk in seeds:
                errs = risk[ix].sum(1)
                best.append(np.array([sum((i - (n - e)) / i for i in range(int(n - e) + 1, n + 1)) / n for e in errs]))
            return a - np.mean(best, 0)
        e = float(metric.split("@")[1])
        covs = []
        for key, risk in seeds:
            rr = running_risk(key, risk, ix)  # [B, n] selective risk at each coverage
            ok = rr <= e + 1e-9
            k = np.where(ok.any(1), ix.shape[1] - np.argmax(ok[:, ::-1], 1), 0)  # the largest coverage within e
            covs.append(k / ix.shape[1])
        return np.mean(covs, 0)
    d = float(stat(seeds_a, full)[0] - stat(seeds_b, full)[0])
    boot = stat(seeds_a, idx) - stat(seeds_b, idx)
    p = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))
    return d, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), max(float(p), 1 / (n_boot + 1))


def hier_boot(a_runs, b_runs, metric, readout="single", n_boot=2000, seed=0, scale="ts"):
    """robustness check (section 7): A minus B with seeds (within each arm) and items resampled; (mean, 95% interval)"""
    if metric not in ("aurc",) + tuple(PER_ITEM):
        raise ValueError(f"hier_boot: {metric} ({', '.join(PER_ITEM)} or aurc)")
    ua = a_runs[0][readout]["uids"]
    assert all(r[readout]["uids"] == ua for r in a_runs + b_runs), "different test items"
    A = [r[readout][scale] for r in a_runs]
    B = [r[readout][scale] for r in b_runs]
    m = len(ua)
    rng = np.random.default_rng(seed)
    key = PER_ITEM.get(metric)
    vals = []
    for _ in range(n_boot):
        sa = rng.integers(0, len(A), len(A))
        sb = rng.integers(0, len(B), len(B))
        ix = rng.integers(0, m, (1, m))
        if key is not None:
            xa = np.mean([np.asarray(A[i][key])[ix[0]] for i in sa])
            xb = np.mean([np.asarray(B[i][key])[ix[0]] for i in sb])
            vals.append(xa - xb)
        else:  # aurc
            seeds_a = [(np.asarray(A[i]["per_item_rank"]), 1 - np.asarray(A[i]["per_item_correct"], float)) for i in sa]
            seeds_b = [(np.asarray(B[i]["per_item_rank"]), 1 - np.asarray(B[i]["per_item_correct"], float)) for i in sb]
            vals.append(float(aurc_boot(seeds_a, ix)[0] - aurc_boot(seeds_b, ix)[0]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def better(d, p_adj, lower_is_better=True, alpha=0.05):
    """+1 if A is significantly better than B after Holm, -1 if significantly worse, 0 otherwise"""
    if p_adj >= alpha:
        return 0
    return 1 if (d < 0) == lower_is_better else -1


def decide_central(results, mp_cells, alpha=0.05, left_out=()):
    """section 8, claim C. results: {(cell, metric): (d, lo, hi, p)} for stage 1 minus GRPO+TS, metric in brier/aurc,
    over the family of all cells given (Holm inside). A cell without a comparison (every seed of an arm collapsed)
    carries d = nan and is outside the Holm family; if it is in left_out (the comparator collapsed: amendment A5), it
    counts for neither verdict, otherwise (stage 1 collapsed) it counts as 'better in neither'. Returns (verdict,
    per-cell outcome)."""
    keys = sorted(results)
    tested = [k for k in keys if not math.isnan(results[k][0])]
    adj = {k: 1.0 for k in keys} | dict(zip(tested, holm([results[k][3] for k in tested])))
    outcome = {k: better(results[k][0], adj[k], True, alpha) for k in keys}
    both = [c for c in mp_cells if outcome.get((c, "brier")) == 1 and outcome.get((c, "aurc")) == 1]
    worse = [c for c in mp_cells if -1 in (outcome.get((c, "brier")), outcome.get((c, "aurc")))]
    neither = [c for c in mp_cells if outcome.get((c, "brier")) != 1 and outcome.get((c, "aurc")) != 1
               and c not in left_out]
    if len(both) >= 3 and not worse:
        verdict = "supported"
    elif len(neither) >= 3:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return verdict, outcome


def _select(paths, verbal=False, split="test"):
    """section 5's rule over settings {tag: [run file per cell]}: the lowest mean single-query Brier after TS over the
    cells; within 0.002 of the best, the lowest mean AURC. A setting whose rationales collapse in any cell cannot be
    selected. Returns (choice or None, rows {tag: (mean Brier, mean AURC, per-cell values or a status)}, status), with
    status 'incomplete' (some run missing or unfinished), 'all collapsed' or 'selected'."""
    rows = {}
    for tag, files in paths.items():
        if not all(f.exists() and '"test": {' in f.read_text() for f in files):
            rows[tag] = (None, None, "incomplete")
            continue
        scores = [score_run(f, verbal=verbal)["single"] for f in files]
        assert all(x["split"] == split for x in scores), tag
        if any(x["collapsed"] for x in scores):
            rows[tag] = (float("inf"), float("inf"), "collapsed")
            continue
        vals = [(x["ts"]["brier"], x["ts"]["aurc"]) for x in scores]
        rows[tag] = (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])), vals)
    if any(r[2] == "incomplete" for r in rows.values()):
        return None, rows, "incomplete"
    best = min(r[0] for r in rows.values())
    if best == float("inf"):
        return None, rows, "all collapsed"
    near = [t for t, r in rows.items() if r[0] <= best + 0.002]
    return min(near, key=lambda t: rows[t][1]), rows, "selected"


TUNE_TAGS = ("b0", "b0.01", "b0.04", "b0.1", "b0.04_kt0.01")
TUNE_FLAGS = {"b0": "--kl-beta 0", "b0.01": "--kl-beta 0.01", "b0.04": "--kl-beta 0.04", "b0.1": "--kl-beta 0.1",
              "b0.04_kt0.01": "--kl-beta 0.04 --kl-tokens 0.01"}


def select_grpo(root=ROOT / "runs/npj_tune", tags=TUNE_TAGS):
    """section 5: the GRPO+TS setting, from the seed-17 tuning runs on val over the four Qwen3 cells"""
    return _select({tag: [root / m / f"{PX[t]}_grpo_{tag}_seed17.json" for m, t in CELLS_TUNE] for tag in tags})


def select_rlcr(root=ROOT):
    """section 5: GRPO-normalized or leave-one-out advantages for RLCR+TS, by the same rule on the seed-17 runs"""
    paths = {"grpo": [root / "runs/extra_baselines" / m / f"{PX[t]}_rlcr_seed17.json" for m, t in CELLS_TUNE],
             "loo": [root / "runs/rlcr_loo" / m / f"{PX[t]}_rlcrloo_seed17.json" for m, t in CELLS_TUNE]}
    return _select(paths, verbal=True)


def main():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # score against the cached dataset revision (labels), as the runs
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select-grpo", "select-rlcr"])
    a = ap.parse_args()
    choice, rows, status = select_grpo() if a.what == "select-grpo" else select_rlcr()
    for tag, (b, au, vals) in rows.items():
        if isinstance(vals, str):
            print(f"{tag:14s} {vals}")
            continue
        cells = "  ".join(f"{m[6:]}/{PX[t]} {v[0]:.4f}/{v[1]:.4f}" for (m, t), v in zip(CELLS_TUNE, vals))
        print(f"{tag:14s} mean Brier {b:.4f}  mean AURC {au:.4f}   per cell (Brier/AURC): {cells}")
    if status == "incomplete":
        print("selected: none yet -- not all runs complete")
    elif status == "all collapsed":
        print("selected: none -- every setting collapsed in some cell (protocol section 5: an amendment is needed)")
    else:
        print("selected:", choice)
        out = ROOT / "runs/npj_tune" / f"selection_{a.what.split('-')[1]}.json"  # read by scripts/plan_npj.py
        out.parent.mkdir(parents=True, exist_ok=True)
        flags = TUNE_FLAGS[choice] if a.what == "select-grpo" else ("--rlcr-advantage loo" if choice == "loo" else "")
        out.write_text(json.dumps(dict(selected=choice, flags=flags, rows={
            t: dict(mean_brier=r[0], mean_aurc=r[1]) for t, r in rows.items()}), indent=1))
        print("written", out)
    sys.exit(0 if status == "selected" else 1)


if __name__ == "__main__":
    main()

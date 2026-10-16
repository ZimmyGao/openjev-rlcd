"""Scoring and paired statistics for the epistemic tasks (one correct option: MMLU-Pro, GSM8K-Verify).

Every arm is recomputed from its saved per-item distributions with the same functions, raw and after
temperature scaling fitted on the dev items (options beyond an item's count stay at -inf).
  score metrics : accuracy, Brier (multi-class, vs one-hot), NLL, ECE (10 bins, confidence vs correct)
  decisions     : AURC, label-free automation cov@e / realized risk@e, oracle coverage (selective_metrics);
                  items are ranked by temperature-scaled max p, ties broken by the model's own log-margin
                  (padded_logits floors probabilities at 1e-12, which would otherwise tie all saturated items),
                  and items that still tie share their group's mean risk (the expectation over random orders)
Paired tests on test items (per-item values averaged over seeds first): mean difference with a 95%
bootstrap CI and a two-sided sign-flip permutation p-value -- tests of the MEAN, which is what an
expected proper score is.
"""
import math

import numpy as np
import torch

from .metrics import fit_temperature, selective_metrics

BUDGETS = (0.05, 0.1, 0.2, 0.3)
SPLITS, KMAX = None, 10  # SPLITS: the task's splits, set by the caller (TASKS[task]["load"]())


def padded_logits(dists):
    z = torch.full((len(dists), KMAX), float("-inf"), dtype=torch.float64)
    for i, p in enumerate(dists):
        z[i, :len(p)] = torch.tensor(p, dtype=torch.float64).clamp_min(1e-12).log()
    return z


def lists(z, items):
    return [row[:len(it["q"])] for row, it in zip(z.softmax(-1).tolist(), items)]


def collapsed(rationale):
    """the exclusion rule for every arm: the policy's test rationales have collapsed -- mean <= 5 tokens (no reasoning
    left) or every rationale at the length limit with repetition >= 0.9 (a loop); rationale = the run's test["rationale"]
    statistics (absent for direct-answer models, which are never excluded)"""
    if not rationale:
        return False
    tokens, limit, rep = (rationale.get(k) for k in ("mean_rationale_tokens", "length_limit_rate", "repetition_ratio"))
    return (tokens is not None and tokens <= 5) or (limit is not None and rep is not None and limit >= 0.99 and rep >= 0.9)


def margins(dists):
    """the model's own log-margin between its two most probable options, from the saved probabilities with only
    exact zeros floored: it orders the saturated items that the 1e-12 floor of padded_logits makes equal"""
    out = []
    for p in dists:
        top = sorted((max(x, 1e-45) for x in p), reverse=True)
        out.append(math.log(top[0]) - math.log(top[1]))
    return out


def rank_key(conf, tiebreak):
    """dense rank of the pairs (conf, tiebreak): higher = more confident, equal = a genuine tie"""
    conf, tiebreak = np.asarray(conf, dtype=float), np.asarray(tiebreak, dtype=float)
    order = np.lexsort((tiebreak, conf))
    c, k = conf[order], tiebreak[order]
    new = np.ones(len(c), bool)
    new[1:] = (c[1:] != c[:-1]) | (k[1:] != k[:-1])
    rank = np.empty(len(c))
    rank[order] = np.cumsum(new)
    return rank.tolist()


def score(items, probs, tiebreak=None):
    brier, nll, acc = [], [], []
    for it, p in zip(items, probs):
        y = it["q"].index(1.0)
        brier.append(sum((a - (i == y)) ** 2 for i, a in enumerate(p)))
        nll.append(-math.log(max(p[y], 1e-12)))
        acc.append(int(p.index(max(p)) == y))
    n = len(items)
    conf = [max(p) for p in probs]
    rank = conf if tiebreak is None else rank_key(conf, tiebreak)
    return dict(acc=sum(acc) / n, brier=sum(brier) / n, nll=sum(nll) / n, maxp=sum(conf) / n,
                **selective_metrics(items, probs, BUDGETS, rank=rank), per_item_brier=brier, per_item_correct=acc,
                per_item_conf=conf, per_item_rank=rank)


def scored(test, dev, seed):
    ti, di = SPLITS["test"][:len(test)], SPLITS["dev"][:len(dev)]
    zt, zd = padded_logits(test), padded_logits(dev)
    t = fit_temperature(list(zd.float()), di, seed)
    m = margins(test)
    return score(ti, lists(zt, ti), m), score(ti, lists(zt / t, ti), m), t


def running_risk(key, risk, idx):
    """risk-coverage curves [B, n] of the items idx [B, n]: ranked by key (higher = more confident; per_item_rank),
    items with an equal key sharing their group's mean risk, as in metrics.selective_metrics"""
    c, r = key[idx], risk[idx]
    order = np.argsort(-c, axis=1, kind="stable")
    cs, rs = np.take_along_axis(c, order, 1), np.take_along_axis(r, order, 1)
    b, n = cs.shape
    new = np.ones((b, n), bool)
    new[:, 1:] = cs[:, 1:] != cs[:, :-1]
    group = np.cumsum(new, 1) - 1 + (np.arange(b) * n)[:, None]  # tie group, unique across rows
    sums = np.bincount(group.ravel(), weights=rs.ravel(), minlength=b * n)
    sizes = np.bincount(group.ravel(), minlength=b * n)
    return np.cumsum((sums / np.maximum(sizes, 1))[group], 1) / np.arange(1, n + 1)


def aurc_boot(seeds, idx):
    """AURC on resampled items idx [B, n], averaged over seeds [(key, 0/1 risk)] -> [B]; the point estimate on the
    full index and the bootstrap rows use the same tie rule (running_risk)"""
    return sum(running_risk(key, risk, idx).mean(1) for key, risk in seeds) / len(seeds)


def paired_aurc(x, y, n=4000, seed=0):
    m = len(x["brier"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, (n, m))
    full = np.arange(m)[None]
    d = aurc_boot(x["seeds"], full)[0] - aurc_boot(y["seeds"], full)[0]
    boot = aurc_boot(x["seeds"], idx) - aurc_boot(y["seeds"], idx)
    p = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))  # capped: identical arms give boot == 0 everywhere
    return d, np.percentile(boot, 2.5), np.percentile(boot, 97.5), p


def paired(a, b, n=20000, seed=0):
    """Mean of a - b, 95% bootstrap CI over items, two-sided sign-flip permutation p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    flips = (rng.integers(0, 2, (n, len(d))) * 2 - 1) @ d / len(d)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), (np.abs(flips) >= abs(d.mean())).mean()

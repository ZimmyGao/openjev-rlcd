"""Selective metrics with tied rankings. Items with an equal ranking key share their group's mean risk: AURC and
risk@e are then exactly the average over all orders inside tie groups of the position-ranked values, oracle_cov@e is
the largest top-ranked set whose tie-averaged risk is <= e, and nothing depends on item order. The bootstrap AURC
(evaluation.aurc_boot, many rows with repeated items) agrees with metrics.selective_metrics row by row, and
evaluation.scored breaks the ties made by the 1e-12 probability floor with the model's own log-margin."""
import itertools
import math
import random

import numpy as np
import pytest

from openjev_rlcd import evaluation as S
from openjev_rlcd.evaluation import aurc_boot, margins, rank_key
from openjev_rlcd.metrics import selective_metrics


def items_probs(confs, correct):
    """two-option items whose max p is conf, on the right option iff correct"""
    items = [dict(q=[1.0, 0.0]) for _ in confs]
    probs = [[c, 1 - c] if ok else [1 - c, c] for c, ok in zip(confs, correct)]
    return items, probs


def ranked(confs, risks, e=None):
    """position-ranked AURC (the old definition), or the realized risk of the label-free cov@e set"""
    rows = sorted(zip(confs, risks), key=lambda r: -r[0])
    cum, total, pred, at_e = 0.0, 0.0, 0.0, 0.0
    for k, (c, r) in enumerate(rows, 1):
        cum += r
        total += cum / k
        pred += 1 - c
        if e is not None and pred / k <= e:
            at_e = cum / k
    return total / len(rows) if e is None else at_e


def order_average(confs, risks, e=None):
    orders = list(itertools.permutations(range(len(confs))))
    return sum(ranked([confs[i] for i in o], [risks[i] for i in o], e) for o in orders) / len(orders)


def test_aurc_and_risk_equal_the_average_over_all_orders_inside_tie_groups():
    confs = [0.9, 0.9, 0.9, 0.7, 0.7, 0.6]
    correct = [1, 0, 1, 0, 1, 0]
    risks = [1 - c for c in correct]
    items, probs = items_probs(confs, correct)
    assert selective_metrics(items, probs)["aurc"] == pytest.approx(order_average(confs, risks), abs=1e-12)
    assert ranked(confs, risks) != pytest.approx(order_average(confs, risks), abs=1e-6)  # the case depends on order
    # cov@0.1 = 2 of 3 items, i.e. it stops inside the 0.8 group: realized risk 0 or 1/2 depending on the order
    confs, correct = [1.0, 0.8, 0.8], [1, 1, 0]
    items, probs = items_probs(confs, correct)
    got = selective_metrics(items, probs, budgets=(0.1,))
    assert got["cov@0.1"] == pytest.approx(2 / 3)
    assert got["risk@0.1"] == pytest.approx(order_average(confs, [1 - c for c in correct], e=0.1), abs=1e-12)
    assert got["risk@0.1"] == pytest.approx(0.25)


def test_oracle_coverage_uses_the_tie_averaged_risk():
    # two tied items, one wrong: either half of the group is 0% or 100% wrong; tie-averaged it is 50% -> nothing at 40%
    items, probs = items_probs([0.9, 0.9, 0.6], [1, 0, 0])
    assert selective_metrics(items, probs, budgets=(0.4,))["oracle_cov@0.4"] == 0.0
    items, probs = items_probs([0.9, 0.9, 0.9, 0.9, 0.6], [1, 1, 1, 0, 0])  # group risk 25%: the whole group
    assert selective_metrics(items, probs, budgets=(0.25,))["oracle_cov@0.25"] == pytest.approx(0.8)


def test_does_not_depend_on_item_order():
    rng = random.Random(0)
    confs = [rng.choice([0.55, 0.8, 0.95, 1.0]) for _ in range(200)]
    correct = [int(rng.random() < c) for c in confs]
    items, probs = items_probs(confs, correct)
    ref = selective_metrics(items, probs, budgets=(0.05, 0.2))
    perm = list(range(len(confs)))
    for _ in range(5):
        rng.shuffle(perm)
        got = selective_metrics([items[i] for i in perm], [probs[i] for i in perm], budgets=(0.05, 0.2))
        assert got == pytest.approx(ref, abs=1e-12)


def test_without_ties_nothing_changes():
    rng = random.Random(1)
    confs = [0.5 + 0.5 * rng.random() for _ in range(100)]
    correct = [int(rng.random() < c) for c in confs]
    items, probs = items_probs(confs, correct)
    assert selective_metrics(items, probs)["aurc"] == pytest.approx(ranked(confs, [1 - c for c in correct]), abs=1e-12)


def test_bootstrap_rows_match_selective_metrics_on_the_resampled_items():
    rng = np.random.default_rng(2)
    n = 120
    seeds = []
    for _ in range(3):
        conf = rng.choice([0.6, 0.75, 0.9, 1.0], size=n)
        seeds.append((conf, (rng.random(n) > conf).astype(float)))
    idx = np.stack([np.arange(n), rng.permutation(n)] + [rng.integers(0, n, n) for _ in range(4)])  # with repeats
    got = aurc_boot(seeds, idx)
    for row, value in zip(idx, got):
        want = np.mean([selective_metrics(*items_probs(c[row].tolist(), (1 - r[row]).astype(int).tolist()))["aurc"]
                        for c, r in seeds])
        assert value == pytest.approx(want, abs=1e-12)


def test_saturated_items_are_ranked_by_their_own_margin():
    # minority probabilities below the 1e-12 floor: equal after padded_logits, ordered by the saved dists
    dists = [[1 - 1e-14, 1e-14], [1 - 1e-20, 1e-20], [1.0, 0.0], [0.9, 0.1]]
    m = margins(dists)
    assert m[2] > m[1] > m[0] > m[3]
    probs = S.lists(S.padded_logits(dists), [dict(q=[1.0, 0.0])] * 4)
    conf = [max(p) for p in probs]
    assert conf[0] == conf[1] == conf[2] > conf[3]  # the floor ties the three saturated items
    rank = rank_key(conf, m)
    assert rank[2] > rank[1] > rank[0] > rank[3]
    assert rank_key([0.9, 0.9], [3.0, 3.0])[0] == rank_key([0.9, 0.9], [3.0, 3.0])[1]  # a genuine tie stays a tie
    assert math.isfinite(m[2])

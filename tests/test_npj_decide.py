"""openjev_rlcd.npj_decide on synthetic fresh-split run files with known effects."""
import json
import math
import random

import pytest

from openjev_rlcd import npj_decide as D
from openjev_rlcd import npj_report as R

N_TEST, N_DEV = 60, 30
HEALTHY = dict(mean_rationale_tokens=50, length_limit_rate=0.0, repetition_ratio=0.1)
COLLAPSED = dict(mean_rationale_tokens=3, length_limit_rate=0.0, repetition_ratio=0.0)


def readout(rng, its, acc, conf_lo, conf_hi):
    out = []
    for it in its:
        k, y = len(it["q"]), it["q"].index(max(it["q"]))
        a = y if rng.random() < acc else rng.choice([j for j in range(k) if j != y])
        c = rng.uniform(conf_lo, conf_hi)
        out.append([c if j == a else (1 - c) / (k - 1) for j in range(k)])
    return out


def write_run(path, task, seed, rng, acc, lo, hi, disperse=0.0, split="fresh", rationale=HEALTHY, n_test=N_TEST):
    tests = R.items(task, split, n_test)
    devs = R.items(task, "dev", N_DEV)
    single_t, single_d = readout(rng, tests, acc, lo, hi), readout(rng, devs, acc, lo, hi)

    def mix_of(single):  # the mixture is the single readout pulled towards uniform by `disperse`
        return [[(1 - disperse) * x + disperse / len(u) for x in u] for u in single]
    mix_t, mix_d = mix_of(single_t), mix_of(single_d)
    logs = lambda d: [[math.log(x) for x in u] for u in d]  # noqa: E731
    run = dict(config=dict(task=task, seed=seed, test_split=split, objective="rlcd"),
               dev_dists=dict(single=single_d, mixture=mix_d), dev_logps=dict(single=logs(single_d), mixture=logs(mix_d)),
               dev_uids=[it["uid"] for it in devs],
               test=dict(dists=dict(single=single_t, mixture=mix_t), logps=dict(single=logs(single_t), mixture=logs(mix_t)),
                         uids=[it["uid"] for it in tests], rationale=rationale))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        R.splits("mmlupro")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"MMLU-Pro not cached: {e}")
    rng = random.Random(0)
    arms = {"warm400": (0.85, 0.6, 0.95, 0.0), "refrat": (0.45, 0.6, 0.95, 0.0),
            "mix": (0.6, 0.6, 0.95, 0.4), "lin": (0.85, 0.97, 0.999, 0.0)}
    for m, px in D.QWEN_CELLS:
        task = "gsm8kv" if px == "gv" else "mmlupro"
        for arm, (acc, lo, hi, disp) in arms.items():
            for s in D.SEEDS:
                write_run(tmp_path / "npj" / m / f"{px}_{arm}_seed{s}.json", task, s, rng, acc, lo, hi, disp)
    D._CACHE.clear()
    return tmp_path / "npj"


def test_decisions_on_synthetic_runs(root):
    out = D.decide(root)
    assert out["C"] == "pending" and out["Q1a"] == "pending" and out["stage2_reward"] == "pending"
    assert out["Q3b"] == "pending" and out["ChaosNLI"] == "pending"
    assert out["Q1b"]["verdict"] == "supported"  # the offline arm is far less accurate
    assert out["Q2"]["P2b"] is True  # the mixture objective's single readouts disperse from their mean
    assert out["Q2"]["P2a_cells"] >= 3  # its less accurate single readouts score worse after TS
    for row in out["Q1b"]["family"].values():  # per-seed signs and the hierarchical bootstrap are reported
        assert sorted(row["seed_signs"]) == list(D.SEEDS) and len(row["hier_boot"]) == 3
    json.dumps(out, default=str)


def test_q1c_is_decided_on_aurc_alone(root):
    out = D.decide(root)["Q1c"]
    assert out["family"] and all(k.endswith(" aurc") for k in out["family"])  # accuracy is not in the Holm family
    for met in ("acc", "nll", "eaurc"):  # reported, each in a Holm family of its own
        assert out["secondary"][met] and all(k.endswith(f" {met}") for k in out["secondary"][met])


def test_p2b_uses_the_seeds_kept_by_the_exclusion_rule(root):
    m, px = D.QWEN_CELLS[0]
    task = "gsm8kv" if px == "gv" else "mmlupro"
    write_run(root / m / f"{px}_mix_seed29.json", task, 29, random.Random(1), 0.6, 0.6, 0.95, 0.95, rationale=COLLAPSED)
    D._CACHE.clear()
    out = D.decide(root)["Q2"]
    kept = [D.dispersion_of(root / m / f"{px}_mix_seed{s}.json") for s in (17, 23)]
    assert out["dispersion_per_rationale_mixture"][f"{m}/{px}"][1] == pytest.approx(sum(kept) / 2)


def test_cells_without_comparison(root):
    # every GRPO seed collapsed in one cell: no comparison there (outside Holm, never a win); in the other three cells
    # GRPO is clearly worse, so all three p-values sit at the sign-flip floor and Holm over 3 tests gives 3/20001 each
    # (over 4, with the left-out cell inside the family, it would give 4/20001)
    for m, px in D.QWEN_CELLS:
        task = "gsm8kv" if px == "gv" else "mmlupro"
        for s in D.SEEDS:
            write_run(root / m / f"{px}_grpo_seed{s}.json", task, s, random.Random(s), 0.3, 0.6, 0.95,
                      rationale=COLLAPSED if (m, px) == D.QWEN_CELLS[0] else HEALTHY)
    D._CACHE.clear()
    G = lambda c, arm: D.runs(root, c[0], c[1], arm)  # noqa: E731
    F = D.family(D.QWEN_CELLS, G, "warm400", "grpo", ("brier",), hier=False)
    assert F["left_out"] == {D.QWEN_CELLS[0]: "grpo"} and F["adj"][(D.QWEN_CELLS[0], "brier")] == 1.0
    assert D.outcome(F, D.QWEN_CELLS[0], "brier") == 0
    assert all(F["adj"][(c, "brier")] == pytest.approx(3 / 20001) for c in D.QWEN_CELLS[1:])
    assert all(D.outcome(F, c, "brier") == 1 for c in D.QWEN_CELLS[1:])
    for s in D.SEEDS:  # stage 1 collapsed as well: both arms left out
        p = root / D.QWEN_CELLS[0][0] / f"{D.QWEN_CELLS[0][1]}_warm400_seed{s}.json"
        write_run(p, "gsm8kv" if D.QWEN_CELLS[0][1] == "gv" else "mmlupro", s, random.Random(s), 0.85, 0.6, 0.95,
                  rationale=COLLAPSED)
    D._CACHE.clear()
    assert D.family(D.QWEN_CELLS, G, "warm400", "grpo", ("brier",), hier=False)["left_out"] == {D.QWEN_CELLS[0]: "both"}


def test_decide_central_with_cells_left_out():
    nan = (float("nan"),) * 3 + (1.0,)
    cells = ["a", "b", "c", "d"]
    res = {(c, m): (0.0, -0.1, 0.1, 0.9) for c in cells[:2] for m in ("brier", "aurc")}
    res |= {(c, m): nan for c in cells[2:] for m in ("brier", "aurc")}
    assert R.decide_central(res, cells)[0] == "refuted"  # the stage-1 arm collapsed in c and d: counts against it
    assert R.decide_central(res, cells, left_out=["c", "d"])[0] == "inconclusive"  # the comparator collapsed
    # six tested p = 0.008: Holm over 6 gives 0.048 (significant); with the NaN cell inside the family (8 tests) it
    # would give 0.064, nothing would be significant and the verdict would be 'refuted'
    close = {(c, m): (-0.1, -0.2, -0.01, 0.008) for c in cells[:3] for m in ("brier", "aurc")}
    close |= {("d", m): nan for m in ("brier", "aurc")}
    assert R.decide_central(close, cells, left_out=["d"])[0] == "supported"
    assert R.decide_central(close, cells)[0] == "supported"  # d adds one 'better in neither'; 3 of 4 still support


def write_arm(root, arm, cells, acc, lo=0.6, hi=0.95, collapsed_cells=(), same_as=None):
    """three seeds of an arm in the given cells; same_as: the readouts of that arm's generator (identical runs)"""
    for m, px in cells:
        task = "gsm8kv" if px == "gv" else "mmlupro"
        for s in D.SEEDS:
            rng = random.Random(f"{same_as or arm}-{m}-{px}-{s}")
            write_run(root / m / f"{px}_{arm}_seed{s}.json", task, s, rng, acc, lo, hi,
                      rationale=COLLAPSED if (m, px) in collapsed_cells else HEALTHY)


def test_claim_c_q3a_stage2_and_the_direction_of_accuracy(root):
    write_arm(root, "warm400", D.CELLS_C, 0.85)
    write_arm(root, "grpo", D.CELLS_C, 0.45)
    write_arm(root, "direct_lr2e-6", D.MP_CELLS, 0.45)
    write_arm(root, "fork_rl", D.QWEN_CELLS, 0.85)
    write_arm(root, "fork_correct", [("qwen3-1.7b", "gv"), ("qwen3-1.7b", "mp")], 0.45)
    D._CACHE.clear()
    out = D.decide(root)
    assert out["C"]["verdict"] == "supported" and out["C"]["cells_without_comparison"] == {}
    for k, row in out["C"]["secondary"]["acc"].items():  # stage 1 is far more accurate: +1, with a positive difference
        assert row["diff"] > 0 and row["outcome"] == 1, k
    assert sorted(out["C"]["secondary"]) == ["acc", "eaurc", "nll"]
    assert out["Q3a_vs_grpo"]["verdict"] == "supported" and out["Q3a_vs_direct_lr2e-6"]["verdict"] == "supported"
    assert out["stage2_reward"]["verdict"] == "proper reward better on both tasks"
    for k, row in out["stage2_vs_stage1"].items():
        assert row["outcome"] in (-1, 0, 1)
    json.dumps(D.finite(out), allow_nan=False)
    # stage 1 no better than GRPO+TS on qwen3-0.6b/mp and phi-4-mini/mp (identical runs); SmolLM3-3B without a comparison
    same = [("qwen3-0.6b", "mp"), ("phi-4-mini", "mp")]
    write_arm(root, "grpo", same, 0.85, same_as="warm400")
    smol = [("smollm3-3b", "mp")]
    for collapsed_arm, verdict in (("grpo", "inconclusive"), ("warm400", "refuted"), ("both", "refuted")):
        write_arm(root, "warm400", smol, 0.85, collapsed_cells=smol if collapsed_arm in ("warm400", "both") else ())
        write_arm(root, "grpo", smol, 0.45, collapsed_cells=smol if collapsed_arm in ("grpo", "both") else ())
        D._CACHE.clear()
        c = D.decide(root)["C"]
        assert c["cells_without_comparison"] == {"smollm3-3b/mp": collapsed_arm}, collapsed_arm
        assert c["verdict"] == verdict, (collapsed_arm, c["verdict"])
        assert c["cells_counting_for_neither_verdict"] == (["smollm3-3b/mp"] if collapsed_arm == "grpo" else [])


def test_q3b_excludes_collapsed_source_seeds(root, monkeypatch):
    try:
        R.splits("arc")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ARC not cached: {e}")
    monkeypatch.setattr(D, "TRANSFER_CELLS", [(m, "arc") for m in D.QWEN])
    rng = random.Random(2)
    for m in D.QWEN:
        for arm, acc in (("warm400", 0.85), ("grpo", 0.6)):
            for s in D.SEEDS:
                write_run(root / m / f"mp_{arm}_seed{s}.json", "mmlupro", s, rng, acc, 0.6, 0.95,
                          rationale=COLLAPSED if (arm, s) == ("warm400", 23) else HEALTHY)
                write_run(root.parent / "npj_transfer" / m / f"arc_{arm}_seed{s}.json", "arc", s, rng, acc, 0.6, 0.95,
                          split="test", n_test=50)
    D._CACHE.clear()
    xs = D.transfer_runs(root, D.QWEN[0], "arc", "warm400")
    assert sorted(xs) == [17, 29]  # the transfer run of seed 23 is healthy, but its source run collapsed
    out = D.decide(root)["Q3b"]
    row = out["family"][f"{D.QWEN[0]}/arc aurc"]
    assert sorted(row["seed_signs"]) == [17, 29] and out["target_temperature_reported"]
    assert out["arms_reported"][f"{D.QWEN[0]}/arc base"] == "pending"


def chaos_run(path, seed, rng, spread, disperse, rationale=HEALTHY):
    """a ChaosNLI run: readouts = q mixed with noise (`spread`), the mixture pulled to uniform by `disperse`"""
    tests, devs = R.items("chaosnli", "test", 40), R.items("chaosnli", "dev", 20)

    def single(its):
        out = []
        for it in its:
            noise = [rng.random() for _ in it["q"]]
            u = [(1 - spread) * a + spread * b / sum(noise) for a, b in zip(it["q"], noise)]
            out.append([max(x, 1e-6) / sum(max(y, 1e-6) for y in u) for x in u])
        return out
    st, sd = single(tests), single(devs)
    mix = lambda d: [[(1 - disperse) * x + disperse / 3 for x in u] for u in d]  # noqa: E731
    logs = lambda d: [[math.log(x) for x in u] for u in d]  # noqa: E731
    run = dict(config=dict(task="chaosnli", seed=seed, test_split="test", objective="rlcd"),
               dev_dists=dict(single=sd, mixture=mix(sd)), dev_logps=dict(single=logs(sd), mixture=logs(mix(sd))),
               dev_uids=[it["uid"] for it in devs],
               test=dict(dists=dict(single=st, mixture=mix(st)), logps=dict(single=logs(st), mixture=logs(mix(st))),
                         uids=[it["uid"] for it in tests], rationale=rationale))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))


def test_chaosnli_soft_label_analysis(root):
    try:
        R.splits("chaosnli")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ChaosNLI not cached: {e}")
    rng = random.Random(3)
    for s in D.SEEDS:  # lambda 0 close to q (and collapsed in one seed: no exclusion on ChaosNLI), lambda 1 noisy
        chaos_run(root / "qwen3-1.7b" / f"cn_s1_lam0_seed{s}.json", s, rng, 0.1, 0.0,
                  rationale=COLLAPSED if s == 29 else HEALTHY)
        chaos_run(root / "qwen3-1.7b" / f"cn_s1_lam1_seed{s}.json", s, rng, 0.9, 0.5)
    D._CACHE.clear()
    out = D.decide(root)["ChaosNLI"]
    assert out["prediction_single_query_per_rationale_better"] is True
    assert out["prediction_mixture_spreads_rationales"] is True
    assert sorted(out["tests"]) == ["raw", "ts"] and len(out["tests"]["raw"]) == 4
    assert sorted(out["per_seed"]["lambda0"]) == list(D.SEEDS)  # the collapsed seed is kept
    maj = out["majority_accuracy_reported"]
    assert sorted(maj) == ["mixture majority_acc", "single majority_acc"]
    row = maj["single majority_acc"]  # lambda 0 is closer to q: higher majority accuracy, and +1 when significant
    assert row["diff"] > 0 and row["outcome"] == (1 if row["p_holm"] < 0.05 else 0)
    x = R.score_soft(root / "qwen3-1.7b" / "cn_s1_lam0_seed17.json")
    it, u = R.items("chaosnli", "test", 40)[0], json.loads((root / "qwen3-1.7b" / "cn_s1_lam0_seed17.json")
                                                         .read_text())["test"]["dists"]["single"][0]
    assert x["single"]["raw"]["per_item_sqdist"][0] == pytest.approx(sum((a - b) ** 2 for a, b in zip(u, it["q"])))

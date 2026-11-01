"""openjev_rlcd.npj_report: scoring from saved log-probabilities on a run's own split, agreement with the earlier
scoring (evaluation.scored) on a released run when the probability floor is 1e-45, E-AURC, Holm, the paired
differences and the selection rule."""
import json
import math

import numpy as np
import pytest
import torch

from openjev_rlcd import evaluation as S
from openjev_rlcd import npj_report as R
from openjev_rlcd.tasks import TASKS

RUN = R.ROOT / "results" / "mp_grpokl_seed17.json"  # a released GRPO+TS run (MMLU-Pro, Qwen3-1.7B)


@pytest.fixture
def released(monkeypatch):
    if not RUN.exists():
        pytest.skip("released run files not present")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        R.splits("mmlupro")
    except Exception as e:  # pragma: no cover - only without the cached dataset
        pytest.skip(f"MMLU-Pro not cached: {e}")
    return json.loads(RUN.read_text())


def test_agrees_with_the_earlier_scoring_at_the_low_floor(released, monkeypatch):
    run = released
    monkeypatch.setattr(S, "SPLITS", TASKS["mmlupro"]["load"]())

    def low(dists):
        z = torch.full((len(dists), S.KMAX), float("-inf"), dtype=torch.float64)
        for i, p in enumerate(dists):
            z[i, :len(p)] = torch.tensor(p, dtype=torch.float64).clamp_min(1e-45).log()
        return z
    monkeypatch.setattr(S, "padded_logits", low)
    raw, ts, t = S.scored(run["test"]["dists"]["single"], run["dev_dists"]["single"], run["config"]["seed"])
    new = R.score_run(RUN)["single"]
    assert new["split"] == "test" and math.isclose(new["T"], t)
    for k in ("acc", "brier", "aurc", "ece", "oracle_cov@0.2", "cov@0.2"):
        assert math.isclose(new["ts"][k], ts[k], rel_tol=1e-9, abs_tol=1e-12), k
        assert math.isclose(new["raw"][k], raw[k], rel_tol=1e-9, abs_tol=1e-12), k


def test_logps_are_used_without_a_floor(monkeypatch):
    """two-option items whose wrong option has log-probability -60: probabilities underflow below the old 1e-12 floor
    but the saved log-probabilities keep the order and the temperature sees them"""
    its = [dict(uid=str(i), q=[1.0, 0.0] if i % 4 else [0.0, 1.0], counts=[1, 0] if i % 4 else [0, 1])
           for i in range(40)]
    rows = [[0.0, -60.0 - i] for i in range(40)]  # always confident in option 0; wrong on every 4th item
    z = R.pad(rows, [2] * 40)
    m = R.margins_of(z)
    out = R.score_logits(z, its, m)
    assert out["acc"] == 0.75 and len(set(out["per_item_rank"])) == 40  # no ties: the margins differ
    t = R.fit_temperature(list(z.float()), its, 17)
    assert t > 10  # overconfident: TS softens a lot


def test_eaurc_and_decomposition():
    assert R.eaurc(sum((i - 7) / i for i in range(8, 11)) / 10, 3, 10) == pytest.approx(0.0)
    d = R.toplabel_decomposition([0.95, 0.95, 0.55, 0.55], [1, 1, 1, 0])
    assert d["uncertainty"] == pytest.approx(0.75 * 0.25)
    assert d["reliability"] == pytest.approx(0.5 * 0.05 ** 2 + 0.5 * 0.05 ** 2)


def test_holm():
    adj = R.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_paired_diff_of_an_arm_with_itself(released):
    s = R.score_run(RUN)
    for metric in ("acc", "brier", "aurc", "eaurc", "oracle_cov@0.2"):
        d, lo, hi, p = R.paired_diff([s], [s], metric, n_boot=200)
        assert d == 0 and lo == 0 and hi == 0 and p == 1.0
    with pytest.raises(ValueError):
        R.paired_diff([s], [s], "acurc")


def test_verbal_clip():
    lp = R.from_probs([[1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], verbal=True)
    assert np.allclose(np.exp(lp[0]), [0.999, 0.001])
    assert np.allclose(np.exp(lp[1]), [0.001 / 3, 0.001 / 3, 0.999, 0.001 / 3])


def test_clopper_pearson_upper_bound():
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((0, 10), (3, 50), (20, 100), (99, 100)):
        want = stats.beta.ppf(0.9, k + 1, n - k)
        assert R.cp_upper(k, n, 0.1) == pytest.approx(want, abs=1e-6)
    assert R.cp_upper(5, 5, 0.1) == 1.0 and R.cp_upper(0, 0, 0.1) == 1.0


def test_risk_threshold_on_a_fixed_grid():
    # 400 items: confidence 0.99 and always right (200), 0.80 and 10% wrong (100), 0.60 and half wrong (100)
    conf = [0.99] * 200 + [0.80] * 100 + [0.60] * 100
    wrong = [0] * 200 + [int(i < 10) for i in range(100)] + [int(i < 50) for i in range(100)]
    lam = R.risk_threshold(conf, wrong, eps=0.2, delta=0.1)
    assert lam == 0.65  # 0.65-0.80 select the same 300 items; 0.60 adds 50 errors (bound above 0.2)
    cov, risk = R.apply_threshold(conf, wrong, lam)
    assert cov == 0.75 and risk == pytest.approx(10 / 300)
    assert R.risk_threshold([0.9] * 10, [1] * 10, eps=0.2) is None  # nothing passes
    assert R.apply_threshold(conf, wrong, None) == (0.0, None)
    assert R.risk_threshold(conf, wrong, eps=0.01, delta=0.1) == 0.99 or R.risk_threshold(conf, wrong, eps=0.01) is None


def test_decide_central():
    cells = ["a", "b", "c", "d"]
    good = {(c, m): (-0.1, -0.2, -0.05, 1e-6) for c in cells for m in ("brier", "aurc")}
    assert R.decide_central(good, cells)[0] == "supported"
    mixed = dict(good)
    mixed[("d", "brier")] = (0.1, 0.05, 0.2, 1e-6)  # one significant reversal on an MMLU-Pro backbone
    assert R.decide_central(mixed, cells)[0] == "inconclusive"
    null = {(c, m): (0.0, -0.1, 0.1, 0.9) for c in cells for m in ("brier", "aurc")}
    assert R.decide_central(null, cells)[0] == "refuted"


def test_hierarchical_bootstrap_of_an_arm_with_itself(released):
    s = R.score_run(RUN)
    for metric in ("brier", "aurc"):
        d, lo, hi = R.hier_boot([s], [s], metric, n_boot=50)
        assert d == pytest.approx(0.0) and lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_hier_boot_rejects_other_metrics(released):
    s = R.score_run(RUN)
    for bad in ("eaurc", "oracle_cov@0.2", "acurc"):
        with pytest.raises(ValueError):
            R.hier_boot([s], [s], bad, n_boot=5)


def test_selection_refuses_when_every_setting_collapsed(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "score_run", lambda p, verbal=None: {"single": dict(collapsed=True, split="test")})
    for m, t in R.CELLS_TUNE:
        d = tmp_path / m
        d.mkdir(exist_ok=True)
        for tag in ("b0", "b0.01"):
            (d / f"{R.PX[t]}_grpo_{tag}_seed17.json").write_text('{"test": {}}')
    choice, rows, status = R.select_grpo(root=tmp_path, tags=("b0", "b0.01"))
    assert choice is None and status == "all collapsed" and all(r[0] == float("inf") for r in rows.values())
    (tmp_path / "qwen3-1.7b" / "mp_grpo_b0.01_seed17.json").rename(tmp_path / "moved.json")  # one run unfinished
    choice, rows, status = R.select_grpo(root=tmp_path, tags=("b0", "b0.01"))
    assert choice is None and status == "incomplete" and rows["b0.01"][2] == "incomplete"

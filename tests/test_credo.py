"""CREDO (train.py --objective credo; amendment A4 of paper/npj/PROTOCOL.md): its settings and loss by hand; the cut of a
generated analysis, including left-padded batches, and the forced slot after it; the slot readout of Rationales.forward
(the answer readout u, the rationale and the analysis log-probabilities are those of a full forward pass at exactly the
right positions; the forced ids are not scored; the calibration loss reaches the body); a training step whose arguments
to the loss are checked against the sampled rollouts; an eval-only pass; the argument checks; the analysis-length probe
and the cap rule; the report's scoring from the saved slot gap (K > 2 and K = 2); the closest-methods families of
npj_decide; the job plan; and the prompts of every other arm, byte-identical to the second install's."""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch

from openjev_rlcd import credo_probe
from openjev_rlcd import npj_decide as D
from openjev_rlcd import npj_report as R
from openjev_rlcd import train
from openjev_rlcd.tasks import CONFIDENCE_REQUEST, CREDO_REQUEST, TASKS, prompt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_baselines import _tiny  # noqa: E402
from test_npj_decide import write_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _tok():
    from transformers import AutoTokenizer
    snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("Qwen3 tokenizer not cached")
    return AutoTokenizer.from_pretrained(str(snaps[0].parent))


def test_settings_are_the_amendments():
    assert train.CREDO == dict(gamma=0.5, alpha=0.03, kappa=0.25, w_min=0.5, w_max=2.0, beta=0.5, warmup=30)
    assert train.CREDO_SLOT_WORDS == ("high", "low")
    assert (train.CREDO_OPEN, train.CREDO_CLOSE, train.CREDO_SLOT) == ("\n<analysis>", "</analysis>", "<confidence>")
    assert credo_probe.CAPS == (64, 128, 192, 256) and credo_probe.SHARE == 0.9


GOLDEN = {  # sha256[:16] of prompt() in the second install (code_version_20261019-114145), Qwen3 template, dev item 0
    ('gsm8kv', False, False): '5c2043e19b6763f8', ('gsm8kv', False, True): 'bf8a99a4ca0d2268',
    ('gsm8kv', True, False): '6fbc9f2fa49b2188', ('gsm8kv', True, True): 'ef59764f5daf23b5',
    ('mmlupro', False, False): 'fd5e403ffe6c0820', ('mmlupro', False, True): '107a9c669fa1c594',
    ('mmlupro', True, False): '56f0828c55f0e6c9', ('mmlupro', True, True): 'e7b8718c55abbdb9',
    ('arc', False, False): '391e59a655f9dec5', ('arc', False, True): '6c151dbc27fb4db4',
    ('arc', True, False): 'a6d5e29bbd742890', ('arc', True, True): '56364cda1d39a828',
    ('chaosnli', False, False): 'aef6fd860e8e0308', ('chaosnli', False, True): '6632243ba06c713d',
    ('chaosnli', True, False): 'b7d7c3ada5474665', ('chaosnli', True, True): 'e745b4b9830d9878',
}


def test_other_prompts_are_unchanged(monkeypatch):
    tok = _tok()
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    for (task, conf, think), want in GOLDEN.items():
        try:
            it = TASKS[task]["load"]()["dev"][0]
        except Exception as e:  # pragma: no cover - only without the cached datasets
            pytest.skip(f"{task} not cached: {e}")
        got = hashlib.sha256(prompt(tok, it, task, confidence=conf, thinking=think).encode()).hexdigest()[:16]
        assert got == want, (task, conf, think)
    it = TASKS["gsm8kv"]["load"]()["dev"][0]
    p = prompt(tok, it, "gsm8kv", credo=True)
    assert CREDO_REQUEST in p and CONFIDENCE_REQUEST not in p and "<confidence>high</confidence>" in p
    with pytest.raises(AssertionError):
        prompt(tok, it, "gsm8kv", confidence=True, credo=True)


def test_credo_loss_by_hand():
    a = torch.tensor([1.0, 0.0, 1.0, 0.0])
    c = torch.tensor([0.9, 0.6, 0.4, 0.2], requires_grad=True)
    f = torch.tensor([1.0, 1.0, 0.0, 1.0])
    lpa = torch.tensor([-3.0, -5.0, -4.0, -6.0], requires_grad=True)
    lpn = torch.tensor([-1.0, -2.0, -1.5, -0.5], requires_grad=True)
    n = torch.tensor([10.0, 20.0, 30.0, 40.0])
    loss, aux = train.credo_loss(a, c, f, lpa, lpn, n, ramp=1.0)
    t = [0.75, 0.25, 0.75, 0.25]  # 0.5 a_i + 0.5 * (the group's accuracy 0.5)
    assert torch.allclose(aux["t"], torch.tensor(t))
    r_ans = [1.5, 0.5, 1.0, 0.5]  # a_i + 0.5 F_i
    r_conf = [1 - 0.15 ** 2, 1 - 0.35 ** 2, 0.0, 1 - 0.05 ** 2]  # F_i (1 - (t_i - c_i)^2)

    def norm(r):
        mu = sum(r) / len(r)
        sd = math.sqrt(sum((x - mu) ** 2 for x in r) / len(r))
        return [(x - mu) / (sd + 1e-4) for x in r]
    dis = [0.1, 0.6, 0.6, 0.2]  # |a_i - c_i|, mean 0.375
    w = [1 + 0.25 * (d - 0.375) for d in dis]  # within [0.5, 2]: no clip
    assert torch.allclose(aux["w"], torch.tensor(w))
    pg = [(wi * aa * x + ac * y) / ni for wi, aa, x, ac, y, ni in
          zip(w, norm(r_ans), lpa.tolist(), norm(r_conf), lpn.tolist(), n.tolist())]
    cal = sum((ci - ti) ** 2 for ci, ti in zip(c.tolist(), t)) / 4
    assert math.isclose(loss.item(), -sum(pg) / 4 + 0.03 * cal, rel_tol=1e-5, abs_tol=1e-6)
    loss.backward()
    # the readout gets its gradient from the calibration loss only (the rewards use it without gradient)
    torch.testing.assert_close(c.grad, 0.03 * 2 * (c.detach() - torch.tensor(t)) / 4)
    torch.testing.assert_close(lpa.grad, -torch.tensor([wi * aa / ni for wi, aa, ni in zip(w, norm(r_ans), n.tolist())]) / 4)
    torch.testing.assert_close(lpn.grad, -torch.tensor([ac / ni for ac, ni in zip(norm(r_conf), n.tolist())]) / 4)


def test_credo_warmup_and_clip(monkeypatch):
    a, f = torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.ones(4)
    c = torch.tensor([0.9, 0.6, 0.4, 0.2], requires_grad=True)
    lp, n = torch.zeros(4), torch.ones(4)
    loss, aux = train.credo_loss(a, c, f, lp, lp, n, ramp=0.0)  # step 0 of the warm-up: no weights, no calibration
    assert torch.equal(aux["w"], torch.ones(4)) and loss.item() == 0.0
    monkeypatch.setitem(train.CREDO, "kappa", 10.0)  # a strength at which the clip binds
    _, aux = train.credo_loss(a, c, f, lp, lp, n, ramp=1.0)
    assert aux["w"].min().item() == 0.5 and aux["w"].max().item() == 2.0


def test_cut_analysis_and_slot_suffix():
    tok = _tok()
    rat = train.Rationales(tok, "gsm8kv", credo=True, analysis_cap=64)
    stop = set(rat.eos)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad = tok.pad_token_id
    closed = enc(" I am fairly sure.</analysis>")
    assert rat.cut_analysis(closed + enc(" more text") + [pad, pad], stop) == (closed, True)
    eos = enc(" I think") + [tok.eos_token_id]
    assert rat.cut_analysis(eos + [pad], stop) == (eos, False)
    capped = enc(" this goes on and on")
    assert rat.cut_analysis(capped, stop) == (capped, False)
    tail = lambda ids, ok: tok.decode(ids + rat.slot_suffix(ids, ok))  # noqa: E731
    assert tail(closed, True).endswith("sure.</analysis>\n<confidence>")
    nl = enc(" fine.</analysis>\n")  # the closing token carries the newline
    assert rat.cut_analysis(nl, stop)[1] and tail(nl, True).endswith("fine.</analysis>\n<confidence>")
    run_on = enc(" ok.</analysis><confidence>")  # the closing token '><' already starts the slot
    cut, ok = rat.cut_analysis(run_on, stop)
    assert ok and tok.decode(cut).endswith("</analysis><") and len(cut) < len(run_on)
    assert tail(cut, True).endswith("ok.</analysis><confidence>")
    assert tail(capped, False).endswith("on and on\n</analysis>\n<confidence>")
    assert tail(eos, False).endswith("<|im_end|>\n</analysis>\n<confidence>")
    (t1, f1, e1), (t2, f2, e2) = rat.slot_tails([0, 1], [(closed, True), (capped, False)])
    head = lambda a: [rat.label_ids[a]] + rat.credo_open  # noqa: E731
    assert f1 == f2 == 1 + len(rat.credo_open)
    assert t1 == head(0) + closed + rat.slot_suffix(closed, True) and e1 == len(rat.slot_suffix(closed, True))
    assert t2 == head(1) + capped + rat.slot_suffix(capped, False) and e2 == len(rat.slot_suffix(capped, False))
    assert tok.decode(rat.slot_ids[0]) == "high" and tok.decode(rat.slot_ids[1]) == "low"


def test_analyses_of_a_left_padded_batch(monkeypatch):
    """analyses() on contexts of unequal length: the generated block of each row is cut at its close, its EOS or the cap,
    and the padding after a finished row is dropped"""
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "gsm8kv", micro=8, credo=True, analysis_cap=6)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad, eos = tok.pad_token_id, tok.eos_token_id
    closed, said = enc(" ok.</analysis>"), enc(" maybe")
    gen_rows = [closed + [pad] * (6 - len(closed)), said + [eos] + [pad] * (5 - len(said)), enc(" a b c d e f")[:6]]
    assert all(len(r) == 6 for r in gen_rows)
    seen = {}

    def fake_generate(input_ids, attention_mask, **kw):
        seen.update(ids=input_ids.clone(), mask=attention_mask.clone(), kw=kw)
        return torch.cat([input_ids, torch.tensor(gen_rows)], 1)
    monkeypatch.setattr(model, "generate", fake_generate)
    rows = [([1, 2, 3], [11, 12, 13, 14], False), ([1, 2], [11], True), ([5], [11, 12], False)]
    out = rat.analyses(model, rows, [0, 1, 0], sample=False)
    assert out == [(closed, True), (said + [eos], False), (gen_rows[2], False)]
    ctx = [p + r + (rat.suffix_for(r) if f else []) + [rat.label_ids[a]] + rat.credo_open
           for (p, r, f), a in zip(rows, [0, 1, 0])]
    width = max(map(len, ctx))
    for j, x in enumerate(ctx):  # left-padded
        assert seen["ids"][j, width - len(x):].tolist() == x and seen["mask"][j].sum().item() == len(x)
    assert seen["kw"]["max_new_tokens"] == 6 and seen["kw"]["stop_strings"] == ["</analysis>"]
    assert seen["kw"]["do_sample"] is False


def test_forward_slot_readout(monkeypatch):
    tok, model = _tiny(monkeypatch)
    with torch.no_grad():  # sharpen the tiny model so that every token's log-probability differs visibly
        model.lm_head.weight.mul_(100)
    rat = train.Rationales(tok, "gsm8kv", credo=True, analysis_cap=64)
    g = torch.Generator().manual_seed(1)
    rnd = lambda n: torch.randint(10, 5000, (n,), generator=g).tolist()  # noqa: E731
    group = [(rnd(7), rnd(5), False), (rnd(4), rnd(9), True), (rnd(6), rnd(3), False)]
    ans = [(rnd(4), True), (rnd(6), False), (rnd(2), True)]
    tails = rat.slot_tails([0, 1, 1], ans)
    u, lp = rat.forward(model, group, grad=False, need_logp=True, k=2)
    u2, lp2, lpt, zs = rat.forward(model, group, grad=False, need_logp=True, k=2, tails=tails, slot=rat.slot_ids)
    torch.testing.assert_close(u2, u, atol=2e-2, rtol=0)  # bf16 autocast: other padding widths round differently
    torch.testing.assert_close(lp2, lp, rtol=2e-2, atol=1.0)
    for j, ((p, r, f), (tail, fx, end)) in enumerate(zip(group, tails)):  # one sequence at a time
        seq = p + r + (rat.suffix if f else []) + tail
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=torch.tensor([seq])).logits[0].float()
        tok_lp = [logits[i - 1].log_softmax(-1)[seq[i]].item() for i in range(1, len(seq))]
        start, stop = len(seq) - len(tail) + fx, len(seq) - end
        assert stop - start == len(ans[j][0])
        window = lambda a, b: sum(tok_lp[a - 1:b - 1])  # noqa: E731  (log-probabilities of seq[a:b])
        want = window(start, stop)
        err = abs(lpt[j].item() - want)
        assert err < 0.02 * abs(want) + 1.0
        assert err < abs(window(start - 1, stop - 1) - want) and err < abs(window(start + 1, stop + 1) - want)
        torch.testing.assert_close(zs[j], logits[-1][list(rat.slot_ids)], rtol=2e-2, atol=1.0)
    u3, none, zs3 = rat.forward(model, group, grad=False, k=2, tails=tails, slot=rat.slot_ids)  # readout-only pass
    assert none is None and zs3.shape == (3, 2)
    torch.testing.assert_close(zs3, zs, atol=1e-4, rtol=1e-4)
    *_, zs4 = rat.forward(model, group, grad=True, k=2, tails=tails, slot=rat.slot_ids)
    model.zero_grad()
    (zs4[:, 0] - zs4[:, 1]).sigmoid().sum().backward()  # the calibration loss reaches the body through c
    w = model.model.layers[0].mlp.down_proj.weight.grad
    assert w is not None and w.abs().sum() > 0


def _saved(tok, model, tmp_path):
    tok.save_pretrained(tmp_path / "m")
    model.save_pretrained(tmp_path / "m")
    return str(tmp_path / "m")


def test_credo_training_step_and_eval_only(monkeypatch, tmp_path):
    tok, model = _tiny(monkeypatch)
    path = _saved(tok, model, tmp_path)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    orig_sample, orig_analyses, orig_forward, orig_loss = (train.Rationales.sample, train.Rationales.analyses,
                                                           train.Rationales.forward, train.credo_loss)
    rec = dict(groups=[], analyses=[], forward=[], loss=[])

    def sample(self, gen_model, items, m, max_new_tokens, micro=None):
        out = orig_sample(self, gen_model, items, m, max_new_tokens, micro)
        if m == 4:  # training: the first rationale of each question wrote the marker itself
            out = [[(p, r[:2] + enc("\nAnswer:"), False) if i == 0 else (p, r, f) for i, (p, r, f) in enumerate(g)]
                   for g in out]
            rec["groups"] += out
        return out

    def analyses(self, gen_model, rows, answers, sample):
        out = orig_analyses(self, gen_model, rows, answers, sample)
        if sample:  # training: the first two analyses closed
            out = [(enc(" sure.</analysis>"), True) if i < 2 else x for i, x in enumerate(out)]
            rec["analyses"].append(out)
        return out

    def forward(self, model_, group, grad, *a, **kw):
        res = orig_forward(self, model_, group, grad, *a, **kw)
        if grad and kw.get("slot") is not None:
            rec["forward"].append(res)
        return res
    monkeypatch.setattr(train.Rationales, "sample", sample)
    monkeypatch.setattr(train.Rationales, "analyses", analyses)
    monkeypatch.setattr(train.Rationales, "forward", forward)
    monkeypatch.setattr(train, "credo_loss", lambda *a, **kw: rec["loss"].append(a) or orig_loss(*a, **kw))
    base = ["train", "--task", "gsm8kv", "--model", path, "--items-per-step", "2", "--m", "4", "--m-eval", "2",
            "--max-new-tokens", "5", "--dev-cap", "2", "--test-cap", "2", "--objective", "credo",
            "--credo-analysis-tokens", "7"]
    out, ck = tmp_path / "credo.json", tmp_path / "credo.pt"
    monkeypatch.setattr(sys, "argv", base + ["--steps", "2", "--eval-every", "2", "--save-checkpoint", str(ck),
                                             "--output", str(out)])
    train.main()
    assert [a[-1] for a in rec["loss"]] == [1 / 30, 1 / 30, 2 / 30, 2 / 30]  # one call per question, linear warm-up
    for (correct, c, formatted, logp_ans, logp_an, n_tok, _), g, an, res in zip(
            rec["loss"], rec["groups"], rec["analyses"], rec["forward"]):
        assert formatted.tolist() == [1.0, 0.0, 0.0, 0.0]  # marker written (row 0) and analysis closed (rows 0, 1)
        assert n_tok.tolist() == [len(r) + 1 + len(ids) for (_, r, _), (ids, _) in zip(g, an)]
        assert torch.equal(logp_an, res[2]) and torch.equal(c, (res[3][:, 0] - res[3][:, 1]).sigmoid())
        assert set(correct.tolist()) <= {0.0, 1.0} and c.requires_grad and logp_ans.requires_grad
    log = json.loads(out.read_text())
    assert log["config"]["objective"] == "credo" and log["config"]["credo_analysis_tokens"] == 7
    assert log["config"]["kl_beta"] == 0.0 and log["credo"]["analysis_tokens"] == 7 and log["credo"]["alpha"] == 0.03
    test = log["test"]
    assert "logps" not in test and "dev_logps" not in log and len(test["dists"]["single"]) == 2
    for sl in (test["slot"], log["dev_slot"]):  # every rationale's decision and gap (M = 2)
        assert len(sl["answer"]) == len(sl["gap"]) == 2 and all(len(x) == 2 for x in sl["answer"] + sl["gap"])
    for p, a, gp in zip(test["dists"]["single"], test["slot"]["answer"], test["slot"]["gap"]):
        c = 1 / (1 + math.exp(-gp[0]))  # the saved readout is c on the decision
        assert math.isclose(p[a[0]], c, rel_tol=1e-5) and math.isclose(sum(p), 1.0, abs_tol=1e-6)
    assert 0 <= test["rationale"]["analysis_unclosed_rate"] <= 1 and 0 < test["rationale"]["mean_slot_confidence"] < 1
    assert all("slot" not in c["dev"] for c in log["curve"])
    ev = tmp_path / "ev.json"
    monkeypatch.setattr(sys, "argv", base + ["--eval-only", "--checkpoint", str(ck), "--output", str(ev)])
    train.main()
    e = json.loads(ev.read_text())
    assert e["credo"]["gamma"] == 0.5 and "logps" not in e["test"] and "slot" in e["test"] and "dev_slot" in e


@pytest.mark.parametrize("bad", [
    [],  # the analysis cap is required
    ["--credo-analysis-tokens", "0"],
    ["--credo-analysis-tokens", "64", "--kl-beta", "0.04"],  # CREDO has no KL
    ["--credo-analysis-tokens", "64", "--kl-beta", "0.04", "--kl-tokens", "0.01"],
    ["--credo-analysis-tokens", "64", "--thinking"],
    ["--credo-analysis-tokens", "64", "--no-score-function"],
    ["--credo-analysis-tokens", "64", "--sf-coef", "0.3"],
    ["--credo-analysis-tokens", "64", "--pathwise-warmup", "5"],
    ["--credo-analysis-tokens", "64", "--rlcr-advantage", "loo"],
    ["--credo-analysis-tokens", "64", "--readout-score", "log"],
    ["--credo-analysis-tokens", "64", "--init-checkpoint", "x.pt"],
    ["--credo-analysis-tokens", "64", "--snapshot-steps", "1", "--steps", "2", "--save-checkpoint", "x.pt"],
])
def test_credo_argument_checks(monkeypatch, bad):
    monkeypatch.setattr(sys, "argv", ["train", "--objective", "credo"] + bad + ["--output", "x.json"])
    with pytest.raises(SystemExit) as e:
        train.main()
    assert e.value.code == 2


def test_cap_only_with_credo(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--objective", "rlcr", "--credo-analysis-tokens", "64", "--output", "x"])
    with pytest.raises(SystemExit) as e:
        train.main()
    assert e.value.code == 2
    sys.path.insert(0, str(ROOT / "scripts"))
    import plan_npj
    for objective in ("rlcd", "grpo", "rlcr"):  # the parsed configuration of every other arm gains no key
        new, _ = plan_npj.parse("train", f"--task gsm8kv --objective {objective}")
        assert "credo_analysis_tokens" not in new
    new, _ = plan_npj.parse("train", "--task gsm8kv --objective credo --credo-analysis-tokens 64")
    assert new["credo_analysis_tokens"] == 64


def test_probe_rule_and_run(monkeypatch, tmp_path):
    assert credo_probe.close_lengths([([1, 2, 3], True), ([1] * 9, False)]) == [3, None]
    shares = credo_probe.shares_within([10, 70, 130, None] * 25)
    assert shares == {64: 0.25, 128: 0.5, 192: 0.75, 256: 0.75} and credo_probe.choose(shares) is None
    assert credo_probe.choose({64: 0.5, 128: 0.9, 192: 0.95, 256: 1.0}) == 128
    tok, model = _tiny(monkeypatch)
    path = _saved(tok, model, tmp_path)
    out = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["credo_probe", "--task", "gsm8kv", "--model", path, "--n", "2", "--m", "2",
                                      "--max-new-tokens", "4", "--output", str(out)])
    credo_probe.main()
    d = json.loads(out.read_text())
    assert d["analyses"] == 4 and len(d["close_lengths_sampled"]) == len(d["close_lengths_greedy"]) == 4
    assert set(d["test"]["share_closed_within_sampled"]) == {str(c) for c in credo_probe.CAPS}
    assert '"test": {' in out.read_text()


def test_credo_cap_and_plan(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import plan_npj
    probe = tmp_path / "runs/npj/credo_probe"
    probe.mkdir(parents=True)
    monkeypatch.setattr(plan_npj, "ROOT", tmp_path)
    picks = {("gv", "qwen3-0.6b"): 64, ("mp", "qwen3-0.6b"): 128, ("gv", "qwen3-1.7b"): 64, ("mp", "qwen3-1.7b"): None}
    for (px, m), cap in list(picks.items())[:3]:
        (probe / f"{px}_{m}.json").write_text(json.dumps(dict(test={}, chosen_cap=cap)))
    with pytest.raises(SystemExit):
        plan_npj.credo_cap()  # a cell has not been probed
    (probe / "mp_qwen3-1.7b.json").write_text(json.dumps(dict(test={}, chosen_cap=None)))
    assert plan_npj.credo_cap() == 256  # a cell that reaches 90% nowhere takes the largest candidate
    (probe / "mp_qwen3-1.7b.json").write_text(json.dumps(dict(test={}, chosen_cap=64)))
    assert plan_npj.credo_cap() == 128  # the largest pick over the cells
    jobs = plan_npj.credo_probe_jobs()
    assert [j.name for j in jobs] == ["gv_qwen3-0.6b", "mp_qwen3-0.6b", "gv_qwen3-1.7b", "mp_qwen3-1.7b"]
    assert all(j.out == "runs/npj/credo_probe" and j.module == "credo_probe" for j in jobs)
    assert "--max-new-tokens 320" in jobs[0].args and "--max-new-tokens 256" in jobs[1].args
    if not plan_npj.ref_path("credo", "qwen3-0.6b", "gv", 17).exists():
        pytest.skip("the paper's RLCR runs are not present (set OPENJEV_REF_ROOT)")
    jobs = [plan_npj.fresh_jobs("credo", m, t, s) for s in plan_npj.SEEDS for m in plan_npj.QWEN
            for t in ("gsm8kv", "mmlupro")]
    assert len(jobs) == 12 and len({j.name + j.out for j in jobs}) == 12
    assert not plan_npj.check(jobs)  # the parity check against the RLCR runs passes
    for j in jobs:
        new, _ = plan_npj.parse(j.module, j.args)
        assert new["objective"] == "credo" and new["credo_analysis_tokens"] == 128 and new["kl_beta"] == 0.0
        assert new["test_split"] == "fresh" and new["extra_splits"] == "" and new["save_checkpoint"].endswith(f"/{j.name}.pt")


def _credo_run(path, task, seed, rng, acc, n_test=60, n_dev=30, split="fresh"):
    """a synthetic CREDO run file: per item M = 2 rationales with a decision and a slot gap"""
    def block(its):
        ans, gaps, single = [], [], []
        for it in its:
            k, y = len(it["q"]), it["q"].index(1.0)
            a = [y if rng.random() < acc else rng.choice([j for j in range(k) if j != y]) for _ in range(2)]
            g = [rng.uniform(-3, 9) for _ in range(2)]
            c = 1 / (1 + math.exp(-g[0]))
            ans.append(a)
            gaps.append(g)
            single.append([c if j == a[0] else (1 - c) / (k - 1) for j in range(k)])
        return dict(answer=ans, gap=gaps), single
    tests, devs = R.items(task, split, n_test), R.items(task, "dev", n_dev)
    st, dt = block(tests), block(devs)
    run = dict(config=dict(task=task, seed=seed, test_split=split, objective="credo"),
               dev_dists=dict(single=dt[1], mixture=dt[1]), dev_slot=dt[0], dev_uids=[it["uid"] for it in devs],
               test=dict(dists=dict(single=st[1], mixture=st[1]), slot=st[0], uids=[it["uid"] for it in tests],
                         rationale=dict(mean_rationale_tokens=50, length_limit_rate=0.0, repetition_ratio=0.1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))
    return run


@pytest.fixture
def cached(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        R.splits("mmlupro")
        R.splits("gsm8kv")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"datasets not cached: {e}")


def test_report_scores_credo_from_the_slot_gap(tmp_path, cached):
    lsig = lambda x: -math.log1p(math.exp(-x))  # noqa: E731
    slot = dict(answer=[[2, 0], [1, 1], [0, 3]], gap=[[30.0, 0.0], [-4.0, 1.0], [0.5, 2.0]])
    lp = R.from_slot(slot, [4, 2, 2])
    assert lp[0][2] == pytest.approx(lsig(30.0)) and lp[0][0] == pytest.approx(lsig(-30.0) - math.log(3))
    assert lp[0][2] > math.log(0.999)  # no clip: a confident slot keeps its gap
    assert lp[1][1] == pytest.approx(lsig(-4.0)) and lp[1][0] == pytest.approx(lsig(4.0))  # K = 2: the other answer wins
    lpk = R.from_slot(dict(answer=[[3]], gap=[[-3.0]]), [10])[0]  # K > 2 and c < 1/K: raised, a stays the decision
    assert max(range(10), key=lambda j: lpk[j]) == 3 and math.exp(lpk[3]) == pytest.approx(0.1 + 1e-6)
    for zero in (0.0, -0.0):  # K = 2, c = 0.5 exactly (bf16 logits can tie): a tie, decided as the first option
        tie = R.from_slot(dict(answer=[[1]], gap=[[zero]]), [2])[0]
        assert tie[0] == tie[1] == pytest.approx(-math.log(2)) and tie.index(max(tie)) == 0
    rng = random.Random(1)
    for task in ("mmlupro", "gsm8kv"):
        p = tmp_path / f"{task}.json"
        run = _credo_run(p, task, 17, rng, acc=0.7)
        out = R.score_run(p)
        assert set(out) == {"single"}
        its = R.items(task, "fresh", 60)
        dec = []  # the decision: a on K > 2, the more probable of a and the other answer on K = 2
        for it, a, g in zip(its, run["test"]["slot"]["answer"], run["test"]["slot"]["gap"]):
            k = len(it["q"])
            dec.append(a[0] if (k > 2 or g[0] > 0) else (1 - a[0] if g[0] < 0 else 0))  # a tie: the first option
        acc = sum(d == it["q"].index(1.0) for d, it in zip(dec, its)) / 60
        assert out["single"]["raw"]["acc"] == pytest.approx(acc)
    bad = dict(json.loads(p.read_text()))
    bad["test"] = {k: v for k, v in bad["test"].items() if k != "slot"}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        R.score_run(p)


def test_closest_methods_families(tmp_path, cached):
    rng = random.Random(0)
    root = tmp_path / "npj"
    for m, px in D.QWEN_CELLS:
        task = "gsm8kv" if px == "gv" else "mmlupro"
        for s in D.SEEDS:
            write_run(root / m / f"{px}_warm400_seed{s}.json", task, s, rng, 0.85, 0.6, 0.95)
            _credo_run(root / m / f"{px}_credo_seed{s}.json", task, s, rng, 0.5)
    D._CACHE.clear()
    out = D.decide(root)
    assert out["closest_rlcr"] == "pending"
    fam = out["closest_credo"]["family"]
    assert len(fam) == 8 and all(k.split()[1] in ("brier", "aurc") for k in fam)
    assert set(out["closest_credo"]["secondary"]) == {"acc", "nll", "eaurc"}
    assert all(row["outcome"] == 1 for k, row in fam.items() if k.endswith("brier"))  # the better arm wins
    json.dumps(out, default=str)

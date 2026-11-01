"""The training-side additions of the npj protocol (paper/npj/PROTOCOL.md): the fresh and calib splits, the saved
log-probabilities of the readouts, the readout-only linear score, the --sf-reward correct control, snapshots of the
weights with a test pass at intermediate steps, --test-split, and the checkpoint of the direct-answer baseline."""
import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from openjev_rlcd import train, train_direct
from openjev_rlcd.estimators import single_surrogate
from openjev_rlcd.tasks import gsm8k_verify, mmlu_pro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_baselines import _tiny  # noqa: E402


def _disjoint(splits, names):
    seen = {}
    for name in names:
        uids = [it["uid"] for it in splits[name]]
        assert len(uids) == len(set(uids)), name
        for other, prev in seen.items():
            assert not prev & set(uids), (name, other)
        seen[name] = set(uids)


def test_gsm8kv_fresh_and_calib_are_unused_train_problems():
    s = gsm8k_verify.load_items()
    assert len(s["fresh"]) == len(s["test"]) == gsm8k_verify.FRESH == 1260
    assert len(s["calib"]) == len(s["pool"]) - 1260 > 1000
    assert s["pool"][:1260] == s["fresh"] and s["pool"][1260:] == s["calib"]
    _disjoint(s, ["train", "dev", "test", "fresh", "calib"])
    assert all(it["split"] == "train" for it in s["fresh"] + s["calib"])
    again = gsm8k_verify.load_items()
    assert [it["uid"] for it in again["fresh"]] == [it["uid"] for it in s["fresh"]]  # deterministic


def test_mmlupro_fresh_and_calib_are_unused_questions(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        s = mmlu_pro.load_items()
    except Exception as e:  # pragma: no cover - only without the cached dataset
        pytest.skip(f"MMLU-Pro not cached: {e}")
    assert len(s["fresh"]) == len(s["calib"]) == 1000 and len(s["pool"]) == 6032
    seen = {mmlu_pro._stem(it["question"]) for it in s["train"] + s["dev"]}
    unseen = [it for it in s["pool"] if mmlu_pro._stem(it["question"]) not in seen]
    assert unseen[:1000] == s["fresh"] and unseen[1000:2000] == s["calib"]  # pool order, repeated stems skipped
    assert len(unseen) < len(s["pool"])  # MMLU-Pro does repeat some train/dev stems in the pool
    _disjoint(s, ["train", "dev", "test", "fresh", "calib"])
    assert len({it["category"] for it in s["fresh"]}) == 14  # every subject is represented


def test_forward_log_readout(monkeypatch):
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "mmlupro")
    g = torch.Generator().manual_seed(3)
    group = [(torch.randint(10, 5000, (6,), generator=g).tolist(), torch.randint(10, 5000, (4,), generator=g).tolist(),
              f) for f in (False, True, False)]
    u, none = rat.forward(model, group, grad=False, k=[4, 10, 7])
    u2, none2, lu = rat.forward(model, group, grad=False, k=[4, 10, 7], return_logu=True)
    assert none is None and none2 is None and torch.equal(u, u2)  # the readout itself is unchanged
    torch.testing.assert_close(lu.exp(), u, atol=1e-6, rtol=1e-5)
    assert torch.isinf(lu[0, 4:]).all() and torch.isfinite(lu[1]).all() and torch.isinf(lu[2, 7:]).all()
    with pytest.raises(AssertionError):
        rat.forward(model, group, grad=True, k=4, return_logu=True)


def test_saved_log_probabilities_match_the_readouts(monkeypatch):
    tok, model = _tiny(monkeypatch)
    items = [dict(uid=str(i), question=f"What is {i} + {i}?", proposed=str(2 * i), q=[1.0, 0.0], counts=[1, 0])
             for i in range(3)]
    out = train.evaluate(model, train.Rationales(tok, "gsm8kv", micro=8), items, 3, 5, keep_dists=True)
    for ro in ("single", "mixture"):
        for p, lp in zip(out["dists"][ro], out["logps"][ro]):
            assert len(lp) == len(p) == 2
            assert all(math.isclose(math.exp(b), a, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(p, lp))
    rlcr = train.evaluate(model, train.Rationales(tok, "gsm8kv", micro=8, confidence=True), items, 2, 5, keep_dists=True)
    assert "logps" not in rlcr and len(rlcr["dists"]["single"]) == 3  # a stated confidence has no logits


def test_sampled_answer_surrogate():
    torch.manual_seed(0)
    m, k, y = 4, 3, 1
    z = torch.randn(m, k, requires_grad=True)
    lp = torch.randn(m, requires_grad=True)
    extra = torch.randn(m) * 0.1
    a = torch.tensor([1, 0, 1, 2])
    # the pathwise (readout) gradient is the one of single_surrogate
    g1 = torch.autograd.grad(train.sampled_answer_surrogate(z.softmax(-1), lp.detach(), y, a, extra), z)[0]
    g2 = torch.autograd.grad(single_surrogate(z.softmax(-1), lp.detach(), y, extra), z)[0]
    torch.testing.assert_close(g1, g2)
    # the score-function part: leave-one-out advantages of (2 * 1[a_i = y] - 1) / m + extra_i
    g = torch.autograd.grad(train.sampled_answer_surrogate(z.softmax(-1).detach(), lp, y, a, extra), lp)[0]
    local = (2 * (a == y).float() - 1) / m + extra
    torch.testing.assert_close(-g, local - (local.sum() - local) / (m - 1))
    # in expectation over the sampled answers the reward is the linear score 2 u_Y - 1 (exact enumeration, M = 2)
    u = torch.tensor([[0.2, 0.7, 0.1], [0.5, 0.3, 0.2]])
    lp2 = torch.zeros(2, requires_grad=True)
    exp_grad = torch.zeros(2)
    for a2 in itertools.product(range(k), repeat=2):
        w = u[0, a2[0]] * u[1, a2[1]]
        exp_grad += w * torch.autograd.grad(train.sampled_answer_surrogate(u, lp2, y, torch.tensor(a2)), lp2)[0]
    local = (2 * u[:, y] - 1) / 2
    torch.testing.assert_close(-exp_grad, local - local.flip(0), atol=1e-6, rtol=0)


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["train"] + argv)
    train.main()


@pytest.mark.parametrize("bad", [
    ["--readout-score", "linear"],  # needs the readout-only objective
    ["--sf-reward", "correct", "--diversity-weight", "1"],
    ["--sf-reward", "correct", "--diversity-weight", "0", "--no-score-function"],
    ["--sf-reward", "correct", "--diversity-weight", "0", "--objective", "grpo"],
    ["--snapshot-steps", "1", "--steps", "2"],  # needs --save-checkpoint
    ["--snapshot-steps", "2", "--steps", "2", "--save-checkpoint", "x.pt"],  # below --steps
    ["--snapshot-steps", "a", "--steps", "2", "--save-checkpoint", "x.pt"],
    ["--rationale-source", "ref", "--no-score-function"],  # needs the reference model (--kl-beta > 0)
    ["--rationale-source", "ref", "--kl-beta", "0.04"],  # needs the readout-only objective
    ["--rationale-source", "ref", "--kl-beta", "0.04", "--no-score-function", "--objective", "grpo"],
    ["--snapshot-steps", "1", "--steps", "2", "--save-checkpoint", "x.pt", "--test-split", "fresh"],  # val only
    ["--snapshot-steps", "1", "--steps", "2", "--save-checkpoint", "x.pt", "--offload-optimizer"],
    ["--kl-tokens", "0.01"],  # the anchor needs the reference model (--kl-beta > 0)
    ["--sf-reward", "correct", "--diversity-weight", "0", "--sf-coef", "0"],
    ["--sf-reward", "correct", "--diversity-weight", "0", "--grad-diag", "1"],
    ["--extra-splits", "test"],  # the default --test-split
    ["--extra-splits", "dev"],
    ["--extra-splits", "calib,calib", "--test-split", "fresh"],
])
def test_argument_checks(monkeypatch, bad):
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, bad + ["--output", "x.json"])
    assert e.value.code == 2


def _saved(tok, model, tmp_path):
    tok.save_pretrained(tmp_path / "m")
    model.save_pretrained(tmp_path / "m")
    return str(tmp_path / "m")


def test_snapshots_fresh_split_and_controls(monkeypatch, tmp_path):
    tok, model = _tiny(monkeypatch)
    path = _saved(tok, model, tmp_path)
    sp = gsm8k_verify.load_items()
    val, fresh, calib = ({it["uid"] for it in sp[x][:2]} for x in ("test", "fresh", "calib"))
    base = ["--task", "gsm8kv", "--model", path, "--items-per-step", "2", "--m", "4", "--m-eval", "2",
            "--max-new-tokens", "5", "--dev-cap", "2", "--test-cap", "2"]
    # 1. stage 1 (linear control) on the validation split with snapshots; the checkpoint directory is created
    out, ck = tmp_path / "s1.json", tmp_path / "ck" / "s1.pt"
    _run(monkeypatch, base + ["--steps", "3", "--eval-every", "2", "--no-score-function", "--readout-score", "linear",
                              "--snapshot-steps", "1,2", "--save-checkpoint", str(ck), "--output", str(out)])
    log = json.loads(out.read_text())
    assert [c["step"] for c in log["curve"]] == [0, 1, 2, 3]
    assert set(log["snapshots"]) == {"1", "2"} and ck.exists() and not list(tmp_path.glob("**/*.tmp"))
    for s in ("1", "2"):
        snap = log["snapshots"][s]
        assert Path(snap["checkpoint"]) == tmp_path / "ck" / f"s1_step{s}.pt" and Path(snap["checkpoint"]).exists()
        assert len(snap["dev_dists"]["single"]) == len(snap["dev_logps"]["single"]) == 2
        assert "test" not in snap and set(snap["test_eval"]["uids"]) == val
        assert len(snap["test_eval"]["logps"]["mixture"]) == 2
        state = torch.load(snap["checkpoint"])
        assert state.keys() == model.state_dict().keys() and all(v.dtype == torch.bfloat16 for v in state.values())
    assert set(log["test"]["uids"]) == val and "logps" in log["test"] and len(log["dev_uids"]) == 2
    assert not any("dists" in c["dev"] or "logps" in c["dev"] or "uids" in c["dev"] for c in log["curve"])
    # a run file written at a snapshot, before the final test pass, must not count as finished
    partial = dict(log)
    partial.pop("test")
    assert '"test": {' not in json.dumps(partial, indent=2) and '"test": {' in out.read_text()
    # 2. stage 2 from the step-1 weights with the reasoning-reward control, on fresh, calib in the same pass
    out2 = tmp_path / "s2.json"
    _run(monkeypatch, base + ["--steps", "1", "--eval-every", "1", "--diversity-weight", "0", "--sf-coef", "0.3",
                              "--sf-reward", "correct", "--kl-beta", "0.04", "--test-split", "fresh",
                              "--extra-splits", "calib", "--init-checkpoint", log["snapshots"]["1"]["checkpoint"],
                              "--output", str(out2)])
    log2 = json.loads(out2.read_text())
    assert log2["config"]["sf_reward"] == "correct" and set(log2["test"]["uids"]) == fresh
    assert set(log2["eval_calib"]["uids"]) == calib and len(log2["eval_calib"]["logps"]["single"]) == 2
    assert '"eval_calib": {' in out2.read_text() and log2["dev_uids"] == log["dev_uids"]
    # 3. eval-only of the final weights on calib, fresh as an extra split
    out3 = tmp_path / "ev.json"
    _run(monkeypatch, base + ["--test-split", "calib", "--extra-splits", "fresh", "--eval-only", "--checkpoint", str(ck),
                              "--output", str(out3)])
    ev = json.loads(out3.read_text())
    assert set(ev["test"]["uids"]) == calib and set(ev["eval_fresh"]["uids"]) == fresh
    assert "dev_logps" in ev and "logps" in ev["test"] and len(ev["dev_uids"]) == 2


def test_split_checks(tmp_path, monkeypatch):
    sp = gsm8k_verify.load_items()
    ref = {"gsm8kv": {"fresh": {"2": __import__("hashlib").sha256(
        "\n".join(it["uid"] for it in sp["fresh"][:2]).encode()).hexdigest()}}}
    f = tmp_path / "npj_splits.json"
    f.write_text(json.dumps(ref))
    monkeypatch.setattr(train, "NPJ_SPLITS", f)
    train.check_splits("gsm8kv", sp, [("fresh", 2), ("calib", 2)])  # matches; calib not recorded at n=2
    moved = dict(sp, fresh=sp["fresh"][1:])
    with pytest.raises(SystemExit):
        train.check_splits("gsm8kv", moved, [("fresh", 2)])
    with pytest.raises(SystemExit):
        train.check_splits("gsm8kv", dict(sp, calib=[]), [("calib", 2)])
    with pytest.raises(SystemExit):
        train.check_splits("chaosnli", {"train": [1]}, [("fresh", 2)])


def test_repository_split_file_matches_the_loaders(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    ref = json.loads((Path(train.__file__).resolve().parents[1] / "data" / "npj_splits.json").read_text())
    sp = gsm8k_verify.load_items()
    train.check_splits("gsm8kv", sp, [(n, int(k)) for n, d in ref["gsm8kv"].items() for k in d])
    try:
        mp = mmlu_pro.load_items()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"MMLU-Pro not cached: {e}")
    train.check_splits("mmlupro", mp, [(n, int(k)) for n, d in ref["mmlupro"].items() for k in d])


def test_save_weights_keeps_tied_weights_tied_and_write_json_is_atomic(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
    net = Qwen3ForCausalLM(cfg).float()
    assert net.lm_head.weight.data_ptr() == net.model.embed_tokens.weight.data_ptr()
    train.save_weights(net, tmp_path / "w.pt")
    st = torch.load(tmp_path / "w.pt")
    assert all(v.dtype == torch.bfloat16 for v in st.values())
    if "lm_head.weight" in st:
        assert st["lm_head.weight"].data_ptr() == st["model.embed_tokens.weight"].data_ptr()  # stored once
    train.write_json(tmp_path / "r.json", {"test": {"a": 1}})
    assert json.loads((tmp_path / "r.json").read_text()) == {"test": {"a": 1}} and not (tmp_path / "r.json.tmp").exists()


def test_direct_baseline_checkpoint_and_split(monkeypatch, tmp_path):
    tok, model = _tiny(monkeypatch)
    monkeypatch.setattr(train_direct, "DEVICE", "cpu")
    path = _saved(tok, model, tmp_path)
    out, ck = tmp_path / "sft.json", tmp_path / "sft.pt"
    monkeypatch.setattr(sys, "argv", ["train_direct", "--task", "gsm8kv", "--model", path, "--labels", "fresh",
                                      "--steps", "1", "--items-per-step", "2", "--eval-every", "1", "--dev-cap", "3",
                                      "--test-cap", "3", "--test-split", "fresh", "--save-checkpoint", str(ck),
                                      "--extra-splits", "calib", "--output", str(out)])
    train_direct.main()
    log = json.loads(out.read_text())
    assert len(log["test_logps"]) == len(log["test_dists"]) == 3 and len(log["dev_logps"]) == 3
    for p, lp in zip(log["test_dists"], log["test_logps"]):
        assert all(math.isclose(math.exp(b), a, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(p, lp))
    state = torch.load(ck)
    assert state.keys() == model.state_dict().keys() and all(v.dtype == torch.bfloat16 for v in state.values())
    assert log["config"]["test_split"] == "fresh"
    sp = gsm8k_verify.load_items()
    assert log["test_uids"] == [it["uid"] for it in sp["fresh"][:3]] and len(log["dev_uids"]) == 3
    assert log["eval_calib"]["uids"] == [it["uid"] for it in sp["calib"][:3]] and len(log["eval_calib"]["logps"]) == 3


def test_offline_rationales_come_from_the_reference_model(monkeypatch, tmp_path):
    """--rationale-source ref: every training step samples its rationales from the frozen reference model, every
    evaluation from the policy"""
    tok, model = _tiny(monkeypatch)
    path = _saved(tok, model, tmp_path)
    seen, orig = [], train.Rationales.sample

    def spy(self, net, items, m, max_new_tokens, micro=None):
        seen.append(all(not q.requires_grad for q in net.parameters()))  # the reference model is frozen
        return orig(self, net, items, m, max_new_tokens, micro)
    monkeypatch.setattr(train.Rationales, "sample", spy)
    out = tmp_path / "off.json"
    _run(monkeypatch, ["--task", "gsm8kv", "--model", path, "--items-per-step", "2", "--m", "4", "--m-eval", "2",
                       "--max-new-tokens", "5", "--dev-cap", "2", "--test-cap", "2", "--steps", "2", "--eval-every", "2",
                       "--no-score-function", "--kl-beta", "0.04", "--kl-tokens", "0.01", "--diversity-weight", "0",
                       "--rationale-source", "ref", "--output", str(out)])
    # step 0 dev pass (policy), steps 1 and 2 (reference), step 2 dev pass and the test pass (policy)
    assert seen == [False, True, True, False, False]
    assert json.loads(out.read_text())["config"]["rationale_source"] == "ref"


def test_thinking_mode(monkeypatch):
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "mmlupro", micro=8, thinking=True)
    assert rat.think_end == tok.convert_tokens_to_ids("</think>") and rat.think_end in rat.suffix_think
    it = dict(uid="0", question="2 + 2?", options=["3", "4"], category="math", q=[0.0, 1.0], counts=[0, 1])
    p_think, p_plain = (train.prompt(tok, it, "mmlupro", thinking=t) for t in (True, False))
    assert p_plain.endswith("<think>\n\n</think>\n\n") and not p_think.endswith("</think>\n\n")
    # the forced suffix closes the think block only when the rationale did not
    assert rat.suffix_for([11, 12]) == rat.suffix_think
    assert rat.suffix_for([11, rat.think_end, 12]) == rat.suffix
    plain = train.Rationales(tok, "mmlupro")
    assert plain.suffix_for([11, 12]) == plain.suffix  # non-thinking: unchanged
    # the stopping rule: the marker counts only after </think>
    ans = tok.encode("\nAnswer:", add_special_tokens=False)
    rows = [[5, 6] + ans + [7], [5, rat.think_end, 8] + ans, [5] + ans + [rat.think_end, 9]]
    width = 2
    ids = torch.full((3, width + max(map(len, rows))), tok.pad_token_id)
    for i, r in enumerate(rows):
        ids[i, width:width + len(r)] = torch.tensor(r)
    done = train._AnswerAfterThink(tok, rat.think_end, width)(ids, None)
    assert done.tolist() == [False, True, False]
    # sampling: a random tiny model never closes the block -> forced, and the readout reads after the closing suffix
    g = rat.sample(model, [it], 2, 6)[0]
    assert all(f for _, _, f in g)
    u, _ = rat.forward(model, g, grad=False, k=2)
    assert u.shape == (2, 10) and torch.allclose(u[:, :2].sum(1), torch.ones(2), atol=1e-5)


def test_thinking_rejected_for_rlcr(monkeypatch):
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, ["--objective", "rlcr", "--thinking", "--output", "x.json"])
    assert e.value.code == 2


def test_think_probe(monkeypatch, tmp_path):
    from openjev_rlcd import think_probe
    shares, cap = think_probe.choose_cap([100, 300, 600, 2048], [True, True, True, False], 2048)
    assert shares[256] == 0.25 and shares[512] == 0.5 and shares[768] == 0.75 and cap is None
    assert think_probe.choose_cap([10] * 9 + [700], [True] * 10, 1024)[1] == 256  # 90% within 256
    assert think_probe.choose_cap([10] * 8 + [700, 700], [True] * 10, 1024)[1] == 768
    tok, model = _tiny(monkeypatch)
    path = _saved(tok, model, tmp_path)
    out = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["think_probe", "--task", "gsm8kv", "--model", path, "--n", "2", "--cap", "8",
                                      "--gen-batch", "2", "--output", str(out)])
    think_probe.main()
    d = json.loads(out.read_text())
    assert len(d["lengths"]) == 2 and '"test": {' in out.read_text() and set(d["test"]) == {"chosen_cap", "share_finished_within"}


def test_thinking_marker_inside_a_token(monkeypatch):
    tok, _ = _tiny(monkeypatch)
    rat = train.Rationales(tok, "mmlupro", thinking=True)
    body = tok.encode("Let me see.", add_special_tokens=False)
    after = tok.encode("\n\n**Answer:** B", add_special_tokens=False)
    ids = body + [rat.think_end] + after
    cut = train.cut_after_marker(tok, ids, len(body) + 1)
    assert cut is not None and "Answer:" in tok.decode(cut[len(body) + 1:]) and len(cut) < len(ids)
    assert train.cut_after_marker(tok, body + [rat.think_end], len(body) + 1) is None
    width = 1
    x = torch.tensor([[tok.pad_token_id] + ids])
    assert train._AnswerAfterThink(tok, rat.think_end, width)(x, None).tolist() == [True]


def test_thinking_marker_with_the_answer_in_the_same_token(monkeypatch):
    tok, _ = _tiny(monkeypatch)
    rat = train.Rationales(tok, "mmlupro", thinking=True)
    ids = tok.encode("ok", add_special_tokens=False) + [rat.think_end] + tok.encode("\n\nAnswer:B", add_special_tokens=False)
    assert tok.convert_ids_to_tokens(ids[-1]) == ":B"  # Qwen3 merges the colon with the letter
    assert train.cut_after_marker(tok, ids, ids.index(rat.think_end) + 1) is None  # never read after the letter
    rule = train._AnswerAfterThink(tok, rat.think_end, 0)
    x = torch.tensor([ids, ids])
    assert rule(x, None).tolist() == [True, True]  # generation still stops there
    assert rule.done.tolist() == [True, True]  # and finished rows are not decoded again


def test_hub_offline_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    train.hub_offline()
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_DATASETS_OFFLINE"] == "0"  # explicit setting kept


def test_thinking_post_processing_cases(monkeypatch):
    """Rationales.sample in thinking mode on fixed generations (generate is replaced): the marker written, a refused
    'Answer:B' (forced, unsampled pad dropped), a sampled <|endoftext|> end (kept), and the cap"""
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "mmlupro", micro=8, thinking=True)
    it = dict(uid="0", question="2 + 2?", options=["3", "4"], category="math", q=[0.0, 1.0], counts=[0, 1])
    te, pad = rat.think_end, tok.pad_token_id
    enc = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    rows = [enc("hm") + [te] + enc("\n\nAnswer:"),                    # marker written
            enc("hm") + [te] + enc("\n\nAnswer:B") + [pad, pad],      # refused marker, pad fill after the rule stop
            enc("hm") + [te] + enc("\n\nIt is B.") + [pad],            # sampled <|endoftext|> (= pad id) as the end
            enc(" hm" * 30)]                                           # the longest row: the cap, inside the block

    def fake_generate(input_ids=None, attention_mask=None, **kw):
        w = max(map(len, rows))
        out = torch.full((len(rows), input_ids.shape[1] + w), pad)
        out[:, :input_ids.shape[1]] = input_ids[0]
        for i, r in enumerate(rows):
            out[i, input_ids.shape[1]:input_ids.shape[1] + len(r)] = torch.tensor(r)
        return out
    monkeypatch.setattr(model, "generate", fake_generate)
    g = rat.sample(model, [it], 4, 64)[0]
    (_, r0, f0), (_, r1, f1), (_, r2, f2), (_, r3, f3) = g
    assert not f0 and tok.decode(r0).endswith("Answer:")
    assert f1 and r1[-1] != pad and tok.decode(r1).endswith("Answer:B")  # forced; the unsampled pad dropped
    assert f2 and r2[-1] == pad  # a sampled end token is kept
    assert f3 and te not in r3 and rat.suffix_for(r3) == rat.suffix_think
    assert r3 == rows[3] and r3[-1] != pad  # cut by the cap: nothing appended, nothing dropped

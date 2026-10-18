"""The rloo and rlcr baselines (train.py): parsing and scoring of the verbalized confidence, and the tail option of
Rationales.forward -- appending the answer token and the confidence line changes neither the readout u nor the
rationale log-probability, and the tail's log-probability is the sum over its scored tokens."""
import math
from pathlib import Path

import pytest
import torch

from openjev_rlcd import train
from openjev_rlcd.train import parse_confidence, rlcr_reward, verbal_dist


def test_parse_confidence():
    assert parse_confidence(" 0.85\n") == 0.85
    assert parse_confidence("0.9.") == 0.9
    assert parse_confidence(" .3") == 0.3
    assert parse_confidence(" 85%") == 0.85
    assert parse_confidence(" 85") == 0.85  # a bare number above 1 is read as a percentage
    assert parse_confidence(" 1") == 1.0 and parse_confidence(" 0") == 0.0
    assert parse_confidence(" 150") is None and parse_confidence(" high") is None and parse_confidence("") is None


def test_verbal_dist_and_reward():
    p = verbal_dist(0.7, 2, 4)
    assert p[2] == 0.7 and math.isclose(sum(p), 1.0) and p[0] == p[1] == p[3]
    assert verbal_dist(0.3, 0, 2) == [0.3, 0.7]  # a low stated confidence moves the mass to the other answer
    r = rlcr_reward(torch.tensor([1.0, 0.0, 1.0, 0.0]), [0.8, 0.8, None, None])
    torch.testing.assert_close(r, torch.tensor([1 - 0.04, -0.64, 0.0, -1.0]))


def _tiny(monkeypatch):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("Qwen3 tokenizer not cached")
    monkeypatch.setattr(train, "DEVICE", "cpu")
    tok = AutoTokenizer.from_pretrained(str(snaps[0].parent))
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    return tok, Qwen3ForCausalLM(cfg).float().eval()


def test_forward_tail_leaves_readout_and_rationale_unchanged(monkeypatch):
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "gsm8kv", confidence=True)
    g = torch.Generator().manual_seed(1)
    group = [(torch.randint(10, 5000, (7,), generator=g).tolist(), torch.randint(10, 5000, (5,), generator=g).tolist(), False),
             (torch.randint(10, 5000, (4,), generator=g).tolist(), torch.randint(10, 5000, (9,), generator=g).tolist(), True)]
    conf = [tok.encode(" 0.75\n", add_special_tokens=False), tok.encode(" 0.2\n", add_special_tokens=False)]
    fixed = 1 + len(rat.conf_prefix)
    tails = [([rat.label_ids[a]] + rat.conf_prefix + c, fixed) for a, c in zip((0, 1), conf)]
    u, lp = rat.forward(model, group, grad=False, need_logp=True, k=2)
    u2, lp2, lpt = rat.forward(model, group, grad=False, need_logp=True, k=2, tails=tails)
    torch.testing.assert_close(u2, u, atol=2e-2, rtol=0)  # bf16 autocast: other padding widths round differently
    torch.testing.assert_close(lp2, lp, atol=5e-2, rtol=1e-2)
    for j, ((p, r, f), (tail, fx)) in enumerate(zip(group, tails)):  # the tail's log-prob, one sequence at a time
        seq = p + r + (rat.suffix if f else []) + tail
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=torch.tensor([seq])).logits[0].float()
        start = len(seq) - len(tail) + fx
        want = sum(logits[i - 1].log_softmax(-1)[seq[i]].item() for i in range(start, len(seq)))
        assert abs(lpt[j].item() - want) < 5e-2 * max(1.0, abs(want))
    _, lp3, lpt3 = rat.forward(model, group, grad=True, k=2, tails=tails)  # gradients reach the tail term
    (lp3.sum() + lpt3.sum()).backward()
    assert model.lm_head.weight.grad is not None and model.lm_head.weight.grad.abs().sum() > 0


def test_confidence_lines_and_rlcr_readout(monkeypatch):
    tok, model = _tiny(monkeypatch)
    rat = train.Rationales(tok, "gsm8kv", micro=8, confidence=True)
    items = [dict(uid=str(i), question=f"What is {i} + {i}?", proposed=str(2 * i), q=[1.0, 0.0], counts=[1, 0])
             for i in range(3)]
    stats = {}
    mix, vote, single, forced = train.decision_dists(model, rat, items, 2, 6, stats=stats, examples=[])
    for p in mix + single:
        assert len(p) == 2 and math.isclose(sum(p), 1.0, abs_tol=1e-6)
    assert 0 <= stats["confidence_unparsed_rate"] <= 1
    rows = [(rat.tok(train.prompt(tok, items[0], "gsm8kv", confidence=True))["input_ids"], [11, 12], True)]
    (toks, q), = rat.confidences(model, rows, [1], sample=True)
    assert 0 < len(toks) <= train.CONF_MAX_TOKENS and (q is None or 0 <= q <= 1)


def test_verbal_decision_keeps_the_stated_answer():
    from openjev_rlcd.evaluation import verbal_decision
    p = verbal_dist(0.0, 3, 10)  # "Confidence: 0" on option 3: every other option gets 1/9
    q = verbal_decision(p)
    assert q.index(max(q)) == 3 and math.isclose(sum(q), 1.0) and q[3] > 0.1 > q[0]
    assert verbal_decision(verbal_dist(0.6, 3, 10)) == verbal_dist(0.6, 3, 10)  # C >= 1/K: unchanged
    assert verbal_decision([0.2, 0.8]) == [0.2, 0.8]  # K = 2 is exact: the other answer is the decision
    assert verbal_decision([0.25] * 4) == [0.25] * 4


def test_flat_dev_nll_leaves_the_temperature_at_one():
    from openjev_rlcd.metrics import fit_temperature
    items = [dict(uid=str(i), counts=[1, 0, 0, 0]) for i in range(20)]
    flat = [torch.zeros(4) for _ in items]  # uniform readouts: the NLL is log 4 at every T
    assert fit_temperature(flat, items, 17) == 1.0
    sharp = [torch.tensor([30.0, 0.0, 0.0, 0.0]) if i % 2 else torch.tensor([0.0, 30.0, 0.0, 0.0]) for i in range(20)]
    assert fit_temperature(sharp, items, 17) > 1.0  # overconfident and half wrong: soften

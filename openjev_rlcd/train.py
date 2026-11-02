#!/usr/bin/env python3
"""RLCD for a reasoning model, and the RL baselines, in one training loop (see estimators.py).

Only the rationale is sampled. Generation stops at "Answer:"; the model's explicit next-token
distribution over the answer-option tokens is then READ, not sampled: u_i = softmax(logits[option ids])
(options beyond an item's own count are masked). If a rationale ends without the marker (EOS or length
limit), "\\nAnswer:" is appended before reading u, so there are no invalid answers by construction.

Per question: M rationales and one outcome Y (ChaosNLI: a fresh annotator; MMLU-Pro / GSM8K-Verify: the
answer key). Objectives (--objective):
  rlcd : estimators.lambda_surrogate -- exact (pathwise) gradient through u plus REINFORCE for the
         rationales with leave-one-out baselines. --diversity-weight lambda: 1 = Brier of the mixture
         p = E_r u (pays a bonus Var_r(u) for disagreeing rationales), 0 = per-rationale score
         (optimum u(r) = q for every rationale). The paper uses lambda = 0 and
           stage 1 (calibrate):  --no-score-function                  (pathwise only)
           stage 2 (reinforce):  --init-checkpoint <stage 1> --sf-coef 0.3 --kl-beta 0.04
  grpo : RLVR -- answer A_i ~ u_i sampled, reward 1[A_i = Y], group-normalized advantage.
  rloo : the same reward with a leave-one-out baseline and no division by the group's std (RLOO / Dr. GRPO).
  rlcr : verbalized confidence (RLCR): after A_i ~ u_i the model writes "Confidence: C_i"; reward
         1[A_i = Y] - (C_i - 1[A_i = Y])^2, group-normalized as in grpo (--rlcr-advantage loo: leave-one-out
         advantages without the division by the group's std, as in rloo); the stated confidence is the readout
         (C_i on A_i, the rest spread evenly over the other options).
  credo: CREDO (Fan et al. 2026, arXiv:2609.34857; amendment A4 of paper/npj/PROTOCOL.md): after A_i ~ u_i the model
         writes "<analysis>...</analysis>" and the confidence c_i = P(high) / (P(high) + P(low)) is READ at the slot
         "<confidence>" that follows; GRPO with two group-normalized advantages (the rationale and A_i: reward
         a_i + beta F_i, scaled by the discrepancy weight w_i; the analysis: F_i (1 - (t_i - c_i)^2)), per-rollout token
         means, no KL, plus alpha * mean_i (c_i - t_i)^2 through c, with the hybrid target t_i = gamma a_i +
         (1 - gamma) mean_j a_j (CREDO below). The readout is c_i on A_i, the rest spread evenly, as rlcr's.
  rft  : STaR / rejection-sampling fine-tuning on the trajectories whose sampled answer is correct.
Readouts at evaluation:
  mixture : mean of u_i over M rationales
  vote    : frequency of argmax u_i
  single  : u_1 from ONE rationale (a single query)
With keep_dists the probabilities of the mixture and single readouts are saved, and their log-probabilities as well
(logps; log_softmax of the option logits, so no probability underflows; absent for the confidences of rlcr and credo).

  python -m openjev_rlcd.train --task gsm8kv ... --output results/x.json   (see scripts/ for every run)
"""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .estimators import lambda_surrogate
from .metrics import readout_metrics
from .offload import OffloadedAdamW, bf16_matmul_weights
from .tasks import TASKS, prompt

MARKER = "Answer:"
CONF_PREFIX = "\nConfidence:"  # rlcr: the line the model writes after its answer token
CONF_MAX_TOKENS = 10
# credo (amendment A4): the analysis is opened after the answer token and sampled until it closes or reaches its cap
# (--credo-analysis-tokens, set by A4's probe, credo_probe.py); the slot is then appended and read over two existing
# tokens (the paper adds two reserved tokens whose embeddings copy the rows of "high" and "low", which equals reading
# those tokens until training moves them apart)
CREDO_OPEN, CREDO_CLOSE, CREDO_SLOT = "\n<analysis>", "</analysis>", "<confidence>"
CREDO_SLOT_WORDS = ("high", "low")
# the paper's settings (math): target mixture gamma, calibration weight alpha, discrepancy strength kappa and its clip
# range, format weight beta, and the warm-up steps of alpha and kappa (the paper does not give the ramp's shape; here it
# is linear from 0, min(1, step / 30))
CREDO = dict(gamma=0.5, alpha=0.03, kappa=0.25, w_min=0.5, w_max=2.0, beta=0.5, warmup=30)
DEVICE = os.environ.get("OPENJEV_DEVICE", "cuda")  # "cpu" only for smoke tests
_CONF = re.compile(r"\s*(\d+(?:\.\d*)?|\.\d+)\s*(%?)")


def parse_confidence(text):
    """the probability stated after 'Confidence:' (rlcr): a number in [0, 1], or a percentage ('85%', or a bare
    number in (1, 100]); None when the text does not start with a number or the number is out of range"""
    m = _CONF.match(text)
    if not m:
        return None
    v = float(m.group(1))
    if m.group(2) or v > 1:
        v /= 100
    return v if 0 <= v <= 1 else None


def verbal_dist(conf, a, k):
    """the answer distribution of a verbalized confidence: conf on the stated answer a, the rest spread evenly"""
    return [conf if j == a else (1 - conf) / (k - 1) for j in range(k)]


def rlcr_reward(correct, confs):
    """RLCR reward 1[A = Y] - (C - 1[A = Y])^2 per rationale; an unparseable confidence (None) costs the largest
    Brier term, 1"""
    return torch.tensor([c - (1.0 if q is None else (q - c) ** 2) for c, q in zip(correct.tolist(), confs)],
                        device=correct.device)


def group_advantages(r, kind):
    """advantages of the M rewards of one question: 'grpo' = group-normalized, (r - mean) / (std + 1e-4) as in GRPO;
    'loo' = leave-one-out, r_i minus the mean of the other rewards, without the division by the group's std (RLOO)"""
    if kind == "grpo":
        return (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    if kind == "loo":
        return r - (r.sum() - r) / (len(r) - 1)
    raise ValueError(kind)


def credo_loss(correct, c, formatted, logp_ans, logp_an, n_tok, ramp):
    """CREDO's loss for the M rollouts of one question (amendment A4; settings in CREDO). correct [M]: 1[A_i = Y];
    c [M]: the slot readouts, with gradient; formatted [M]: F_i (1 if the rollout wrote the marker and closed its
    analysis); logp_ans [M]: log-probability of the rationale and its sampled answer; logp_an [M]: that of the sampled
    analysis; n_tok [M]: sampled tokens of the rollout (each rollout's token mean); ramp in [0, 1]: the warm-up factor
    of alpha and kappa. The hybrid target t_i = gamma a_i + (1 - gamma) mean_j a_j; answer reward a_i + beta F_i and
    confidence reward F_i (1 - (t_i - c_i)^2), each group-normalized; the answer advantage is scaled by the discrepancy
    weight w_i = clip(1 + kappa (|a_i - c_i| - mean_j |a_j - c_j|), w_min, w_max); one on-policy update (PPO's ratio is
    1); no KL. Returns (loss, dict(t, w, cal))."""
    s = CREDO
    cd = c.detach()
    t = s["gamma"] * correct + (1 - s["gamma"]) * correct.mean()
    r_ans = correct + s["beta"] * formatted
    r_conf = formatted * (1 - (t - cd) ** 2)
    a_ans, a_conf = group_advantages(r_ans, "grpo"), group_advantages(r_conf, "grpo")
    dis = (correct - cd).abs()
    w = (1 + s["kappa"] * ramp * (dis - dis.mean())).clamp(s["w_min"], s["w_max"])
    pg = (w * a_ans * logp_ans + a_conf * logp_an) / n_tok
    cal = ((c - t) ** 2).mean()
    return -pg.mean() + s["alpha"] * ramp * cal, dict(t=t, w=w, cal=cal.detach())


def sampled_answer_surrogate(u, logp_r, y, a, extra_local=None):
    """the reasoning-reward control (--sf-reward correct): the pathwise term of estimators.single_surrogate on the
    readout, unchanged, but each rationale is rewarded by the Brier score of ONE answer a_i sampled from its readout,
    2 * 1[a_i = Y] - 1 (an affine map of RLVR's 1[a_i = Y]; in expectation over a_i the improper linear score
    2 u_i[Y] - 1 instead of the proper 2 u_i[Y] - ||u_i||^2), with the same leave-one-out baseline and shaping terms.
    u [M, K], logp_r [M], a [M] sampled option indices. Returns a scalar loss."""
    m = u.shape[0]
    if m < 2:
        raise ValueError("needs M >= 2 rationales")
    r = 2 * u[:, y] - (u * u).sum(1)
    local = (2 * (a == y).to(u.dtype) - 1) / m
    if extra_local is not None:
        local = local + extra_local.detach().to(local.dtype)
    adv = (local - (local.sum() - local) / (m - 1)).detach()
    return -(r.mean() + (adv * logp_r).sum())


def _token_logp(lm_head, h, tgt):
    return lm_head(h).float().log_softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)


def _token_kl(lm_head, ref_head, h, h_ref):
    """KL(pi || pi_ref) between the next-token distributions at each position (rows of h, h_ref)."""
    lp = lm_head(h).float().log_softmax(-1)
    lq = ref_head(h_ref).float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


class _AnswerAfterThink:
    """stopping rule of thinking mode, per row: stop once the text after </think> contains the marker (the marker
    inside the think block does not count; a token that runs past it, as in '**Answer:**', counts). Only the last 16
    tokens are decoded at each step (the first occurrence stops the row, so it is always among the newest tokens) and
    finished rows are not decoded again. width: the prompt width of the batch"""
    def __init__(self, tok, think_end, width):
        self.tok, self.think_end, self.width = tok, think_end, width
        self.done = None

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[:, self.width:]
        if self.done is None:
            self.done = torch.zeros(gen.shape[0], dtype=torch.bool)
        closed = (gen == self.think_end).any(1).cpu()
        tails = gen[:, -16:].cpu().tolist()
        for i in (closed & ~self.done).nonzero().flatten().tolist():
            w = tails[i]
            if self.think_end in w:  # the block closed within the window: only the text after it counts
                w = w[w.index(self.think_end) + 1:]
            self.done[i] = MARKER in self.tok.decode(w)
        return self.done.to(input_ids.device)


def cut_after_marker(tok, ids, start):
    """ids up to and including the token that completes the first MARKER in the text of ids[start:]; None if there
    is none, or if that token already carries the answer (Qwen3 has ':B', ':no' tokens: 'Answer:B' must not be read
    after its letter; such a rationale is treated as not having written the marker, as in non-thinking mode)"""
    for i in range(start, len(ids)):
        text = tok.decode(ids[start:i + 1])
        if MARKER in text:
            after = text.split(MARKER, 1)[1]
            return None if any(ch.isalnum() for ch in after) else ids[:i + 1]
    return None


class Rationales:
    def __init__(self, tok, task="chaosnli", micro=64, confidence=False, thinking=False, credo=False, analysis_cap=None):
        self.tok, self.task, self.micro = tok, task, micro  # micro: sequences generated per batch
        self.confidence = confidence  # rlcr: the prompt asks for a 'Confidence: C' line after the answer
        self.credo = credo  # credo: the prompt asks for an analysis and a high/low confidence slot after the answer
        self.analysis_cap = analysis_cap  # credo: the analysis' token cap
        assert not (confidence and credo) and (not credo or (analysis_cap or 0) > 0)
        self.thinking = thinking  # the chat template's thinking mode (--thinking)
        if tok.pad_token_id is None:  # padding is masked out everywhere; the pad id only has to exist
            tok.pad_token = tok.eos_token
        self.label_ids = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in TASKS[task]["labels"]]
        assert len(set(self.label_ids)) == len(self.label_ids)
        self.eos = {tok.eos_token_id, tok.pad_token_id}
        self.suffix = tok.encode("\n" + MARKER, add_special_tokens=False)
        self.conf_prefix = tok.encode(CONF_PREFIX, add_special_tokens=False)
        if credo:
            self.credo_open = tok.encode(CREDO_OPEN, add_special_tokens=False)
            slot = [tok.encode(w, add_special_tokens=False) for w in CREDO_SLOT_WORDS]
            assert all(len(s) == 1 for s in slot) and slot[0] != slot[1], "the slot words must be single tokens"
            self.slot_ids = (slot[0][0], slot[1][0])  # c = P(high) / (P(high) + P(low)) at the slot
        if thinking:  # one special token closes the think block (Qwen3: 151668, SmolLM3: 128003)
            self.think_end = tok.convert_tokens_to_ids("</think>")
            assert isinstance(self.think_end, int) and self.think_end != tok.unk_token_id, "no </think> token"
            self.suffix_think = tok.encode("\n</think>\n\n" + MARKER, add_special_tokens=False)
            assert self.think_end in self.suffix_think

    def suffix_for(self, rationale):
        """the marker appended after a rationale that did not write it (forced): in thinking mode a rationale cut inside
        the think block also gets the block closed"""
        if self.thinking and self.think_end not in rationale:
            return self.suffix_think
        return self.suffix

    def k(self, item):
        return TASKS[self.task]["k"](item)

    @torch.no_grad()
    def confidences(self, gen_model, rows, answers, sample):
        """rlcr: the 'Confidence: C' line written after each rationale and its answer token. rows: (prompt_ids,
        rationale_ids, forced); answers: option index per row; sample: temperature 1 (training) or greedy
        (evaluation). Returns per row the generated ids (through the first newline or EOS) and the parsed C (or None)."""
        gen_model.eval()
        stop = set(self.eos)
        cfg = getattr(gen_model.generation_config, "eos_token_id", None)
        stop |= set(cfg if isinstance(cfg, (list, tuple)) else [cfg]) - {None}
        ctxs = [p + r + (self.suffix_for(r) if f else []) + [self.label_ids[a]] + self.conf_prefix
                for (p, r, f), a in zip(rows, answers)]
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        out = []
        for s in range(0, len(ctxs), self.micro):
            chunk = ctxs[s:s + self.micro]
            width = max(map(len, chunk))
            ids = torch.full((len(chunk), width), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, x in enumerate(chunk):  # left-padded, as in sample
                ids[j, width - len(x):] = torch.tensor(x)
                mask[j, width - len(x):] = 1
            mixed = getattr(gen_model, "openjev_mixed_precision", False)
            with torch.autocast(DEVICE, dtype=torch.bfloat16) if mixed else contextlib.nullcontext():
                gen = gen_model.generate(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                                         max_new_tokens=CONF_MAX_TOKENS, stop_strings=["\n"], tokenizer=self.tok,
                                         pad_token_id=self.tok.pad_token_id, **kw)
            for row in gen[:, width:].tolist():
                cut = next((j for j, t in enumerate(row) if t in stop or "\n" in self.tok.decode([t])), None)
                toks = row if cut is None else row[:cut + 1]
                out.append((toks, parse_confidence(self.tok.decode(toks))))
        return out

    @torch.no_grad()
    def analyses(self, gen_model, rows, answers, sample):
        """credo: the analysis written after each rationale and its answer token, opened for the model (CREDO_OPEN).
        rows: (prompt_ids, rationale_ids, forced); answers: option index per row; sample: temperature 1 (training) or
        greedy (evaluation). Returns per row (ids, closed): the generated ids through the token that completes
        CREDO_CLOSE (closed), through a sampled EOS, or all analysis_cap of them (not closed)."""
        gen_model.eval()
        stop = set(self.eos)
        cfg = getattr(gen_model.generation_config, "eos_token_id", None)
        stop |= set(cfg if isinstance(cfg, (list, tuple)) else [cfg]) - {None}
        ctxs = [p + r + (self.suffix_for(r) if f else []) + [self.label_ids[a]] + self.credo_open
                for (p, r, f), a in zip(rows, answers)]
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        out = []
        for s in range(0, len(ctxs), self.micro):
            chunk = ctxs[s:s + self.micro]
            width = max(map(len, chunk))
            ids = torch.full((len(chunk), width), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, x in enumerate(chunk):  # left-padded, as in sample
                ids[j, width - len(x):] = torch.tensor(x)
                mask[j, width - len(x):] = 1
            mixed = getattr(gen_model, "openjev_mixed_precision", False)
            with torch.autocast(DEVICE, dtype=torch.bfloat16) if mixed else contextlib.nullcontext():
                gen = gen_model.generate(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                                         max_new_tokens=self.analysis_cap, stop_strings=[CREDO_CLOSE],
                                         tokenizer=self.tok, pad_token_id=self.tok.pad_token_id, **kw)
            out += [self.cut_analysis(row, stop) for row in gen[:, width:].tolist()]
        return out

    def cut_analysis(self, row, stop):
        """(ids, closed) of one generated analysis row (see analyses); the padding after a finished row is never kept"""
        for j, t in enumerate(row):
            if t in stop:
                return row[:j + 1], False
            if CREDO_CLOSE in self.tok.decode(row[:j + 1]):
                return row[:j + 1], True
        return row, False

    def slot_suffix(self, ids, closed):
        """credo: the forced text after a generated analysis, ending in the slot '<confidence>': an analysis that did
        not close (cap or EOS) is closed first; after a closed one the slot goes on a new line, unless the token that
        closed it already started the line or the slot ('</analysis>\\n', '</analysis><'), which is then completed"""
        if not closed:
            return self.tok.encode("\n" + CREDO_CLOSE + "\n" + CREDO_SLOT, add_special_tokens=False)
        text = self.tok.decode(ids)
        rest = text.split(CREDO_CLOSE, 1)[1] if CREDO_CLOSE in text else ""  # what the closing token wrote after it
        head = rest.lstrip()
        if head and CREDO_SLOT.startswith(head) and head != CREDO_SLOT:
            text = CREDO_SLOT[len(head):]
        else:
            text = CREDO_SLOT if rest.endswith("\n") else "\n" + CREDO_SLOT
        return self.tok.encode(text, add_special_tokens=False)

    def slot_tails(self, answers, analyses):
        """credo: per row the tail for forward (with slot=self.slot_ids): the answer token, the opened analysis, its
        generated ids and the forced slot (slot_suffix). Only the generated analysis ids are scored: (tail, n leading
        ids not scored, n trailing ids not scored)."""
        out = []
        for a, (ids, closed) in zip(answers, analyses):
            suf = self.slot_suffix(ids, closed)
            out.append(([self.label_ids[a]] + self.credo_open + ids + suf, 1 + len(self.credo_open), len(suf)))
        return out

    @torch.no_grad()
    def sample(self, gen_model, items, m, max_new_tokens, micro=None):
        """Per item, m tuples (prompt_ids, rationale_ids, forced: bool)."""
        gen_model.eval()
        micro = micro or self.micro
        reqs = [(i, prompt(self.tok, it, self.task, confidence=self.confidence, thinking=self.thinking,
                           **({"credo": True} if self.credo else {})))
                for i, it in enumerate(items) for _ in range(m)]
        out = [[] for _ in items]
        stop = set(self.eos)  # the tokenizer's EOS/pad and every EOS id the model's generation config stops at
        cfg = getattr(gen_model.generation_config, "eos_token_id", None)
        stop |= set(cfg if isinstance(cfg, (list, tuple)) else [cfg]) - {None}
        for s in range(0, len(reqs), micro):
            chunk = reqs[s:s + micro]
            enc = self.tok([p for _, p in chunk], return_tensors="pt", padding=True, padding_side="left",
                           add_special_tokens=False).to(DEVICE)  # the chat template already holds BOS etc.
            mixed = getattr(gen_model, "openjev_mixed_precision", False)  # bf16 Linear + fp32 rest: needs autocast
            width = enc["input_ids"].shape[1]
            if self.thinking:  # "Answer:" inside the think block must not stop the rationale
                from transformers import StoppingCriteriaList
                stop_kw = dict(stopping_criteria=StoppingCriteriaList([_AnswerAfterThink(self.tok, self.think_end, width)]))
            else:
                stop_kw = dict(stop_strings=[MARKER], tokenizer=self.tok)
            with torch.autocast(DEVICE, dtype=torch.bfloat16) if mixed else contextlib.nullcontext():
                gen = gen_model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                         max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, **stop_kw)
            for (i, _), row, prow, pmask in zip(chunk, gen, enc["input_ids"], enc["attention_mask"]):
                comp = row[width:].tolist()
                cut = next((k for k, t in enumerate(comp) if t in stop), None)
                head = comp if cut is None else comp[:cut]
                done = self.tok.decode(head).rstrip().endswith(MARKER)
                rule_stop = False
                if self.thinking:  # the marker counts only after the think block, wherever it completes there
                    after = head.index(self.think_end) + 1 if self.think_end in head else None
                    cut_ids = cut_after_marker(self.tok, head, after) if after is not None else None
                    done = cut_ids is not None
                    rule_stop = after is not None and MARKER in self.tok.decode(head[after:])
                    if done:
                        head = cut_ids
                if done:
                    rat, forced = head, False           # stopped at the marker
                elif rule_stop and cut is not None and comp[cut] == self.tok.pad_token_id:
                    rat, forced = comp[:cut], True      # the rule stopped a refused marker: the pad was not sampled
                else:
                    rat, forced = (comp if cut is None else comp[:cut + 1]), True  # keep the sampled EOS
                p_ids = prow[pmask.bool()].tolist()
                out[i].append((p_ids, rat, forced))
        return out

    def forward(self, model, group, grad, need_logp=None, k=None, chunk=512, return_hidden=False, tails=None,
                return_logu=False, slot=None):
        """group: list of (prompt_ids, rationale_ids, forced). Returns u [M, K] (zero beyond each
        row's option count k: int or per-row list) and logp_r [M] (None unless grad or need_logp),
        plus (hidden states, row index, position) of the rationale positions with return_hidden.
        Vocabulary logits are computed only at the answer position and, in checkpointed chunks,
        at the rationale positions -- never over the prompt.
        tails (rlcr, credo): per row (token ids appended after the answer position, number of leading ids not scored[,
        number of trailing ids not scored]); u is still read at the answer position (causal attention: the tail cannot
        change it), and the sum of log-probabilities of the scored tail ids is returned as one more last element,
        logp_tail [M].
        slot (credo): two token ids; their logits at each row's last position (which predicts the slot token) are
        returned as the very last element, [M, 2] (also without need_logp).
        return_logu (evaluation; needs need_logp False): also return log u = log_softmax of the same option logits,
        (u, None, log u), which keeps the information that u loses when a probability underflows in float32."""
        need_logp = grad if need_logp is None else need_logp
        assert not (return_logu and need_logp), "return_logu is for the readout-only forward pass"
        seqs, spans, tspans = [], [], []
        for j, (p, r, forced) in enumerate(group):
            ctx = p + r + (self.suffix_for(r) if forced else [])
            if tails is not None:
                tail, fixed, end = tails[j] if len(tails[j]) == 3 else (*tails[j], 0)
                tspans.append((len(ctx) + fixed, len(ctx) + len(tail) - end, len(ctx) - 1))
                ctx = ctx + tail
            seqs.append(ctx)
            spans.append((len(p), len(p) + len(r)))
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        rows = torch.arange(len(seqs), device=DEVICE)
        last = torch.tensor([len(x) - 1 for x in seqs] if tails is None else [c for _, _, c in tspans], device=DEVICE)
        with torch.set_grad_enabled(grad), torch.autocast(DEVICE, dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            z = model.lm_head(h[rows, last])[:, self.label_ids].float()
            if k is not None:
                ks = torch.tensor([k] * len(seqs) if isinstance(k, int) else k, device=DEVICE)
                z = z.masked_fill(torch.arange(z.shape[1], device=DEVICE)[None] >= ks[:, None], float("-inf"))
            u = z.softmax(-1)
            zs = None
            if slot is not None:  # credo: the two slot tokens' logits at each row's last position
                ends = torch.tensor([len(x) - 1 for x in seqs], device=DEVICE)
                zs = model.lm_head(h[rows, ends])[:, list(slot)].float()
            if not need_logp:
                res = (u, None, z.log_softmax(-1)) if return_logu else (u, None)
                return res if zs is None else res + (zs,)
            pos = torch.arange(width - 1, device=DEVICE)[None]
            start = torch.tensor([a for a, _ in spans], device=DEVICE)[:, None]
            end = torch.tensor([b for _, b in spans], device=DEVICE)[:, None]
            b, t = ((pos >= start - 1) & (pos < end - 1)).nonzero(as_tuple=True)  # logits at t predict token t+1
            tgt = ids[b, t + 1]
            parts = []
            for s in range(0, len(b), chunk):
                hs = h[b[s:s + chunk], t[s:s + chunk]]
                if grad:
                    parts.append(checkpoint(_token_logp, model.lm_head, hs, tgt[s:s + chunk], use_reentrant=False))
                else:
                    parts.append(_token_logp(model.lm_head, hs, tgt[s:s + chunk]))
            lp = torch.cat(parts) if parts else torch.zeros(0, device=DEVICE)
            if tails is not None:  # the scored tail ids (rlcr: the generated confidence), same chunked log-probs
                t0 = torch.tensor([a for a, _, _ in tspans], device=DEVICE)[:, None]
                t1 = torch.tensor([b_ for _, b_, _ in tspans], device=DEVICE)[:, None]
                b2, t2 = ((pos >= t0 - 1) & (pos < t1 - 1)).nonzero(as_tuple=True)
                tgt2, parts2 = ids[b2, t2 + 1], []
                for s in range(0, len(b2), chunk):
                    hs = h[b2[s:s + chunk], t2[s:s + chunk]]
                    if grad:
                        parts2.append(checkpoint(_token_logp, model.lm_head, hs, tgt2[s:s + chunk], use_reentrant=False))
                    else:
                        parts2.append(_token_logp(model.lm_head, hs, tgt2[s:s + chunk]))
                lp2 = torch.cat(parts2) if parts2 else torch.zeros(0, device=DEVICE)
        logp = torch.zeros(len(seqs), device=DEVICE).index_add(0, b, lp)
        out = (u, logp, (h, b, t)) if return_hidden else (u, logp)
        if tails is not None:
            out = out + (torch.zeros(len(seqs), device=DEVICE).index_add(0, b2, lp2),)
        if zs is not None:
            out = out + (zs,)
        return out

    def token_kl(self, model, aux, ref_model, ref_aux, chunk=256):
        """Per rationale, the sum over its positions of KL(pi || pi_ref) between next-token distributions,
        differentiable through pi (pathwise). This anchors the rationale distribution without going
        through REINFORCE, so it also acts in stage 1, where the rationales get no score-function
        gradient. aux / ref_aux: the return_hidden outputs of forward on the same group."""
        (h, b, t), (hr, br, tr) = aux, ref_aux
        assert torch.equal(b, br) and torch.equal(t, tr)
        parts = []
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            for s in range(0, len(b), chunk):
                hs, hq = h[b[s:s + chunk], t[s:s + chunk]], hr[b[s:s + chunk], t[s:s + chunk]]
                parts.append(checkpoint(_token_kl, model.lm_head, ref_model.lm_head, hs, hq, use_reentrant=False))
        kl = torch.cat(parts) if parts else torch.zeros(0, device=DEVICE)
        return torch.zeros(h.shape[0], device=DEVICE).index_add(0, b, kl)


def repetition(text):
    words = text.split()
    return 1 - len(set(words)) / max(1, len(words))


@torch.no_grad()
def decision_dists(model, rat, items, m, max_new_tokens, stats=None, examples=None, logs=None):
    """logs: optional dict, filled with the log-probabilities of the mixture and single readouts (per item; None for the
    stated-confidence readout of rlcr)"""
    groups = rat.sample(model, items, m, max_new_tokens)
    lmix, lsingle = [], []
    mix, vote, single, forced = [], [], [], 0
    if stats is not None:
        texts = [rat.tok.decode(r) for g in groups for _, r, _ in g]
        stats.update(mean_rationale_tokens=sum(len(r) for g in groups for _, r, _ in g) / len(texts),
                     length_limit_rate=sum(len(r) >= max_new_tokens for g in groups for _, r, _ in g) / len(texts),
                     repetition_ratio=sum(map(repetition, texts)) / len(texts))
    conf_stats = [0, 0.0, 0]  # rlcr / credo: unparseable confidences (credo: unclosed analyses), sum of confidences, rationales
    slot_answers, slot_gaps = [], []  # credo: per item and rationale the decided option and z_high - z_low at the slot
    for it, g in zip(items, groups):
        k = rat.k(it)
        outs = [rat.forward(model, g[s:s + 16], grad=False, k=k, return_logu=True) for s in range(0, len(g), 16)]
        u = torch.cat([o[0] for o in outs]).cpu()[:, :k]
        lu = torch.cat([o[2] for o in outs]).cpu()[:, :k].double()
        confs = None
        if rat.confidence:  # rlcr: the readout is the stated confidence (greedy) on the most probable answer
            answers = u.argmax(1).tolist()
            confs = rat.confidences(model, g, answers, sample=False)
            u = torch.tensor([verbal_dist(1.0 / k if q is None else q, a, k) for a, (_, q) in zip(answers, confs)])
            lu = None  # a stated confidence has no logits; the analysis scores the saved probabilities
            conf_stats[0] += sum(q is None for _, q in confs)
            conf_stats[1] += sum(q for _, q in confs if q is not None)
            conf_stats[2] += len(confs)
        elif rat.credo:  # credo: the confidence read at the slot after a greedy analysis, on the most probable answer
            answers = u.argmax(1).tolist()
            ans = rat.analyses(model, g, answers, sample=False)
            tails = rat.slot_tails(answers, ans)
            zs = torch.cat([rat.forward(model, g[s:s + 16], grad=False, k=k, tails=tails[s:s + 16], slot=rat.slot_ids)[2]
                            for s in range(0, len(g), 16)]).cpu()
            gaps = (zs[:, 0] - zs[:, 1]).double()
            cs = gaps.sigmoid().tolist()
            confs = [(ids, c) for (ids, _), c in zip(ans, cs)]
            u = torch.tensor([verbal_dist(c, a, k) for a, c in zip(answers, cs)])
            lu = None  # the analysis scores c from the saved gap (logs['slot'], amendment A4), not from these probabilities
            slot_answers.append(answers)
            slot_gaps.append(gaps.tolist())
            conf_stats[0] += sum(not closed for _, closed in ans)
            conf_stats[1] += sum(cs)
            conf_stats[2] += len(cs)
        mix.append(u.mean(0).tolist())
        vote.append((torch.bincount(u.argmax(1), minlength=k).float() / len(u)).tolist())
        single.append(u[0].tolist())
        lsingle.append(None if lu is None else lu[0].tolist())
        lmix.append(None if lu is None else (torch.logsumexp(lu, 0) - math.log(len(lu))).tolist())
        forced += sum(f for _, _, f in g)
        if examples is not None and len(examples) < 6:
            examples += [dict(uid=it["uid"], q=it["q"], rationale=rat.tok.decode(r),
                              u=[round(x, 3) for x in uu.tolist()],
                              **({} if confs is None else dict(confidence=rat.tok.decode(confs[j][0]))))
                         for j, ((_, r, _), uu) in enumerate(zip(g[:2], u[:2]))]
    if stats is not None and rat.confidence:
        stats.update(confidence_unparsed_rate=conf_stats[0] / max(1, conf_stats[2]),
                     mean_stated_confidence=conf_stats[1] / max(1, conf_stats[2] - conf_stats[0]))
    if stats is not None and rat.credo:
        stats.update(analysis_unclosed_rate=conf_stats[0] / max(1, conf_stats[2]),
                     mean_slot_confidence=conf_stats[1] / max(1, conf_stats[2]))
    if logs is not None:
        logs.update(mixture=lmix, single=lsingle)
        if rat.credo:
            logs["slot"] = dict(answer=slot_answers, gap=slot_gaps)
    return mix, vote, single, forced / (m * len(items))


def evaluate(model, rat, items, m, max_new_tokens, keep_dists=False):
    stats, examples, logs = {}, [], {}
    mix, vote, single, forced = decision_dists(model, rat, items, m, max_new_tokens, stats, examples, logs)
    out = dict(mixture=readout_metrics(items, mix), vote=readout_metrics(items, vote),
               single=readout_metrics(items, single), forced_marker_rate=forced, rationale=stats, examples=examples)
    if keep_dists:
        out["dists"] = dict(mixture=mix, single=single)
        if not (rat.confidence or rat.credo):
            out["logps"] = logs
        if rat.credo:  # the decided option and the slot's logit gap of every rationale (npj_report scores c from them)
            out["slot"] = logs["slot"]
        out["uids"] = [it["uid"] for it in items]
    return out


def save_weights(net, path):
    """the weights in bf16 (the sampling copy's precision), written through <path>.tmp: a killed save never leaves a
    truncated checkpoint under the real name. bf16 tensors are written as they are (torch.save moves one storage at a
    time to the host); other floating tensors are converted on their device once per storage, so tied weights stay
    tied. Free an optimizer that lives in host memory (--offload-optimizer) before calling it."""
    state, conv = {}, {}
    for k, v in net.state_dict().items():
        v = v.detach()
        if v.is_floating_point() and v.dtype != torch.bfloat16:
            key = (v.data_ptr(), tuple(v.shape), v.dtype)
            if key not in conv:
                conv[key] = v.to(torch.bfloat16)
            v = conv[key]
        state[k] = v
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    """the run file, written through <path>.tmp: a run killed while writing never leaves a truncated file, which the
    job queue would take as finished if it already held '"test": {'"""
    tmp = f"{path}.tmp"
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp).write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


NPJ_SPLITS = Path(__file__).resolve().parents[1] / "data" / "npj_splits.json"


def check_splits(task, splits, wanted):
    """wanted: [(split, n items)]. Every split exists and yields n > 0 items; where data/npj_splits.json records
    the sha256 of the first n uids of a split (the npj protocol's splits), they must match: a changed dataset or loader
    stops the run before training instead of silently moving items between splits."""
    ref = json.loads(NPJ_SPLITS.read_text()).get(task, {}) if NPJ_SPLITS.exists() else {}
    for name, n in wanted:
        if name not in splits or not splits[name][:n]:
            raise SystemExit(f"{task}: split '{name}' is missing or empty (n={n})")
        h = ref.get(name, {}).get(str(len(splits[name][:n])))
        if h is not None:
            got = hashlib.sha256("\n".join(it["uid"] for it in splits[name][:n]).encode()).hexdigest()
            if got != h:
                raise SystemExit(f"{task}: split '{name}' ({n} items) differs from data/npj_splits.json")


def prepare_checkpoint_dir(path):
    """create the directory of a checkpoint path at start, so a missing directory fails in seconds, not hours"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if not os.access(d, os.W_OK):
        raise SystemExit(f"checkpoint directory {d} is not writable")


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_item_jsd"} if isinstance(v, dict) else v for k, v in m.items()}


def draw(rng, item):
    """One outcome: a random annotator (ChaosNLI) or the answer key (MMLU-Pro: counts are one-hot)."""
    return rng.choices(range(len(item["counts"])), weights=item["counts"])[0]


def check_args(ap, args):
    """the checks of a parsed command line that argparse cannot express (ap.error on failure); returns the sorted
    snapshot steps. Job planners call it to validate a job line without starting the run."""
    if args.rlcr_advantage != "grpo" and args.objective != "rlcr":
        ap.error("--rlcr-advantage applies to --objective rlcr only")
    if args.readout_score != "brier" and not (args.objective == "rlcd" and args.no_score_function):
        ap.error("--readout-score log/linear is defined for the readout-only objective (rlcd --no-score-function)")
    if args.sf_reward != "proper" and not (args.objective == "rlcd" and not args.no_score_function
                                           and args.sf_reader == "self" and args.diversity_weight == 0.0
                                           and not args.pathwise_warmup and not args.accuracy_bonus):
        ap.error("--sf-reward correct needs the per-rationale rlcd objective with its score-function term "
                 "(--diversity-weight 0, --sf-reader self, no --no-score-function, --pathwise-warmup or --accuracy-bonus)")
    if args.thinking and args.objective == "rlcr":
        ap.error("--thinking is not implemented for --objective rlcr")
    if args.objective == "credo" and (args.kl_beta or args.kl_tokens or args.thinking or args.no_score_function
                                      or args.pathwise_warmup or args.sf_coef != 1.0 or args.sf_reader != "self"
                                      or args.accuracy_bonus or args.length_penalty or args.rationale_source != "policy"
                                      or args.init_checkpoint or args.snapshot_steps.strip()):
        ap.error("--objective credo is CREDO as specified in amendment A4: no KL (--kl-beta 0, --kl-tokens 0), no "
                 "--thinking, and none of --no-score-function, --pathwise-warmup, --sf-coef, --sf-reader, "
                 "--accuracy-bonus, --length-penalty, --rationale-source, --init-checkpoint or --snapshot-steps")
    cap = getattr(args, "credo_analysis_tokens", None)  # absent unless given (other arms' run files are unchanged)
    if (args.objective == "credo") != (cap is not None) or (cap is not None and cap < 1):
        ap.error("--credo-analysis-tokens N (N >= 1, set by amendment A4's probe) is required by --objective credo and "
                 "applies to it only")
    if args.kl_tokens > 0 and not args.kl_beta > 0:
        ap.error("--kl-tokens needs --kl-beta > 0 (the reference model); without it the anchor would be off")
    if args.sf_reward != "proper" and (args.sf_coef == 0 or args.grad_diag):
        ap.error("--sf-reward correct: --sf-coef must be > 0, and --grad-diag is not computed for it")
    extra = [x for x in args.extra_splits.split(",") if x.strip()]
    if any(x not in ("test", "fresh", "calib") for x in extra) or len(set(extra)) != len(extra) \
            or args.test_split in extra:
        ap.error("--extra-splits: distinct names among test, fresh, calib, other than --test-split")
    if args.rationale_source == "ref" and not (args.objective == "rlcd" and args.no_score_function and args.kl_beta > 0
                                               and not args.pathwise_warmup and not args.eval_only):
        ap.error("--rationale-source ref needs the readout-only rlcd objective (--no-score-function) and the reference "
                 "model (--kl-beta > 0), without --pathwise-warmup")
    try:
        snaps = sorted({int(s) for s in args.snapshot_steps.split(",") if s.strip()})
    except ValueError:
        ap.error("--snapshot-steps: comma-separated integers")
    if snaps and (args.eval_only or not args.save_checkpoint or snaps[0] < 1 or snaps[-1] >= args.steps):
        ap.error("--snapshot-steps needs --save-checkpoint and steps in [1, --steps), and does not apply to --eval-only")
    if snaps and (args.test_split != "test" or args.offload_optimizer):
        ap.error("--snapshot-steps: only on the validation split (--test-split test) and without --offload-optimizer "
                 "(a fresh-split number per snapshot would invite choices on the final test set; a save next to the "
                 "offloaded optimizer state can exhaust host memory)")
    return snaps


def hub_offline():
    """use the cached model and dataset revisions (an upstream push cannot change a model mid-protocol); set
    HF_HUB_OFFLINE=0 / HF_DATASETS_OFFLINE=0 explicitly to resolve online"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main():
    hub_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="chaosnli")
    ap.add_argument("--objective", choices=["rlcd", "grpo", "rloo", "rlcr", "rft", "credo"], default="rlcd",
                    help="rft = STaR / rejection-sampling fine-tuning: behaviour cloning (rationale + answer token) of "
                         "this step's sampled trajectories whose sampled answer is correct; no baseline, no KL. "
                         "rloo / rlcr / credo: see the module docstring")
    ap.add_argument("--credo-analysis-tokens", type=int, default=argparse.SUPPRESS,
                    help="credo only (required): token cap of the analysis, fixed by amendment A4's probe "
                         "(python -m openjev_rlcd.credo_probe)")
    ap.add_argument("--rlcr-advantage", choices=["grpo", "loo"], default="grpo",
                    help="rlcr only: 'grpo' = group-normalized advantages, as in the RLCR recipe; 'loo' = leave-one-out "
                         "advantages without the division by the group's std, as --objective rloo")
    ap.add_argument("--readout-score", choices=["brier", "log", "linear"], default="brier",
                    help="score for the readout-only objective (with --no-score-function): 'log' = CE on the answer "
                         "distribution read after the model's own rationale; 'linear' = control: 2 u_Y - 1, the expected "
                         "Brier score of an answer sampled from u, i.e. the Brier score 2 u_Y - ||u||^2 without its "
                         "-||u||^2 term: a dense pathwise signal on the same scale, but improper (its optimum is one-hot)")
    ap.add_argument("--sf-reward", choices=["proper", "correct"], default="proper",
                    help="rlcd with the score-function term: the reward of a rationale. 'proper' = the Brier score of its "
                         "own readout (RLCD-RL); 'correct' = control: the Brier score of ONE answer sampled from that "
                         "readout, 2 * 1[A = Y] - 1 (the RLVR reward up to an affine map); the pathwise readout term, "
                         "the leave-one-out baseline and the KL shaping are unchanged")
    ap.add_argument("--rationale-source", choices=["policy", "ref"], default="policy",
                    help="whose rationales a training step reads the answer distribution after: 'policy' = the model's "
                         "own current rationales (the deployed setting); 'ref' = control: rationales sampled from the "
                         "frozen initial model (the reference; needs --kl-beta > 0), readout-only objective, so the "
                         "readout is trained after reasoning the deployed model does not produce. Evaluation always "
                         "samples the policy's own rationales")
    ap.add_argument("--thinking", action="store_true",
                    help="the chat template's thinking mode (Qwen3, SmolLM3): the rationale is the think block and the "
                         "text after it, up to the marker written after </think>; set --max-new-tokens accordingly")
    ap.add_argument("--test-split", choices=["test", "fresh", "calib"], default="test",
                    help="split of the final test pass (and of the snapshots): 'test' = the split of the earlier runs "
                         "(the validation split of the npj protocol); 'fresh' = the independent final test set; 'calib' "
                         "= held-out items for decision thresholds (paper/npj/PROTOCOL.md)")
    ap.add_argument("--extra-splits", default="",
                    help="comma-separated splits evaluated as well, in the same process, after the final dev pass and "
                         "before the final test pass (one temperature for all): log['eval_<split>'] (e.g. calib)")
    ap.add_argument("--snapshot-steps", default="",
                    help="comma-separated steps below --steps at which the weights are saved as <--save-checkpoint "
                         "without .pt>_step<N>.pt and the dev and test splits are evaluated (log['snapshots'][N]: "
                         "dev_dists, dev_logps, test_eval); only with --test-split test (the validation split) and "
                         "without --offload-optimizer")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--items-per-step", type=int, default=8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--m-eval", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--dev-cap", type=int, default=199)
    ap.add_argument("--test-cap", type=int, default=400)
    ap.add_argument("--kl-beta", type=float, default=0.0, help="KL(pi || pi_ref) on rationales, via reward shaping")
    ap.add_argument("--kl-tokens", type=float, default=0.0,
                    help="beta for a pathwise KL anchor: beta * sum over rationale positions of KL(pi || pi_ref) of the "
                         "next-token distributions (needs --kl-beta > 0 for the reference model). Unlike the REINFORCE "
                         "KL shaping it also acts in stage 1 (--no-score-function), where the readout gradient otherwise "
                         "lets the rationales drift")
    ap.add_argument("--length-penalty", type=float, default=0.0, help="penalty for rationales that never write the marker")
    ap.add_argument("--diversity-weight", type=float, default=1.0,
                    help="lambda in J_single + lambda Var_r(u): 1 = mixture objective, 0 = per-rationale objective")
    ap.add_argument("--fixed-label", action="store_true",
                    help="ONE fixed annotator label per train item instead of a fresh draw per visit (ChaosNLI)")
    ap.add_argument("--no-score-function", action="store_true",
                    help="ablation: drop the REINFORCE term for rationales (answer-head/pathwise gradient only)")
    ap.add_argument("--sf-coef", type=float, default=1.0,
                    help="scale of the REINFORCE (score-function) term for the rationales, KL shaping included; "
                         "1 = unbiased, 0 = --no-score-function. Under Adam a noisy term dilutes the pathwise signal")
    ap.add_argument("--sf-reader", choices=["self", "ref", "none"], default="self",
                    help="whose answer distribution scores a rationale in the REINFORCE term: the policy's own readout "
                         "(unbiased gradient of E_r S(u_r, Y)) or the frozen reference model's (a calibrated external "
                         "reader after --pathwise-warmup: rewards informative reasoning, not self-assertive reasoning); "
                         "the policy's readout is always trained on its own u by the pathwise term; 'none' = control: "
                         "the REINFORCE term carries only the shaping (KL anchor), no outcome signal")
    ap.add_argument("--grad-diag", type=int, default=0,
                    help="every N steps log the pathwise vs score-function gradient norms (first item, 3 MLP weights)")
    ap.add_argument("--accuracy-bonus", type=float, default=0.0,
                    help="alpha: per-rationale reward J(u_i) + alpha 1[argmax u_i = Y]. The optimum is still u(r) = q "
                         "(argmax q maximizes the bonus), but the reasoning gets a correctness signal (RLCR-style)")
    ap.add_argument("--pathwise-warmup", type=int, default=0,
                    help="first N steps without the REINFORCE term: calibrate the readout after the model's own "
                         "reasoning before optimizing the reasoning itself. With --kl-beta the reference is reset to "
                         "the model at step N, so the KL anchors the RL phase at the calibrated readout (the warm-up "
                         "drifts far from the initial model because the anchor only acts through REINFORCE)")
    ap.add_argument("--save-checkpoint", default="")
    ap.add_argument("--init-checkpoint", default="",
                    help="start training from this saved (bf16) teacher; the KL reference is set to it as well, so "
                         "several continuations fork from an identical point (fresh optimizer state for all)")
    ap.add_argument("--checkpoint", default="", help="weights for --eval-only")
    ap.add_argument("--offload-optimizer", action="store_true",
                    help="for models whose fp32 weights + AdamW state do not fit on the GPU (3-4B on 48 GB): one GPU "
                         "copy (bf16 Linear weights, fp32 rest) for sampling, forward and backward, fp32 gradient "
                         "accumulation on the GPU, fp32 master weights and AdamW on the CPU (offload.py)")
    ap.add_argument("--gen-batch", type=int, default=64,
                    help="sequences generated per batch (smaller for long prompts on a full GPU; same distribution)")
    ap.add_argument("--gpu-headroom-gb", type=float, default=8.0,
                    help="with --offload-optimizer: GPU memory left free for activations and generation; the "
                         "optimizer state of as many parameters as fit in the rest stays on the GPU")
    ap.add_argument("--eval-only", action="store_true",
                    help="re-evaluate --checkpoint (or the base model) on dev + test with the current metrics")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    snaps = check_args(ap, args)
    if args.save_checkpoint:
        prepare_checkpoint_dir(args.save_checkpoint)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    credo = args.objective == "credo"
    rat = Rationales(tok, args.task, micro=args.gen_batch, confidence=args.objective == "rlcr", thinking=args.thinking,
                     credo=credo, analysis_cap=args.credo_analysis_tokens if credo else None)
    load = TASKS[args.task]["load"]
    settings = dict(credo=dict(CREDO, analysis_tokens=args.credo_analysis_tokens, slot_words=list(CREDO_SLOT_WORDS))) \
        if credo else {}  # recorded in the run file

    if args.eval_only:  # same bf16 weights the training loop samples from and saves
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        splits = load()
        extras = [x for x in args.extra_splits.split(",") if x.strip()]
        check_splits(args.task, splits, [("dev", args.dev_cap)] + [(x, args.test_cap) for x in [args.test_split] + extras])
        d = evaluate(model, rat, splits["dev"][:args.dev_cap], args.m_eval, args.max_new_tokens, keep_dists=True)
        out = dict(config=vars(args), **settings, dev_dists=d.pop("dists"), dev_uids=d.pop("uids"))
        if "logps" in d:
            out["dev_logps"] = d.pop("logps")
        if "slot" in d:
            out["dev_slot"] = d.pop("slot")
        out["dev"] = d
        for x in extras:  # the same temperature as the test split: fitted later on this run's dev readouts
            out[f"eval_{x}"] = evaluate(model, rat, splits[x][:args.test_cap], args.m_eval, args.max_new_tokens,
                                        keep_dists=True)
            write_json(args.output, out)
        out["test"] = evaluate(model, rat, splits[args.test_split][:args.test_cap], args.m_eval, args.max_new_tokens,
                               keep_dists=True)
        write_json(args.output, out)
        print(json.dumps({k: v for k, v in strip(out["test"]).items() if k not in ("dists", "logps", "uids", "examples")}),
              flush=True)
        return

    if args.offload_optimizer:  # one GPU copy: sampled from, trained, refreshed from the CPU master weights
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa",
                                                     device_map=DEVICE)
        model = bf16_matmul_weights(model)
        gen_model = model
    else:  # fp32 weights for training, a bf16 copy for sampling, synced after every step
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
        gen_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
    model.gradient_checkpointing_enable()
    gp, tp = list(gen_model.parameters()), list(model.parameters())
    assert [p.shape for p in gp] == [p.shape for p in tp]

    ref_model = None
    if args.kl_beta > 0:  # frozen initial policy for the KL regularizer
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        ref_model.eval().requires_grad_(False)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=DEVICE)
        for net in {id(n): n for n in [model, gen_model] + ([ref_model] if ref_model is not None else [])}.values():
            net.load_state_dict(state)
        del state

    def sync():
        if gen_model is model:  # the optimizer already refreshed the bf16 weights
            return
        with torch.no_grad():
            for g, p in zip(gp, tp):
                g.copy_(p)

    if args.offload_optimizer:  # after the reference model is loaded: the GPU share of the optimizer uses what is free
        opt = OffloadedAdamW(model, lr=args.lr, weight_decay=0.0, gpu_headroom_gb=args.gpu_headroom_gb)
        print(json.dumps(dict(optimizer_on_gpu_b=opt.gpu_params_b, optimizer_on_cpu_b=opt.cpu_params_b)), flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    splits = load()
    extras = [x for x in args.extra_splits.split(",") if x.strip()]
    check_splits(args.task, splits, [("train", None), ("dev", args.dev_cap)]
                 + [(x, args.test_cap) for x in [args.test_split] + extras])
    dev, test = splits["dev"][:args.dev_cap], splits[args.test_split][:args.test_cap]
    log, t0 = dict(config=vars(args), **settings, curve=[]), time.perf_counter()

    def record(step, keep=False):
        """dev pass at step; with keep (snapshots) or at the last step it returns (dists, logps or None)"""
        m = strip(evaluate(gen_model, rat, dev, args.m_eval, args.max_new_tokens, keep_dists=keep or step == args.steps))
        kept = (m.pop("dists"), m.pop("logps", None)) if "dists" in m else None
        uids = m.pop("uids", None)
        slot = m.pop("slot", None)  # credo
        if kept is not None and step == args.steps:  # final dev distributions: every arm is temperature-scaled on dev
            log["dev_dists"] = kept[0]
            if kept[1] is not None:
                log["dev_logps"] = kept[1]
            log["dev_uids"] = uids
            if slot is not None:
                log["dev_slot"] = slot
        for ex in m["examples"][:2]:
            print(json.dumps(ex)[:600], flush=True)
        log["curve"].append(dict(step=step, dev=m, minutes=(time.perf_counter() - t0) / 60))
        print(json.dumps(log["curve"][-1]), flush=True)
        write_json(args.output, log)
        return kept

    fixed = {it["uid"]: draw(random.Random(f"{args.seed}-{it['uid']}"), it) for it in splits["train"]}
    record(0)
    for step in range(1, args.steps + 1):
        if ref_model is not None and args.pathwise_warmup and step == args.pathwise_warmup + 1:
            ref_model.load_state_dict(gen_model.state_dict())
            print(json.dumps(dict(step=step, event="reference reset to the warmed-up model")), flush=True)
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        groups = rat.sample(gen_model if args.rationale_source == "policy" else ref_model, items, args.m,
                            args.max_new_tokens)
        model.train()
        opt.zero_grad(set_to_none=True)
        hit, forced, kl_sum, umax, var, kl_tok_sum = 0.0, 0, 0.0, 0.0, 0.0, 0.0
        conf_log = [0.0, 0, 0]  # rlcr: sum of parsed confidences, unparseable ones, all
        credo_log = [0.0, 0, 0, 0.0]  # credo: sum of slot confidences, unclosed analyses, all, sum of calibration losses
        tok_anchor = args.kl_tokens > 0 and ref_model is not None
        for it, g in zip(items, groups):
            y = draw(rng, it)  # one fresh outcome (always drawn: same item order across arms)
            if args.fixed_label:
                y = fixed[it["uid"]]
            k = rat.k(it)
            tails = None
            if args.objective == "rlcr":  # the answer and its confidence line are sampled first; one pass scores both
                with torch.no_grad():
                    u0 = rat.forward(gen_model, g, grad=False, k=k)[0][:, :k].float()
                a_cr = torch.multinomial(u0, 1).squeeze(1)
                confs = rat.confidences(gen_model, g, a_cr.tolist(), sample=True)
                tails = [([rat.label_ids[a]] + rat.conf_prefix + toks, 1 + len(rat.conf_prefix))
                         for a, (toks, _) in zip(a_cr.tolist(), confs)]
                model.train()  # confidences() put the sampler in eval mode; with --offload-optimizer it is the model
                conf_log[0] += sum(q for _, q in confs if q is not None)
                conf_log[1] += sum(q is None for _, q in confs)
                conf_log[2] += len(confs)
            if args.objective == "credo":  # the answer and its analysis are sampled first; one pass scores both
                with torch.no_grad():
                    u0 = rat.forward(gen_model, g, grad=False, k=k)[0][:, :k].float()
                a_cr = torch.multinomial(u0, 1).squeeze(1)
                ans = rat.analyses(gen_model, g, a_cr.tolist(), sample=True)
                tails = rat.slot_tails(a_cr.tolist(), ans)
                model.train()  # analyses() put the sampler in eval mode
                credo_log[1] += sum(not closed for _, closed in ans)
                credo_log[2] += len(ans)
            out = rat.forward(model, g, grad=True, k=k, return_hidden=tok_anchor, tails=tails,
                              slot=rat.slot_ids if args.objective == "credo" else None)
            u, logp_r = out[0][:, :k], out[1]
            if args.no_score_function or step <= args.pathwise_warmup:
                logp_r = logp_r.detach()
            m = len(g)
            extra = torch.zeros(m, device=DEVICE)  # per-rationale shaping terms (KL anchor, length penalty)
            u_ref = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_out = rat.forward(ref_model, g, grad=False, need_logp=True, k=k, return_hidden=tok_anchor)
                u_ref, ref_lp = ref_out[0][:, :k], ref_out[1]
                logratio = logp_r.detach() - ref_lp
                extra -= (args.kl_beta / m) * logratio
                kl_sum += logratio.mean().item()
            if args.length_penalty:
                extra -= (args.length_penalty / m) * torch.tensor([float(f) for _, _, f in g], device=DEVICE)
            if args.accuracy_bonus:  # depends on r_i only through argmax u_i: score-function part only
                extra += (args.accuracy_bonus / m) * (u.detach().argmax(1) == y).float()
            logp_sf = logp_r * args.sf_coef if args.sf_coef != 1.0 else logp_r
            if args.objective == "rft":
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                keep = (a == y).float()  # clone only the trajectories whose sampled answer is right
                loss = -(keep * (logp_r + u[:, y].clamp_min(1e-12).log())).sum() / m
                if ref_model is not None:  # optional KL anchor (--kl-beta), same shaping as the other arms
                    reg = extra - (extra.sum() - extra) / (m - 1)
                    loss = loss - (reg * logp_r).sum()
            elif args.readout_score == "log":  # readout-only with the log score (pathwise; no REINFORCE)
                assert args.no_score_function, "--readout-score log is only defined for the readout-only objective"
                loss = -u[:, y].clamp_min(1e-12).log().mean()
            elif args.readout_score == "linear":  # control: the Brier score without -||u||^2 (improper)
                assert args.no_score_function
                loss = -(2 * u[:, y] - 1).mean()
            elif args.objective == "grpo":
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                r = (a == y).float()
                adv = group_advantages(r, "grpo")
                reg = extra - (extra.sum() - extra) / (m - 1)  # same shaping, leave-one-out baseline (unbiased)
                loss = (-(adv * (logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.objective == "rloo":  # grpo's reward, leave-one-out baseline, no division by the group std
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                r = (a == y).float()
                adv = group_advantages(r, "loo")
                reg = extra - (extra.sum() - extra) / (m - 1)
                loss = (-(adv * (logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.objective == "rlcr":  # grpo with the RLCR reward; the confidence tokens are part of the action
                a = a_cr.to(DEVICE)
                r = rlcr_reward((a == y).float(), [q for _, q in confs])
                adv = group_advantages(r, args.rlcr_advantage)
                reg = extra - (extra.sum() - extra) / (m - 1)
                logp_c = out[-1] * args.sf_coef if args.sf_coef != 1.0 else out[-1]
                loss = (-(adv * (logp_sf + logp_c + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.objective == "credo":  # out: (u, logp_r, logp of the analyses, slot logits [M, 2])
                a = a_cr.to(DEVICE)
                c = (out[-1][:, 0] - out[-1][:, 1]).sigmoid()  # P(high) / (P(high) + P(low)), with gradient
                formatted = torch.tensor([float(not f and closed) for (_, _, f), (_, closed) in zip(g, ans)],
                                         device=DEVICE)
                n_tok = torch.tensor([len(r) + 1 + len(ids) for (_, r, _), (ids, _) in zip(g, ans)], device=DEVICE,
                                     dtype=torch.float32)  # sampled tokens: rationale, answer, analysis
                logp_ans = logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log()
                loss, cr = credo_loss((a == y).float(), c, formatted, logp_ans, out[-2], n_tok,
                                      min(1.0, step / CREDO["warmup"]))
                credo_log[0] += c.detach().sum().item()
                credo_log[3] += cr["cal"].item()
            elif args.sf_reward == "correct":  # control: the reasoning rewarded by one sampled answer's score
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                loss = sampled_answer_surrogate(u, logp_sf, y, a, extra_local=extra)
            elif args.sf_reader == "none":  # anchor-only control: equal rewards -> only the shaping terms remain
                loss = (lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight)
                        + lambda_surrogate(torch.full_like(u, 1.0 / k), logp_sf, y, args.diversity_weight, extra_local=extra))
            elif args.sf_reader == "ref" and u_ref is not None:  # pathwise on own u; rationales scored by the reader
                loss = (lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight)
                        + lambda_surrogate(u_ref, logp_sf, y, args.diversity_weight, extra_local=extra))
            else:
                loss = lambda_surrogate(u, logp_sf, y, args.diversity_weight, extra_local=extra)
                if args.grad_diag and step % args.grad_diag == 0 and it is items[0] and logp_sf.requires_grad:
                    layers = model.model.layers
                    sub = [layers[i].mlp.down_proj.weight for i in (0, len(layers) // 2, len(layers) - 1)]
                    d_path = torch.autograd.grad(lambda_surrogate(u, logp_sf.detach(), y, args.diversity_weight, extra),
                                                 sub, retain_graph=True, allow_unused=True)
                    d_sf = torch.autograd.grad(lambda_surrogate(u.detach(), logp_sf, y, args.diversity_weight, extra),
                                               sub, retain_graph=True, allow_unused=True)
                    vp = torch.cat([(d if d is not None else torch.zeros_like(w)).flatten() for d, w in zip(d_path, sub)])
                    vs = torch.cat([(d if d is not None else torch.zeros_like(w)).flatten() for d, w in zip(d_sf, sub)])
                    print(json.dumps(dict(step=step, diag_path_norm=vp.norm().item(), diag_sf_norm=vs.norm().item(),
                                          diag_ratio=(vs.norm() / vp.norm().clamp_min(1e-12)).item(),
                                          diag_cos=torch.nn.functional.cosine_similarity(vp, vs, dim=0).item())),
                          flush=True)
            if tok_anchor:  # pathwise KL anchor on the rationale distribution
                kl_tok = rat.token_kl(model, out[2], ref_model, ref_out[2])
                loss = loss + args.kl_tokens * kl_tok.mean()
                kl_tok_sum += kl_tok.mean().item()
            (loss / len(items)).backward()
            hit += u[:, y].mean().item()
            ud = u.detach()
            umax += ud.max(1).values.mean().item()
            var += ((ud - ud.mean(0)) ** 2).sum(1).mean().item()
            forced += sum(f for _, _, f in g)
        if args.offload_optimizer:
            opt.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync()
        if step % 10 == 0:
            print(json.dumps(dict(step=step, mean_p_of_outcome=hit / len(items),
                                  forced_marker_rate=forced / (args.m * len(items)), kl_est=kl_sum / len(items),
                                  kl_tokens=kl_tok_sum / len(items),
                                  mean_u_max=umax / len(items), var_u=var / len(items),
                                  **({} if not conf_log[2] else dict(
                                      mean_stated_confidence=conf_log[0] / max(1, conf_log[2] - conf_log[1]),
                                      confidence_unparsed_rate=conf_log[1] / conf_log[2])),
                                  **({} if not credo_log[2] else dict(
                                      mean_slot_confidence=credo_log[0] / credo_log[2],
                                      analysis_unclosed_rate=credo_log[1] / credo_log[2],
                                      calibration_loss=credo_log[3] / len(items))),
                                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0,
                                  minutes=(time.perf_counter() - t0) / 60)), flush=True)
        if step % args.eval_every == 0 or step in snaps or step == args.steps:  # the last step: dev_dists for TS
            kept = record(step, keep=step in snaps)
        if step in snaps:  # weights and a full test pass at an intermediate step
            ck = f"{args.save_checkpoint.removesuffix('.pt')}_step{step}.pt"
            save_weights(gen_model, ck)
            t_snap = evaluate(gen_model, rat, test, args.m_eval, args.max_new_tokens, keep_dists=True)
            print(json.dumps(dict(snapshot=step, **{k: v for k, v in strip(t_snap).items()
                                                    if k not in ("dists", "logps", "uids", "examples")})), flush=True)
            log.setdefault("snapshots", {})[str(step)] = dict(
                checkpoint=ck, dev_dists=kept[0], **({} if kept[1] is None else dict(dev_logps=kept[1])),
                test_eval=t_snap,  # never "test": a run file holding '"test": {' counts as finished (jobqueue, run())
                minutes=(time.perf_counter() - t0) / 60)
            write_json(args.output, log)
    # free training-only memory (optimizer state, gradients, reference model) before the checkpoint and the long test
    # pass: with --offload-optimizer the host master weights + AdamW state are ~56 GB for Phi-4-mini (25 GB pinned), and
    # writing the checkpoint on top of them got the run killed under host-memory pressure (2026-10-15)
    model.zero_grad(set_to_none=True)
    del opt, ref_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if args.save_checkpoint:  # bf16, as the default recipe's sampling copy; saved before the test pass
        state = gen_model.state_dict()
        if args.offload_optimizer:
            state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        tmp = f"{args.save_checkpoint}.tmp"  # a killed save must not leave a truncated checkpoint under the real name
        torch.save(state, tmp)
        os.replace(tmp, args.save_checkpoint)
        del state
    for x in extras:  # same weights and the same final dev pass (one temperature) as the test split
        log[f"eval_{x}"] = evaluate(gen_model, rat, splits[x][:args.test_cap], args.m_eval, args.max_new_tokens,
                                    keep_dists=True)
        write_json(args.output, log)
    log["test"] = evaluate(gen_model, rat, test, args.m_eval, args.max_new_tokens, keep_dists=True)
    print(json.dumps({k: v for k, v in strip(log["test"]).items() if k not in ("dists", "logps", "uids", "examples")}),
          flush=True)
    write_json(args.output, log)


if __name__ == "__main__":
    main()

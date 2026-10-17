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
         1[A_i = Y] - (C_i - 1[A_i = Y])^2, group-normalized as in grpo; the stated confidence is the readout
         (C_i on A_i, the rest spread evenly over the other options).
  rft  : STaR / rejection-sampling fine-tuning on the trajectories whose sampled answer is correct.
Readouts at evaluation:
  mixture : mean of u_i over M rationales
  vote    : frequency of argmax u_i
  single  : u_1 from ONE rationale (a single query)

  python -m openjev_rlcd.train --task gsm8kv ... --output results/x.json   (see scripts/ for every run)
"""
import argparse
import contextlib
import gc
import json
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


def _token_logp(lm_head, h, tgt):
    return lm_head(h).float().log_softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)


def _token_kl(lm_head, ref_head, h, h_ref):
    """KL(pi || pi_ref) between the next-token distributions at each position (rows of h, h_ref)."""
    lp = lm_head(h).float().log_softmax(-1)
    lq = ref_head(h_ref).float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


class Rationales:
    def __init__(self, tok, task="chaosnli", micro=64, confidence=False):
        self.tok, self.task, self.micro = tok, task, micro  # micro: sequences generated per batch
        self.confidence = confidence  # rlcr: the prompt asks for a 'Confidence: C' line after the answer
        if tok.pad_token_id is None:  # padding is masked out everywhere; the pad id only has to exist
            tok.pad_token = tok.eos_token
        self.label_ids = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in TASKS[task]["labels"]]
        assert len(set(self.label_ids)) == len(self.label_ids)
        self.eos = {tok.eos_token_id, tok.pad_token_id}
        self.suffix = tok.encode("\n" + MARKER, add_special_tokens=False)
        self.conf_prefix = tok.encode(CONF_PREFIX, add_special_tokens=False)

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
        ctxs = [p + r + (self.suffix if f else []) + [self.label_ids[a]] + self.conf_prefix
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
    def sample(self, gen_model, items, m, max_new_tokens, micro=None):
        """Per item, m tuples (prompt_ids, rationale_ids, forced: bool)."""
        gen_model.eval()
        micro = micro or self.micro
        reqs = [(i, prompt(self.tok, it, self.task, confidence=self.confidence)) for i, it in enumerate(items)
                for _ in range(m)]
        out = [[] for _ in items]
        stop = set(self.eos)  # the tokenizer's EOS/pad and every EOS id the model's generation config stops at
        cfg = getattr(gen_model.generation_config, "eos_token_id", None)
        stop |= set(cfg if isinstance(cfg, (list, tuple)) else [cfg]) - {None}
        for s in range(0, len(reqs), micro):
            chunk = reqs[s:s + micro]
            enc = self.tok([p for _, p in chunk], return_tensors="pt", padding=True, padding_side="left",
                           add_special_tokens=False).to(DEVICE)  # the chat template already holds BOS etc.
            mixed = getattr(gen_model, "openjev_mixed_precision", False)  # bf16 Linear + fp32 rest: needs autocast
            with torch.autocast(DEVICE, dtype=torch.bfloat16) if mixed else contextlib.nullcontext():
                gen = gen_model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                         max_new_tokens=max_new_tokens, stop_strings=[MARKER], tokenizer=self.tok,
                                         pad_token_id=self.tok.pad_token_id)
            width = enc["input_ids"].shape[1]
            for (i, _), row, prow, pmask in zip(chunk, gen, enc["input_ids"], enc["attention_mask"]):
                comp = row[width:].tolist()
                cut = next((k for k, t in enumerate(comp) if t in stop), None)
                head = comp if cut is None else comp[:cut]
                if self.tok.decode(head).rstrip().endswith(MARKER):
                    rat, forced = head, False           # stopped at the marker
                else:
                    rat, forced = (comp if cut is None else comp[:cut + 1]), True  # keep the sampled EOS
                p_ids = prow[pmask.bool()].tolist()
                out[i].append((p_ids, rat, forced))
        return out

    def forward(self, model, group, grad, need_logp=None, k=None, chunk=512, return_hidden=False, tails=None):
        """group: list of (prompt_ids, rationale_ids, forced). Returns u [M, K] (zero beyond each
        row's option count k: int or per-row list) and logp_r [M] (None unless grad or need_logp),
        plus (hidden states, row index, position) of the rationale positions with return_hidden.
        Vocabulary logits are computed only at the answer position and, in checkpointed chunks,
        at the rationale positions -- never over the prompt.
        tails (rlcr): per row (token ids appended after the answer position, number of leading ids not scored);
        u is still read at the answer position (causal attention: the tail cannot change it), and the sum of
        log-probabilities of the scored tail ids is returned as one more last element, logp_tail [M]."""
        need_logp = grad if need_logp is None else need_logp
        seqs, spans, tspans = [], [], []
        for j, (p, r, forced) in enumerate(group):
            ctx = p + r + (self.suffix if forced else [])
            if tails is not None:
                tail, fixed = tails[j]
                tspans.append((len(ctx) + fixed, len(ctx) + len(tail), len(ctx) - 1))
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
            if not need_logp:
                return u, None
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
def decision_dists(model, rat, items, m, max_new_tokens, stats=None, examples=None):
    groups = rat.sample(model, items, m, max_new_tokens)
    mix, vote, single, forced = [], [], [], 0
    if stats is not None:
        texts = [rat.tok.decode(r) for g in groups for _, r, _ in g]
        stats.update(mean_rationale_tokens=sum(len(r) for g in groups for _, r, _ in g) / len(texts),
                     length_limit_rate=sum(len(r) >= max_new_tokens for g in groups for _, r, _ in g) / len(texts),
                     repetition_ratio=sum(map(repetition, texts)) / len(texts))
    conf_stats = [0, 0.0, 0]  # rlcr: unparseable confidences, sum of parsed confidences, rationales
    for it, g in zip(items, groups):
        k = rat.k(it)
        u = torch.cat([rat.forward(model, g[s:s + 16], grad=False, k=k)[0] for s in range(0, len(g), 16)]).cpu()[:, :k]
        confs = None
        if rat.confidence:  # rlcr: the readout is the stated confidence (greedy) on the most probable answer
            answers = u.argmax(1).tolist()
            confs = rat.confidences(model, g, answers, sample=False)
            u = torch.tensor([verbal_dist(1.0 / k if q is None else q, a, k) for a, (_, q) in zip(answers, confs)])
            conf_stats[0] += sum(q is None for _, q in confs)
            conf_stats[1] += sum(q for _, q in confs if q is not None)
            conf_stats[2] += len(confs)
        mix.append(u.mean(0).tolist())
        vote.append((torch.bincount(u.argmax(1), minlength=k).float() / len(u)).tolist())
        single.append(u[0].tolist())
        forced += sum(f for _, _, f in g)
        if examples is not None and len(examples) < 6:
            examples += [dict(uid=it["uid"], q=it["q"], rationale=rat.tok.decode(r),
                              u=[round(x, 3) for x in uu.tolist()],
                              **({} if confs is None else dict(confidence=rat.tok.decode(confs[j][0]))))
                         for j, ((_, r, _), uu) in enumerate(zip(g[:2], u[:2]))]
    if stats is not None and rat.confidence:
        stats.update(confidence_unparsed_rate=conf_stats[0] / max(1, conf_stats[2]),
                     mean_stated_confidence=conf_stats[1] / max(1, conf_stats[2] - conf_stats[0]))
    return mix, vote, single, forced / (m * len(items))


def evaluate(model, rat, items, m, max_new_tokens, keep_dists=False):
    stats, examples = {}, []
    mix, vote, single, forced = decision_dists(model, rat, items, m, max_new_tokens, stats, examples)
    out = dict(mixture=readout_metrics(items, mix), vote=readout_metrics(items, vote),
               single=readout_metrics(items, single), forced_marker_rate=forced, rationale=stats, examples=examples)
    if keep_dists:
        out["dists"] = dict(mixture=mix, single=single)
    return out


def strip(m):
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_item_jsd"} if isinstance(v, dict) else v for k, v in m.items()}


def draw(rng, item):
    """One outcome: a random annotator (ChaosNLI) or the answer key (MMLU-Pro: counts are one-hot)."""
    return rng.choices(range(len(item["counts"])), weights=item["counts"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="chaosnli")
    ap.add_argument("--objective", choices=["rlcd", "grpo", "rloo", "rlcr", "rft"], default="rlcd",
                    help="rft = STaR / rejection-sampling fine-tuning: behaviour cloning (rationale + answer token) of "
                         "this step's sampled trajectories whose sampled answer is correct; no baseline, no KL. "
                         "rloo / rlcr: see the module docstring")
    ap.add_argument("--readout-score", choices=["brier", "log"], default="brier",
                    help="proper score for the readout-only objective (with --no-score-function): 'log' = CE on the "
                         "answer distribution read after the model's own rationale")
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rat = Rationales(tok, args.task, micro=args.gen_batch, confidence=args.objective == "rlcr")
    load = TASKS[args.task]["load"]

    if args.eval_only:  # same bf16 weights the training loop samples from and saves
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        splits = load()
        d = evaluate(model, rat, splits["dev"][:args.dev_cap], args.m_eval, args.max_new_tokens, keep_dists=True)
        t = evaluate(model, rat, splits["test"][:args.test_cap], args.m_eval, args.max_new_tokens, keep_dists=True)
        Path(args.output).write_text(json.dumps(dict(config=vars(args), dev_dists=d.pop("dists"), dev=d, test=t), indent=2))
        print(json.dumps({k: v for k, v in strip(t).items() if k not in ("dists", "examples")}), flush=True)
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
    dev, test = splits["dev"][:args.dev_cap], splits["test"][:args.test_cap]
    log, t0 = dict(config=vars(args), curve=[]), time.perf_counter()

    def record(step):
        m = strip(evaluate(gen_model, rat, dev, args.m_eval, args.max_new_tokens, keep_dists=step == args.steps))
        if "dists" in m:  # final dev distributions: lets every arm be temperature-scaled on dev afterwards
            log["dev_dists"] = m.pop("dists")
        for ex in m["examples"][:2]:
            print(json.dumps(ex)[:600], flush=True)
        log["curve"].append(dict(step=step, dev=m, minutes=(time.perf_counter() - t0) / 60))
        print(json.dumps(log["curve"][-1]), flush=True)
        Path(args.output).write_text(json.dumps(log, indent=2))

    fixed = {it["uid"]: draw(random.Random(f"{args.seed}-{it['uid']}"), it) for it in splits["train"]}
    record(0)
    for step in range(1, args.steps + 1):
        if ref_model is not None and args.pathwise_warmup and step == args.pathwise_warmup + 1:
            ref_model.load_state_dict(gen_model.state_dict())
            print(json.dumps(dict(step=step, event="reference reset to the warmed-up model")), flush=True)
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        groups = rat.sample(gen_model, items, args.m, args.max_new_tokens)
        model.train()
        opt.zero_grad(set_to_none=True)
        hit, forced, kl_sum, umax, var, kl_tok_sum = 0.0, 0, 0.0, 0.0, 0.0, 0.0
        conf_log = [0.0, 0, 0]  # rlcr: sum of parsed confidences, unparseable ones, all
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
            out = rat.forward(model, g, grad=True, k=k, return_hidden=tok_anchor, tails=tails)
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
            elif args.objective == "grpo":
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                r = (a == y).float()
                adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
                reg = extra - (extra.sum() - extra) / (m - 1)  # same shaping, leave-one-out baseline (unbiased)
                loss = (-(adv * (logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.objective == "rloo":  # grpo's reward, leave-one-out baseline, no division by the group std
                a = torch.multinomial(u.detach(), 1).squeeze(1)
                r = (a == y).float()
                adv = r - (r.sum() - r) / (m - 1)
                reg = extra - (extra.sum() - extra) / (m - 1)
                loss = (-(adv * (logp_sf + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
            elif args.objective == "rlcr":  # grpo with the RLCR reward; the confidence tokens are part of the action
                a = a_cr.to(DEVICE)
                r = rlcr_reward((a == y).float(), [q for _, q in confs])
                adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
                reg = extra - (extra.sum() - extra) / (m - 1)
                logp_c = out[-1] * args.sf_coef if args.sf_coef != 1.0 else out[-1]
                loss = (-(adv * (logp_sf + logp_c + u[torch.arange(m, device=DEVICE), a].clamp_min(1e-12).log())).mean()
                        - (reg * logp_sf).sum())
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
                                  gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0,
                                  minutes=(time.perf_counter() - t0) / 60)), flush=True)
        if step % args.eval_every == 0:
            record(step)
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
    log["test"] = evaluate(gen_model, rat, test, args.m_eval, args.max_new_tokens, keep_dists=True)
    print(json.dumps({k: v for k, v in strip(log["test"]).items() if k not in ("dists", "examples")}), flush=True)
    Path(args.output).write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()

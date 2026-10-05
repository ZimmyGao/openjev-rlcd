#!/usr/bin/env python3
"""Zero-shot screening of a backbone before any training (dev split only).

  python -m openjev_rlcd.probe --model HuggingFaceTB/SmolLM3-3B --task gsm8kv --output runs/probe/x.json

Checks that the recipe applies to the model:
  tokenizer  : every answer label is ONE token (" A".." J", " yes"/" no"), the labels are distinct, a pad id exists
  template   : the chat prompt renders in non-thinking mode (shows its tail)
  label mass : at the position after "Answer:", the full-vocabulary probability on the label tokens. The
               readout renormalizes over the labels, so a small mass means the model wanted to write something
               else (e.g. "**B**").
and measures zero-shot quality, raw (no temperature scaling):
  direct     : "Answer:" right after the prompt (the SFT baseline's input)
  reasoning  : one sampled rationale, then the readout (the RL arms' input); forced-marker rate, length
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .metrics import selective_metrics
from .tasks import TASKS, prompt
from .train import DEVICE, MARKER, Rationales


@torch.no_grad()
def read(model, rat, seqs, ks, bs=16):
    """Next-token distribution after each sequence: (u renormalized over the item's labels, label mass)."""
    us, masses = [], []
    for s in range(0, len(seqs), bs):
        chunk, kk = seqs[s:s + bs], ks[s:s + bs]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), rat.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, x in enumerate(chunk):
            ids[j, :len(x)], mask[j, :len(x)] = torch.tensor(x), 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = torch.tensor([len(x) - 1 for x in chunk], device=DEVICE)
            p = model.lm_head(h[torch.arange(len(chunk), device=DEVICE), last]).float().softmax(-1)
        for row, k in zip(p, kk):
            lab = row[rat.label_ids[:k]]
            masses.append(float(lab.sum()))
            us.append((lab / lab.sum()).tolist())
    return us, masses


def quality(items, probs):
    n = len(items)
    ys = [it["q"].index(max(it["q"])) for it in items]
    acc = sum(p.index(max(p)) == y for p, y in zip(probs, ys)) / n
    brier = sum(sum((a - (i == y)) ** 2 for i, a in enumerate(p)) for p, y in zip(probs, ys)) / n
    sel = selective_metrics(items, probs, budgets=(0.05, 0.2))
    return dict(acc=acc, brier=brier, mean_max_p=sum(max(p) for p in probs) / n, ece=sel["ece"], aurc=sel["aurc"],
                **{k: sel[k] for k in ("oracle_cov@0.05", "oracle_cov@0.2")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["gsm8kv", "mmlupro"], required=True)
    ap.add_argument("--n", type=int, default=300, help="dev items")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: 320 (gsm8kv) / 256 (mmlupro)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    max_new = args.max_new_tokens or {"gsm8kv": 320, "mmlupro": 256}[args.task]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    had_pad = tok.pad_token_id is not None
    rat = Rationales(tok, args.task)
    labels = TASKS[args.task]["labels"]
    enc = {lab: tok.encode(" " + lab, add_special_tokens=False) for lab in labels}
    checks = dict(single_token_labels=all(len(v) == 1 for v in enc.values()),
                  distinct_labels=len(set(rat.label_ids)) == len(labels), had_pad_token=had_pad,
                  pad_token=tok.pad_token, eos_token=tok.eos_token,
                  label_tokens={lab: tok.convert_ids_to_tokens(v) for lab, v in enc.items()})
    items = TASKS[args.task]["load"]()["dev"][:args.n]
    rendered = prompt(tok, items[0], args.task)
    checks.update(prompt_tail=rendered[-160:], thinking_in_prompt="<think>" in rendered and "</think>" not in rendered)
    print(json.dumps(checks), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
    ks = [rat.k(it) for it in items]
    marker = tok.encode(MARKER, add_special_tokens=False)
    out = dict(config=vars(args), checks=checks, params_b=sum(p.numel() for p in model.parameters()) / 1e9)

    ctx = [tok(prompt(tok, it, args.task), add_special_tokens=False)["input_ids"] + marker for it in items]
    u, mass = read(model, rat, ctx, ks)
    out["direct"] = dict(**quality(items, u), label_mass=sum(mass) / len(mass))
    print("direct", json.dumps(out["direct"]), flush=True)

    t0 = time.perf_counter()
    groups = rat.sample(model, items, 1, max_new)
    secs = time.perf_counter() - t0
    trip = [g[0] for g in groups]
    ctx = [p + r + (rat.suffix if forced else []) for p, r, forced in trip]
    u, mass = read(model, rat, ctx, ks)
    out["reasoning"] = dict(**quality(items, u), label_mass=sum(mass) / len(mass),
                            forced_marker_rate=sum(f for _, _, f in trip) / len(trip),
                            mean_rationale_tokens=sum(len(r) for _, r, _ in trip) / len(trip),
                            length_limit_rate=sum(len(r) >= max_new for _, r, _ in trip) / len(trip),
                            sample_seconds_per_item=secs / len(items))
    out["examples"] = [dict(rationale=tok.decode(r), u=[round(x, 3) for x in uu], q=it["q"])
                       for (_, r, _), uu, it in zip(trip[:4], u[:4], items[:4])]
    print("reasoning", json.dumps(out["reasoning"]), flush=True)
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

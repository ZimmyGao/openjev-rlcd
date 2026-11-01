"""Analysis-length probe of amendment A4 (CREDO; paper/npj/PROTOCOL.md). The zero-shot model answers the first --n dev
items (default 100) with --m rationales each (default 4) under the CREDO prompt, sampled as a CREDO training step samples
them: the rationale at temperature 1, the answer token from the readout u, then the analysis at temperature 1. As the
evaluation does, it also writes a greedy analysis after the readout's most probable answer. Every analysis is generated
with a cap of --cap tokens (default 256). An analysis closes within c tokens if it wrote '</analysis>' by its c-th
token; the first c tokens of a capped analysis are those of an analysis capped at c, so one pass gives every smaller
cap. Reported: the share closed within each of CAPS, for the sampled and the greedy analyses, and the cap A4's rule
picks for this cell -- the smallest of CAPS at which at least 90% of the sampled analyses close (None if none does).
Every CREDO run uses the largest cap picked over the four Qwen3 cells, max(CAPS) for a cell that picks none
(scripts/plan_npj.py credo_cap). Only dev items are read.
  python -m openjev_rlcd.credo_probe --task gsm8kv --model Qwen/Qwen3-1.7B --max-new-tokens 320 \
      --output runs/npj/credo_probe/gv_qwen3-1.7b.json
"""
import argparse
import json
import time

import torch

from . import train
from .tasks import TASKS

CAPS = (64, 128, 192, 256)
SHARE = 0.9


def close_lengths(analyses):
    """per analysis (ids, closed) the number of tokens through the one that closed it, or None (EOS or cap first)"""
    return [len(ids) if closed else None for ids, closed in analyses]


def shares_within(lengths):
    """{cap: share of the analyses closed within cap tokens}"""
    return {c: sum(n is not None and n <= c for n in lengths) / len(lengths) for c in CAPS}


def choose(shares):
    """A4's rule for one cell: the smallest cap with a closed share of at least SHARE, or None"""
    ok = [c for c in CAPS if shares[c] >= SHARE]
    return min(ok) if ok else None


def main():
    train.hub_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8kv", "mmlupro"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, required=True, help="the task's rationale cap (256 / 320)")
    ap.add_argument("--cap", type=int, default=max(CAPS))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.cap < max(CAPS):
        ap.error(f"--cap must be at least {max(CAPS)}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(train.DEVICE).eval()
    rat = train.Rationales(tok, args.task, micro=args.gen_batch, credo=True, analysis_cap=args.cap)
    dev = TASKS[args.task]["load"]()["dev"][:args.n]
    t0 = time.perf_counter()
    groups = rat.sample(model, dev, args.m, args.max_new_tokens)
    rows, sampled_answers, greedy_answers = [], [], []
    for it, g in zip(dev, groups):
        k = rat.k(it)
        with torch.no_grad():
            u = rat.forward(model, g, grad=False, k=k)[0][:, :k].float()
        sampled_answers += torch.multinomial(u, 1).squeeze(1).tolist()
        greedy_answers += u.argmax(1).tolist()
        rows += g
    sampled = rat.analyses(model, rows, sampled_answers, sample=True)
    greedy = rat.analyses(model, rows, greedy_answers, sample=False)
    ls, lg = close_lengths(sampled), close_lengths(greedy)
    sh_s, sh_g = shares_within(ls), shares_within(lg)
    marker = [not f for _, _, f in rows]
    formatted = {c: sum(w and n is not None and n <= c for w, n in zip(marker, ls)) / len(ls) for c in CAPS}
    chosen = choose(sh_s)
    # "test" holds the result: the job queue counts a run file with a test block as finished
    out = dict(config=vars(args), test=dict(chosen_cap=chosen, share_closed_within_sampled=sh_s,
                                            share_closed_within_greedy=sh_g, share_formatted_within_sampled=formatted),
               chosen_cap=chosen, minutes=(time.perf_counter() - t0) / 60, analyses=len(rows),
               share_wrote_marker=sum(marker) / len(marker), close_lengths_sampled=ls, close_lengths_greedy=lg,
               examples=[dict(rationale=tok.decode(r), analysis_sampled=tok.decode(s[0]), analysis_greedy=tok.decode(q[0]))
                         for (_, r, _), s, q in list(zip(rows, sampled, greedy))[:4]])
    train.write_json(args.output, out)
    print(json.dumps(dict(task=args.task, model=args.model, chosen_cap=chosen, minutes=out["minutes"],
                          sampled=sh_s, greedy=sh_g, formatted=formatted,
                          share_wrote_marker=out["share_wrote_marker"])), flush=True)


if __name__ == "__main__":
    main()

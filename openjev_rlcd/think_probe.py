"""Length probe for the thinking-mode arms (paper/npj/PROTOCOL.md, section 4): the zero-shot model writes one
thinking-mode rationale per dev item (the first --n, default 300) with a cap of --cap tokens (default 2,048). A
rationale counts as finished when it closed the think block and ended before the cap, with the marker or with EOS (an
answer written in another form, e.g. '\\boxed{B}', still finishes). Reported: the share finished within each multiple
of 256 tokens, the shares that closed the block and that wrote the marker, and the cap the protocol's rule picks: the
smallest multiple of 256 at which at least 90% finish (at most --cap). Only dev items are read.
  python -m openjev_rlcd.think_probe --task mmlupro --model Qwen/Qwen3-1.7B --output runs/npj/think_probe/mp_qwen3-1.7b.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import train
from .tasks import TASKS


def choose_cap(lengths, finished, cap, share=0.9, step=256):
    """(per-cap shares of rationales finished within the cap, the smallest cap reaching share, or None)"""
    n = len(lengths)
    shares = {c: sum(f and l <= c for l, f in zip(lengths, finished)) / n for c in range(step, cap + 1, step)}
    ok = [c for c, s in shares.items() if s >= share]
    return shares, (min(ok) if ok else None)


def main():
    train.hub_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(train.DEVICE).eval()
    rat = train.Rationales(tok, args.task, micro=args.gen_batch, thinking=True)
    dev = TASKS[args.task]["load"]()["dev"][:args.n]
    t0 = time.perf_counter()
    groups = rat.sample(model, dev, 1, args.cap)
    lengths = [len(g[0][1]) for g in groups]
    closed = [rat.think_end in g[0][1] for g in groups]
    # finished: the think block closed and the rationale ended before the cap -- with the marker, with EOS, or stopped
    # by the rule at a marker whose token carries the answer; only a rationale that used all --cap tokens was cut short
    def rule_stop(r):  # the rule stops at the first marker after </think> (also a refused one, e.g. 'Answer:B')
        return rat.think_end in r and train.MARKER in tok.decode(r[r.index(rat.think_end) + 1:])
    ended = [not g[0][2] or len(g[0][1]) < args.cap or (bool(g[0][1]) and g[0][1][-1] in rat.eos) or rule_stop(g[0][1])
             for g in groups]
    finished = [c and e for c, e in zip(closed, ended)]
    marker = [not g[0][2] for g in groups]
    shares, chosen = choose_cap(lengths, finished, args.cap)
    # "test" holds the result: the job queue counts a run file with a test block as finished
    out = dict(config=vars(args), test=dict(chosen_cap=chosen, share_finished_within=shares),
               minutes=(time.perf_counter() - t0) / 60, lengths=lengths, finished=finished,
               closed_think=closed, wrote_marker=marker, share_finished_within=shares, chosen_cap=chosen,
               gen_batch=args.gen_batch, rationales=len(groups),
               examples=[tok.decode(g[0][1])[-600:] for g in groups[:3]])
    train.write_json(args.output, out)
    print(json.dumps(dict(task=args.task, model=args.model, chosen_cap=chosen, minutes=out["minutes"],
                          share_finished=sum(finished) / len(finished), share_closed=sum(closed) / len(closed),
                          share_marker=sum(marker) / len(marker), shares=shares)), flush=True)


if __name__ == "__main__":
    main()

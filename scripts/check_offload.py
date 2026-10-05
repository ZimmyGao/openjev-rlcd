#!/usr/bin/env python3
"""How much does --offload-optimizer change the gradient on a real model?

The default recipe trains fp32 weights under bf16 autocast. The offloaded recipe keeps bf16 Linear
weights and fp32 everything else on the GPU (fp32 master weights on the CPU). On the same sampled
rationales and outcomes, compute the stage-2 RLCD gradient both ways and compare them (cosine, relative
L2 difference). The offloaded gradient is accumulated in fp32 by OffloadedAdamW's hooks, exactly as in
training. A second fp32 pass gives the floor set by GPU nondeterminism.

  python scripts/check_offload.py --model Qwen/Qwen3-0.6B --task gsm8kv
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openjev_rlcd.estimators import lambda_surrogate  # noqa: E402
from openjev_rlcd.offload import OffloadedAdamW, bf16_matmul_weights  # noqa: E402
from openjev_rlcd.tasks import TASKS  # noqa: E402
from openjev_rlcd.train import DEVICE, Rationales, draw  # noqa: E402


def loss_and_backward(model, rat, items, groups, ys, sf_coef=0.3):
    for it, g, y in zip(items, groups, ys):
        k = rat.k(it)
        u, logp = rat.forward(model, g, grad=True, k=k)
        loss = lambda_surrogate(u[:, :k], logp * sf_coef, y, 0.0)
        (loss / len(items)).backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--task", default="gsm8kv")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    rat = Rationales(tok, args.task)
    items = rng.sample(TASKS[args.task]["load"]()["train"], args.items)
    ys = [draw(rng, it) for it in items]

    bf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    bf = bf16_matmul_weights(bf).to(DEVICE)
    groups = rat.sample(bf, items, 4, args.max_new_tokens)
    bf.gradient_checkpointing_enable()
    bf.train()
    opt = OffloadedAdamW(bf, lr=2e-6)
    loss_and_backward(bf, rat, items, groups, ys)
    g_off = torch.cat([b.flatten() for b in opt.gpu_grad])

    fp = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
    fp.gradient_checkpointing_enable()
    fp.train()
    loss_and_backward(fp, rat, items, groups, ys)
    g_fp = torch.cat([p.grad.flatten() for p in fp.parameters()])
    fp.zero_grad(set_to_none=True)
    loss_and_backward(fp, rat, items, groups, ys)
    g_fp2 = torch.cat([p.grad.flatten() for p in fp.parameters()])

    def cmp(a, b):
        return dict(cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(), rel_l2_diff=((a - b).norm() / b.norm()).item())
    res = dict(model=args.model, task=args.task, items=args.items, offload_vs_fp32=cmp(g_off, g_fp),
               fp32_vs_fp32_rerun=cmp(g_fp2, g_fp), grad_norm_fp32=g_fp.norm().item(), grad_norm_offload=g_off.norm().item())
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline: SFT + temperature scaling (direct answer, no rationale).

The SAME instruct model as the RLCD runs (Qwen3-1.7B), the same prompt and the same
label tokens, but the assistant answers immediately ("Answer:" with an empty rationale) and is
fine-tuned with cross-entropy on the label token -- i.e. the RLCD model with an empty rationale,
trained by SFT instead of RL. Afterwards a temperature is fitted on dev with ONE annotator label per dev
item (fit_temperature, same protocol as every other arm).
  --labels single : ONE fixed label per train item (same draw as train.py --fixed-label)
  --labels fresh  : a fresh label per visit, the exact (item, label) stream the RL runs see
                    for the same seed and --items-per-step

  python -m openjev_rlcd.train_direct --task mmlupro --labels fresh --steps 400 --output results/x.json
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from .metrics import fit_temperature, readout_metrics
from .offload import OffloadedAdamW, bf16_matmul_weights
from .tasks import TASKS, prompt
from .train import (DEVICE, MARKER, Rationales, check_splits, draw, hub_offline, prepare_checkpoint_dir, save_weights,
                    write_json)


def logits_of(u, ks):
    """log u with the options beyond each row's count at -inf (so temperature scaling cannot revive them)."""
    z = u.clamp_min(1e-12).log()
    return z.masked_fill(torch.arange(u.shape[1])[None] >= torch.tensor(ks)[:, None], float("-inf"))


def main():
    hub_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="chaosnli")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--labels", choices=["single", "fresh"], required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--items-per-step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--dev-cap", type=int, default=0, help="0 = whole split")
    ap.add_argument("--test-cap", type=int, default=0, help="0 = whole split")
    ap.add_argument("--offload-optimizer", action="store_true",
                    help="bf16 Linear weights on the GPU, fp32 master weights + AdamW on the CPU (see offload.py)")
    ap.add_argument("--gpu-headroom-gb", type=float, default=8.0, help="see train.py")
    ap.add_argument("--gen-batch", type=int, default=64, help="unused (no generation); accepted for the scripts")
    ap.add_argument("--test-split", choices=["test", "fresh", "calib"], default="test", help="see train.py")
    ap.add_argument("--save-checkpoint", default="", help="the final weights (bf16), as train.py")
    ap.add_argument("--extra-splits", default="", help="comma-separated splits predicted as well (log['eval_<split>']), "
                                                        "with the same --test-cap; see train.py")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: predict dev + test (+ --extra-splits) with --checkpoint (or the base model), with "
                         "the forward pass of the training run's final predictions (e.g. the transfer tasks, which have "
                         "no training split)")
    ap.add_argument("--checkpoint", default="", help="weights for --eval-only")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    extras = [x for x in args.extra_splits.split(",") if x.strip()]
    if any(x not in ("test", "fresh", "calib") for x in extras) or len(set(extras)) != len(extras) \
            or args.test_split in extras:
        ap.error("--extra-splits: distinct names among test, fresh, calib, other than --test-split")
    if args.save_checkpoint:
        prepare_checkpoint_dir(args.save_checkpoint)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rat = Rationales(tok, args.task)
    answer = tok.encode(MARKER, add_special_tokens=False)
    splits = TASKS[args.task]["load"]()
    check_splits(args.task, splits, ([] if args.eval_only else [("train", None)]) + [("dev", args.dev_cap or None)]
                 + [(x, args.test_cap or None) for x in [args.test_split] + extras])
    for x in extras:
        splits[f"extra_{x}"] = splits[x][:args.test_cap or None]
    splits["dev"] = splits["dev"][:args.dev_cap or None]
    splits["test"] = splits[args.test_split][:args.test_cap or None]  # the run's test split, whichever --test-split
    fixed = {it["uid"]: draw(random.Random(f"{args.seed}-{it['uid']}"), it) for it in splits["train"]}
    cache = {}

    def group(items):
        for it in items:
            if it["uid"] not in cache:
                cache[it["uid"]] = tok(prompt(tok, it, args.task), add_special_tokens=False)["input_ids"]
        return [(cache[it["uid"]], answer, False) for it in items]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa",
                                                 device_map=DEVICE)
    if args.eval_only and args.checkpoint:  # the saved bf16 weights upcast into the fp32 model the training run
        # predicted with: under bf16 autocast every Linear layer computes exactly as at the end of training (its weights
        # are rounded to bf16 either way); only the embedding and norm weights keep the checkpoint's bf16 rounding
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    if args.offload_optimizer and not args.eval_only:
        model = bf16_matmul_weights(model)
    model.gradient_checkpointing_enable()
    if args.eval_only:
        opt = None
    elif args.offload_optimizer:
        opt = OffloadedAdamW(model, lr=args.lr, weight_decay=0.0, gpu_headroom_gb=args.gpu_headroom_gb)
        print(json.dumps(dict(optimizer_on_gpu_b=opt.gpu_params_b, optimizer_on_cpu_b=opt.cpu_params_b)), flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    @torch.no_grad()
    def predict(items, logs=False):
        """u [N, K] (zero beyond each item's option count) and the per-item counts; with logs also log u [N, K]."""
        model.eval()
        ks = [rat.k(it) for it in items]
        outs = [rat.forward(model, group(items[s:s + 16]), grad=False, k=ks[s:s + 16], return_logu=True)
                for s in range(0, len(items), 16)]
        u = torch.cat([o[0] for o in outs]).cpu()
        return (u, ks, torch.cat([o[2] for o in outs]).cpu().double()) if logs else (u, ks)

    def lists(u, ks):
        return [row[:k] for row, k in zip(u.tolist(), ks)]

    log, t0 = dict(config=vars(args), curve=[]), time.perf_counter()

    def record(step):
        m = readout_metrics(splits["dev"], lists(*predict(splits["dev"])))
        m.pop("per_item_jsd")
        log["curve"].append(dict(step=step, dev=m, minutes=(time.perf_counter() - t0) / 60))
        print(json.dumps(log["curve"][-1]), flush=True)

    if not args.eval_only:
        record(0)
    for step in range(1, 0 if args.eval_only else args.steps + 1):
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        ys = [draw(rng, it) for it in items]  # the RL runs' stream, same rng order
        if args.labels == "single":
            ys = [fixed[it["uid"]] for it in items]
        model.train()
        u, _ = rat.forward(model, group(items), grad=True, k=[rat.k(it) for it in items])
        loss = -u.clamp_min(1e-12).log()[torch.arange(len(items), device=DEVICE), torch.tensor(ys, device=DEVICE)].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.offload_optimizer:
            opt.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0:
            record(step)

    model.zero_grad(set_to_none=True)
    del opt  # with --offload-optimizer its state lives in host memory: free it before the checkpoint is written
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if args.save_checkpoint and not args.eval_only:
        save_weights(model, args.save_checkpoint)
    (test_u, test_k, test_lu), (dev_u, dev_k, dev_lu) = predict(splits["test"], True), predict(splits["dev"], True)
    t_fit = fit_temperature(list(logits_of(dev_u, dev_k)), splits["dev"], args.seed)
    log["temperature"] = t_fit
    log["test"] = readout_metrics(splits["test"], lists(test_u, test_k))
    log["test_ts"] = readout_metrics(splits["test"], lists((logits_of(test_u, test_k) / t_fit).softmax(-1), test_k))
    log["test_dists"], log["dev_dists"] = lists(test_u, test_k), lists(dev_u, dev_k)
    log["test_logps"], log["dev_logps"] = lists(test_lu, test_k), lists(dev_lu, dev_k)
    log["test_uids"], log["dev_uids"] = [it["uid"] for it in splits["test"]], [it["uid"] for it in splits["dev"]]
    for x in extras:  # the same model and the same dev readouts (one temperature) as the test split
        u, k, lu = predict(splits[f"extra_{x}"], True)
        log[f"eval_{x}"] = dict(dists=lists(u, k), logps=lists(lu, k), uids=[it["uid"] for it in splits[f"extra_{x}"]])
    print(json.dumps({k: {kk: vv for kk, vv in log[k].items() if kk != "per_item_jsd"} for k in ["test", "test_ts"]}
                     | dict(temperature=t_fit)), flush=True)
    write_json(args.output, log)


if __name__ == "__main__":
    main()

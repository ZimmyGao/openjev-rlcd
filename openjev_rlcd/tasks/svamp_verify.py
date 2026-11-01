"""SVAMP-Verify: the GSM8K-Verify construction on SVAMP (Patel et al. 2021), an out-of-domain transfer test for the
models trained on GSM8K-Verify (npj protocol, Q3b). Each of the 1,000 SVAMP problems (train and test, none used in
training) gets ONE proposed final answer from the same solver as GSM8K-Verify (Qwen3-0.6B, a sampled brief chain of
thought, seed 0, the same prompt and parsing); the question is whether the proposal is correct.

  python -m openjev_rlcd.tasks.svamp_verify --build --output <summary.json>   builds data/svamp_verify.json (one GPU)
The build runs once, offline (the cached solver and dataset revisions, as every job of the protocol), and refuses to
replace an existing data/svamp_verify.json (--force does); if the data file exists but the summary is missing (a build
interrupted between the two writes), --build --output writes the summary of the existing file. A uid carries the
proposed answer, so a run evaluated on one build cannot be scored against another (npj_report.check_uids).
Splits (seed 0): dev = 200 problems (a target-domain temperature, secondary), test = the rest.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch

from . import gsm8k_verify

DATA = Path(__file__).resolve().parents[2] / "data" / "svamp_verify.json"
LABELS = gsm8k_verify.LABELS
N_DEV = 200


@torch.no_grad()
def build(model_name="Qwen/Qwen3-0.6B", seed=0, batch=64, max_new_tokens=320, force=False):
    if DATA.exists() and not force:
        raise SystemExit(f"{DATA} exists: SVAMP-Verify is built once (--force replaces it)")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ds = load_dataset("ChilleD/SVAMP")
    rows = [(split, i, x) for split in ("train", "test") for i, x in enumerate(ds[split])]
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{x['question_concat']}\n\nSolve the problem. Think step by step briefly, "
                                         "then finish with exactly one line of the form 'Answer: <number>'."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, _, x in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        for (split, i, x), row in zip(chunk, gen):
            text = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            cand = gsm8k_verify._number(text.split("Answer:")[-1]) if "Answer:" in text else gsm8k_verify._number(text)
            gold = float(x["Answer"])
            gold = int(gold) if gold == int(gold) else round(gold, 4)
            if cand is None:
                continue
            proposed = gsm8k_verify._fmt(cand)
            out.append(dict(uid=f"svamp-{split}-{i}-p{proposed}", question=x["question_concat"],
                            gold=gsm8k_verify._fmt(gold), proposed=proposed,
                            correct=bool(abs(float(cand) - float(gold)) < 1e-6)))
    tmp = DATA.with_suffix(".tmp")  # a reader never sees a partial file
    tmp.write_text(json.dumps(dict(solver=model_name, seed=seed, n_problems=len(rows), items=out)))
    os.replace(tmp, DATA)
    return out


def summary():
    """the summary of the built data file (the build job's 'test' block)"""
    rows = json.loads(DATA.read_text())["items"]
    s = load_items()
    return dict(n=len(rows), share_correct=sum(r["correct"] for r in rows) / len(rows), n_dev=len(s["dev"]),
                n_test=len(s["test"]), share_correct_test=sum(it["q"][0] for it in s["test"]) / len(s["test"]),
                sha256=hashlib.sha256(DATA.read_bytes()).hexdigest())


def built(summary_path):
    """the build job finished (its summary has a test block) and the data file is the one it summarized"""
    p = Path(summary_path)
    if not (p.exists() and DATA.exists() and '"test": {' in p.read_text()):
        return False
    return json.loads(p.read_text())["test"].get("sha256") == hashlib.sha256(DATA.read_bytes()).hexdigest()


def load_items(split_seed=0, n_dev=N_DEV):
    rows = json.loads(DATA.read_text())["items"]
    items = [dict(uid=r["uid"], question=r["question"], proposed=r["proposed"],
                  q=[1.0, 0.0] if r["correct"] else [0.0, 1.0], counts=[1.0, 0.0] if r["correct"] else [0.0, 1.0])
             for r in rows]
    random.Random(split_seed).shuffle(items)
    return dict(train=[], dev=items[:n_dev], test=items[n_dev:])


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the cached solver and dataset revisions (as train.hub_offline)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace an existing data/svamp_verify.json")
    ap.add_argument("--output", default="")
    a = ap.parse_args(argv)
    if a.build:
        if DATA.exists() and not a.force and a.output and not built(a.output):
            # an interrupted build: DATA is complete (written atomically), only the summary is missing
            print(f"{DATA} exists without its summary: summarizing it (no rebuild)", flush=True)
        else:
            build(force=a.force)
        res = summary()
        print(json.dumps(res), flush=True)
        if a.output:  # "test" holds the summary: the job queue counts a run file with a test block as finished
            tmp = Path(a.output).with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(config=vars(a), test=res), indent=2))
            os.replace(tmp, a.output)


if __name__ == "__main__":
    main()

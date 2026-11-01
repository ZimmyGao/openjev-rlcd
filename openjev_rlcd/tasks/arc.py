"""ARC-Challenge (Clark et al. 2018) as an out-of-domain multiple-choice transfer test for the models trained on
MMLU-Pro (npj protocol, Q3b): grade-school science questions with 3-5 options, never used in training. Same prompt,
option letters and readout as MMLU-Pro; options are relabelled A, B, C, ... in their dataset order (some items label
them 1-4). Splits: dev = the 299 validation questions (a target-domain temperature, reported as secondary), test = the
1,172 test questions; no training split.
"""
from . import mmlu_pro

LETTERS = mmlu_pro.LETTERS


def load_items():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def conv(x):
        labels, texts = list(x["choices"]["label"]), list(x["choices"]["text"])
        y = labels.index(x["answerKey"])
        q = [1.0 if i == y else 0.0 for i in range(len(texts))]
        return dict(uid=x["id"], question=x["question"], options=texts, category="arc", q=q, counts=q)
    return dict(train=[], dev=[conv(x) for x in ds["validation"]], test=[conv(x) for x in ds["test"]])

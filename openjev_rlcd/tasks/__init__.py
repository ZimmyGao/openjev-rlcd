"""Task registry. Every item has uid, q (outcome distribution) and counts (weights for drawing Y).
  chaosnli     : aleatoric, 3 labels, q = 100-annotator distribution
  mmlupro      : epistemic, up to 10 options (options beyond an item's count are masked)
  mmlupro_stem : MMLU-Pro restricted to computation-heavy subjects
  gsm8kv       : epistemic yes/no -- is a weaker solver's GSM8K answer correct?
"""
import re

from . import chaosnli, gsm8k_verify, mmlu_pro

# Some chat templates insert today's date (SmolLM3: "Today Date: ..."); pin it so that a run's prompts do not
# change from one day to the next.
FIXED_DATE = "05 October 2026"

TASKS = {
    "chaosnli": dict(labels=chaosnli.LABELS, message=chaosnli.user_message, load=chaosnli.load_items,
                     k=lambda it: len(chaosnli.LABELS)),
    "mmlupro": dict(labels=list(mmlu_pro.LETTERS), message=mmlu_pro.user_message, load=mmlu_pro.load_items,
                    k=lambda it: len(it["options"])),
    "mmlupro_stem": dict(labels=list(mmlu_pro.LETTERS), message=mmlu_pro.user_message,
                         load=lambda: mmlu_pro.load_items(categories=mmlu_pro.STEM, n_train=3000, n_dev=300),
                         k=lambda it: len(it["options"])),
    "gsm8kv": dict(labels=gsm8k_verify.LABELS, message=gsm8k_verify.user_message, load=gsm8k_verify.load_items,
                   k=lambda it: len(gsm8k_verify.LABELS)),
}


CONFIDENCE_REQUEST = ("After that line, write one more line of the form 'Confidence: C', where C is your probability "
                      "(a number from 0 to 1) that your answer is correct.")


def prompt(tok, item, task, confidence=False):
    """Chat prompt (Qwen3 non-thinking mode); the rationale is generated after it. confidence: also ask for a
    'Confidence: C' line after the answer (the RLCR baseline, train.py --objective rlcr)."""
    message = TASKS[task]["message"](item) + (" " + CONFIDENCE_REQUEST if confidence else "")
    text = tok.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return re.sub(r"(Today Date: )[^\n]*", lambda m: m.group(1) + FIXED_DATE, text)

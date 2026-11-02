# openjev-rlcd

Code, run scripts and data for

- **Beyond outcome rewards: Post-rationale distribution optimization for selective prediction in reasoning models**
  (manuscript, 2026). Its code is tagged `npj-experiments`: the study's final code install (2026-10-26), which also
  ran the analysis. The two earlier installs that ran the first runs differ from it only by later additions, such as
  the transfer evaluation and CREDO+TS. The paper's data deposit records the code version of every run.
- **OpenJev-RLCD: A Working RLCD Implementation** ([arXiv:2609.38850](https://arxiv.org/abs/2609.38850)). Its code
  is tagged `arxiv-v1`; see [OpenJev-RLCD (arXiv v1)](#openjev-rlcd-arxiv-v1) below.

## Post-rationale distribution optimization (PRDO)

A reasoning model answers a closed-form question by sampling a rationale. At the marker `Answer:` it then gives a
next-token distribution `u` over the answer options, the readout. A deployed single query takes the readout's argmax
as its decision and its maximum as the confidence that decides whether to answer or defer. PRDO trains this readout
directly:

- For each training question, it samples M = 4 rationales from the current model and reads `u` after each one.
- It minimizes the negative of a strictly proper score of each readout (Brier `2u_y - ||u||^2` or log `log u_y`),
  averaged over the rationales, plus a token-level KL anchor (λ = 0.01) to the initial model.
- Gradients flow through the readouts and the anchor only. The rationales get no reward.

Outcome-reward RL (GRPO) instead samples an answer from `u` and reinforces the rationale and the answer when it is
correct. Its expected reward is linear in `u`, so it is maximized by putting all probability on one answer. Scoring the
average of the readouts (the mixture objective) would also reward disagreement between rationales, which a single
query never shows.

In the code, PRDO is stage 1 of the training loop:
`python -m openjev_rlcd.train --diversity-weight 0 --no-score-function --kl-beta 0.04 --kl-tokens 0.01`.
- `--diversity-weight 1`, the default, is the mixture objective.
- `--kl-beta` only loads the initial model for the anchor; without the score-function term it has no other effect.
- Run files are named `*_warm400_*` (Brier score) and `*_ce_*` (log score).
- Full commands are [below](#reproduce-the-npj-experiments).

## Main results

Setup:
- MMLU-Pro on four models (Qwen3-0.6B, Qwen3-1.7B, SmolLM3-3B, Phi-4-mini-instruct);
- GSM8K-Verify on the two Qwen3 models;
- three seeds per arm, every arm temperature-scaled on development questions;
- paired comparisons on held-out test questions (1,000 MMLU-Pro, 1,260 GSM8K-Verify) with budget-matched GRPO
  followed by temperature scaling (GRPO+TS).

PRDO with the Brier score, against GRPO+TS:
- **Brier score and AURC (MMLU-Pro).** On every model the Brier score is 0.068–0.080 lower, and the area under the
  risk–coverage curve (AURC) is 0.106–0.161 lower (Holm-adjusted P < 0.05).
- **Accuracy (MMLU-Pro).** Higher by 2.6–4.1 points.
- **Coverage.** At a realized error of at most 20%, 12.3–38.2 percentage points more MMLU-Pro questions are answered
  (oracle coverage: thresholds set with the test labels).
- **Transfer.** AURC is lower in all four transfer cells:
  - Qwen3 models trained on MMLU-Pro, evaluated on ARC-Challenge;
  - Qwen3 models trained on GSM8K-Verify, evaluated on SVAMP-Verify;
  - both with the source task's temperature.
  On SVAMP-Verify, Qwen3-0.6B loses 3.4 points of accuracy.
- **Answer verification (GSM8K-Verify).** Qwen3-0.6B lowers AURC but loses 2.7 points of accuracy. Qwen3-1.7B shows no
  difference.

## Layout

```
openjev_rlcd/
  train.py           one training loop: PRDO (stage 1), the optional second stage that reinforces the rationales,
                     GRPO, RLOO, RFT/STaR, RLCR, CREDO and the controls  (python -m openjev_rlcd.train --help)
  train_direct.py    SFT baseline: direct answer, cross-entropy
  estimators.py      per-rationale and mixture estimators, lambda-interpolation
  metrics.py         temperature scaling, Brier score, ECE, AURC, coverage at a risk level
  evaluation.py      per-item scoring after temperature scaling, paired bootstrap and sign-flip tests
  offload.py         fp32 master weights and AdamW state in host memory (3-4B models on one 48 GB GPU)
  npj_report.py      scoring of the npj runs; selection of the baselines' settings on the validation split
  npj_decide.py      the decision rules of the npj analysis plan: Holm families, seed agreement, seed-and-item
                     hierarchical bootstrap
  think_probe.py, credo_probe.py   length probes on development questions (thinking mode, CREDO's analysis)
  final_tables.py    tables of the arXiv version (scripts/make_tables.sh)
  probe.py, backbone_report.py     zero-shot screening of a backbone; report of the multi-backbone and anchor pilots
  tasks/             mmlu_pro, gsm8k_verify, chaosnli, arc (transfer), svamp_verify (transfer)
scripts/
  plan_npj.py        job lines of every npj run, each checked against the saved run it mirrors
  plan_posthoc.py    job lines of the exploratory follow-up experiments
  jobqueue.py, queue_worker.sh, queue_watchdog.sh, watch_queue.sh   job queue for the local GPUs
  smoke_npj.sh, smoke_test.sh      a few steps of every training mode, as a GPU check
  two_stage.sh, baselines.sh, ablations.sh, make_tables.sh   arXiv runs and tables
  backbone.sh, plan_anchored.py    multi-backbone and anchor pilot runs (plan_npj.py imports plan_anchored's flags)
tests/               unit tests, including exact-enumeration checks of every estimator
data/                gsm8k_verify.json and svamp_verify.json (constructed sets), npj_splits.json (split manifests)
results/             per-item result files behind the tables of the arXiv version
```

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # the pins in requirements.txt need Python 3.12
pip install -r requirements.txt        # versions the experiments ran with (CUDA 12.4)
# or: pip install -e ".[analysis,test]"  (editable only: the code reads data/ and results/ from the checkout)
python -m pytest tests                 # CPU; tests that need a cached model or dataset are skipped without it
```

Each run uses one 48 GB GPU; the papers used RTX 6000 Ada. A 400-step run takes about 0.7–2.3 h on Qwen3 and 3–4.5 h
on SmolLM3-3B or Phi-4-mini, plus the test evaluation.

**Offline by default.** These entry points run offline and use the cached model and dataset revisions:
- `train` and `train_direct`;
- `npj_report` and `npj_decide`;
- the probes and the SVAMP-Verify build.

`make_tables.sh` (`final_tables`), `backbone_report` and `probe` do not run offline; give them
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Fill the cache once by running with `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`.
The code pins no revision, so this fetches the Hub's current ones. The npj runs used these:

| Model or dataset | Revision |
|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| HuggingFaceTB/SmolLM3-3B | `a07cc9a04f16550a088caea529712d1d335b0ac1` |
| microsoft/Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| TIGER-Lab/MMLU-Pro | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| allenai/ai2_arc | `210d026faf9955653af8916fad021475a3f00453` |
| ChilleD/SVAMP | `5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef` |
| tasksource/chaos-mnli-ambiguity | `64e9c55ec72f36df266479ac200069cc3613d2ff` |

## Reproduce the npj experiments

**Splits.** `data/npj_splits.json` fixes the items of every split by hash:
- `--test-split fresh` is the final test set;
- `--extra-splits calib` also scores the calibration split used for the risk-controlled thresholds;
- the default `--test-split test` is the earlier test split, which serves as the validation split in this study.

Each run writes `--output` as a JSON file with:
- every argument (`config`);
- the per-item log-probabilities of the single-query and mixture readouts on the development and test items;
- its metrics.

Examples (Qwen3-1.7B, MMLU-Pro, seed 17):

```bash
C="--task mmlupro --model Qwen/Qwen3-1.7B --m 4 --m-eval 4 --max-new-tokens 256 --steps 400 --dev-cap 300 \
   --test-cap 1000 --diversity-weight 0 --seed 17 --test-split fresh"
# PRDO, Brier score
python -m openjev_rlcd.train $C --extra-splits calib --kl-beta 0.04 --kl-tokens 0.01 --no-score-function \
  --eval-every 400 --save-checkpoint checkpoints/mp_warm400_seed17.pt --output runs/npj/qwen3-1.7b/mp_warm400_seed17.json
# GRPO+TS
python -m openjev_rlcd.train $C --extra-splits calib --objective grpo --kl-beta 0.04 --kl-tokens 0.01 --eval-every 200 \
  --save-checkpoint checkpoints/mp_grpo_seed17.pt --output runs/npj/qwen3-1.7b/mp_grpo_seed17.json
# SFT+TS
python -m openjev_rlcd.train_direct --task mmlupro --model Qwen/Qwen3-1.7B --labels fresh --lr 2e-6 --seed 17 \
  --steps 400 --dev-cap 300 --test-cap 1000 --test-split fresh --extra-splits calib \
  --save-checkpoint checkpoints/mp_direct_lr2e-6_seed17.pt --output runs/npj/qwen3-1.7b/mp_direct_lr2e-6_seed17.json
```

**Log score and controls.** Use the PRDO line without `--extra-splits calib`. Only the zero-shot, SFT+TS, PRDO,
two-stage and GRPO+TS runs scored the calibration split. Rename the run in `--save-checkpoint` and `--output`
(`warm400` → `ce`, `lin`, `refrat` or `mix`) and add one of:
- `--readout-score log`: the log score (`ce`);
- `--readout-score linear`: the improper linear score (`lin`);
- `--rationale-source ref`: rationales sampled from the frozen initial model (`refrat`);
- `--diversity-weight 1`: the mixture objective (`mix`).

**3-4B models.** SmolLM3-3B and Phi-4-mini add `--offload-optimizer --gpu-headroom-gb 24 --gen-batch 16`. The fp32
master weights and AdamW state then live in host memory: 16 bytes per parameter, about 50–60 GB per run (the papers'
machine had 125 GB).

**Other arms.** `python -m openjev_rlcd.train --help` lists every option. These include the second stage
(`--init-checkpoint`, `--sf-coef`, `--sf-reward`), RLCR and CREDO (`--objective rlcr|credo`), thinking mode
(`--thinking`) and evaluation of saved weights (`--eval-only --checkpoint`).

**Every run of the study.** `scripts/plan_npj.py` writes the job line of every run, for example
`python3 scripts/plan_npj.py tier1`; the plans are listed at the top of the file. Final weights go to
`$OPENJEV_CKPT_DIR` (default `checkpoints/`).
- **Reference runs.** Before writing a line, the planner checks it against the saved configuration of the earlier run
  it mirrors. Point `OPENJEV_REF_ROOT` at a folder holding those run files: `runs/anchored01/`, `runs/backbones/`,
  `runs/extra_baselines/` and `results/`.
- **Only `results/` is in this repository.** The other folders come with the paper's data deposit. Without them, the
  plans `tune`, `tier1` and `tier2` print no job lines and exit 1; `probe`, `credo_probe`, `transfer` and `chaos` need
  only `results/`.
- **Plans that need earlier outputs:**
  - `grpo` and `rlcr` need the selections written by `python -m openjev_rlcd.npj_report select-grpo` and
    `select-rlcr`, from the tuning runs;
  - `credo` and `thinking` need the `credo_probe` and `probe` runs;
  - `transfer_svamp` needs the SVAMP-Verify build of plan `transfer`.
- **Follow-up experiments.** `scripts/plan_posthoc.py` writes the job lines of the exploratory follow-up experiments.
  Push them only after the `tier1` and `chaos` runs are complete, because they start from those checkpoints.

The job lines run on a small queue, one worker per GPU:

```bash
python3 scripts/plan_npj.py chaos > lines && python3 scripts/jobqueue.py push < lines
GPU=0 PY=.venv/bin/python bash scripts/queue_worker.sh      # one worker per GPU; GPU=1 ... for the second
python3 scripts/jobqueue.py show                             # queued and running jobs
```

**Analysis.** `python -m openjev_rlcd.npj_decide --root runs/npj --out decisions.json` recomputes the analysis plan's
decision rules from the run files:
- Holm-adjusted tests within each family;
- the agreement of the seeds;
- the seed-and-item bootstrap.

The thinking-mode comparisons are computed by the paper's own analysis scripts, not by this module.

### Run names and labels

Run files are named `<task>_<arm>_seed<s>.json`.

Task prefixes:

| Prefix | Task |
|---|---|
| `mp` | MMLU-Pro |
| `gv` | GSM8K-Verify |
| `cn` | ChaosNLI (`s1_lam0`: per-rationale objective, `s1_lam1`: mixture objective) |
| `arc` | ARC-Challenge |
| `svampv` | SVAMP-Verify |

Arms:

| Arm in the file name | In the paper |
|---|---|
| `warm400` | PRDO (Brier score) |
| `ce` | PRDO (log score) |
| `lin` | linear-score control |
| `refrat` | offline rationales (from the frozen initial model) |
| `mix` | mixture objective |
| `fork_rl` | two-stage: a second stage that rewards each rationale with the Brier score of its readout |
| `fork_correct` | second stage with a sampled-answer reward |
| `fork_grpo` | GRPO continuation of PRDO |
| `grpo` | GRPO+TS (`grpo_b*`: settings tried on the validation split) |
| `direct_lr2e-6`, `direct` | SFT+TS |
| `base_eval` | zero-shot model |
| `rlcr`, `credo` | RLCR+TS, CREDO+TS |
| `think_warm400`, `think_grpo` | thinking mode, trained in it |
| `thinkeval_*`, `think_base_eval` | thinking mode, evaluated in it only |
| `s1cont`, `noanchor`, `noanchor_refrat`, `cn_s2_*` | exploratory follow-up experiments |

Code comments cite the study's analysis plan (`paper/npj/PROTOCOL.md`) and the designs of its follow-up experiments
(`paper/npj/POSTHOC_20261029.md`) by their labels. Both will be published with the paper's data deposit; they are not
in this repository.

| Label | Meaning |
|---|---|
| C | the central comparison with GRPO+TS |
| Q1a, Q1b, Q1c | Brier vs log score, own vs offline rationales, Brier vs linear score |
| Q2 | per-rationale vs mixture objective (P2a–P2c in `npj_decide` are its three predictions) |
| Q3a | coverage at a fixed error |
| Q3b | transfer |
| A1–A5 | the plan's extensions: transfer tasks, ChaosNLI, thinking mode, CREDO+TS, the decision code |
| P1–P3, X1–X4 | the exploratory follow-up entries and experiments |

"Registered" in a comment means pre-specified.

## OpenJev-RLCD (arXiv v1)

The arXiv paper trains Qwen3-1.7B with a two-stage recipe, **calibrate, then reinforce**:

1. **Stage 1 (calibrate the decision).** This is PRDO above, without the token anchor.
2. **Stage 2 (reinforce the reasoning).** Starting from the stage-1 model, it applies REINFORCE to the rationales, using
   the same proper score of the model's own readout as the reward. It uses a score-function weight of 0.3,
   leave-one-out baselines and a KL anchor (β = 0.04) at the stage-1 model.

Results with Qwen3-1.7B (3 seeds, single query, temperature-scaled on dev):

| | GSM8K-Verify acc / Brier / AURC / Cov@5% | MMLU-Pro acc / Brier / AURC / Cov@20% |
|---|---|---|
| SFT + TS | 70.4 / 0.382 / 0.174 / 5% | 42.3 / 0.711 / 0.386 / 16% |
| RFT/STaR + KL + TS | 91.4 / 0.160 / 0.073 / 11% | 40.9 / 0.761 / 0.494 / 0% |
| GRPO + KL + TS | 91.8 / 0.150 / 0.067 / 10% | 42.4 / 0.753 / 0.481 / 0% |
| **RLCD two-stage** | **92.2 / 0.135 / 0.034 / 81%** | **47.7 / 0.658 / 0.304 / 28%** |

**Changes since arXiv v1 (2026-10-16): metrics.**
- **Ties.** AURC and Cov@ε used to rank items of equal confidence by their position in the test set. Items are now
  ranked by temperature-scaled confidence, with ties broken by the model's own log-probability margin. Items that
  still tie share their group's error rate.
- **Effect.** This mainly moves the GSM8K-Verify numbers of saturated baselines (GRPO AURC 0.061 → 0.067, Cov@5% 19%
  → 10%). The two-stage Cov@5% moves by less than one point.
- **Collapsed runs.** Runs whose rationales collapse are now excluded from every GSM8K-Verify and MMLU-Pro table and
  test; on MMLU-Pro this removes RFT/STaR without KL, all three seeds. Collapses that are themselves results (Figure 1,
  the ChaosNLI table) are still reported.

Rebuild the GSM8K-Verify and MMLU-Pro tables (every arm after temperature scaling, with paired tests) from the released
per-item results, no GPU needed:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash scripts/make_tables.sh
```

The ChaosNLI and score-function-weight tables and the figures were built by the paper sources, which are not in this
repository; their run files are in `results/`.

Rerun the experiments; each run writes `runs/<name>.json` and finished runs are skipped:

```bash
bash scripts/smoke_test.sh                                     # every training mode, 2 steps (~5 min)
CUDA_VISIBLE_DEVICES=0 bash scripts/two_stage.sh gsm8kv        # stage 1 + four forked stage-2 arms, 3 seeds
CUDA_VISIBLE_DEVICES=1 bash scripts/two_stage.sh mmlupro
CUDA_VISIBLE_DEVICES=0 bash scripts/baselines.sh gsm8kv        # base, SFT, GRPO, RFT(+KL), log-score stage 1
CUDA_VISIBLE_DEVICES=1 bash scripts/baselines.sh mmlupro
CUDA_VISIBLE_DEVICES=0 bash scripts/ablations.sh               # Figure 1, Table 3 (ChaosNLI), Table 4
RESULTS=runs bash scripts/make_tables.sh
```

Sampling on GPUs is not bitwise deterministic, so reruns match the papers up to seed-level noise.

Result files hold:
- `config`: all arguments;
- `curve`: dev metrics during training;
- `dev_dists` and `test.dists`: per-item answer distributions of the `mixture` and `single` readouts;
- `test`: raw metrics and rationale statistics.

File names:
- `gv_` is GSM8K-Verify, `mp_` is MMLU-Pro, and `teacher*` / `direct_fresh*` are ChaosNLI;
- the arms are named as in `openjev_rlcd/final_tables.py`: `grpokl` is GRPO+KL, `rft` / `rftkl` is RFT/STaR without /
  with KL, plus `fork_ro`, `fork_anchor` and `direct800`.

Every reported metric is recomputed from the per-item distributions after temperature scaling on dev
(`openjev_rlcd/evaluation.py`).

## Citation

```bibtex
@misc{gao2026openjevrlcd,
  title         = {{OpenJev-RLCD}: A Working {RLCD} Implementation},
  author        = {Gao, Zhimin and Wang, Pichao},
  year          = {2026},
  eprint        = {2609.38850},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.38850},
}
```

A citation for the PRDO manuscript will be added when it is published.

## License

Code: [MIT](LICENSE).

Datasets keep their own licenses:
- GSM8K: MIT;
- MMLU-Pro: MIT;
- ARC: CC BY-SA 4.0;
- SVAMP: MIT;
- ChaosNLI: CC BY-NC 4.0.

`data/gsm8k_verify.json` and `data/svamp_verify.json` are derived from GSM8K and SVAMP, with proposed answers written
by Qwen3-0.6B. The upstream license texts are in [`data/THIRD_PARTY_LICENSES.md`](data/THIRD_PARTY_LICENSES.md).

Jev is a product of TypeSafe AI. This repository is an independent implementation and makes no claim about Jev's
internal training algorithm.

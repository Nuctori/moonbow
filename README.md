# Can a Model Judge Whether a Task Is Done? A Negative-Result Study on Obligation-Closure Judgment

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## TL;DR

We spent an extended effort trying to build a classifier that, given a user obligation
and an agent's utterance, judges whether the obligation is *closed* (done). We failed
to beat a trivial baseline — and we argue this is not a modeling failure but a
**task-definition problem**: under a sentence-level tri-value definition, the task is
not reliably labelable even by large general-purpose language models (≈70% agreement
between independent model instances), so no classifier can exceed that ceiling.

This repository contains the **methodology, reproducible evaluation harness, and
label-only gold standards** — but **no real session text**, to protect the privacy of
the original work. All results are reproducible from the provided scripts + your own
session data.

## What we did (the exclusion chain)

We tested five families of approaches, each with a causal experiment, all excluded:

| # | Approach | Outcome | Evidence |
|---|----------|---------|-----------|
| 1 | Swap backbone (linear-attention / pre-trained linear base) | Below envelope | architecture sweep + Route 6/6b |
| 2 | More data (full untruncated obligations) | No line-break | data-volume experiments |
| 3 | Input reform (truncation constant / candidate filtering) | All negative | reconstruction tests |
| 4 | Anchored training (adversarial "must read obligation") | Recall collapse | v14.0/14.1, incl. causal ablation |
| 5 | Two-stage training (match-then-judge) | Recall collapses harder | §163/§164 causal control |

We then collected a **new, uncontaminated gold standard** (897 items, dual-blind
annotation + third-party arbitration, 361 positives) and ran a **DeLong paired-AUROC**
test at adequate statistical power (n_pos = 361 ≥ 280):

| Judge | AUROC |
|-------|-------|
| **v12.1** (287M fine-tuned mDeBERTa) | **0.8294** |
| zero-semantic baseline (count completion words) | 0.7525 |
| **anchor model** (reads obligation) | 0.7621 |
| hand-crafted interpretable features | 0.6573 |

- The anchor model is **statistically indistinguishable from the zero-semantic baseline**
  (DeLong p = 0.82) and **not significantly better** than v12.1 (p = 0.073, boundary).
- v12.1 beats hand-crafted features (+0.17) but the gap to the zero-semantic baseline
  is small (+0.077).

## The decisive finding: it is a task problem, not a model problem

We asked a large general-purpose LLM (zero-shot, no fine-tuning) to perform the *same
annotation task* on a 200-item blind set, using the identical protocol given to human
annotators. Its three-way agreement with the gold was **73.0%** — essentially the same
as the **70.2%** agreement between two independent model annotators on the same task.

We then put both judges on the **same ruler** (same 200 items, same binary definition):

| Judge | acc | P | R | F1 |
|-------|-----|---|---|----|
| general LLM (zero-shot) | 0.765 | 0.894 | 0.500 | 0.641 |
| v12.1 (287M fine-tuned) | 0.740 | 0.648 | 0.833 | 0.729 |
| (matched positive count) | 0.745 | 0.851 | 0.476 | 0.611 |

- Accuracy gap = **2.5pp**; at matched operating point the P/R nearly coincide.
- Inter-judge binary agreement = **64.5%** (a third of the time they disagree).

**Conclusion:** scaling the model 5–10× does not help. The bottleneck is not model
capacity; the task, under its current sentence-level definition, does not support
reliable labeling.

## Why (mechanism)

1. The model **does not read the obligation** (ablating it flips only 9.6% of judgments).
2. Forcing it to read the obligation **collapses recall** (causal ablation shows this).
3. The model relies on **completion-word density**, but that signal is *inverted*:
   sentences containing completion words are *less* likely positive (P=0.396 vs 0.818).
4. When evidence is richest, the model **loses discrimination** (R=1.0 / P=0.5).
5. The gold standard has a **structural contamination history** (negatives lacked
   evidence; a campaign gold had 106 machine-payload items labeled positive; its
   positive labels only survived re-labeling 23.8% of the time).
6. The aggregation layer is **already cross-window** (`unit_fired = any(window)`),
   so there is no aggregation to fix.

Net: closure-judgment failure is a **structural mismatch** between the task definition
(unit/binary closure) and the data's nature (closure is a continuum, evidence is
scattered, and completion phrasing is inversely correlated with actual completion).

## Reproducibility

```
pipeline/                 # evaluation + data-collection scripts
protocol/                # annotation protocol (v4→v6)
data/                     # label-only gold (no text) + synthetic demo
results/                  # DeLong / stratified / size-vs-task outputs
```

All scripts load a gold standard of the form `{"<pid>": {"label": ...}}` and an
item file with `obligation` / `sentence` text. We provide
`data/synthetic_demo.json` so the pipeline runs end-to-end without any private data.
To reproduce on your own sessions, replace the data with yours and keep the label ids.

Key scripts:
- `research_v173_delong_final.py` — DeLong paired AUROC (the primary test)
- `research_v175_stratified_eval.py` — stratified by obligation length × enumeration
- `research_v178_size_vs_task.py` — same-ruler comparison (287M vs general LLM)

Model weights are **not** included (distributed via the application side); the scripts
expect a local model path.

## Limitations (read before citing)

- **Gold standards are model-produced.** Every label was produced by model annotators
  (subagents); **no human annotation was performed**. The "73% vs 70%" comparison is
  between model instances, **not** model-vs-human. A human-annotation sample (≥100
  items) is the single missing number and is required before any strong claim about
  human labelability.
- **Single seed / single batch** for most experiments; effect sizes are reported but
  CIs are partial.
- **OR-aggregation assumption tested at 61%**: the historical `unit_fired = any(...)`
  aggregation was directly tested on 72 mixed-label obligations and matched the
  obligation-level truth only 61% of the time. All historical metrics sit on top of
  this aggregation.
- Results are reproducible from the provided scripts + **your own** session data.

## Citation

If you use this work, please cite the methodology and the label-only gold:

```bibtex
@misc{obligation-closure-study-2026,
  title  = {Can a Model Judge Whether a Task Is Done? A Negative-Result
            Study on Obligation-Closure Judgment},
  author = {The Authors}, year = {2026},
  howpublished = {GitHub repository}
}
```

Structured metadata is in `CITATION.cff`; the same entry plus the GLiNER2
reference is in `CITATION.bib`.

**References still to verify before submission:**
- **GLiNER2 / gliner2** — the span-extraction library used for v12.1; the BibTeX
  above is a placeholder author/year and must be confirmed against the actual release.
- **DeLong paired-AUROC** — cite the original DeLong (1988, Biometrika) method paper,
  not a software package.
- The model backbone is a **mDeBERTa** fine-tune; cite the DeBERTa paper if the
  architecture is discussed.
- The general LLM comparison used a Qwen2.5-Instruct family model; cite the Qwen2.5
  technical report if that comparison is included in the final paper.

All other named works in the project logs (FEVER, various author-year mentions) are
**not** cited here because they were not used as building blocks of this study.

## License

Apache-2.0. See [LICENSE](LICENSE).

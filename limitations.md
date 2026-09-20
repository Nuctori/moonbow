# Limitations (honest boundary)

This document states what this study does **not** establish. It is part of the
submission and should be read before any claim is made about the results.

## 1. No human annotation

Every label in this study — including the gold standards `data/labels_only_gold.json`
(897 items) and `data/labels_only_l2_gold.json` (460 items) — was produced by **model
annotators (subagents)**. There is **no human-labeled data** anywhere in the project.

Consequences:
- The "73.0% vs 70.2%" comparison in the paper is **between two model instances**, NOT
  model-vs-human.
- We cannot state the **human labelability ceiling** for this task. The statement
  "the task is not reliably labelable" is supported only at the level of "model
  annotators cannot agree"; it is **not** established for human annotators.
- **Required before any strong claim**: a human-annotation sample (≥100 items,
  dual-blind, with inter-annotator agreement reported) on the same protocol.

## 2. Single seed / single batch

Most experiments use one random seed and one annotation batch. Effect sizes are reported,
but confidence intervals are partial. The DeLong test on the new gold (n_pos = 361) is
adequately powered; the stratified and size-comparison analyses have wider CIs and should
be read as exploratory.

## 3. The OR-aggregation assumption is only 61% correct

All historical metrics (P/R/AUROC) rest on `unit_fired = any(window fired)`. We directly
tested this aggregation on 72 mixed-label obligations against obligation-level truth:
agreement = **61.1%** (OR), 55.6% (majority), 38.9% (AND). The historical numbers
therefore measure "sentence-level judgment + OR aggregation", **not** "obligation
closure state".

## 4. Gold-standard contamination history

The gold standards are not pristine:
- Old gold: negatives had **0/63** evidence (now 100/100 after re-annotation).
- A campaign gold had **106 machine-payload items labeled positive**; its positive labels
  survived re-labeling only **23.8%** of the time (WHOLE_CLOSE 7.7%).
- The new gold (`nc_gold_all.json`) was built with session-level isolation and
  machine-payload pre-filtering; its remaining ambiguity is 6/897 (0.7%).

## 5. Data privacy

To protect the privacy of the original work, **no real session text is published**. The
repository contains label-only gold (`{pid: label}`), synthetic demo data, evaluation
scripts, and the annotation protocol. End-to-end reproduction requires the user's own
session data.

## 6. What this study DOES establish

- A complete, experiment-numbered **exclusion chain** for five modeling approaches.
- A statistical (DeLong) test at adequate power showing the anchor model is
  indistinguishable from a zero-semantic baseline.
- A same-ruler comparison showing a 287M fine-tuned model and a large general LLM
  perform within 2.5pp on the same task.
- A methodological contribution: the G-3 gate (negatives must carry evidence).

## 7. SWE-bench: requested, executed qualitatively, not quantified

SWE-bench Lite was the requested online benchmark. A pilot infrastructure was
built and run end-to-end on a real harness and real repos (closed Pi
distribution, per-instance venvs, isolated worktrees, anti-reward-hacking
judgement protocol); the audit surfaced three evaluation-invalidating defects,
and the guard chain was verified live (see `maps/SWE_EVAL*.md`).

**No quantitative SWE-bench score is claimed anywhere in this repository.**
Full runs are not tractable on the available local compute (single Intel Arc
A770; Spark-X2.5-4B at ~9 min/task before backend tuning; per-repo dependency
environments). All SWE-related findings reported here are qualitative:
the reward-hacking specimen, the guard delivery defect, and two blind-spot
specimens (half-fix, non-evidence closure).

The quantitative path is prepared and automated: the `e2e-full` GitHub Actions
job runs the full weight-backed evaluation whenever production weights are
attached via the `MODEL_RELEASE_URL` repository variable.

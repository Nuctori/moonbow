# Adversarial audit of our own negative result — and why we retract it

This repository's headline is a **negative** result: that evidence-strength grading needs no
model because mechanical rules match the model. Earlier commits built that conclusion up
across three arms. This file audits it. **The conclusion does not survive.**

## Why this audit exists

A reviewer asked the obvious question: *did you adversarially audit the negative
conclusion?* We had audited our **positive** claims repeatedly — retracting three of four
earlier failure verdicts as metric artifacts (see `MEASUREMENT_AUDIT.md`). We had not
audited the claim that *terminates the project*.

That is an asymmetric standard, and it is the wrong way round. A negative result that ends
the work carries more risk than a positive one, not less.

## The finding: the program arm is contaminated

We tested five adversarial hypotheses. One lands.

**N1 — is the program rule favoured by the evaluation set?**

| Set | Program-rule accuracy (binary subset) |
|---|---|
| **Training** (n=925) | **0.589** |
| **Evaluation** (n=779) | **0.839** |

A 0.250 gap. The error *patterns* differ too:

| Error direction | Train | Eval |
|---|---|---|
| gold=MEDIUM predicted STRONG | 0.096 | **0.215** |
| gold=STRONG predicted MEDIUM | 0.120 | **0.008** |
| gold=MEDIUM predicted NONE | **283/700 = 0.404** | 75/532 = 0.141 |

On the evaluation set the program almost never mispredicts `STRONG` (0.008) and rarely
mislabels a `MEDIUM` as `NONE` (0.141). On the training set those same errors run at 0.120
and 0.404.

**Root cause:** the token table *is* the eval-set annotation criterion. It was written by
the annotators while labelling these very items and then frozen. The program rule is
calibrated on the set it is being scored against.

## The other four hypotheses

| Hypothesis | Result |
|---|---|
| N2 model underfit (training OOM'd at epoch 2.9/8) | **Not excluded** — only 3 checkpoints survived; no epoch-wise comparison was run |
| N3 model degenerated to a constant predictor | Fails — normalised entropy 0.614, not 0 |
| N4 model overtakes on weak-token strata | Fails — no stratum shows a significant overtake (strong_token +0.043 p=0.133; med_token −0.135 p=0.0000; diag_only −0.118 p=0.0010) |
| N5 binarisation unfair to the model | Fails — NONE→MEDIUM, NONE→STRONG, and abstain all give 0.779, since the model never emits NONE |

## What is retracted, and what survives

| Claim | Status |
|---|---|
| "The model adds nothing over mechanical rules" | **RETRACTED** — the comparison arm is contaminated by a 0.250 self-serving advantage |
| "Program strictly better" | **Also not available** — same contamination, opposite direction |
| "Strength grading is mechanically decidable" | **RETRACTED** — rests entirely on the contaminated comparison |
| "Hybrid (program gate + model) is worse than the model alone" | **Also contaminated** — same program gate |
| Model exceeds chance on STRONG/MEDIUM (exact +0.17..+0.32, Spearman +0.52..+0.65) | **Survives** — measured against a random baseline, no program involved |
| Model performs intent↔evidence matching (counterfactual swap −28.7pp) | **Survives** — model-internal control, no program involved |

## What is actually undetermined

**Who is better — model or program — cannot be decided from this data.** The two arms are
not on equal footing, and the contamination cannot be removed post hoc.

Settling it requires a program rule calibrated on samples **disjoint** from the evaluation
set, or a leave-one-evaluation-set-out rebuild of the token table. Neither exists here,
because the token table and the evaluation set share an origin.

## Methodological takeaway

**Audit symmetrically.** This project's failure mode was to treat the negative conclusion
as the safe one — the low-risk, no-audit-needed direction — and to spend all scrutiny on
whether the model *worked*. The negative conclusion is exactly the one that ends
everything, and therefore the one that most needs auditing first.

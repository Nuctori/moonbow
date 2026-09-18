# Results (reproducible outputs)

All files here are machine-readable outputs of the `pipeline/` scripts, produced on the
author's own session data. They contain **labels and statistics only**; no text.

## Main study

| File | Content |
|------|---------|
| `delong_final.json` | DeLong paired AUROC: v12.1 vs anchor vs zero-semantic baseline (n_pos=361) |
| `stratified_eval.json` | AUROC stratified by obligation length × enumeration |
| `size_vs_task.json` | Same-ruler comparison: 287M fine-tuned vs general LLM (200 items) |

## Measurement audit (post-hoc; see `MEASUREMENT_AUDIT.md`)

These document three failure verdicts that turned out to be **artifacts of the metric**
rather than model failures. They are the methodological contribution of the project.

| File | Content |
|------|---------|
| `MEASUREMENT_AUDIT.md` | Full write-up: the instrument-admissibility test and its three applications |
| `multimetric_final.json` | Model vs trivial vs random on the corrected metric family (n=209) |
| `adjacent_diag.json` | Direct demonstration that `exact` and `adjacent` are mutually exclusive |
| `reverse_gate_result.json` | Skew-corrected reverse-rate gate (three sub-gates) |
| `holdout_check.json` | Train/eval leakage check: 0 id overlap, 0 text overlap |
| `label_drift_audit.json` | Grade-boundary drift: 91/1138 (8.0%) concentrated on one bucket |

### Headline numbers from the audit

| Object | exact | within-1 | reverse | Spearman |
|---|---|---|---|---|
| 287M fine-tune | **0.780** | 0.995 | 0.005 | **+0.658** |
| Trivial (majority class) | 0.703 | 1.000 | 0.000 | +0.000 |
| Random | 0.572 | 0.989 | 0.011 | −0.043 |

Read with the limits in `MEASUREMENT_AUDIT.md`: `within-1` is passed by the trivial
predictor, and the `WEAK` grade has no cross-batch evidence.

## Three-value demotion (see `THREE_VALUE_DEMOTION.md`)

| File | Content |
|------|---------|
| `THREE_VALUE_DEMOTION.md` | Why the 4th grade was removed, and the collapse of the NONE boundary |
| `three_value_eval.json` | Three-grade model on four independent held-out label sets |
| `three_value_evidence.json` | Cross-batch evidence table; drift drop and WEAK merge counts |

Headline: the 4th grade (`WEAK`) was never a defined category (its evidence field held one
annotator's private vocabulary). After demoting to three grades and retraining, exact
exceeds chance by **+0.168 to +0.317** across four independent label sets — but the model
**never emits `NONE`** (0/24 recall), because `NONE` is a mechanical scan, not a semantic
match. This independently reproduces the repository's main claim: *the program decides,
the model only extracts.*

To regenerate: place your own gold (`{"<pid>": {"label": ...}}`) and item file
(`obligation` / `sentence`) under `data/`, then run the corresponding script in
`pipeline/`.

## Adversarial audit of the negative result (see `NEGATIVE_CONCLUSION_AUDIT.md`)

| File | Content |
|------|---------|
| `NEGATIVE_CONCLUSION_AUDIT.md` | Five adversarial hypotheses against the "no model needed" conclusion |
| `audit_negative.json` | Raw output: program rule 0.589 on train vs 0.839 on eval; stratum breakdown |

**Result: the negative conclusion is RETRACTED, and the clean redo reverses it.**
The program rule was re-calibrated on the training set (see `FINAL_COMPARISON.md`): the
model then leads in **3 of 4** label sets. Original: The program arm is contaminated — the
token table is derived from the evaluation set's own annotation criterion, so the program
is scored on the set it was calibrated on (gap 0.250). Neither "model adds nothing" nor
"program is better" is available. What survives is what never involved the program: the
model exceeds chance (exact +0.17..+0.32) and does intent↔evidence matching (counterfactual
swap −28.7pp).

**Methodological point:** the audit was asymmetric. We audited the positive claims
repeatedly and the project-terminating negative claim not at all. That is backwards.

## Clean comparison and final criterion

| File | Content |
|------|---------|
| `FINAL_COMPARISON.md` | Contamination removed; model leads 3/4 sets (2 significant) |
| `final3arm_result.json` | Three-arm clean comparison with McNemar tests |
| `indep_prog_result.json` | Feature statistics and rule learned on the training set |
| `protocol/COVERAGE_CRITERION_v7.md` | Formal revision: layer responsibilities + two-valued model layer |

**Final position:** the model beats chance (exact +0.17..+0.32, Spearman +0.52..+0.65) and
beats an independently calibrated program rule in 3 of 4 sets. The `NONE` boundary is
unreliable for **both** arms (model recall 0; program mispredicts 0.404 of MEDIUM as NONE)
and is excluded from acceptance pending a dedicated annotation experiment.

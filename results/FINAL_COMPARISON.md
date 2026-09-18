# Clean three-arm comparison: the retraction reverses the conclusion

Follow-up to `NEGATIVE_CONCLUSION_AUDIT.md`. That audit found the program arm contaminated:
its token table was derived from the evaluation set's own annotation criterion, giving it a
self-serving advantage (train 0.589 vs eval 0.839).

This file removes the contamination and redoes the comparison.

## Removing the contamination

The program rule keeps its shape but is **re-calibrated on the training set** (which shares
zero item ids with the evaluation set). Feature statistics learned on train:

| Feature | Coverage n | P(STRONG \| present) | P(STRONG \| absent) |
|---|---|---|---|
| **f_commit** | 198 | **0.793** | 0.094 |
| f_test | 89 | 0.697 | 0.195 |
| f_build | 2 | 1.000 | 0.242 |
| f_deploy | 9 | 1.000 | 0.236 |
| f_diag | 456 | 0.294 | 0.194 |
| f_write_file | 340 | 0.382 | 0.162 |

Rule selected (highest P among features with coverage ≥ 20): **`f_commit` ⇒ `STRONG`**;
no token at all ⇒ `NONE`; otherwise `MEDIUM`.

**The construction validates itself.** The independent rule scores **0.671 on train** vs the
contaminated rule's **0.628** — and on the evaluation sets the ordering flips
(independent 0.675–0.774, contaminated 0.690–0.844). The contaminated arm's advantage exists
*only on the evaluation set*; the independent arm's advantage exists *only on train*. That is
the exact fingerprint of the bias described in the audit.

## Clean comparison (four independent held-out label sets)

| Label set | n | Contaminated program | **Independent program** | **Model** | Δ(model − indep) | McNemar p |
|---|---|---|---|---|---|---|
| v5b_A | 203 | 0.690 | 0.675 | **0.768** | **+0.094** | **0.0127** |
| v5b_B | 199 | 0.844 | 0.774 | **0.814** | +0.040 | 0.3317 |
| v5c_A | 196 | 0.719 | 0.730 | **0.816** | **+0.087** | **0.0241** |
| v5c_B | 205 | 0.771 | 0.741 | 0.717 | −0.024 | 0.5966 |

The model leads in **3 of 4** sets, significantly in 2. The one set where the program leads
is not significant (p = 0.60). Pooled (pseudo-replicated, indicative only): Δ = **+0.049**,
p = 0.0091.

## Revised conclusions

| Claim | Before | **Now** |
|---|---|---|
| Model vs program | equivalent | **model leads in 3/4 sets, 2 significant** |
| "The model adds nothing" | held | **retracted** — it was an artifact of the contaminated arm |
| "`NONE` belongs to the program" | held | **retracted** (see below) |

**On `NONE`:** the model never emits it (0/24 recall), which was read as the model failing
that boundary. But the independent program rule *also* mispredicts `MEDIUM` as `NONE` at
0.404 on train. The `NONE` boundary is hard for the program too — it is not a model-specific
defect, and handing it to the program does not fix it.

## What is established, and what is not

Established:
- The model beats chance on STRONG/MEDIUM: exact excess **+0.17 to +0.32**, Spearman **+0.52 to +0.65**
- The model performs intent↔evidence matching: counterfactual swap **−28.7pp**
- The model beats an **independently calibrated** program rule in 3 of 4 sets

Not established:
- Any claim about intrinsic model-vs-program superiority — the lead (+0.049 pooled) has not
  been validated on fully independent samples
- Which of the successive narrowings (four grades → three → two) reflect real capability
  limits versus label noise

The original goal — training evidence-strength grading into a ~200M model — is therefore
**no longer a negative result**. The revised answer: the model works on this task and beats
a same-origin program rule, but its absolute level (binary exact ~0.72–0.82) is still
bounded by label quality.

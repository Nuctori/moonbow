# Measurement audit: three failure verdicts that were artifacts of the metric

This is a self-contained record of a methodological finding that emerged **after** the
main study. It is included because it changes how the earlier negative results should be
read.

## Summary

Across the project's history, four conclusions were recorded as failures of the **model**.
On re-audit, **three of them were failures of the measurement instrument**, not the model:

| # | Original verdict | Actual cause |
|---|---|---|
| 1 | "Containing a completion word is a *negative* signal" (P=0.396 vs 0.818) | Conditional direction had been **reversed**; correct values are 0.640 vs 0.183 |
| 2 | "Semantic capture fails: Jaccard agreement 0.272" | **Triple measurement failure**: 48% of inputs were injected machine payloads, 48.7% were truncated, and exact-string Jaccard is boundary-intolerant. Boundary-free agreement is 0.932 |
| 3 | "Four-grade discrimination not learned: adjacent agreement 0.215 < random 0.420" | `adjacent` and `exact` are **mutually exclusive** under few, adjacent-valued classes — the metric penalizes correct answers |

## The instrument-admissibility test (the contribution)

A metric is admissible for acceptance **iff it satisfies all three**:

1. **Trivial fails.** A constant predictor (always emit the majority class) must not pass.
2. **Random is comparable.** A well-defined, reproducible chance baseline must exist.
3. **Correct is not penalized.** The metric must not decrease as the model gets *more* accurate.

Criterion 3 is the one that is normally skipped, and it is the one that produced verdict 3:

```
Model   exact 0.780   adjacent 0.215
Random  exact 0.572   adjacent 0.417
                    ↑ the +0.207 exact gain is ~exactly the −0.202 adjacent loss
```

Because the label space has only three grades and 93% of mass sits on two **adjacent**
grades, a random prediction lands on an adjacent grade 41.7% of the time *for free*.
A model that gets the exact grade right therefore *loses* adjacent credit.
Under this metric, **being more correct is punished.**

## Same task, admissible metrics

`within-1` (exact + adjacent) is the monotone replacement.

| Object | exact | within-1 | reverse | Spearman |
|---|---|---|---|---|
| 287M fine-tune | **0.780** | 0.995 | 0.005 | **+0.658** |
| Trivial (majority) | 0.703 | 1.000 | 0.000 | +0.000 |
| Random | 0.572 | 0.989 | 0.011 | −0.043 |

Note the second row: **trivial scores 1.000 on within-1.** This is why within-1 cannot be
used alone — it is reported here to show that criterion 1 (trivial fails) must be applied
per-metric, not per-study.

Rejection is confirmed by a counterfactual swap: pairing intent A with work-log B drops
exact from 0.780 to 0.493 (−28.7pp), while "prediction matches the work-log's own label"
stays at 0.780. The model reads the pairing, not the template.

## Label-side audit

The grade `WEAK` is **not usable** in the current data. Of 76 `WEAK` labels, 49 (64%)
come from a single batch that mapped a mechanically-decidable bucket ("no artifact token")
onto `WEAK`; the frozen criterion mandates `NONE` for that bucket. Drift is
**91/1138 = 8.0%** and is concentrated entirely on that one bucket — not spread as noise.
After removing it, 27 `WEAK` labels survive, all from one batch: **no cross-batch evidence
for this grade exists.** The effective ordinal scale is three-valued, not four.

## Leakage check

Training and evaluation are disjoint: 900 vs 200 item ids, **0 overlap**, and
**0/202 overlap** on the first 120 characters of item text. Evaluation is held-out.

## Reproduce

```
python pipeline/research_v198_gate.py       # corrected reverse-rate gate
python pipeline/research_v200_adjacent.py   # exact/adjacent mutual exclusion
python pipeline/research_v201_metrics.py    # admissible metric family
python pipeline/research_v202_holdout.py    # leakage check
```

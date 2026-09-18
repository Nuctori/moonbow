> **RETRACTED (see `NEGATIVE_CONCLUSION_AUDIT.md`).** The program arm in this comparison
> scores 0.839 on the evaluation set but only 0.589 on the training set, because the token
> table is derived from the evaluation set's own annotation criterion. The program is
> scored against the set it was calibrated on. The conclusion below is therefore not
> supported, and neither is its reverse. Kept for audit trail.

# Does the program/model split help on evidence-strength grading? No.

Follow-up to `THREE_VALUE_DEMOTION.md`. Its diagnosis implied that the `NONE` boundary
should be handed to the program while the model kept the rest. That implication is
**falsified**.

## Setup

Three arms, same 208 items, evaluated on four independent held-out label sets:

- **H0** — pure 287M model, three grades
- **H1** — program decides `NONE` when the work-log has no artifact token; model decides
  `STRONG` vs `MEDIUM` otherwise
- **H2** — pure program, no model at all (mechanical token rules)

## Result

| Label set | H0 pure model | H1 program gate + model | H2 pure program |
|---|---|---|---|
| v5b_A | **0.768** | 0.675 | 0.690 |
| v5b_B | 0.814 | 0.739 | **0.844** |
| v5c_A | **0.816** | 0.730 | 0.719 |
| v5c_B | 0.717 | 0.746 | **0.771** |
| **mean** | **0.779** | 0.722 | 0.756 |
| **NONE recall (mean)** | **0.000** | 0.857 | 0.857 |

## Reading

Three facts hold at once, and the third is the interesting one:

1. The program gate **does** recover `NONE`: recall 0.000 → **0.857**. The diagnosis that
   `NONE` is mechanical was correct.
2. **But H1 is worse than H0** (0.722 vs 0.779). On the two sets with no gold `NONE`,
   the gate fires on items that should be `MEDIUM`/`STRONG`, and the loss outweighs the gain.
3. **H2 (pure program, no model) beats H1** (0.756 vs 0.722) and matches H0 within 0.023.

The model wins on two sets (+0.079, +0.097) and loses on two (−0.030, −0.054). That is a
no-stable-advantage distribution, not a capability.

**So the split is not "program does `NONE`, model does the rest" — it is that the model
adds nothing over mechanical rules at this level.** The whole evidence-strength grading
task falls inside the reach of mechanical rules.

## Relationship to the main study (do not conflate)

The main study established that *program builds the graph, LLM does semantic capture*
(accuracy 0.878 vs 0.634 baseline). **This result does not contradict that** — the tasks
differ:

| | Main study | This result |
|---|---|---|
| Task | intent ↔ evidence **coverage** (was it done?) | evidence **strength grading** (strong vs medium) |
| Verdict | LLM semantic capture works | program rules ≡ model |

Reading this as a refutation of the main study is a misreading. The correct reading:
*"semantic capture" works; "strength grading," one level finer, is already within
mechanical reach.*

## Consequence for the project's original goal

The original goal was to train evidence-strength grading into a ~200M model. That goal is
**falsified — not because the model is too small, but because the task is mechanically
decidable.** A negative result about the task, not the model.

## Paired significance tests (added after the initial write-up)

The means above were reported without a test, so the original write-up could only claim
"no stable advantage." Paired per-item McNemar exact tests plus 10,000 item-resampled
bootstrap CIs sharpen two of the three claims:

| Label set | n | H0 | H1 | H2 | H0−H1 Δ [95% CI] | McNemar p | H0−H2 Δ [95% CI] | p |
|---|---|---|---|---|---|---|---|---|
| v5b_A | 203 | 0.768 | 0.675 | 0.690 | **+0.094** [+0.054,+0.138] | **0.0000** | +0.079 [+0.005,+0.153] | **0.0479** |
| v5b_B | 199 | 0.814 | 0.739 | 0.844 | **+0.075** [+0.030,+0.121] | **0.0015** | −0.030 [−0.106,+0.045] | 0.5044 |
| v5c_A | 196 | 0.816 | 0.730 | 0.719 | **+0.087** [+0.051,+0.128] | **0.0000** | +0.097 [+0.026,+0.168] | **0.0127** |
| v5c_B | 205 | 0.717 | 0.746 | 0.771 | −0.029 [−0.078,+0.015] | 0.3075 | −0.054 [−0.127,+0.020] | 0.1925 |

Three corrections to the text above:

1. **"H1 is worse than H0" is now statistically supported and is stronger than stated.**
   It is significant in 3 of 4 sets with a consistent sign, and on v5b_A / v5c_A the
   discordant counts are b=19/c=0 and b=17/c=0 — **zero counterexamples**. The gate did
   not merely trade gains for losses; on sets without gold `NONE` it is pure damage.
2. **"H2 beats H1" is NOT significant** (p = 0.75 / 0.0005 / 0.87 / 0.50; only one set
   significant). Do not claim pure-program superiority over the hybrid.
3. **"Model adds nothing over program" survives.** H0 vs H2 is significant in only 2 of 4
   sets, with *opposite* signs — so the correct claim remains equivalence, not program
   superiority.

The headline conclusion — that strength grading records **no stable model advantage** —
is unaffected. What changed is which secondary claims are permitted.

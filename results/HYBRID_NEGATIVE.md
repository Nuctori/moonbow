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

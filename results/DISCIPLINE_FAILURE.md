# Why five negative results were all wrong: the audit was applied asymmetrically

This file documents a **process** defect, not a result. It is included because it is the most
transferable lesson in this repository.

## The observation

Five conclusions in this project were **negative** — they said the system, the labels, or the
task was worse than hoped. All five were recorded. All five were later overturned.

| § | Negative conclusion | Audited? | Overturned by | Cause |
|---|---|---|---|---|
| 249 | Three batches drifted 8.0% | **No** | §266 | Circular: the checker read the annotator's own field saying `"NONE"` and then declared the annotator violated a token rule |
| 250 | Grade `WEAK` has no cross-batch evidence | **No** | §266 | Same circularity; `WEAK` is used independently by two annotators on the same items |
| 254 | Demote four grades to three | **No** | §266 | Premise (249/250) did not hold |
| 255 | The model adds nothing over mechanical rules | **No** | §258 | The program arm was calibrated on the evaluation set (train 0.589 vs eval 0.839) |
| 261 | `NONE` has zero inter-annotator consensus | **No** | §262 | A hand-written token table with a **96.3% false-negative rate** used as ground truth to judge human labels |

**Five negative conclusions, zero of them audited, five of them wrong (0/5).**

Meanwhile the **positive** claims were audited repeatedly and several were corrected
(§251 metric artifact, §258 contaminated baseline arm).

## The diagnosis

The project already required that *any* conclusion pass adversarial audit. The requirement
was real and was written down. **It was applied asymmetrically: positive claims got audited,
negative claims did not.**

The likely reason is motivational rather than technical: a negative result feels like an
*endpoint* — the work stops, the pressure lifts — while a positive result feels like an
*obligation to keep proving*. So the negative claim escaped scrutiny exactly when it
deserved the most, because a negative claim is the one that terminates the project.

## The second, independent defect: instruments were never cross-audited with the data

Two whole classes of check were simply never performed:

1. **Scripts vs data.** The scripts were assumed to interpret fields correctly. They did not:
   the `tokens` column contains a **mixture of three semantics** in one field — real evidence
   (`b4348ab`, `scripts/ci/run_core_tests.sh:18`), the annotator's own verdict (`"NONE"`),
   and free-text description (`"process-description"`) — with no metadata distinguishing them.
   One script treated the annotator's verdict as its own independent detection.

2. **Training data provenance.** `strength3_train.json` (1047 rows) is **100% single-annotator**;
   its three batches label **pairwise-disjoint** item sets, so **no row can be cross-verified**.
   The data was audited downstream (metrics) but never as a source.

## The rules now in force

Written into `protocol/COVERAGE_CRITERION_v7.md` §7 as mandatory gates:

```
D1  A negative conclusion is untrusted by default. Before recording one, state what
    would be observed if it were FALSE, and go measure that.
D2  An instrument may not judge another system until it has itself passed a recall
    audit (FN < 20%, validated against >= 100 hand-labelled items).
D3  Training data must report (a) annotator count, (b) per-grade one-vs-rest
    agreement, (c) share that cannot be cross-verified. If (a)=1 or (c)>50%, every
    conclusion must be tagged "single-annotator, not cross-verified".
D4  Cross-audit scripts against data: verify each field's actual semantics by sampling.
    Watch for multi-semantic columns and for treating others' conclusions as
    independent measurements.
D5  Log every conclusion with whether it was audited, how, and by whom.
```

Plus a concrete prohibition: **the `tokens` field may never be used to decide whether
evidence exists** (mixed semantics, includes self-report). Evidence presence must come from
independently scanning the work record — the widened table has a 0.002 false-negative rate
versus the original's 0.124.

## Transferable point

The instrument-admissibility test, the recall audit, and the split-parity check are all
useful. But none of them would have caught the actual failure, because the failure was not
in any instrument — it was in **when scrutiny was applied**. A toolchain cannot fix a
discipline that exempts the conclusions which end the work.

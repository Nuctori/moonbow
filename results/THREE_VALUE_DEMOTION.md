# The four-grade criterion was demoted to three, and the remaining boundary collapsed

Follow-up to `MEASUREMENT_AUDIT.md`. Two findings, both negative for the model.

## 1. The fourth grade was never defined

The grade `WEAK` had no cross-batch evidence. Re-inspecting the 27 surviving `WEAK`
labels showed their evidence field contained `process-description`,
`analysis-no-locator`, `process-name-only` — **one annotator's private vocabulary**, not a
grade definition any other annotator shared.

So `WEAK` was not "under-sampled"; **it was never a defined category.** The criterion was
demoted to three grades (`STRONG` / `MEDIUM` / `NONE`); `WEAK`'s semantics were merged into
`MEDIUM` (merging into `NONE` would have broken `NONE`'s mechanical gate, which is the one
boundary that had been stable all along).

Two mechanical changes to the criterion:
- Boundaries went from **three to two** (the `MEDIUM↔WEAK` and `WEAK↔NONE` boundaries are
  gone). Removing a boundary is a *subtraction*; the "threshold conservation" pathology
  observed earlier only appears when a boundary is added or tightened.
- A new acceptance rule: **every grade must appear in at least two annotation batches.**
  This is the direct guard against repeating the `WEAK` failure. It passes
  (`STRONG` 3 batches, `MEDIUM` 3, `NONE` 2).

## 2. Retrained on three grades: better, but the remaining boundary collapsed

Trained a 287M classifier on 1047 cleaned rows (three grades). Loss 0.65 → 0.207.
Evaluated on four independent held-out label sets:

| Label set | exact | chance | **excess** | Spearman | NONE recall |
|---|---|---|---|---|---|
| v5b_A | 0.768 | 0.600 | **+0.168** | +0.518 | — (no gold NONE) |
| v5b_B | 0.814 | 0.498 | **+0.317** | +0.646 | **0.000** (n=3) |
| v5c_A | 0.816 | 0.588 | **+0.228** | +0.626 | — (no gold NONE) |
| v5c_B | 0.717 | 0.446 | **+0.272** | +0.627 | **0.000** (n=21) |

The three-grade model beats chance by a wider margin than the four-grade model did
(+0.17 to +0.32, consistent across four independent label sets).

**But it never emits `NONE`.** Across 203 items the output distribution is
`MEDIUM` 124 / `STRONG` 84 / **`NONE` 0** — 0 of 24 gold `NONE` items are recovered.

## 3. What this means

`NONE` depends on a **mechanical scan of the whole work-log** ("does it contain any
artifact token?"), not on intent↔evidence semantic matching. The earlier counterfactual
test showed the model *does* learn the latter. So the mechanical gate is the part the
model cannot carry.

The capability boundary is therefore narrower than the three-grade criterion suggests:

```
program (deterministic):  no artifact token  => NONE      # no model needed
model   (semantic):       STRONG vs MEDIUM                # excess +0.17 .. +0.32
```

This independently reproduces the repository's main architectural claim — *the program
decides, the model only extracts* — by arriving at it from the opposite direction: a
boundary that is mechanically decidable is exactly a boundary the model fails to learn.

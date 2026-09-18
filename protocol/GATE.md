# The gate as executable code

Five negative conclusions in this project were recorded without audit and all five were
wrong. Two subsequent audits then **independently** overstated their own headline numbers
(92.6% and 96.3% false-negative rates that hand-adjudication revised to ~33%).

The project already had these rules written as **prose** in
`protocol/COVERAGE_CRITERION_v7.md` §7. Prose failed twice. So the same rules now exist as
**executable code** that a pre-commit hook enforces:

```
python pipeline/gate.py           # check working tree
python pipeline/gate.py --all     # check everything
python pipeline/gate.py --explain G1
```

## Rules

| Rule | Severity | Catches |
|---|---|---|
| **G1** | **BLOCK** | Using the `tokens` field's value as evidence of fact. That column mixes three semantics (real evidence, the annotator's own verdict, free text), and treating the annotator's verdict as an independent measurement is what produced the §249 circular argument. |
| **G2** | warn | A sentence asserting a negative conclusion with no audit marker within ±12 lines. Only checks **newly appended** content — otherwise FINDINGS.md's 970+ historical lines drown the signal. |
| **G3** | warn | Training data with no `annotator` / `criterion` metadata. `strength3_train.json` (1047 rows) is 100% single-annotator with pairwise-disjoint batches, so no row is cross-verifiable. |
| **G4** | warn | A reported rate with no hand-adjudication or interval evidence. This is the §270 gate: automatic statistics from a regex can overstate a rate several-fold. |
| **G5** | warn | A rule-based classifier not declared in the `INSTRUMENTS` registry, i.e. never subject to the recall audit the project requires of such tools. |

## Design notes

- **Noise kills gates.** The first version emitted 1081 warnings; nobody reads a gate that
  fires constantly. It now emits ~60, entirely by restricting G2/G3 to *active* outputs and
  newly-appended content rather than the whole history.
- **Import is not the sin.** G1 blocks *using* `toks_present`'s return value as a fact,
  not importing it for audit or reproduction. Exempt context markers are supported.
- **Self-tested.** A deliberately violating file is caught (`token == 'NONE'` comparison);
  the test file is then removed.

## Honest limits

This gate cannot tell whether an auditor's hand-adjudication was itself honest, whether a
claimed interval is right, or whether a negative conclusion's audit was competent. It only
enforces that the *markers exist*. A gate is a floor, not a substitute for judgement — the
underlying failure was about *when scrutiny is applied*, and no linter fixes that.

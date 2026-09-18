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

To regenerate: place your own gold (`{"<pid>": {"label": ...}}`) and item file
(`obligation` / `sentence`) under `data/`, then run the corresponding script in
`pipeline/`.

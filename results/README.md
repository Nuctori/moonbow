# Results (reproducible outputs)

All files here are machine-readable outputs of the `pipeline/` scripts, produced on the
author's own session data. They contain **labels and statistics only**; no text.

| File | Content |
|------|---------|
| `delong_final.json` | DeLong paired AUROC: v12.1 vs anchor vs zero-semantic baseline (n_pos=361) |
| `stratified_eval.json` | AUROC stratified by obligation length × enumeration |
| `size_vs_task.json` | Same-ruler comparison: 287M fine-tuned vs general LLM (200 items) |

To regenerate: place your own gold (`{"<pid>": {"label": ...}}`) and item file
(`obligation` / `sentence`) under `data/`, then run the corresponding script in
`pipeline/`.

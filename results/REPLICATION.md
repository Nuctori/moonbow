# Independent replication of the two audits (falsification attempt)

**Scope.** Replicates `maps/_labelaudit/LABEL_AUDIT.md` (§266) and `maps/_toolaudit/TOOLCHAIN_AUDIT.md` (§264).
Method: derive every number from raw label files + the project's own source scripts, without reading
the audit reports first. Machine-readable results: `maps/_repl/replication.json`.
Work dir: `maps/_repl/`. Interpreter: `.venv-xpu/Scripts/python.exe`.

**Privacy.** `pipeline/research_v196_scrub.py` run over all 29 files in `maps/_repl/`:
3 hits, all of them already-redacted `<REDACTED>` markers quoted from source data. No live
credentials, tokens, internal IPs, private repo slugs or emails were written.

**Verdict tally: CONFIRMED 4 · CONFIRMED-WITH-CAVEAT 5 · OVERSTATED 2 · REFUTED 1.**

---

## Claim set A — label audit

### A1 — §249's violation check was circular · CONFIRMED-WITH-CAVEAT
Reproduced exactly from `maps/v5b_label_train_{1,2,3}.json` (n=1138 bindings):
**91 violations, of which 63 have `tokens` literally `"NONE"` (69.2% of violations, 5.5% of rows).**
Audit said 91 and 63 — exact agreement.

The *characterization* is half-right. Two things are true:
1. `research_v205_hybrid.py:69` filters the eval set with `toks_present(b['tokens']) and grade!=NONE` — the rule that builds the eval set is the same rule the audit charges. That is a genuine circularity.
2. The 63 literal-`NONE` rows are rejected by `toks_present()` **by definition** (`'NONE'` is in its reject tuple), so calling them "violations" is definitional, not empirical.

But "circular" **mis-describes the annotator**. The counter-case is strong: an annotator writing
`tokens="NONE"` while grading STRONG/MEDIUM is not self-contradicting — they are recording
"I found the record responsive but did not isolate a token string". That is legitimate. The accurate
charge is *"the rule re-reads the label it is meant to test, so it cannot test it"* — not
"the annotator contradicted themselves".

Independent secondary finding: the 28 residual non-literal-`NONE` violations are **all** grade=STRONG
with short-but-real tokens (`全绿`, `上线`, `部署`, `2/2`, `5/5`) that `toks_present`'s `len>3` gate
rejects. That is a real len>3 defect, independent of the NONE circularity.

### A2 — true no-evidence bucket is ~42 rows (3.7%) · **OVERSTATED**
**42 is exactly reproducible** under the criterion "the work_log contains nothing artifact-ish at all"
(no backtick span, no dotted-extension token, no `n/n`, no PASS/FAIL). But **the criterion is wrong**.

Hand-reading the 42 shows many contain obvious completion evidence, e.g.:
- b2/b3 duplicate rows whose work_log is a **finished** version-update announcement with a delivered table and sign-off;
- b2/WEAK "图标居中约束已经落地并过测试";
- b2/STRONG "**no code diff**; gate not required" with a 70/30 root-cause decomposition;
- b3 "我把局部裁切重新生成一份".

Adding a single completion-verb test collapses the bucket **42 → 1**. Full sensitivity band:
**1** (strict) · **42** (nothing-artifactish) · **197** (my six-signal tier-0) · **367** (any-of-six absent).
The audit's 42 is the *most permissive* threshold they used, i.e. an **upper**, not lower, estimate.
Their "almost entirely b3" (b1 3 / b2 7 / b3 32) is correct.

Corrected statement: *the no-evidence bucket is between 1 and ~42 rows depending on the threshold,
with 42 being the loosest defensible bound; it is not a sharp 3.7%.*

### A3 — largest trustworthy subset = 122 bindings, NONE = 0 · **CONFIRMED**
Exact reproduction: **122 items, STRONG 33 / MEDIUM 89 / NONE 0**, every core item supported by
exactly 2 families.

Caveat the audit under-states: `v3_A/v3_B` uses a **different item pool** (batch `recheck`) with
**zero pid overlap** with v5b/v5c/clean_recheck (200 vs 200, intersection 0). So only v5b+v5c can
ever form a ≥2-family core; v3 silently contributes nothing while §5.1's narrative implies three
families. My v5b+v5c-only count is **123 (STRONG 34 / MEDIUM 89)**; the audit's 122 comes from
index-matching (`min(len(da),len(db))`) rather than key-matching, which drops one binding.
The 1-item difference does not change the conclusion.

### A4 — noise ceiling raw 0.818 / κ 0.622 · **CONFIRMED**
Exact to 4 dp: v5b_A vs v5b_B, n=209, **raw 0.8182, p_e 0.5184, κ 0.6225** (unweighted Cohen,
marginal-product p_e). 4-class raw 0.8134. Robust to alignment choice (pid+index and pid+intent-prefix
both give 0.8182/0.6225). v5c 0.7703/0.5348 and v3 0.7538/0.2531 also reproduce.

### A5 — 1047 rows, single-annotator, pairwise-disjoint · **CONFIRMED-WITH-CAVEAT**
Disjointness **confirmed exactly**: 0 overlap on all three pairings, 900 union pids.
Single-annotator confirmed (one `annotator` field per file).

On resolvability — I found **strong batch-level style signals**: token-field mean length
**107.6 / 20.0 / 133.8**, semicolon usage **181 / 0 / 359**, literal-`NONE` **63 / 32 / 90**, and
**disjoint date windows** (7 / 13 / 6 days). But these are *batch* signals, not *person* signals:
one person annotating three disjoint item pools on three disjoint days with a shifting convention
produces an identical fingerprint to three people. There is no per-row author id, no session log,
no cross-batch calibration item. **I confirm the audit's caveat: identity is genuinely unresolvable.**
The style differences do independently corroborate the "convention changed across batches" reading.

---

## Claim set B — toolchain audit

### B1 — token-table FN rate ~92.6% · **OVERSTATED** (most damaged claim)
I imported the **real** `program_grade()` from `research_v205_hybrid.py`, applied it to
`clean_recheck_200` → 27 NONE rows, built my own independent tiered classifier, then
**hand-adjudicated all 27 by reading them**.

| FN estimate | value |
|---|---|
| audit's regex families (my reproduction) | 25/27 = **0.926** |
| backtick-only (the dominant family) | 25/27 = **0.926** |
| my regex LOOSE | 25/27 = 0.926 |
| **my hand adjudication** | **9/27 = 0.333** |
| regex precision vs hand | **0.36** |

25 of the audit's 25 hits come from **one** family, `backtick_command`, which matches *any* inline
code span — including purely hypothetical ones ("给 E2E 增加 \`wait-idle\`",
"给 RuntimePathResolver 加 \`STRICT\` 开关"). That is a near-trivial criterion.

Hand-reading found 9 rows with a genuine completed action ([1] "已执行 npm install… 验证通过 pnpm -v
8.15.6"; [3] "我把 Setting.ini 的 menu_hide 改成 0 后，窗口变成 Left=1720"; [4] "已完成仓库检查" with
git output; [5] "我刚实测" 187.19MB; [7] "已经帮你配好了" Run-key written; [8] NTFS dedup measurement;
[13] "定位到了…我先删掉"; [19] "已设置完成, origin…"; [0] "我刚做了对比") and 18 that are
plans/advice/explanations ([2] layered plan; [10] "建议用强门禁实现"; [11] nullability norm;
[16] Sentry setup advice; [17] "我会用…技能来做"; [21]/[22]/[24] CI config templates).

**Corrected: the token table's true FN rate on this population is ≈33% (9/27), not 92.6%.**

### B2 — `adjacent` anti-monotone; 2 scripts still gate on it · **CONFIRMED-WITH-CAVEAT**
Arithmetic **exactly reproduced** from `maps/strength_confusion.json` gold (n=209): adjacent
**0.397 at random**, collapsing monotonically to **0.000 at perfect exact**, non-monotone in between
(0.316→0.373 at a=0.2→0.3). Clean, correct result.

The follow-on is **REFUTED**: the audit's census flags `_audit_f_ceiling.py` and
`research_v200_adjacent.py` via `adjacent['\"]?\s*[><=]`. Both hits are **false positives** — the
matched text is `adjacent=` inside an f-string print label. Neither file compares adjacent to a
threshold. `maps/MULTI_METRIC_ACCEPTANCE.md:3` states the `adjacent ≥90%` gate was **retired** in
§251, and the real gate `research_v198_gate.py` thresholds the **reverse** rate, not adjacent.
**No pipeline script currently gates acceptance on `adjacent`.**

### B3 — calibration leak, train 0.628 vs eval 0.68–0.77 · **CONFIRMED-WITH-CAVEAT**
All numbers reproduce exactly: **train 0.6275**, eval **0.6794 / 0.8230 / 0.6794 / 0.7656**
(mean 0.7369), **zero pid overlap and zero 120-char text overlap**.

I actively tested the alternative explanation (base-rate artifact) and it **fails**: on the
STRONG/MEDIUM-only subset, base rate neutralised, train is still **0.589** vs eval **0.679–0.820**.
Mechanism is visible: the rule predicts NONE on **38%** of train vs **13%** of eval.
Temporal caveat: `research_v205_hybrid.py` (22:06) *post-dates* `strength3_train.json` (21:48), so
the rule was not literally fitted on train; but `v209_indep_prog.py`'s own docstring states
"§258 发现：程序 token 表由评估集标注反推". The leak is real and self-acknowledged.
The *attribution share* cannot be pinned from timestamps.

### B4 — 6/6 scripts use hand-written rules with zero recall audit · **CONFIRMED-WITH-CAVEAT**
The list is exactly right: `research_v205_hybrid`, `v206_signif`, `v207_equiv`, `v208_audit_neg`,
`v209_indep_prog`, `v210_final3arm`, all importing the `RE_*` table from v205.
Applicability: `v205` is the definition site (tautological). `v207_equiv` already computes
`L1_prog_recall`. `v206_signif` uses the rule only as the H2 comparison arm, never as ground truth.
A recall audit is **genuinely applicable** to `v208` (program_grade used as oracle to judge human
labels) and `v209`/`v210` (table treated as calibrated truth); largely benign in `v205`/`v206`/`v207`.

### B5 — acceptance doc catches only 1 of 6 incidents · **CONFIRMED**
My independent adjudication gives **1/6 on both** the prompt's incident set (§249/§250/§254/§255/
§261/§251) and the audit's own set (§203/§223/§241/§251/§258/§261). Only **§251** is caught, by
T3 (monotonicity). On the prompt's set I judge **no** incident "partial": §255 is cross-split
leakage (T1–T3 are within-split properties) and §261 needs an instrument-recall requirement that
no test has. Note `DISCIPLINE_FAILURE.md` itself shows a **five**-row table.

---

## Claim set C — self-declared limits

### C1 — topic-overlap flag discarded · **CONFIRMED**
The discard is justified, and my reproduction is if anything stronger. Under the audit's **own**
`topic_ratio` definition the flag fires on **64.4–65.3%** of the 2652 answered rows (they said 57.8%;
same regime). Under my independent character-bigram + latin-token measure it fires on **11.6%**, but
hand-reading those rows shows they are overwhelmingly **correctly** labelled — "Godot Engine v4.6…"
with a root-cause log; "好，麻烦做P0和P1." with a delivered implementation; "先正确提交代码。" with a
crash root-cause. Precision ≈ 0 under either definition.

Mild finding: the audit's *stated reason* ("Chinese intent and Chinese work-log share few literal
tokens") is imprecise — my bigram measure found **31% mean overlap**. The real defect is that low
literal overlap is *uncorrelated with mislabelling*, not that the texts are lexically disjoint.

### C2 — its own regexes make the number a lower bound · **REFUTED**
The claim is **backwards**. A lower bound needs the regexes to *under*-detect; they **over**-detect.
The dominant family matches hypothetical code spans, so it counts plans as completions.
Measured against hand truth the precision is **0.36** and the reported rate **overstates** the true
FN rate by **2.8×**. **Corrected: 92.6% is an UPPER bound; the hand-validated rate is 33.3%.**
This is the same §262 failure mode the audit was criticising, applied to the audit's own metric.

---

## Could not fully reproduce / judge

- **B3 attribution share.** I confirm the gap is real and not base-rate, but cannot quantify how much
  comes from authoring-against-eval vs genuine distribution shift (no git history; v205 post-dates
  both datasets by ~18 min).
- **B2 provenance of the gold.** I used the project's own `strength_confusion.json` with the audit's
  seed, so B2 is an *exact reproduction* of an arithmetic claim, not an independent re-derivation
  from raw model outputs. I did not re-run the model.
- **Model-dependent claims** (v205's H1/H2 arms, the 0.780 headline) were **not** independently
  reproduced — only program-rule and label-statistics claims were.

## What this replication cannot establish

1. **My A2 and B1 rules are my own judgement calls.** The A2 bucket is definition-dependent (1–197);
   no single number is *the* answer. B1's 9/18 split is one careful reading under a stated rule.
2. **A second reader could move 2–3 B1 borderline rows** ([0] comparison-as-deliverable,
   [23] query-executed-but-no-product, [25] endpoints-tested-but-no-artifact), shifting the true
   FN rate between ~0.22 and ~0.44. My 0.333 is a point estimate on a 27-row sample.
3. **My B1 classifier is regex-based, like the audit's.** Its added value is the hand adjudication —
   which is itself subject to D1. I did not escape the discipline I am applying.
4. **A5 annotator identity is genuinely unresolvable** from the files; my verdict records that, it
   does not settle it.

## Bottom line

The **label audit's arithmetic is excellent** (A3, A4 reproduce to 4 dp) but it inherits the very
defect it criticises when it *counts* (A2's 42 rests on a permissive "nothing artifactish" rule).
The **toolchain audit's two headline numbers are the weakest** (B1, C2): its 92.6% is a single
backtick regex with 0.36 precision measured against hand reading (true rate ≈33%), and its
"lower bound" claim is inverted. Its B2 arithmetic is right but its "2 scripts still gate on
adjacent" follow-on is a regex false positive, and the gate was already retired.

Both audits' **headline conclusions survive in direction, but not in magnitude**: negative
conclusions about the labels/rule *are* systematically contaminated (A1's circularity is real,
B3's leak is real, B5's 1/6 is right, C1's discard is right), yet the specific inflated numbers
(92.6% FN, 3.7% no-evidence) should be replaced by the hand-validated values (≈33% FN, ≤42 and
likely far fewer no-evidence rows).

*Speculation (labelled as such):* that both audits share the §262 defect — an unvalidated narrow
instrument treated as truth — suggests the failure is structural to regex-based audit tooling in
this project rather than two independent slip-ups; a shared "hand-validate your classifier on ≥20
rows before reporting its rate" gate would have caught both.

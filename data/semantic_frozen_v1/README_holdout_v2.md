# holdout_v2 — process v2 及后续迭代的新留出集（R6，provisional）

日期：2026-09-30。背景：冻结 test split（frozen_texts_test）已于 2026-09-30
最终验收中首次且唯一一次消费（见 `results/semantic-runtime/final-test/
final_acceptance.md` §0），后续任何迭代不得再以其为留出集；本批按"新 id 段、
与全部既有数据查重、按源任务整组切分"规则新建。

## 状态声明（重要）

- **provisional**：全部文本 agent 原创、标签 agent 生成 + 自审，无人工复核，
  非金标准；每条带 label_basis。
- **仅用于 process.unresolved v2 及后续迭代的留出验收**（以及届时其他 match
  模式的留出复评）；**消费一次后即作废，需再新建 holdout_v3**，不得反复使用。
- **不参与训练、不参与早停、不参与阈值选择**；calib 仅作验证/选阈值，
  最终口径以本集（或其后继）为准。

## 文件与格式

- `holdout_v2_texts.jsonl` / `holdout_v2_labels.jsonl`：120 条
  （id 前缀 `fzh2_0001–0120`），格式与 `frozen_texts_dev.jsonl` /
  `frozen_labels_dev.jsonl` 完全一致（5 pattern 布尔 + ts.capture + label_basis）。
- ts.capture 全部 abstain（沿用 extension 批约定；本集用于 match 模式验收，
  ts.capture 走 rule 的 P7 结论不变）。
- 源任务 `fzT301–fzT420`（fzT300+ 新段），每任务恰 1 条文本（无改写组，
  按源任务整组切分即天然成立），与 dev/calib/test（fzT001–fzT200）、
  dev_extension_r4（fzT201–fzT292）、modality_extension_r4（fzmodT*）、
  process_extension_r5（fzprocT*）**零交叉**。

## 与冻结 test 的分布对照

| 维度 | frozen test (150) | holdout_v2 (120) |
|---|---|---|
| user_request | 75 (50.0%) | 60 (50.0%) |
| conversation_interference | 36 (24.0%) | 29 (24.2%) |
| agent_completion | 24 (16.0%) | 19 (15.8%) |
| interim_report | 15 (10.0%) | 12 (10.0%) |
| zh:en | 99:51 (66:34) | 72:48 (60:40) |
| process.unresolved 正例 | 55 | 38（≥35 达标） |
| process 负例 | 95 | 82（≥50 达标） |

语言比 en 偏高 6 点（FN 家族 F3 为英文形态，需在留出集中保量，如实登记）。
process 正例刻意覆盖 fn_analysis.md 的 FN 家族（完成内嵌未决 / 前置待查 /
英文形态 / 求证问句），负例含强干扰形态（自答疑点、修辞性反问、历史已解决、
外置遗留），可直接检验 v2 的召回修复与 P 不倒退。

## 校验

`python tools/validate_frozen_data.py --holdout-v2`：id 唯一且前缀 fzh2_、
task_id 与全部既有集无交叉、5 pattern 枚举、quote 逐字（本批全 abstain）、
与冻结三组 + dev_extension_r4 + modality_extension_r4 + process_extension_r5 +
semantic_dev_v1 的 bigram Jaccard>0.8 查重、配额统计。

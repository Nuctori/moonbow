# semantic_frozen_v1 — 冻结测试集（provisional 标注）

日期：2026-09-30。对应 `docs/semantic_match_runtime_plan.md` §4-P8（数据方案与首次
质量目标）与 `docs/task_structure_spec.md` §2/§3/§6。本集是 P8 的**首次冻结**批次。

## provisional 声明（重要）

- 本集全部文本为 **agent 原创编写**，未从任何真实会话复制；与
  `data/semantic_dev_v1/` 及 `tests/` 现有样例做了字符 bigram Jaccard>0.8 查重，
  无重复（见 `python tools/validate_frozen_data.py` 输出）。
- 标签为 **agent 生成 + agent 自审的暂定标注（provisional），无人工复核**，
  **不构成人工金标准**（与 task_structure_spec 状态声明一致）。标注一致性未测量。
  任何基于本集的离线指标只能作为暂定评测结果，不得宣称人工金标准验收。
- 每条带 `label_basis` 一句话依据；无法可靠确认的边界样本按规范弃权
  （ts.capture="abstain" 并在 basis 说明），未做猜测性标注。

## 文件

| 文件 | 内容 |
|---|---|
| frozen_texts_dev.jsonl / frozen_labels_dev.jsonl | 开发组 300 条 |
| frozen_texts_calib.jsonl / frozen_labels_calib.jsonl | 校准组 150 条 |
| frozen_texts_test.jsonl / frozen_labels_test.jsonl | 测试组 150 条 |

文本字段：`id`（fz_0001–fz_0600）、`text`、`source_family`、`lang`、`task_id`、
可选 `long`。标签格式沿用 `data/semantic_dev_v1/dev_labels.jsonl`：5 个固定
pattern 布尔值 + `ts.capture`（8 类捕获列表或 "abstain"）+ `label_basis`。

## 拆分与来源组

- 200 个源任务（fzT001–fzT200），按源任务整组切分：dev 100 / calib 50 / test 50；
  **同源任务的文本与改写对必须同组**（validator 强制校验 task_id 不跨组）。
- 每个源任务含 2–3 条文本，其中前两条互为改写对（复述/同义改写），
  第三条为同任务的变体或边界样本。
- 5 个 pattern 的正例覆盖（全集）：completion.asserted 89 / modality.assertive 178 /
  task.object.alignment 429 / process.unresolved 173；ts.capture 捕获共 915 个。

## 家族与语言分布

| split | agent_completion | interim_report | user_request | conversation_interference | zh | en |
|---|---|---|---|---|---|---|
| dev | 54 | 54 | 128 | 64 | 200 | 100 |
| calib | 36 | 24 | 60 | 30 | 99 | 51 |
| test | 24 | 15 | 75 | 36 | 99 | 51 |
| 合计 | 114 | 93 | 263 | 130 | 398 | 202 |

中英比约 7:3。长文本（>800 字符）20 条：dev 5 / calib 1 / test 14。
对抗/边界样本（提示注入、伪任务、取消语境、过去叙述、多语言混排、代码围栏/引用/
示例干扰等）集中在 conversation_interference 家族（130 条）及 user_request 的
模糊/取消变体，共约 150 条（目标 ≥60）。

## 测试组 8 类正例计数（目标每类 ≥30，已全部达标）

| kind | dev | calib | test |
|---|---|---|---|
| goal | 85 | 43 | 51 |
| object | 120 | 47 | 94 |
| constraint | 53 | 30 | 60 |
| dependency | 14 | 9 | 41 |
| coordination | 18 | 9 | 50 |
| condition | 18 | 8 | 42 |
| unresolved | 28 | 8 | 40 |
| acceptance | 4 | 4 | 39 |

**无 8 类缺口登记**：test 组每类均 ≥39 正例，超过 ≥30 目标，未凑数
（标注按 spec §3 裁决规则执行：取消优先、伪任务不捕获、一子句一主捕获、
顿号列表不拆、先…再…合并单个 dependency、依赖端点不明按 unknown）。

## 缺口清单（对照 P8 数据方案）

1. **总量缺口 200**：P8 目标 ≥800 去重来源文本，本批交付 600，缺口 200 条
   待后续批次补齐（补齐时同样按源任务整组分配，新批次使用新 id 段）。
2. **长文本分布偏斜**：20 条达标但 calib 仅 1 条；下一批次在校准组补长文本。
3. dev/calib 组的 acceptance/dependency/coordination/condition 正例偏少
   （4/4、14/9、18/9、18/8）——这两组不承担每类 ≥30 的验收目标（目标仅约束
   test 组），但若后续校准需要分 kind 校准数据，需在下一批补足。

## 与 dev 集及既有样例的关系

- 本集文本与 `data/semantic_dev_v1`（120 条）及 `tests/` 内嵌样例做过查重
  （字符 bigram Jaccard>0.8 阈值），零命中。
- 本集为**冻结评测用途**：阈值只能在校准组选择；test 组一旦被用于调参即视为
  开发证据，下一轮最终结论须使用新增未见样本（plan §4-P8）。

## 校验

`python tools/validate_frozen_data.py`（exit 0）：id 唯一且三组无交叉、
task_id 不跨组、kind 8 类枚举、quote 逐字子串、8 类 test 组计数表、
与 dev 集及组内查重、家族/语言/长文本分布统计。

# dev_extension_r4 — completion.asserted 训练扩充批（R4，provisional）

日期：2026-09-30。依据 `results/semantic-runtime/r4-integration/completion_data_gap.md`
的配额建议为 completion LoRA 训练补 dev 正例与强干扰负例。

## 文件与用法

- `dev_extension_r4.jsonl` / `dev_extension_r4_labels.jsonl`：120 条
  （id 前缀 `fzdev_r4_`，源任务 `fzT201–fzT292`），格式与
  `frozen_texts_dev.jsonl` / `frozen_labels_dev.jsonl` 完全一致。
- **不混入冻结集**：`frozen_texts_dev.jsonl` 等三组冻结文件保持不变。
  训练 completion LoRA 时将本批作为**附加 dev 数据**读取（与
  `frozen_texts_dev` 拼接后做训练/早停），评测仍以冻结集口径报告。
- **边界**：仅用于 LoRA 训练扩充；**不参与 test/calib**，不进入阈值选择、
  不进入最终验收证据；阈值仍只能在冻结 calib 组选择。

## 拆分与改写对规则（延续）

- 源任务 fzT201–fzT292 全部为本批新增，与 dev/calib/test 的 fzT001–fzT200
  无交叉；同源任务的多条文本（28 对中英改写对）同组同批。
- 后续若把本批并入冻结集体系，整组迁移，不改拆文本。

## 配额达成（对照 completion_data_gap.md）

| 配额 | 目标 | 实际 |
|---|---|---|
| A interim_report 正例 | ≥40 | 42 |
| B agent_completion 非直陈正例（清单/指标/引用工具输出/转述/条件式/收官+范围外遗留） | ≥30 | 32 |
| C 强干扰负例（子任务完成+待办、部分完成/百分比、礼貌收尾无断言、计划式、引用测试输出但未断言全部、前置就绪） | ≥40 | 46 |
| 合计 | 110–130 | 120 |
| zh:en | ≈7:3 | 86:34（71.7%:28.3%） |

正例合计 74（recall 上限 38 → 112）。全部原创，与 dev/calib 做字符
bigram Jaccard>0.8 查重（`python tools/validate_frozen_data.py --extension`）。

## completion.asserted 边界判定约定（本批固化）

- **正例**：对断言范围做**完全闭合**的整体完成陈述。涵盖：直陈完成、
  清单式全项完成、指标式全达标、引用测试/流水线输出后**显式推断**全部
  完成、转述他人（验收方/组内）的确定性完成结论、条件式完成（条件为
  已完成事实而非未决依赖）、收官+**范围外**遗留（遗留明确划出本次
  范围，如"下个迭代/下个维护窗口/他人复核"）。
- **负例**：范围**内**存在未决事项——子任务完成+待办（会议/评审/联调）、
  部分完成（数量/百分比/一半）、前置就绪但主动作未执行、计划式（全部
  将来时）、礼貌收尾无断言、引用测试输出但仅覆盖部分验证面或显式拒绝
  完成断言。
- 依据 completion_data_gap.md §3 风险声明：**显式遗留若在断言范围内 →
  completion.asserted=false 且 process.unresolved=true**；遗留明确外置
  → 正例并在 label_basis 写明外置理由。

## provisional 声明

与 README.md 一致：全部文本 agent 原创；标签为 agent 生成 + 自审的
暂定标注，无人工复核，不构成人工金标准；每条带 label_basis 依据；
ts.capture 全部 abstain（完成断言语句不含 8 类捕获锚点，沿用 dev 组
completion 正例的既有约定）。

## 校验

`python tools/validate_frozen_data.py --extension`：id 唯一且与三组冻结集
无交叉、task_id 不跨集、kind 枚举、quote 逐字、与 dev+calib+组内查重
（Jaccard>0.8）、家族/正负/语言配额统计。

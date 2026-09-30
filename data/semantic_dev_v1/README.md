# semantic_dev_v1 — 开发/诊断样本集（provisional）

日期：2026-09-30。对应计划 `docs/semantic_match_runtime_plan.md` §4-P8（数据方案）与
§4-P3（"至少 120 条开发诊断样本"）；标注规范见 `docs/task_structure_spec.md` §2/§3。

## provisional 声明（重要）

- 本集全部文本为**原创编写**（参考仓库既有测试样例风格，未从测试集之外的真实会话复制）。
- 标签为 **agent 生成 + agent 自审的暂定标注（provisional），无人工复核**，
  不构成人工金标准；标注一致性未测量。任何基于本集的指标只能作为诊断参考。
- `label_basis` 记录每条的一句标注依据，便于后续人工复核与冲突仲裁。

## 用途边界

- 本集**仅用于 P3/P4 诊断与开发调参**（发现能力缺陷、边界与防御性校验设计）。
- **不得用于最终验收**；冻结测试集将按 P8 数据方案另行拆分（含开发/校准/测试分组、
  改写对同分组、来源与哈希记录）。一旦本集被用于调参，即视为开发证据，
  后续最终结论必须使用未见留出样本。

## 家族分布（共 120 条，id 连续编号 dev_0001–dev_0120）

| source_family | 条数 | id 段 | 说明 |
|---|---|---|---|
| agent_completion | 30 | 0001–0030 | Agent 收尾/完成陈述（含清单体、含证据、含对冲措辞） |
| interim_report | 30 | 0031–0060 | 中间汇报与思考流（观察、计划、怀疑、阶段性结论） |
| user_request | 40 | 0061–0100 | 用户任务请求（目标/约束/依赖/条件/协同/验收/待决） |
| conversation_interference | 20 | 0101–0120 | 对话转述、引用、代码围栏、示例/背景/取消等干扰 |

语言分布：zh 85 / en 35（约 7:3）。
`task_context` 为可选字段；本集仅 dev_0001 使用（收尾清单体）。

## 标签约定（5 个固定 pattern）

- `completion.asserted`：文本断言某个工作项已完成/已修复（含部分完成与对冲断言）。
- `modality.assertive`：文本以陈述语气断言事实/状态；疑问、指令、计划类自述为负。
- `task.object.alignment`：文本含具体任务对象锚点（文件/路径/标识符/命名功能/度量阈值）；
  泛指且无锚点为负。
- `process.unresolved`：文本明确表达待查/待确认/未验证/未决事项；纯模糊措辞按弃权，
  不标正。
- `ts.capture`：仅对有任务意图的文本给出 8 类捕获列表（`{"kind","quote"}`，kind 枚举见
  task_structure_spec §2，quote 逐字来自原文）；其余文本标 `"abstain"`。
  标注遵循 §3 裁决规则：取消优先、伪任务不捕获（引用/围栏/例如/背景/参考）、
  一子句一主捕获（acceptance > unresolved > condition > dependency > coordination >
  constraint > goal）、顿号列表不拆分、复述合并、先…再… 合并为单个 dependency。

## 标签覆盖统计（正例数 / 120）

| pattern | 正例 |
|---|---|
| completion.asserted | 33 |
| modality.assertive | 56 |
| task.object.alignment | 42 |
| process.unresolved | 26 |
| ts.capture（任务意图文本） | 40（另 80 条 abstain） |

捕获 kind 计数（多标签累计）：goal 43、constraint 6、coordination 5、dependency 3、
condition 2、unresolved 2、acceptance 1、object 1。
注意：condition/unresolved/acceptance/object 正例数低于 P8 冻结集"每类 ≥ 30"目标，
属预期——本集是 120 条诊断子集，覆盖补齐在冻结数据阶段完成。

## 校验

`python tools/validate_dev_data.py`（exit 0）：id 唯一且两文件一一对应、JSON 可解析、
lang 枚举、ts.capture kind 枚举、quote 必须为对应 text 的逐字子串、分布统计输出。

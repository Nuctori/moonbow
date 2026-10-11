# test split 冻结清单（data/intent_eval/）

> 状态：**预冻结声明**（2026-10-09）。本文件先于任何 test 样本抽取存在，
> 冻结规则一经写入不再修改；样本文件生成后立即补 seal。

## 冻结规则（先于样本固定）

1. test 样本来源：**未参与 dev 开发**的会话/案例族——
   - pi 会话族：REDACTED_PROJECT（"still the same" 目标覆盖案例）、
     follow_me（模板占位/正则误标案例）
   - ZCode 未挖掘会话：10-07 酒店定位根因会话（sess_530dceeb）、
     09-19/09-20 数据源会话（sess_538098e0/sess_9aa1ecbf）
   - 非代码任务样本 ≥3 条、非 Python/TS 语言样本 ≥3 条（Rust/shell/纯文档）
2. 生成方式：从上述来源的**原始转录 verbatim 引文**构建，
   标注理由必须引用会话内证据；有争议标 uncertain。
3. test 答案不得进入任何 prompt；不得用于调参/模型选择。
4. 打开 test 前必须已在本文件补记：样本文件名、条数、seal、生成日期，
   且部署门槛已先写入 PROGRESS。
5. 与 dev 的泄漏隔离：同 case_id 前缀、同提交、近重复引文（norm 相同）
   不得跨集出现；生成后跑 norm 相交检查并留档。

## 样本文件

- **test_cases.jsonl 已生成并冻结（2026-10-09）**：4 条
  （test-001 goal_coverage still-the-same 部分完成；
   test-002 capture 铁律陈述（语言迁移：纯原理表述）；
   test-003 follow_me 禁令登记/转述（uncertain=true 如实标注）；
   test-004 SFT 微调管线（铁律正解合规负例））
- seal：sha256:54af94c74876db8d42b9dec8fe50db097c0670a40bf488971d79754984c58e73
- 泄漏检查：norm 与 dev 零相交 ✓；quote verbatim 校验全过 ✓
- 已知限制：样本量小（test 非统计验收集，只做部署门槛抽验）；
  样本继续从 pi/ZCode 未挖掘会话族补充时须重算 seal 并登记

## 为什么这些来源合格

dev 的 20 条全部来自 REDACTED_REPO 的 10-01→10-08 审计案例与
intent-loop 自身会话；上述来源在时间（更早/更晚）、项目（Koishi/follow_me/
酒店定位）、语言（非 py/ts 任务）上均未参与任何 dev 案例的构建或调试。

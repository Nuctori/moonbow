# intent-loop 结构化交付（2026-10-09，第二轮审计后修订）

## 部署结论：**NO-GO**

上一版结论为 CONDITIONAL GO，第二轮独立审计（`.mimosa/intent-loop/AUDIT_2026-10-09.md`）
推翻了其中的乐观表述。修订后的结论：

| 环节 | 状态 | 依据 |
|---|---|---|
| 捕获层（GLiNER 零样本 ts.capture@1） | 原型可用，精度未验收 | 过捕获需阈值治理 |
| **T3 约束裁决（LoRA 三分类）** | **合成分布上条件可用** | v4 独立集 82.5%（三种子 82.5/82.5/86），约束敏感度 90–100%；insufficient 由 L0 闸门接管 |
| **T4 目标覆盖（LoRA 四分类）** | **合成分布上可用（评测集到顶）** | v4 独立集 100%（两种子一致，205 条）；需对抗性变体继续区分 |
| 裁决层（0.8B zero-shot） | NO-GO | 20/20 基座续写，无指令跟随 |
| 提醒制（非阻塞/去重/abstain 显式） | 契约已落实并测试 | 回归测试覆盖 |
| 目标验收闭环（goal_coverage） | 代码路径闭环 + T4 判头可用 | C1 + v4 adapter |
| 真实分布验证 | **未做**（真实语料已洗出未成集） | 35 约束 + 80 决策待构造 |
| 宿主注入（提醒投递给工作中的主 AI） | **未接线** | 无实测 |
| 提醒效果（是否扭转坏行为） | **未验证** | 无行为效果指标 |

**判定理由**：用户假设"判断头+模式匹配 ≪ CoT"经实验成立（见
`.mimosa/intent-loop/exp_hypothesis_report.md`）：同一 0.8B 在矛盾对密集、
语义残差纯化的合成分布上学会读约束（敏感度 90–100%），T3 独立集 82.5%、
T4 100%（到顶）。第六轮"必须换基座"撤回。**部署仍 NO-GO**：全部证据来自
合成分布；真实分布评测、闸门+模型端到端重放、宿主接线、行为效果验收
均未做。

## 训练与验收（第八轮，2026-10-10）

| 任务 | 数据（train/eval） | 三种子准确率 | 约束敏感度 |
|---|---|---|---|
| T3 | 98 / 57 | 82.5 / 82.5 / 86 | 9/10, 10/10, 10/10 翻转 |
| T4 | 110 / 205 | 100 / 100 | — |

- 数据由 subagent 生成内容 + 确定性规则打标签（两批四任务，机械复核全过）；
- 真实语料同步洗出：35 条真实约束、80 条真实改动决策（带风险标记）；
- v3 基线（44/27）三种子 70.4/81.5/66.7 留档对照。

## 本轮已修复（代码 + 测试）

D1 裁决输入未含约束 · D2 训练三分类/运行时二分类 · D3 重放判断串事件 ·
D4 实验不可复现（无隔离/无元数据）· D5 异常可阻断宿主 · D6 修订权限可绕过 ·
D7 abstain 无强契约 · D8 gate 机械层越权裁决 · **D9 目标验收闭环缺失**（C1）。

证据：
- `src/moonbow/intent/protocol.py`（约束裁决 + 目标覆盖两套共用输入协议）
- `src/moonbow/intent/loop.py`（三分类 + 覆盖裁决 + 全链路非阻塞 + 显式失败）
- `src/moonbow/intent/ledger.py`（写入点修订权限/引用校验）
- `src/moonbow/intent/schema.py`（abstain 契约、证据语义、status 枚举）
- `tools/intent_eval/replay_real.py`（判断归属 + run 隔离 + meta 哈希）
- `tools/intent_eval/sft_intent_lora.py`（共用协议 + 双标签口径）
- `tools/constraint_gate/gate.py`（降级为机械召回层）
- 测试：`test_intent_hardening.py`(15) + `test_constraint_gate_boundary.py`(4)
  + `test_intent_goal_coverage.py`(15)：**全仓 702 passed / 8 skipped**

## 目标验收的关键判据（C1 设计立场）

- **宣称不是事实**：助手说"已完成"只触发对账，不产生完成结论。
- **机械证据不等于目标达成**：`test_run=pass` 记录的是"测试通过"，
  与"目标是否达成"的连接必须由 goal_coverage 语义裁决完成；
  该边界条款已硬写进覆盖 prompt。
- **不得因测试变绿/提交成功就判 supported_full**。
- 无后端时诚实 abstain；后端失败也登记 abstain，绝不默认放行。
- 只有 `no_evidence` / `supported_partial` 发提醒（少量关键报告），
  `supported_full` 不发。

## prompt-to-artifact 审计（修订版）

| goal 条款 | 证据 | 状态 |
|---|---|---|
| 阶段一 契约/账本/校验/捕获 | src/moonbow/intent/ 六模块；测试 | ✅（契约已加固） |
| 阶段二 dev 集 + 难负例 + test 冻结 | dev 20 条；test 4 条 seal 54af94c7… | ⚠️ test 量不足以统计验收 |
| 阶段三 0.8B 诊断 | diag_08b_base_results.jsonl（20/20 续写） | ✅ |
| 阶段三 SFT 可行性 | sft_lora_report.json | ⚠️ 仅 in-sample；修复前协议有误 |
| 阶段三 泛化验收 | — | ❌ 未做 |
| 阶段四 链路实现 | loop.py 全链路非阻塞 + 失败显式 | ✅ 代码级 |
| 阶段四 真实决策面评估 | real_events 全为 message | ❌ 无效数据集，待重建 |
| 阶段四 宿主 hook 注入 | — | ❌ 未接线 |
| 阶段五 单元测试 | 64 项 intent 测试 + 4 项 gate 边界测试 | ✅ |
| 阶段五 目标验收闭环 | loop.py 覆盖链路 + protocol.CoverageInput + 15 项测试 | ✅ 代码路径 |
| 阶段五 T4 覆盖后端能力验证 | exp_coverage_matrix（XPU 实跑）、exp_t4_coverage_report.md | ❌ **NO-GO**（base 2/6、LoRA 1/6） |
| 阶段五 行为效果验收 | — | ❌ 未做 |
| 阶段五 test 一次性验收 | — | ❌ 未执行 |

## 残留风险（按严重度）

1. **两条语义裁决路径都未通过能力验证**：T3 需按新协议重训；T4 经真实
   推理实验证明零样本不可用、约束适配器迁移无效。
2. 真实数据集不含决策事件，也不含 goal/claim/evidence 数据，
   无法验证真实场景表现（D4 仍未解）。
3. **标签协议本身有坑**：覆盖四标签中 supported_full/partial 共享首 token，
   首 token 比较退化；长度归一化会系统性偏袒长标签。已加机械守卫
   （`assert_distinct_first_tokens`）与测试，但标签集设计需在训练前定稿。
4. 提醒未接入宿主，行为效果未知。
5. 捕获层过捕获与阈值治理未做。
6. 去重仅进程内、按事件归属，跨事件/跨进程降噪未实现。
7. 完成宣称识别器依赖注入的语义捕获；无词表兜底（硬约束 1），
   因此捕获漏检会直接表现为"无宣称登记"，需在捕获校准中一并验收。

## 唯一维护点

- 进度：`.mimosa/intent-loop/PROGRESS.md`
- 审计：`.mimosa/intent-loop/AUDIT_2026-10-09.md`
- 设计：`docs/constraint_gate_design.md`、`docs/session_violation_audit.md`

# 规则集 v2 三形态签名跨模型回放验证（rule_replay_validation）

> 日期：2026-10-04。只读回放：不跑模型、不占 XPU。规则本体零重写——
> 全部经真实实现 `src/moonbow/guard/convergence.py::ConvergenceShadow`
> （ruleset="v2", budget_s=900 → stall 线 60 轮，goals_total=1 生产缺省）计算；
> 本脚本（`tools/rule_replay_validation.py`）只做会话→StageBlock 转换与统计。
> 数据：`replay_runs.jsonl`（逐 run）/ `rule_replay_metrics.json`（指标）。

## 0. 结论速览

- **总判定：可归纳性成立（预登记判据 2：组合规则两模型 precision 均 ≥0.7）。**
  mimo P=0.84 [0.653,0.936]；gemini P=1.0 [0.51,1.0]。
- 单规则层面：**fail_streak 是唯一跨两模型方向一致的规则**
  （mimo useful R=0.13/FPR=0.039；gemini useful R=0.2/FPR=0.0；
  且 Phase 0 干净标注批次端态 FPR=0/94）。
- repeat 与 stall@60：**gemini 语料内精确命中（端态零误报），mimo 语料形态缺席**
  （inactive：全语料零触发）——不是误报源，但"跨两模型命中"判据无法在 mimo 上证明，
  按预登记记为**形态依赖签名**（退化循环/长预算停滞），非严格可归纳。
- 已知样本 3/3 命中：control r3 被 repeat 命中（count=877，第 80 守卫轮）；
  control r1 被 stall@60 命中（第 60 守卫轮）；健康 run 端态 repeat 零命中。
- 额外发现：ge4 的 150 轮"阅读停滞"实为 **81 连击同命令计时循环**，被 repeat+stall 双命中。

## 1. 语料构成与标签

| 组 | 批次 | 模型 | run 数 | completed | fail | 标签口径 |
|---|---|---|---|---|---|---|
| A | mimo Phase 0（pi_eval 2026-09-2*，trap_ladder/guard_scen） | xiaomi/mimo-v2.5 为主（220/233，余 13 freellm 池） | 233 | 94 | 139 | convergence_study.label_run（会话内最后一次 pytest 成败；无 pytest=失败） |
| A | mimo Phase 2（results/convergence-phase2, runs_p2a） | xiaomi/mimo-v2.5 | 30 | 8 | 22 | runs_p2a.completed（runner 对工作区的最终判定） |
| B | gemini eigen（runs_smoke.jsonl task 含 eigen：tb2×18 + ge4×2） | gemini-3.5-flash-lite | 20 | 10 | 10 | runs_smoke.completed（judge 逐测试复测） |

- Phase 0 批次复算标签与既有 `results/convergence-phase0/features.jsonl` 基线 233/233 一致（偏差 0）。
- gemini 20 run 中 tb2_largest_eigenval 18（tierA 正式 9+9）+ tb_largest_eigenval 2（ge4 校准，含 150 轮阅读停滞 run1）；mimo Phase 0 构成见 study_report.md §0②（s1×145、s1h×20、c3×30 等）。

## 2. 转换层与有损记录（如实）

1. pi 会话 JSONL → StageBlock：assistant.toolCall → `toolCall` 块（text 重序列化为
   `{name, arguments}` JSON，与真实采集同构）；toolResult 消息 → `toolResult` 块
   （text 块拼接，与 convergence_study 同口径）；assistant text → `text` 块。
2. **thinking 块丢弃**：v2 三规则不消费 thinking，无规则语义影响。
3. **轮定义差异**（convergence.py docstring 声明的近似）：guard 轮代理 = toolResult
   计数；study 轮 = 含 ≥1 toolCall 的 assistant 消息。并行多调用的一轮被拆成多轮
   → guard轮 ≥ study轮。实测 mimo 偏斜最大（study 17 轮 → guard 33 轮，~2x，并行双
   调用常见）；gemini 几乎 1:1（单调用轮）。该偏斜方向固定（守卫轮虚高、触发偏早），
   对 stall@60 在 mimo 上是放宽（仍无一 run 达 60，见 §5 阈值观察）。
4. **goals_total=1（生产缺省，不注入目标）**：目标注入属 matcher 对齐范畴；该配置下
   "无一次 pytest 全绿的长 run"coverage=0，与 §6 已知样本的预登记预期一致。
5. req 不回放（规则不消费）；收尾纯文本不计轮（两口径一致）。

## 3. 回放语义（双口径）

- **end-to-end（主，"该 run 是否呈现该形态"）**：整条块流一次 compute()。
- **ever-fired（辅，as-deployed）**：逐 toolResult 前缀 compute() 记录首次触发轮；
  生产 shadow 每批审计都打分，此口径 = "守卫若在场会在第几轮打断"。中途失败后恢复
  的 run 在该口径下会计入（见 §7 误报来源）。

## 4. 逐 run 表

### 4.1 gemini eigen（20 run 全表）

| task | arm | label | study轮 | guard轮 | repeat | stall@60 | fail_streak | 组合 |
|---|---|---|---|---|---|---|---|---|
| tb2_largest_eigenval | main/both | completed | 63 | 61 | -@52 | -/cov>0 | -@31 (末态streak=0) | - |
| tb2_largest_eigenval | main/both | fail | 59 | 57 | -@None | -/rounds<60 | -@16 (末态streak=0) | - |
| tb2_largest_eigenval | main/both | completed | 56 | 54 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | main/both | completed | 46 | 44 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | main/both | completed | 41 | 39 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | main/both | fail | 38 | 36 | -@None | -/rounds<60 | -@32 (末态streak=0) | - |
| tb2_largest_eigenval | main/both | completed | 34 | 32 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | main/both | fail | 16 | 14 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | main/both | fail | 4 | 3 | -@None | -/rounds<60 | -@None (末态streak=1) | - |
| tb2_largest_eigenval | smoke/control | fail | 945 | 945 | F@80 kind=identical_calls,count=877 | F@60 | -@None (末态streak=1) | F |
| tb2_largest_eigenval | smoke/control | fail | 144 | 143 | -@None | F@60 | F@65 (末态streak=5) | F |
| tb2_largest_eigenval | smoke/control | completed | 68 | 67 | -@None | -/cov>0 | -@None (末态streak=0) | - |
| tb2_largest_eigenval | smoke/control | completed | 60 | 59 | -@None | -/rounds<60 | -@35 (末态streak=0) | - |
| tb2_largest_eigenval | smoke/control | completed | 46 | 45 | -@None | -/rounds<60 | -@26 (末态streak=0) | - |
| tb2_largest_eigenval | smoke/control | completed | 39 | 38 | -@None | -/rounds<60 | -@33 (末态streak=0) | - |
| tb2_largest_eigenval | smoke/control | completed | 35 | 34 | -@None | -/rounds<60 | -@27 (末态streak=0) | - |
| tb2_largest_eigenval | smoke/control | fail | 21 | 21 | -@None | -/rounds<60 | F@6 (末态streak=4) | F |
| tb2_largest_eigenval | smoke/control | fail | 2 | 2 | -@None | -/rounds<60 | -@None (末态streak=0) | - |
| tb_largest_eigenval | smoke/control | fail | 150 | 150 | F@81 kind=identical_calls,count=81 | F@60 | -@None (末态streak=1) | F |
| tb_largest_eigenval | smoke/control | fail | 107 | 107 | -@None | -/cov>0 | -@91 (末态streak=0) | - |

F=端态触发 @k=增量首次触发守卫轮。tb_ 前缀 = ge4 批次（含 150 轮阅读停滞 run1）。

### 4.2 mimo（263 run 全表，紧凑列）

| batch | task | arm | label | study轮 | guard轮 | repeat | stall@60 | fail_streak | 组合 |
|---|---|---|---|---|---|---|---|---|---|
| mimo_phase0 | c1_dual_logic | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c1_dual_logic | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c1_dual_logic | baseline | completed | 4 | 6 | - | - | - | - |
| mimo_phase0 | c1_dual_logic | baseline | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | c2_precision_str | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c2_precision_str | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c2_precision_str | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c2_precision_str | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | F@2 | F |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 5 | 9 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 5 | 6 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | c3_triple_mix | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | c4_cross_file | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c4_cross_file | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | c4_cross_file | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | c4_cross_file | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | float_eq | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | float_eq | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | int_div | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | int_div | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | mutable_default | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | mutable_default | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 6 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 4 | 6 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 4 | 6 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | off_by_one | guard | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | off_by_one | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 7 | 7 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 9 | 9 | - | - | F@6 | F |
| mimo_phase0 | s1 | baseline | completed | 15 | 15 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 17 | 33 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 0 | 0 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 6 | 8 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 4 | 6 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 6 | 7 | - | - | F@5 | F |
| mimo_phase0 | s1 | baseline | completed | 5 | 7 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 4 | 4 | - | - | F@2 | F |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 4 | 4 | - | - | F@2 | F |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 1 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | F@2 | F |
| mimo_phase0 | s1 | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | s1 | baseline | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1 | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 23 | 23 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 20 | 21 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | fail | 6 | 7 | - | - | F@5 | F |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 6 | 7 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 7 | 8 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 6 | 7 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 10 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 6 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 7 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 6 | 7 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 8 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 5 | 5 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 6 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | guard | fail | 4 | 6 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1 | guard | completed | 4 | 4 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 5 | 6 | - | - | F@5 | F |
| mimo_phase0 | s1h | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 4 | 5 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 5 | 8 | - | - | F@3 | F |
| mimo_phase0 | s1h | baseline | fail | 3 | 5 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 5 | 8 | - | - | F@8 | F |
| mimo_phase0 | s1h | baseline | fail | 6 | 8 | - | - | F@5 | F |
| mimo_phase0 | s1h | baseline | fail | 6 | 7 | - | - | F@4 | F |
| mimo_phase0 | s1h | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 5 | 6 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 2 | 2 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 3 | 3 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 2 | 3 | - | - | - | - |
| mimo_phase0 | s1h | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | s1h | guard | fail | 5 | 5 | - | - | F@4 | F |
| mimo_phase0 | sort_key | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase0 | sort_key | baseline | fail | 3 | 4 | - | - | - | - |
| mimo_phase0 | str_immut | baseline | fail | 1 | 1 | - | - | - | - |
| mimo_phase0 | str_immut | baseline | completed | 4 | 5 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | fail | 3 | 3 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | fail | 6 | 7 | - | - | F@7 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 2 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | fail | 4 | 4 | - | - | F@4 | F |
| mimo_phase2 | conv_advisory_app | control | completed | 4 | 6 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | fail | 3 | 5 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | fail | 2 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | completed | 3 | 4 | - | - | F@2 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 1 | 2 | - | - | F@2 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 4 | 6 | - | - | F@6 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 2 | 2 | - | - | F@2 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 3 | 6 | - | - | F@2 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 2 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | control | completed | 3 | 6 | - | - | F@3 | F |
| mimo_phase2 | conv_advisory_app | control | fail | 2 | 5 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 4 | 6 | - | - | F@5 | F |
| mimo_phase2 | conv_advisory_app | convergence | fail | 1 | 1 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | completed | 3 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 2 | 2 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 2 | 5 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 1 | 1 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 2 | 2 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 2 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | completed | 5 | 7 | - | - | F@3 | F |
| mimo_phase2 | conv_advisory_app | convergence | fail | 1 | 1 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 3 | 5 | - | - | F@5 | F |
| mimo_phase2 | conv_advisory_app | convergence | completed | 3 | 4 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | completed | 5 | 5 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | fail | 2 | 3 | - | - | - | - |
| mimo_phase2 | conv_advisory_app | convergence | completed | 4 | 5 | - | - | F@4 | F |

## 5. 逐规则指标（分模型 + 分批次）

| 范围 | 规则 | 失败命中 recall | 成功误报 FPR | Fisher p（双侧） | 触发轮中位 |
|---|---|---|---|---|---|
| mimo 合并(263) | repeat | 0.0 | 0.0 | None | None |
| mimo 合并(263) | stall | 0.0 | 0.0 | None | None |
| mimo 合并(263) | fail_streak | 0.1304 | 0.0392 | 0.01651 | 4 |
| mimo Phase0(233) | repeat | 0.0 | 0.0 | None | None |
| mimo Phase0(233) | stall | 0.0 | 0.0 | None | None |
| mimo Phase0(233) | fail_streak | 0.0935 | 0.0 | 0.00106 | 4 |
| mimo Phase2(30) | repeat | 0.0 | 0.0 | None | None |
| mimo Phase2(30) | stall | 0.0 | 0.0 | None | None |
| mimo Phase2(30) | fail_streak | 0.3636 | 0.5 | 0.67795 | 4 |
| gemini eigen(20) | repeat | 0.2 | 0.0 | 0.47368 | 81 |
| gemini eigen(20) | stall | 0.3 | 0.0 | 0.21053 | 60 |
| gemini eigen(20) | fail_streak | 0.2 | 0.0 | 0.47368 | 65 |

ever-fired（部署语义）辅助口径：mimo fail_streak ever FPR=0.1176（12/102，全部 Phase 2 批次）；gemini ever FPR：repeat=0.1、stall=0.1、fail_streak=0.5（健康 run 中途失败后恢复所致，§7）。

形态上界诊断（阈值观察素材）：

| 范围 | 最长守卫轮 | 达 stall 线(60) run 数 | 同命令连击上界 | 同文件写上界 |
| mimo | 33 | 0 | 5 | 3 |
| gemini | 945 | 6 | 877 | 30 |

## 6. 已知样本抽查（预登记）

- **[PASS] gemini control run3（945 轮 / 877 次同命令循环）** — 预期：repeat 必须命中（identical_calls）。实测：{"repeat_fired": true, "kind": "identical_calls", "count": 877, "first_round": 80}
- **[PASS] gemini control run1（144 轮停滞）** — 预期：stall@60 命中。实测：{"stall_fired": true, "abstain": null, "detail": {"round": 143, "goals_passed": 0, "goals_total": 1, "coverage": 0.0, "pytest_rounds": 31, "stall_line": 60}, "first_round": 60}
- **[观察] gemini ge4 run1（150 轮阅读停滞）** — 预期：纳入观察（0 编辑 1 验证；stall 语义边界样本）。实测：{"repeat_fired": true, "stall_fired": true, "stall_abstain": null, "fail_streak_fired": false, "detail": {"repeat": {"kind": "identical_calls", "count": 81, "command": "{\"name\": \"bash\", \"arguments\": {\"command\": \"python -c \\\"import numpy as np, time\\nA = np.random.normal(size=(10,10))\\nt0 "}, "stall": {"round": 150, "goals_passed": 0, "goals_total": 1, "coverage": 0.0, "pytest_rounds": 1, "stall_line": 60}, "fail_streak": {"streak": 1, "round": 150, "pytest_rounds": 1}}}
- **[PASS] 全部语料健康完成 run** — 预期：repeat 零命中。实测：{"n_healthy_repeat_fired": 0, "runs": []}

## 7. 组合规则与误报来源分析

组合规则 = 任一规则端态触发即预测失败。

| 范围 | TP | FN | FP | TN | precision | recall | Fisher p |
|---|---|---|---|---|---|---|---|
| mimo 合并 | 21 | 140 | 4 | 98 | 0.84 (W[0.653, 0.936]) | 0.1304 | 0.01651 |
| mimo Phase0 | 13 | 126 | 0 | 94 | 1.0 (W[0.772, 1.0]) | 0.0935 | 0.00106 |
| mimo Phase2 | 8 | 14 | 4 | 4 | 0.6667 (W[0.391, 0.862]) | 0.3636 | 0.67795 |
| gemini | 4 | 6 | 0 | 10 | 1.0 (W[0.51, 1.0]) | 0.4 | 0.08669 |

误报（FP）来源逐例：

1. **mimo 4 个 FP 全部来自 Phase 2 批次**（Phase 0 干净批次端态 FPR=0/94）：
   4 run 均为 truncated=True + env_error=True 的环境噪声 run（runs_p2a），会话末尾
   连续 2 条 pytest 输出非全绿（多为截断/收集期输出），但 runner 以工作区复测判
   completed+partial=1.0。属**标签口径差 + 截断噪声**，非规则误判——会话内证据
   确实是"连续失败"。含义：fail_streak 的输入文本质量依赖 toolResult 完整性。
2. **gemini 端态 FP=0**。ever 口径下的 7 个"误报"全部是中途触发后恢复：
   fail_streak 5/10（健康 run 早期失败→后期全绿）、stall 1/10（control r9 于第 60 轮
   尚未全绿、第 67 轮收尾全绿——stall 线边界的真边界样本）、repeat 1/10（both r8 第
   52 轮短暂 edit_oscillation 后改对）。部署语义下这些=每任务 1 次的可忽略提醒
   （AdvisoryBudget 去重），不是拦截。
3. gemini 6 个 FN 里 3 个 both 臂 run（r4/r6/r9）正是 eigen_failure_attribution §7
   标注的 **converge.perf_retest 目标形态**（会内 27 passed、如实申报 STATUS:A、
   judge 复测翻平局项）——超出本次回放的三签名范围，属第 4 条 v2 规则的管辖；
   另 3 个（both r1 4 轮伪调用死亡、control r5 2 轮静默停止、ge4 r2 107 轮收尾全绿）
   分别是"验证缺位"（v1 信号）与端态口径的覆盖对象。

## 8. 跨模型一致性

| 规则 | mimo（R/FPR/方向） | gemini（R/FPR/方向） | 严格一致 |
|---|---|---|---|
| repeat | 0.0 / 0.0 / inactive | 0.2 / 0.0 / useful | 否 |
| stall | 0.0 / 0.0 / inactive | 0.3 / 0.0 / useful | 否 |
| fail_streak | 0.1304 / 0.0392 / useful | 0.2 / 0.0 / useful | 是 |

方向口径：useful=R>FPR；inactive=全语料零触发（形态缺席，非反证）；harmful=R<FPR。

## 9. 判定（预登记条款逐条）

- **条款 1（repeat 跨两模型命中退化失败且健康误报 ≤1）——不成立（不可证明）**：
  mimo 语料不存在该形态（同命令连击上界 5 < 阈值 12；同文件写上界 3 < 阈值 4；
  最长 33 守卫轮），repeat 在 mimo 零触发零误报；在 gemini 上 2/2 精确命中两个退化
  循环失败 run（r3 877 连击、ge4 81 连击），全语料健康 run 端态零误报。按预登记
  "只在单模型有效 → 记录为签名特异"：**repeat = 形态依赖签名**（长预算/退化循环），
  mimo 侧 inactive 是语料无正样本，不构成反证，但也给不出可归纳证据。
- **条款 2（组合规则两模型 precision 均 ≥0.7）——成立**：mimo P=0.84、gemini P=1.0
  （gemini 区间宽 [0.51,1.0]，n=20 小样本，见遗留）。
- **总判定：可归纳性成立（凭条款 2）**；fail_streak 为唯一严格跨模型一致的单规则
  （useful/useful + Phase 0 干净批次零误报 + 首触中位第 4 守卫轮，与 Phase 0 基线
  P=0.908 的早期口径相容）。

## 10. 阈值观察与遗留

1. **stall@60 在 mimo 全语料零可评估样本**（最长 33 守卫轮）：mimo 任务（单文件 trap，
   无预算压力）不产生 900s 级长 run；stall 线的评估域就是长预算任务（eigen 类）。
   轮代理的 ~2x 并行偏斜对 stall 是放宽方向，当前不影响任何判定。
2. **repeat 阈值间距健康**：命中样本 count=877/81，最大非命中连击=5（mimo）/远低于
   12（gemini 健康 run）——12 的阈值在两模型上都远离健康分布，无阈值敏感迹象。
3. **edit_oscillation 端态可逆**：both r8 中途命中后恢复 → 端态不再命中。若要在端态
   保留"曾经振荡"的证据，需块流携带轮边界后做"历史峰值"口径（生产为增量打分，无此问题）。
4. gemini 端态 R=0.2–0.3 的单规则召回受 n=20 限制（Fisher p 0.21–0.47，均不显著）；
   判定依赖的条款 2 用的是 precision（B 报告指标），其 gemini Wilson 下界 0.51——
   **n≥20×2 的 eigen 复测（v2 规则臂重跑）仍需补**，本回放只回答"签名可归纳性"，
   不替代干预效果实验。
5. Phase 2 批次的截断噪声（14/15 每臂 env_error）提示：生产 fail_streak 判定前宜对
   截断 toolResult 做显式标记（is_error 已有字段，规则侧未消费——待后续版本）。

## 附：产物与复现

- `tools/rule_replay_validation.py`（转换层+真实 v2 规则回放+统计+本报告生成）
- `replay_runs.jsonl`（逐 run：283 行）/ `rule_replay_metrics.json`（指标+一致性+抽查）
- 复现：`python tools/rule_replay_validation.py`（默认 budget=900；只读历史会话）
- 回归：`python -m pytest tests/test_convergence_shadow.py -q`（60 passed，规则实现零改动）

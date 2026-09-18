# 项目终局状态（2026-09-18）

## 用户的原始目标
把「证据强度分级」能力训练进 200M 级模型，并投入 workshop 级开源。

## 终局回答：**以否证告终，且否证是干净的**

| 层级 | 可判者 | 证据 |
|---|---|---|
| 有无产物 token（NONE 界） | **程序**（机械） | NONE 召回 0.857 vs 模型 0.000 |
| STRONG vs MEDIUM | **程序 ≡ 模型**（无增益） | H2 纯程序 0.756 / H0 纯模型 0.779 / H1 混合 0.722 |
| 意图 ↔ 证据覆盖（§228） | **LLM 语义捕获 + 程序建图** | acc 0.878 vs 0.634 |

**⇒ 证据强度分级这一整个任务落在机械规则射程内，不需要模型。**
否证原因不是"模型太小"，而是**任务本身机械可判**。

## 交付物清单（全部已验证存在）

### 判据与协议
- `maps/COVERAGE_CRITERION_v6_3class.md` —— 三值判据（替代 v5b 四值）
- `maps/REVERSE_GATE_v2.md` —— 偏斜校正反向率门槛（三闸）
- `maps/MULTI_METRIC_ACCEPTANCE.md` —— 口径族 v2（含**口径准入检验**）

### 数据与模型
- `maps/strength3_train.json` —— 1047 行三值训练集（已脱敏）
- `maps/strength3_evidence.json` —— 跨批证据表（KILLER 闸，PASS）
- `models/gliner25_strength3_v1/final` —— 287M 三值模型（1.1G）

### 实验结果
- `maps/strength3_eval_result.json` —— 四独立标注集评估
- `maps/hybrid_result.json` —— 三臂混合架构检验（决定性否证）
- `maps/adjacent_diag.json` / `maps/reverse_gate_result.json` / `maps/holdout_check.json`

### 发布仓 `publish_repo`（8 commit，48 文件）
- commit `a855b91` 测量审计（三个假失败）
- commit `652a5c2` 四值降三值 + NONE 界崩塌
- commit `6dcd775` 混合架构否证

## 安全（用户硬约束）
- **凭据泄漏已修复**：160 文件 / 1481 处，复验 1088 文件**残留 0**
- **publish_repo 含完整 git 历史**（8 commit 全量 rev-list）扫描**命中 0**
- 用户明示不上传项（`.zcode` / `.venv*` / zsh history / 模型权重 / 真实会话文本）均未进入仓库

## 本轮闭环的方法论产出
1. **口径准入检验**（平凡解不过 / 随机可比 / **不惩罚正确**）——本项目四次失败结论中三次是口径假失败
2. **档位跨批证据闸**——每档必须 ≥2 批出现，防 WEAK 事故重演
3. **混合架构反证法**——用"纯程序"对照臂检验"模型是否提供增益"，比直接看模型分数更能证否

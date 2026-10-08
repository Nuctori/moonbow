# constraint_gate — 约束守卫（提醒制）

用户决策 = **目标 + 约束**；守卫在两个检查点用语义模型分析 AI 决策：
①任务结束时判定目标是否闭合；②每个新增决策判定是否违反约束。
**响应策略：全程提醒、不阻断**（SLM/LLM 均有误判率），裁决权在用户。

来源：真实会话审计（[docs/session_violation_audit.md](../../docs/session_violation_audit.md)）
→ 机制设计（[docs/constraint_gate_design.md](../../docs/constraint_gate_design.md)）
→ 首次实验（[lab/lab_report.md](lab/lab_report.md)，2026-10-08 夜）。

## 组件

| 文件 | 角色 | 铁律对应 |
|---|---|---|
| `constraints.yaml` | 约束卡登记：原文逐字 + 边界条款 + 正反判例 + 存量清单 | 禁令≠一句话，需可判定边界 |
| `tracker.py` | 机械追踪层：7 规则，零 LLM，只引 span | 追踪器零 LLM；召回可用机械 |
| `gate.py` | CLI：提醒块生成；`--claim` 进入 D3 对账（宣称 vs 检测矛盾 → CONFLICT） | 描述层只描述不裁决 |
| `lab/` | 首次实验脚本 + 全部结果 + 报告 | 证据落盘 |

## 用法

```bash
# 单提交提醒块
python gate.py --repo D:/react/app --commit 5c42dbedd

# 会话收尾对账（终报宣称 vs 机械检测）
python gate.py --repo D:/react/app --since "2026-10-02 04:00" --until "2026-10-02 09:00" --claim "零关键词表"
# → reconciliation.verdict = CONFLICT 时注入用户必读对账块
```

## 首次实验结论（详见 lab/lab_report.md）

- tracker：召回 5/5，commit 级精度 5/8——高召回低精度，召回层定位实证
- T3 语义裁决（LLM 兜底）：落地 **4/4 全对**，含 P8 换皮识别（"数据驱动的词表仍算词表"）与 span 级引用；4 例因免费窗口限流减员，可断点续跑
- T3 的 SLM 零样本不可用（0.8B：6/10 不可解析 + 2 误报）→ 必须用本管线标注流微调，此前走 LLM 兜底
- D3 假宣称可机械戳穿：10-02 夜巡 78 条证据 vs "零关键词表"宣称 → CONFLICT

## 待办

1. 非高峰补齐 T3 剩余 4 例（`python lab/exp_b_t3.py` 断点续跑），报严格 P/R
2. gate.py 接入 ZCode hook（PostToolUse / Stop）
3. 用标注流（findings + 用户裁决）微调 T3 SLM 后端（XPU）

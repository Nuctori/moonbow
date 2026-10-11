# -*- coding: utf-8 -*-
"""moonbow.intent.applicability — 约束适用范围闸门（L0，零 LLM）。

来源（exp_t3_decomposition_report.md）：T3 拆维度实验证明，机械层无法替代
语义裁决（残差仍占 86%），但其中**适用范围判断**是用户已在约束卡里明确
声明为机械规则的边界，可以零 LLM 可靠地前置执行：

- **scope 不继承**：约束只在声明它的项目内生效；
- **时间不溯及**：约束自宣告时刻起生效，此前提交不受其约束。

这两条在 eval_t3 上覆盖 14%（2/14）且判定全对。把它们固化进运行时，
语义层只会收到"约束确实适用"的样本，残差更干净（也是更公平的基座能力
测试）。

本模块是**机械层**：只做字段比较，不做任何语义判断，不读内容词表。
"""
from dataclasses import dataclass
from typing import Optional

#: 适用范围闸门的结论
GATE_APPLIES = "applies"
GATE_OUT_OF_SCOPE = "out_of_scope"
GATE_NOT_YET_EFFECTIVE = "not_yet_effective"
GATE_UNKNOWN_SCOPE = "unknown_scope"


@dataclass
class Applicability:
    """约束对某决策的适用性判定（机械）。"""
    verdict: str
    reason: str
    constraint_id: str = ""

    @property
    def blocks_semantic_review(self) -> bool:
        """是否应跳过语义裁决（直接弃权）。"""
        return self.verdict != GATE_APPLIES


def check_applicability(constraint, decision_scope: str,
                        decision_ts: str = "",
                        constraint_scope: Optional[str] = None,
                        constraint_effective_at: str = "") -> Applicability:
    """判断约束是否适用于该决策（机械字段比较，无内容分析）。

    constraint_scope / constraint_effective_at 若未显式给出，从约束实体的
    meta 里读（登记时写入）。缺省 = 该约束未声明范围，按"适用"处理——
    不因缺省元数据就静默弃权（弃权必须是显式结论，不能靠缺失推断）。
    """
    meta = getattr(constraint, "meta", None) or {}
    c_scope = constraint_scope if constraint_scope is not None \
        else meta.get("scope")
    c_at = constraint_effective_at or meta.get("effective_at", "")
    cid = getattr(constraint, "entity_id", "")

    # 1) scope 闸门：约束声明了范围且与决策所在范围不一致 → 不适用
    if c_scope and decision_scope and c_scope != decision_scope:
        return Applicability(
            GATE_OUT_OF_SCOPE,
            f"约束 scope={c_scope!r} 与决策 scope={decision_scope!r} 不一致"
            f"（约束不跨项目继承）", cid)

    # 2) 时间闸门：约束生效时刻晚于决策时刻 → 不适用
    if c_at and decision_ts and _lt(decision_ts, c_at):
        return Applicability(
            GATE_NOT_YET_EFFECTIVE,
            f"决策时刻 {decision_ts} 早于约束生效时刻 {c_at}（约束不溯及）",
            cid)

    return Applicability(GATE_APPLIES, "约束适用于该决策", cid)


def _lt(a: str, b: str) -> bool:
    """字典序比较 ISO8601 时间戳（同格式下与时间序一致）。

    不做日期解析——格式不保证统一，字典序对 ISO8601 前缀足够可靠；
    无法确定时返回 False（不阻断），由语义层处理。
    """
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    if len(a) < 10 or len(b) < 10:
        return False
    return a[:19] < b[:19]


def filter_applicable(constraints, decision_scope: str,
                      decision_ts: str = ""):
    """把约束集分成 (适用, 不适用+原因) 两组。"""
    applicable, skipped = [], []
    for c in constraints:
        ap = check_applicability(c, decision_scope, decision_ts)
        (applicable if ap.verdict == GATE_APPLIES else skipped).append(
            (c, ap) if ap.verdict != GATE_APPLIES else c)
    return applicable, skipped

# -*- coding: utf-8 -*-
"""moonbow.task_structure.scoring

计量层：从捕获项计算可解释结构向量，再按透明规则派生等级与建议类型。

设计约束：
- 阈值是**待验证启发式**（本文件顶部集中定义，报告须引用依据），
  不是已校准的实际难度概率。
- 弃权优先：信息不足/截断时输出 unknown，不得解释为"简单"。
- "结构负担高"与"适合拆解"分离：建议只描述结构事实与可选做法，
  不承诺可并行，不构成要求。
"""
from typing import List
import re

from .schema import (
    StructureAnalysis, StructureVector,
    LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_UNKNOWN,
    GOAL, CONSTRAINT, DEPENDENCY, COORDINATION, CONDITION, UNRESOLVED,
    ACCEPTANCE, OBJECT,
)

# 任务意图标记：纯背景/闲聊文本没有结构时**不**建议澄清（避免噪声提醒）
_TASK_INTENT = re.compile(
    r"请|麻烦|帮忙|帮我|需要你|劳驾|帮忙看|please|could you|can you|would you",
    re.IGNORECASE)

# ---------------------------------------------------------------------------
# 透明阈值（v0 启发式；修改须同步评测报告中的阈值依据说明）
# ---------------------------------------------------------------------------
HIGH_GOALS = 3              # 独立交付义务数达到即"结构复杂"
HIGH_BURDEN = 3             # 约束+依赖+协调总数达到即"结构复杂"
HIGH_DEPS = 2               # 显式依赖数达到即"结构复杂"
MEDIUM_GOALS = 2            # 中等起点
LOW_MAX_CONSTRAINTS = 1     # 简单任务允许的约束上限

ADVICE_SPLIT_BY_GOAL = "split_by_goal"          # 多义务、少依赖 → 按交付义务分批
ADVICE_PHASED = "phased_with_integration"       # 依赖多 → 分阶段并保留集成核验
ADVICE_CLARIFY = "clarify_scope"                # 边界不明 → 先明确范围/完成条件
ADVICE_CAUTION = "coordination_caution"         # 协调/约束多 → 建议分别核验


def compute_vector(analysis: StructureAnalysis) -> StructureVector:
    """从去重后的捕获项计数。向量是第一输出；等级只是它的派生。"""
    v = StructureVector()
    for c in analysis.captures:
        if c.kind == GOAL:
            v.goals += 1
        elif c.kind == CONSTRAINT:
            v.constraints += 1
        elif c.kind == DEPENDENCY:
            v.explicit_dependencies += 1
        elif c.kind == COORDINATION:
            v.coordinations += 1
        elif c.kind in (CONDITION, UNRESOLVED):
            v.conditions_unresolved += 1
        elif c.kind == OBJECT:
            v.objects += 1
    return v


def classify(analysis: StructureAnalysis) -> None:
    """就地填充 level / level_reasons / advice。

    规则完全透明：每条结论都附带可读理由。
    """
    reasons: List[str] = []
    advice = None

    # —— 弃权优先 ——
    if analysis.truncated:
        analysis.level = LEVEL_UNKNOWN
        analysis.level_reasons = ["输入被截断，结构可能不完整；不评定复杂度"]
        analysis.abstain = True
        analysis.abstain_reason = "truncated"
        analysis.advice = None
        return
    v = analysis.vector or StructureVector()
    if v.goals == 0 and v.constraints == 0 and v.explicit_dependencies == 0 \
            and v.coordinations == 0:
        analysis.level = LEVEL_UNKNOWN
        analysis.level_reasons = ["未捕获到显式交付义务或结构标记；"
                                  "信息不足，不解释为简单"]
        analysis.abstain = True
        analysis.abstain_reason = analysis.abstain_reason or "no_explicit_task"
        if _TASK_INTENT.search(analysis.text):
            analysis.advice = ADVICE_CLARIFY
        return
    # 模糊目标且无对象锚点 → 边界不明
    if v.goals >= 1 and v.objects == 0 and _vague_only(analysis):
        analysis.level = LEVEL_UNKNOWN
        analysis.level_reasons = ["目标表述模糊且无具体对象锚点；边界不明"]
        analysis.abstain = True
        analysis.abstain_reason = analysis.abstain_reason or "vague_boundary"
        analysis.advice = ADVICE_CLARIFY
        return

    # —— 透明分级 ——
    burden = v.constraints + v.explicit_dependencies + v.coordinations
    if v.goals >= HIGH_GOALS or burden >= HIGH_BURDEN \
            or v.explicit_dependencies >= HIGH_DEPS:
        analysis.level = LEVEL_HIGH
    elif v.goals >= MEDIUM_GOALS or burden >= 1:
        analysis.level = LEVEL_MEDIUM
    elif v.goals == 1 and v.constraints <= LOW_MAX_CONSTRAINTS:
        analysis.level = LEVEL_LOW
    else:
        analysis.level = LEVEL_MEDIUM
    if analysis.level == LEVEL_HIGH:
        if v.goals >= HIGH_GOALS:
            reasons.append(f"捕获到 {v.goals} 项独立交付义务（阈值 {HIGH_GOALS}）")
        if v.explicit_dependencies >= HIGH_DEPS:
            reasons.append(f"捕获到 {v.explicit_dependencies} 处显式依赖（阈值 {HIGH_DEPS}）")
        if v.constraints + v.coordinations >= HIGH_BURDEN:
            reasons.append(f"约束+协调共 {v.constraints + v.coordinations} 项"
                           f"（阈值 {HIGH_BURDEN}）")
    elif analysis.level == LEVEL_MEDIUM:
        reasons.append(f"义务 {v.goals} 项、约束/依赖/协调共 {burden} 项；"
                       "结构负担中等")

    # —— 建议分型（事实描述 + 可选做法；不构成要求，不承诺可并行）——
    if analysis.level == LEVEL_HIGH:
        if v.explicit_dependencies >= HIGH_DEPS:
            advice = ADVICE_PHASED
        elif v.goals >= HIGH_GOALS and v.explicit_dependencies == 0:
            advice = ADVICE_SPLIT_BY_GOAL
        else:
            advice = ADVICE_CAUTION
    elif analysis.level == LEVEL_MEDIUM and v.coordinations >= 2:
        advice = ADVICE_CAUTION
    elif analysis.abstain_reason is None and analysis.level == LEVEL_UNKNOWN:
        advice = ADVICE_CLARIFY

    # 依赖存在但端点不明 → 明示"不可据此并行"
    if v.explicit_dependencies > 0 and "dependency" in analysis.unknowns:
        reasons.append("存在显式顺序/依赖表述，端点未经核验；不构成可并行的依据")

    analysis.level_reasons = reasons or [f"义务 {v.goals} 项；结构负担低"]
    analysis.advice = advice


def _vague_only(analysis: StructureAnalysis) -> bool:
    """所有 goal 捕获的引文是否都只含模糊动词、无具体对象词。"""
    from .extractor import VAGUE_GOALS
    if not analysis.captures:
        return False
    goal_caps = [c for c in analysis.captures if c.kind == GOAL]
    if not goal_caps:
        return False
    for c in goal_caps:
        if not any(g in c.quote for g in VAGUE_GOALS):
            return False
    return True

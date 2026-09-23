# -*- coding: utf-8 -*-
"""moonbow.task_structure.policy

提醒策略层：决定**是否**生成建议消息，并构造事实性措辞。

硬约束（与 Goal 的职责边界一一对应）：
- 建议是可忽略的提示，不是要求；措辞禁止"必须/请务必/需要拆解"。
- 结构指纹去重：同一结构不重复提醒；每个任务版本有提醒上限。
- 依赖未经核验时，提醒必须显式声明"不构成可并行的依据"。
- 采纳与否不影响任何状态（无升级、无阻止、无合规记录）。
"""
import hashlib
from typing import Dict, List, Optional, Set

from .schema import StructureAnalysis, LEVEL_HIGH, LEVEL_UNKNOWN

# 每个任务版本允许的提醒条数上限（默认 1：同一结构只说一次）
DEFAULT_MAX_REMINDERS = 1

# 建议等级门槛：默认仅 high（或 abstain 且可澄清）才提醒
DEFAULT_MIN_LEVEL = LEVEL_HIGH

_ADVICE_TEXT = {
    "split_by_goal": (
        "捕获到多项相互独立的交付义务。可考虑按交付义务分批处理并分别核验；"
        "是否拆解由你决定。"
    ),
    "phased_with_integration": (
        "捕获到显式的先后/依赖表述。可考虑分阶段处理，并在最后做一次整体核验；"
        "是否拆解由你决定。当前分析不代表这些任务可以并行。"
    ),
    "coordination_caution": (
        "捕获到多项约束/协调要求，同时处理可能增加遗漏风险。"
        "可考虑在收尾前逐项对照核验；是否调整做法由你决定。"
    ),
    "clarify_scope": (
        "当前任务描述的边界或完成条件不够明确。"
        "可考虑先明确范围与验收条件再动手；是否澄清由你决定。"
    ),
}

_FACTUAL_PREFIX = "【任务结构（仅提示，非要求）】"


def structure_fingerprint(analysis: StructureAnalysis) -> str:
    """结构指纹：与措辞无关、与结构相关。

    由"来源 + 各 kind 的归一化捕获键排序序列 + 等级"构成。
    同一任务的同义复述 → 捕获键高度重合；指纹对**完全相同**结构稳定，
    近似结构的近重合由 ReminderPolicy 的相似度缓冲兜底。
    """
    keys = sorted(f"{c.kind}:{c.norm or c.quote}" for c in analysis.captures)
    payload = "|".join([analysis.source, str(analysis.level)] + keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ReminderPolicy:
    """有状态提醒器：指纹去重 + 相似缓冲 + 每版本上限。

    用法：对每次分析调用 consider()，返回建议文本或 None。
    状态仅用于抑制重复提醒；永不升级、永不拦截。
    """

    def __init__(self, max_reminders: int = DEFAULT_MAX_REMINDERS,
                 min_level: str = DEFAULT_MIN_LEVEL,
                 similarity_buffer: int = 8,
                 similar_threshold: float = 0.85):
        self.max_reminders = max(1, int(max_reminders))
        self.min_level = min_level
        self.similarity_buffer = similarity_buffer
        self.similar_threshold = similar_threshold
        self.reset()

    def reset(self) -> None:
        self.seen: Set[str] = set()
        self.recent_keys: List[Set[str]] = []   # 最近结构的捕获键集（改写兜底）
        self.counts: Dict[str, int] = {}        # 每任务版本计数（version 作 key）

    def consider(self, analysis: StructureAnalysis) -> Optional[str]:
        """返回提醒文本，或 None（不提醒）。"""
        if analysis.abstain and analysis.level == LEVEL_UNKNOWN:
            # 信息不足：仅当能给出"澄清边界"建议时提醒一次
            if analysis.advice != "clarify_scope":
                return None
        elif analysis.level != self.min_level and analysis.level != LEVEL_HIGH:
            if not (analysis.abstain and analysis.advice == "clarify_scope"):
                return None
        if not analysis.advice:
            return None

        fp = structure_fingerprint(analysis)
        keys = {f"{c.kind}:{c.norm or c.quote}" for c in analysis.captures}
        if fp in self.seen:
            return None
        if any(_jaccard(keys, r) >= self.similar_threshold
               for r in self.recent_keys):
            return None

        version = str(analysis.version)
        if self.counts.get(version, 0) >= self.max_reminders:
            return None

        self.seen.add(fp)
        self.recent_keys.append(keys)
        if len(self.recent_keys) > self.similarity_buffer:
            self.recent_keys.pop(0)
        self.counts[version] = self.counts.get(version, 0) + 1
        return self.build_message(analysis)

    def build_message(self, analysis: StructureAnalysis) -> str:
        v = analysis.vector
        facts = []
        if v:
            if v.goals:
                facts.append(f"{v.goals} 项交付义务")
            if v.constraints:
                facts.append(f"{v.constraints} 项约束")
            if v.explicit_dependencies:
                facts.append(f"{v.explicit_dependencies} 处显式依赖")
            if v.coordinations:
                facts.append(f"{v.coordinations} 项协调要求")
        advice_text = _ADVICE_TEXT.get(analysis.advice or "", "")
        parts = [_FACTUAL_PREFIX]
        if facts:
            parts.append("当前捕获到" + "、".join(facts) + "。")
        if advice_text:
            parts.append(advice_text)
        return "".join(parts)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """捕获键集 Jaccard 相似度：同任务改写的结构近似兜底。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# -*- coding: utf-8 -*-
"""moonbow.task_structure

Task Structure Advisor（任务结构顾问）：

从自然语言消息中捕获显式的任务义务、约束与关系，用确定性规则量化
"可观察结构复杂度"，在有足够依据时向主模型提供**可忽略**的拆解或
分阶段处理建议。

边界（设计即约束，见 docs/task_structure_spec.md）：
- 不是任务真实难度预测器；标签描述任务表述，不代表解决难度。
- 只捕获文本显式表达的结构；未表达依赖不得推断为"独立"。
- 无法可靠确认时弃权；"未捕获到复杂结构"不等于"简单"。
- 建议不构成要求，不拦截执行，不影响 Progress Guard 的完成判定。
"""
from .schema import (
    CAPTURE_KINDS,
    Capture,
    Relation,
    StructureAnalysis,
    StructureVector,
    validate_captures,
)
from .extractor import RuleBackend, analyze_text
from .scoring import compute_vector, classify
from .policy import ReminderPolicy, structure_fingerprint

__all__ = [
    "CAPTURE_KINDS",
    "Capture",
    "Relation",
    "StructureAnalysis",
    "StructureVector",
    "validate_captures",
    "RuleBackend",
    "analyze_text",
    "compute_vector",
    "classify",
    "ReminderPolicy",
    "structure_fingerprint",
]

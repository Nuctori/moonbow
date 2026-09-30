# -*- coding: utf-8 -*-
"""moonbow.semantic

统一语义匹配运行时的契约层（P1 冻结）。
外部概念：文本 + 匹配要求 -> 匹配结果、依据、分数、未知状态。

边界：
1. 文本里"声称已完成"不是现实里"已完成"。
2. 模型识别语义；程序校验证据、协议并决定动作。
3. 未校准分数只是 raw_score，不是命题真值概率。
"""
from moonbow.semantic.schema import (  # noqa: F401
    ValidationError,
    MatchRequest,
    MatchResponse,
    Evidence,
    Provenance,
    PatternSpec,
    validate_response,
    OPERATIONS,
    STATUSES,
    REASON_CODES,
    RELATIONS,
    MAX_TEXT_LENGTH,
)
from moonbow.semantic.patterns import (  # noqa: F401
    PatternRegistry,
    load_builtin_patterns,
)

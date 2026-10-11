# -*- coding: utf-8 -*-
"""moonbow.intent — 目标/约束语义提醒闭环（领域模块）。

事件流 → 语义捕获 → 账本 → 关系判断 → 非阻塞提醒 → 用户反馈。

哲学（与仓库一致）：
- 捕获只声称"文本里出现了什么"，必须携带 verbatim 原文引用；
- 助手宣称不等于事实；用户原文不可被助手转述覆盖；
- 判断必须绑定双方证据引用；模型失败/信息不足必须显式 abstain，
  不得伪装成"未违规/已完成"；
- 提醒永久非阻塞，最终裁决权在用户。
"""
from .schema import (
    SOURCE_ROLES, ENTITY_KINDS, EVENT_KINDS,
    CONSTRAINT_VERDICTS, COVERAGE_VERDICTS,
    Event, Entity, Judgment, Reminder,
)
from .ledger import Ledger
from .verify import verify_quote, store_event_text
from .capture import CaptureStage, SemanticCaptureBackend, FakeCaptureBackend

__all__ = [
    "SOURCE_ROLES", "ENTITY_KINDS", "EVENT_KINDS",
    "CONSTRAINT_VERDICTS", "COVERAGE_VERDICTS",
    "Event", "Entity", "Judgment", "Reminder",
    "Ledger", "verify_quote", "store_event_text",
    "CaptureStage", "SemanticCaptureBackend", "FakeCaptureBackend",
]

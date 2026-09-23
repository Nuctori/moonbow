# -*- coding: utf-8 -*-
"""moonbow.task_structure.service

Task Structure Advisor 的服务载荷入口（被 guard server 的独立端点
`/v1/task-structure` 调用；不参与守卫的任何判定）。

与守卫的隔离保证：
- 守卫 server 仅在本模块可导入且请求命中该端点时才触达本模块；
- 本模块的任何异常只影响该端点的响应，不影响 /check 与 /v1/stage-check。
"""
from typing import Optional

from .extractor import analyze_text, get_backend
from .policy import ReminderPolicy, structure_fingerprint

_SOURCES = ("user", "plan", "revision")

# 会话级提醒状态（进程内；重启即重置——建议不是状态机，丢了也无损）
_POLICY: Optional[ReminderPolicy] = None


def _policy() -> ReminderPolicy:
    global _POLICY
    if _POLICY is None:
        import os
        cap = int(os.environ.get("TASK_STRUCTURE_MAX_REMINDERS", "1"))
        _POLICY = ReminderPolicy(max_reminders=cap)
    return _POLICY


def analyze_payload(payload: dict, decide_reminder: bool = False) -> dict:
    """分析 {text, source?, version?, include_text?} → 结构化结果 dict。

    - text 必须非空（空输入是调用方 bug，返回 400 由端点负责）。
    - 默认不回显原文（text_hash 足够对账）；include_text=true 才返回原文，
      供离线评测使用。
    - decide_reminder=true 时由服务端策略判断是否生成提醒文本（advisory
      模式由插件本地去重，跨会话进程内状态仅兜底）。
    """
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    source = payload.get("source", "user")
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}")
    try:
        version = int(payload.get("version", 1))
    except (TypeError, ValueError):
        raise ValueError("version must be an integer")

    backend = get_backend()
    analysis = analyze_text(text, backend=backend, source=source, version=version)
    out = analysis.to_dict()
    out["backend"] = getattr(backend, "name", "rule")
    out["fingerprint"] = structure_fingerprint(analysis)
    # 候选措辞由服务端统一构造（单一事实来源）；是否发送由客户端
    # 按 advisory/shadow 模式与本地去重决定——服务端状态不构成要求。
    if analysis.advice:
        out["message"] = ReminderPolicy().build_message(analysis)
    if payload.get("include_text"):
        out["text"] = text
    if decide_reminder:
        reminder = _policy().consider(analysis)
        out["reminder"] = reminder
    return out

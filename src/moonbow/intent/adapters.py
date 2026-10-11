# -*- coding: utf-8 -*-
"""moonbow.intent.adapters — ZCode / pi 事件适配器（只转换，不做语义逻辑）。

pi 会话格式（v3）：{"type":"message","message":{"role":...,
"content":[{"type":"text","text":...}, ...],"timestamp":...}}
ZCode 导出格式：每行 {"event_id","kind","source_role","ts","text"}
（由宿主导出器产生；本适配器只做字段映射）。
"""
import json
from pathlib import Path
from typing import Iterator, Optional

from .schema import Event

_ROLE_MAP = {"user": "user_requirement",
             "assistant": "assistant_claim",
             "toolResult": "tool_observation"}


def _mk_event(eid, kind, role, ts, text):
    return Event(event_id=eid, kind=kind, source_role=role, ts=ts, text=text)


def pi_events_from_jsonl(path: str, session_prefix: str = "pi",
                         limit: Optional[int] = None) -> Iterator[Event]:
    """pi 会话 JSONL → Event 流。文本消息映射为 message 事件；
    assistant 的 toolCall → tool_call 事件（决策面）；toolResult 角色映射为
    tool_observation。只转换，不过滤内容。"""
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if limit is not None and n >= limit:
                return
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "message":
                continue
            m = obj.get("message", {})
            role = m.get("role")
            if role not in _ROLE_MAP:
                continue
            content = m.get("content")
            ts = str(obj.get("timestamp", ""))
            texts, tools = [], []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("type")
                    if t == "text":
                        texts.append(c.get("text", ""))
                    elif t in ("toolCall", "tool_use", "tool_call"):
                        tools.append((c.get("name") or c.get("toolName") or "tool",
                                      c.get("arguments") or c.get("input")
                                      or c.get("input_args") or {}))
            if role == "assistant":
                for name, args in tools:
                    n += 1
                    yield _mk_event(f"{session_prefix}_t{n:05d}", "tool_call",
                                    "assistant_claim", ts,
                                    _tool_blob(name, args))
            text = "\n".join(texts).strip()
            if not text:
                continue
            n += 1
            yield _mk_event(f"{session_prefix}_m{n:05d}", "message",
                            _ROLE_MAP[role], ts, text)


def zcode_events_from_jsonl(path: str) -> Iterator[Event]:
    """ZCode 导出 JSONL → Event 流（每行已含五字段）。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            yield _mk_event(d["event_id"], d["kind"], d["source_role"],
                            d.get("ts", ""), d.get("text", ""))


# ── omp / kimi 真实会话适配（真实接入重放用）─────────────────────────────

def _tool_blob(name, args) -> str:
    """工具调用序列化为决策文本（名字 + 参数摘要，截断到 1500 字）。"""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    blob = json.dumps({name: args}, ensure_ascii=False)
    return blob[:1500]


def omp_events_from_jsonl(path: str, session_prefix: str = "omp",
                          limit: Optional[int] = None) -> Iterator[Event]:
    """omp v3 会话 JSONL → Event 流。

    message.content 列表项：text / toolCall（name + arguments）。
    assistant 的 toolCall → tool_call 事件（决策面）；toolResult 角色 →
    tool_result 事件。只转换，不过滤内容。
    """
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if limit is not None and n >= limit:
                return
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "message":
                continue
            m = obj.get("message") or {}
            role = m.get("role")
            content = m.get("content")
            ts = str(obj.get("timestamp", ""))
            texts, tools = [], []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("type")
                    if t == "text":
                        texts.append(c.get("text", ""))
                    elif t in ("toolCall", "tool_use", "tool_call"):
                        tools.append((c.get("name") or c.get("toolName") or "tool",
                                      c.get("arguments") or c.get("input") or {}))
            if role in ("user", "developer") and any(x.strip() for x in texts):
                n += 1
                yield _mk_event(f"{session_prefix}_u{n:05d}", "message",
                                "user_requirement", ts,
                                "\n".join(texts).strip())
            elif role == "assistant":
                if any(x.strip() for x in texts):
                    n += 1
                    yield _mk_event(f"{session_prefix}_a{n:05d}", "message",
                                    "assistant_claim", ts,
                                    "\n".join(texts).strip())
                for name, args in tools:
                    n += 1
                    yield _mk_event(f"{session_prefix}_t{n:05d}", "tool_call",
                                    "assistant_claim", ts,
                                    _tool_blob(name, args))
            elif role in ("toolResult", "tool_result"):
                if any(x.strip() for x in texts):
                    n += 1
                    yield _mk_event(f"{session_prefix}_r{n:05d}", "tool_result",
                                    "tool_observation", ts,
                                    "\n".join(texts).strip()[:4000])


def kimi_events_from_jsonl(path: str, session_prefix: str = "kimi",
                           limit: Optional[int] = None) -> Iterator[Event]:
    """kimi context.jsonl → Event 流。

    行格式 {"role","content"}：content 为字符串或列表（text/think/function）。
    function 项（{"function":{"name","arguments"}}）→ tool_call 事件。
    """
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if limit is not None and n >= limit:
                return
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            role = o.get("role")
            if role in (None, "_system_prompt"):
                continue
            content = o.get("content")
            texts, tools = [], []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("type")
                    if t == "text":
                        texts.append(c.get("text", ""))
                    elif t == "function":
                        fn = c.get("function") or {}
                        tools.append((fn.get("name", "tool"),
                                      fn.get("arguments", {})))
            if role == "assistant":
                for name, args in tools:
                    n += 1
                    yield _mk_event(f"{session_prefix}_t{n:05d}", "tool_call",
                                    "assistant_claim", "",
                                    _tool_blob(name, args))
            if any(x.strip() for x in texts):
                n += 1
                if role in ("user", "developer"):
                    yield _mk_event(f"{session_prefix}_u{n:05d}", "message",
                                    "user_requirement", "",
                                    "\n".join(texts).strip())
                elif role == "assistant":
                    yield _mk_event(f"{session_prefix}_a{n:05d}", "message",
                                    "assistant_claim", "",
                                    "\n".join(texts).strip())
                elif role in ("tool", "toolResult", "tool_result"):
                    yield _mk_event(f"{session_prefix}_r{n:05d}", "tool_result",
                                    "tool_observation", "",
                                    "\n".join(texts).strip()[:4000])

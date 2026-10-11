# -*- coding: utf-8 -*-
"""moonbow.intent.host — 宿主接线组件（提醒投递 + 会话运行器）。

这是"接线"的代码主体：宿主（ZCode/pi hook）只需提供一个 sink（回调/文件/
stdout），本模块把 IntentLoop 产出的提醒块格式化为宿主可注入的 JSON 并投递。
**真实 hook 配置刻意不在此模块内**——接线的最后一厘米是宿主配置（一行
hook 注册），由部署时完成；本模块保证代码路径完整、可单测、可重放。

设计立场：
- 提醒块形状与机械层 gate 产出一致（type=constraint-reminder,
  policy=remind-only），宿主侧无需区分来源层；
- emitter 永不抛出——投递失败记入 failures，不阻断主 AI；
- SessionRunner 把"适配器事件流 → IntentLoop → 提醒投递"串成一行，
  供真实会话重放与未来宿主主循环复用。
"""
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .loop import IntentLoop, LoopReport
from .schema import Event, Reminder


class ReminderEmitter:
    """把提醒块投递给宿主 sink。sink 签名：(dict) -> None。"""

    def __init__(self, sink: Optional[Callable[[dict], None]] = None):
        self._sink = sink
        self.emitted: List[dict] = []
        self.failures: List[dict] = []

    def emit(self, reminder: Reminder, context: dict) -> None:
        block = reminder.to_dict()
        block["context"] = {
            "session": context.get("session", ""),
            "event_id": context.get("event_id", ""),
            "quote": context.get("quote", "")[:200],
        }
        self.emitted.append(block)
        if self._sink is None:
            return
        try:
            self._sink(block)
        except Exception as e:                        # noqa: BLE001 — 不阻断
            self.failures.append({"stage": "emit",
                                  "error": f"{type(e).__name__}: {e}",
                                  "dedup_key": block.get("dedup_key", "")})

    def drain_jsonl(self) -> str:
        return "\n".join(json.dumps(b, ensure_ascii=False)
                         for b in self.emitted)


@dataclass
class SessionReport:
    session: str
    source: str = ""
    events: int = 0
    decisions: int = 0
    judgments: int = 0
    gate_skipped: int = 0
    abstains: int = 0
    conflicts: int = 0
    reminders_deduped: int = 0
    failures: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class SessionRunner:
    """会话事件流 → IntentLoop → 提醒投递。宿主主循环与重放共用。"""

    def __init__(self, loop: IntentLoop, emitter: ReminderEmitter):
        self.loop = loop
        self.emitter = emitter

    def run_event(self, event: Event, session: str = "") -> LoopReport:
        report = self.loop.process_event(event)
        for r in report.reminders_emitted:
            self.emitter.emit(r, {"session": session,
                                  "event_id": event.event_id,
                                  "quote": event.text[:200]})
        return report

    def run_stream(self, events, session: str = "", source: str = ""
                   ) -> SessionReport:
        sr = SessionReport(session=session, source=source)
        for ev in events:
            rep = self.run_event(ev, session=session)
            sr.events += 1
            sr.decisions += 1 if ev.kind in self.loop.DECISION_KINDS else 0
            sr.judgments += rep.judgments
            sr.gate_skipped += rep.applicability_skipped
            sr.conflicts += len(rep.reminders_emitted)
            sr.reminders_deduped += rep.reminders_deduped
            sr.failures += rep.failures
        sr.abstains = sum(
            1 for j in self.loop.ledger.all_judgments() if j.abstain)
        return sr

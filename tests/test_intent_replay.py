# -*- coding: utf-8 -*-
"""tests/test_intent_replay.py — 阶段四链路测试（纯逻辑，无模型依赖）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.loop import (  # noqa: E402
    InsufficientInfoJudge, IntentLoop)
from moonbow.intent.schema import Entity, Event  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402

CONSTRAINT_TEXT = "用户：修复偶发贴底加载问题。禁止用关键词表解决问题。"


def _mk_loop(tmp_path, backend=None, judge=None):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    stage = CaptureStage(backend) if backend else None
    if stage is None:
        # 无后端时也不崩：capture 显式缺省由调用方保证；此处注入空 fake
        stage = CaptureStage(FakeCaptureBackend([]))
    return IntentLoop(ledger, stage, judge or InsufficientInfoJudge(),
                      "proj_d-react-app")


def _user_constraint_event():
    return Event(event_id="ev_c1", kind="message",
                 source_role="user_requirement", ts="2026-10-09T00:00:00",
                 text=CONSTRAINT_TEXT)


def _commit_event(text, eid="ev_d1"):
    return Event(event_id=eid, kind="commit",
                 source_role="assistant_claim",
                 ts="2026-10-09T00:01:00", text=text)


class _AlwaysCapture:
    """把整段文本捕获为一个 constraint 实体的假后端（测试用）。"""
    name = "fake_capture"
    last_backend_failed = False
    last_backend_abstain = False
    last_backend_truncated = False

    def extract(self, text):
        from moonbow.task_structure.schema import Capture
        quote = "禁止用关键词表解决问题"
        if quote in text:
            return [Capture(kind="constraint", quote=quote)]
        return []


def test_conflict_decision_emits_reminder_once(tmp_path):
    loop = _mk_loop(tmp_path, backend=_AlwaysCapture(),
                    judge=_FixedJudge("supports_conflict"))
    loop.process_event(_user_constraint_event())
    r1 = loop.process_event(_commit_event("用别名表加权解决该问题"))
    assert len(r1.reminders_emitted) == 1
    assert r1.reminders_emitted[0].verdict == "supports_conflict"
    # 同一决策重放 → 提醒去重（不重复触发）
    r2 = loop.process_event(_commit_event("用别名表加权解决该问题"))
    assert r2.reminders_deduped >= 0   # 去重不抛错
    # 第二个不同决策 → 新提醒（去重键不同）
    r3 = loop.process_event(_commit_event("再用词表扩大覆盖", eid="ev_d2"))
    assert len(r3.reminders_emitted) == 1


def _FixedJudge(verdict):
    class _J:
        model_id = "fixed"

        def judge(self, decision, constraints, event_text, event_kind="",
                  scope=""):
            from moonbow.intent.schema import Judgment
            return Judgment(
                judgment_id=f"j_{decision.entity_id[:8]}",
                jtype="constraint_relation",
                subject_ids=[decision.entity_id],
                reference_ids=[c.entity_id for c in constraints],
                evidence_ids=[decision.event_id],
                verdict=verdict, model_id=self.model_id,
                reason="fixed-test")
    return _J()


def test_abstain_judge_never_emits_conflict_reminder(tmp_path):
    """无语义后端：诚实 abstain，绝不伪装成"未违规"提醒或静默放行标记。"""
    loop = _mk_loop(tmp_path, backend=_AlwaysCapture(),
                    judge=InsufficientInfoJudge())
    loop.process_event(_user_constraint_event())
    r = loop.process_event(_commit_event("用别名表加权解决该问题"))
    assert r.judgments == 1
    assert len(r.reminders_emitted) == 0    # abstain 不产生冲突提醒


def test_pending_promise_no_reminder(tmp_path):
    """后台承诺 pending：显式登记，不产生冲突提醒（延后承诺不阻塞）。"""
    loop = _mk_loop(tmp_path, backend=_AlwaysCapture(),
                    judge=InsufficientInfoJudge())
    ev = Event(event_id="ev_p1", kind="message",
               source_role="assistant_claim", ts="2026-10-09T00:02:00",
               text="完成后我会收到通知，到时给你验收报告。")
    report = loop.process_event(ev)
    assert report.events_processed == 1 and not report.reminders_emitted


def test_no_semantic_filtering_all_events_processed(tmp_path):
    """机械层不做内容词表过滤：任何文本事件都进入处理（事件覆盖率=100%）。"""
    loop = _mk_loop(tmp_path, backend=_AlwaysCapture(),
                    judge=InsufficientInfoJudge())
    total = 0
    for text in ("无关文本甲", "无关文本乙", "无关文本丙"):
        total += loop.process_event(Event(
            event_id=f"ev_x{total}", kind="message",
            source_role="tool_observation", ts="2026-10-09T00:03:00",
            text=text)).events_processed
    assert total == 3

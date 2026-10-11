# -*- coding: utf-8 -*-
"""tests/test_intent_host.py — 宿主接线组件（ReminderEmitter / SessionRunner）。

接线立场：宿主只需提供 sink；emitter 永不抛出；提醒块形状与机械层一致。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.host import ReminderEmitter, SessionRunner  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import InsufficientInfoJudge, IntentLoop  # noqa: E402
from moonbow.intent.schema import Entity, Event  # noqa: E402


class _Conflict:
    model_id = "fixed"

    def judge(self, decision, constraints, event_text, event_kind="", scope=""):
        from moonbow.intent.schema import Judgment
        return Judgment(judgment_id=f"j_{decision.entity_id[:6]}",
                        jtype="constraint_relation",
                        subject_ids=[decision.entity_id],
                        reference_ids=[c.entity_id for c in constraints],
                        evidence_ids=[decision.event_id],
                        verdict="supports_conflict", model_id="fixed",
                        reason="fixed")


def _mk_loop(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.add_entity(Entity(entity_id="c1", kind="constraint",
                             quote="禁止用词表裁决", span=None, event_id="u1",
                             source_role="user_requirement", scope="s"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      _Conflict(), "s")
    return ledger, loop


def _events():
    yield Event(event_id="u1", kind="message", source_role="user_requirement",
                ts="t", text="禁止用词表裁决")
    yield Event(event_id="d1", kind="tool_call",
                source_role="assistant_claim", ts="t",
                text=json.dumps({"Edit": {"file": "a.ts",
                                          "content": "新增别名表"}}))
    yield Event(event_id="d2", kind="tool_call",
                source_role="assistant_claim", ts="t",
                text=json.dumps({"Edit": {"file": "a.ts",
                                          "content": "新增别名表"}}))


def test_session_runner_emits_host_blocks(tmp_path):
    ledger, loop = _mk_loop(tmp_path)
    sink_blocks = []
    emitter = ReminderEmitter(sink=sink_blocks.append)
    sr = SessionRunner(loop, emitter).run_stream(_events(), session="sess1",
                                                 source="test")
    assert sr.events == 3 and sr.decisions == 2
    assert sr.conflicts == 1 and sr.reminders_deduped == 1
    assert len(sink_blocks) == 1
    block = sink_blocks[0]
    assert block["type"] == "constraint-reminder"
    assert block["policy"] == "remind-only"
    assert block["context"]["session"] == "sess1"
    assert block["context"]["event_id"] == "d1"
    # emitter 自身也留档，可 drain 为 JSONL
    lines = emitter.drain_jsonl().splitlines()
    assert json.loads(lines[0])["dedup_key"] == block["dedup_key"]


def test_emitter_sink_failure_is_nonblocking(tmp_path):
    ledger, loop = _mk_loop(tmp_path)

    def boom(block):
        raise RuntimeError("sink down")

    emitter = ReminderEmitter(sink=boom)
    sr = SessionRunner(loop, emitter).run_stream(_events(), session="s1")
    assert emitter.emitted          # 已留档
    assert emitter.failures and emitter.failures[0]["stage"] == "emit"
    assert sr.events == 3           # 主流程未被打断


def test_runner_without_sink_still_records(tmp_path):
    ledger, loop = _mk_loop(tmp_path)
    emitter = ReminderEmitter(sink=None)
    sr = SessionRunner(loop, emitter).run_stream(_events())
    assert len(emitter.emitted) == 1 and not emitter.failures


def test_adapter_omp_toolcall_becomes_decision_event(tmp_path):
    """omp toolCall → tool_call 事件（决策面），文本消息 → message。"""
    from moonbow.intent.adapters import omp_events_from_jsonl
    import tempfile, os
    lines = [
        json.dumps({"type": "message", "message": {
            "role": "user", "timestamp": "t1",
            "content": [{"type": "text", "text": "禁止用词表裁决"}]}}),
        json.dumps({"type": "message", "message": {
            "role": "assistant", "timestamp": "t2",
            "content": [{"type": "text", "text": "好的"},
                        {"type": "toolCall", "name": "Edit",
                         "arguments": {"file": "a.ts"}}]}}),
        json.dumps({"type": "message", "message": {
            "role": "toolResult", "timestamp": "t3",
            "content": [{"type": "text", "text": "写入成功"}]}}),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        evs = list(omp_events_from_jsonl(path))
    finally:
        os.unlink(path)
    kinds = [e.kind for e in evs]
    assert kinds == ["message", "message", "tool_call", "tool_result"]
    tool_ev = [e for e in evs if e.kind == "tool_call"][0]
    assert "Edit" in tool_ev.text and "a.ts" in tool_ev.text
    assert tool_ev.source_role == "assistant_claim"


def test_adapter_kimi_function_becomes_decision_event(tmp_path):
    """kimi function 项 → tool_call 事件。"""
    from moonbow.intent.adapters import kimi_events_from_jsonl
    import tempfile, os
    lines = [
        json.dumps({"role": "user", "content": "禁止用词表裁决"}),
        json.dumps({"role": "assistant", "content": [
            {"type": "think", "thinking": "..."},
            {"type": "text", "text": "好的"},
            {"type": "function", "id": "t1",
             "function": {"name": "WriteFile",
                          "arguments": "{\"path\": \"b.ts\"}"}}]}),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        evs = list(kimi_events_from_jsonl(path))
    finally:
        os.unlink(path)
    kinds = [e.kind for e in evs]
    assert "tool_call" in kinds
    tool_ev = [e for e in evs if e.kind == "tool_call"][0]
    assert "WriteFile" in tool_ev.text

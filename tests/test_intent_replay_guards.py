# -*- coding: utf-8 -*-
"""tests/test_intent_replay_guards.py — 两级判官架构的回归。

架构（用户 2026-10-11 定名）：LLM 定性捕获 → LLM 定量复核。
- 冲突投递前一律升级第二级复核（KEEP/DROP）；低置信不是丢弃而是升级；
- 数值门槛方案已被否决撤销（margin 不稳定未校准，阈值裁决与词表同族）；
- 复核判官崩溃 → 提醒照发（fail-open 到提醒侧）且失败显式记录；
- 会话预算是防刷屏保险丝：超预算落账不投递。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import IntentLoop  # noqa: E402
from moonbow.intent.schema import Entity, Event, Judgment  # noqa: E402


class _Conflict:
    model_id = "pass1"

    def judge(self, decision, constraints, event_text, event_kind="", scope=""):
        return Judgment(judgment_id=f"j1_{decision.entity_id[:6]}",
                        jtype="constraint_relation",
                        subject_ids=[decision.entity_id],
                        reference_ids=[c.entity_id for c in constraints],
                        evidence_ids=[decision.event_id],
                        verdict="supports_conflict", model_id=self.model_id,
                        reason="pass1 conflict")


class _Verify:
    """可编程复核判官：keep=True → KEEP；keep=False → DROP；boom → 崩溃。"""

    model_id = "pass2/verify"

    def __init__(self, keep=True, boom=False):
        self.keep, self.boom = keep, boom

    def verify(self, decision, constraints, claimed, decision_text=""):
        if self.boom:
            raise RuntimeError("verify down")
        if self.keep:
            return Judgment(
                judgment_id=f"j2_{decision.entity_id[:6]}",
                jtype="constraint_relation",
                subject_ids=claimed.subject_ids,
                reference_ids=claimed.reference_ids,
                evidence_ids=claimed.evidence_ids,
                verdict="supports_conflict", model_id=self.model_id,
                reason="复核KEEP（维持第一级结论）")
        return Judgment(
            judgment_id=f"j2_{decision.entity_id[:6]}",
            jtype="constraint_relation",
            subject_ids=claimed.subject_ids,
            reference_ids=claimed.reference_ids,
            evidence_ids=[], verdict="insufficient_info",
            model_id=self.model_id, abstain=True,
            abstain_reason="第二级复核判定第一级冲突结论不可靠",
            reason="复核DROP")


def _loop(tmp_path, verify=None, max_per_session=0):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.add_entity(Entity(entity_id="c1", kind="constraint",
                             quote="禁止用词表裁决", span=None, event_id="u1",
                             source_role="user_requirement", scope="s"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      _Conflict(), "s", verify_judge=verify,
                      max_per_session=max_per_session)
    return ledger, loop


def _commit(text, eid):
    return Event(event_id=eid, kind="tool_call",
                 source_role="assistant_claim", ts="t", text=text)


def test_conflict_verified_keep_surfaces_and_logs_both(tmp_path):
    ledger, loop = _loop(tmp_path, verify=_Verify(keep=True))
    rep = loop.process_event(_commit("违规改动", "d1"))
    assert rep.judgments == 2                    # 一级 + 复核都落账
    assert len(rep.reminders_emitted) == 1
    js = ledger.all_judgments()
    assert js[0].verdict == "supports_conflict"
    assert js[1].model_id.endswith("/verify")


def test_conflict_verified_drop_suppressed_but_logged(tmp_path):
    """复核 DROP → 提醒降级为弃权、不投递；两级判定都在账本可追溯。"""
    ledger, loop = _loop(tmp_path, verify=_Verify(keep=False))
    rep = loop.process_event(_commit("疑似违规改动", "d1"))
    assert rep.judgments == 2 and not rep.reminders_emitted
    js = ledger.all_judgments()
    assert js[0].verdict == "supports_conflict"  # 第一级结论保留
    assert js[1].verdict == "insufficient_info" and js[1].abstain
    assert "不可靠" in js[1].abstain_reason


def test_verify_crash_fails_open_to_reminder_with_log(tmp_path):
    """复核崩溃：提醒照发（宁可误报不可漏报）且失败显式记录。"""
    ledger, loop = _loop(tmp_path, verify=_Verify(boom=True))
    rep = loop.process_event(_commit("违规改动", "d1"))
    assert rep.failures and rep.failures[0]["stage"] == "verify"
    assert len(rep.reminders_emitted) == 1


def test_no_verify_judge_single_level_compat(tmp_path):
    ledger, loop = _loop(tmp_path, verify=None)
    rep = loop.process_event(_commit("违规改动", "d1"))
    assert rep.judgments == 1 and len(rep.reminders_emitted) == 1


def test_session_budget_suppresses_but_logs(tmp_path):
    ledger, loop = _loop(tmp_path, verify=_Verify(keep=True),
                         max_per_session=1)
    r1 = loop.process_event(_commit("违规改动甲", "d1"))
    r2 = loop.process_event(_commit("违规改动乙", "d2"))
    assert len(r1.reminders_emitted) == 1
    assert not r2.reminders_emitted and r2.judgments == 2  # 判定照常落账


def test_dedup_across_verify_rounds(tmp_path):
    """同一决策文本经复核后只投一次（内容基去重在复核之后仍生效）。"""
    ledger, loop = _loop(tmp_path, verify=_Verify(keep=True))
    r1 = loop.process_event(_commit("同样的违规改动", "d1"))
    r2 = loop.process_event(_commit("同样的违规改动", "d2"))
    assert len(r1.reminders_emitted) == 1
    assert not r2.reminders_emitted and r2.reminders_deduped == 1

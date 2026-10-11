# -*- coding: utf-8 -*-
"""tests/test_intent_goal_coverage.py — 目标验收闭环（C1 / 审计 D9）。

覆盖用户点名的五类场景：
  1. 测试通过但目标未完成（机械证据 ≠ 目标达成）
  2. 助手声称完成但无产物（no_evidence）
  3. 后台承诺尚未兑现（pending_promise 不产生完成结论）
  4. 部分完成（supported_partial）
  5. 用户修订目标后按新版本对账

全部为纯逻辑测试（无模型）：语义裁决后端用可注入的 Fake，
默认后端（无后端时）必须是诚实 abstain，绝不默认"已完成"。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import (  # noqa: E402
    CoverageJudge, InsufficientInfoJudge, IntentLoop)
from moonbow.intent.schema import Entity, Event, Judgment  # noqa: E402


def _mk(tmp_path, coverage_judge=None, detector=lambda t: "已完成"):
    ledger = Ledger(tmp_path / "l.jsonl")
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      InsufficientInfoJudge(), "s",
                      coverage_judge=coverage_judge, claim_detector=detector)
    return ledger, loop


def _goal(ledger, quote="修复偶发贴底加载，并给出根因", eid="g1",
          source_role="user_requirement"):
    g = Entity(entity_id=eid, kind="goal", quote=quote, span=None,
               event_id="u1", source_role=source_role, scope="s")
    return ledger.add_entity(g)


def _claim_event(text="已修复，问题解决。", eid="m1"):
    return Event(event_id=eid, kind="message", source_role="assistant_claim",
                 ts="t", text=text)


class _FixedCoverage:
    model_id = "fixed-coverage"

    def __init__(self, verdict):
        self.verdict = verdict

    def judge(self, goal, claim, evidence, goals_all=None):
        return Judgment(judgment_id=f"jc_{self.verdict}",
                        jtype="goal_coverage",
                        subject_ids=[claim.entity_id],
                        reference_ids=[goal.entity_id],
                        evidence_ids=[e["evidence_id"] for e in evidence],
                        verdict=self.verdict, model_id=self.model_id,
                        reason=f"fixed:{self.verdict}",
                        evidence_kinds=[e["kind"] for e in evidence],
                        evidence_results=[e["result"] for e in evidence])


def test_default_backend_is_honest_abstain_not_completed(tmp_path):
    """无语义后端时：宣称被登记，但绝不默认判定"已完成"。"""
    ledger, loop = _mk(tmp_path)
    _goal(ledger)
    rep = loop.process_event(_claim_event())
    assert rep.claims_registered == 1
    js = ledger.all_judgments()
    assert len(js) == 1 and js[0].verdict == "insufficient_info"
    assert js[0].abstain and js[0].abstain_reason
    assert not rep.reminders_emitted      # abstain 不产生"未完成"指控


def test_claim_registered_even_without_goal(tmp_path):
    """没有目标登记时宣称仍须留痕（否则等于静默放行）。"""
    ledger, loop = _mk(tmp_path)
    rep = loop.process_event(_claim_event())
    assert rep.claims_registered == 1
    kinds = [e.kind for e in ledger.all_entities()]
    assert "completion_claim" in kinds
    assert rep.judgments == 0             # 无目标可对账


# ── 场景 1：测试通过但目标未完成 ────────────────────────────────────────

def test_green_tests_do_not_mean_goal_met(tmp_path):
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("supported_partial"))
    _goal(ledger)
    loop.register_evidence("tr1", "test_run", "pass", note="pytest 687 passed")
    rep = loop.process_event(_claim_event())
    js = ledger.all_judgments()
    assert js[0].verdict == "supported_partial"        # 不是 supported_full
    assert js[0].evidence_results == ["pass"]          # 机械事实如实记录
    assert rep.reminders_emitted                        # 部分完成 → 提醒
    assert rep.reminders_emitted[0].verdict == "supported_partial"


def test_coverage_input_makes_boundary_explicit():
    """覆盖裁决的 prompt 必须显式写明"测试通过≠目标达成"边界。"""
    from moonbow.intent import protocol
    g = Entity(entity_id="g1", kind="goal", quote="修复并给出根因", span=None,
               event_id="u1", source_role="user_requirement", scope="s")
    c = Entity(entity_id="c1", kind="completion_claim", quote="已修复", span=None,
               event_id="m1", source_role="assistant_claim", scope="s")
    ci = protocol.build_coverage_input(
        g, c, [{"evidence_id": "e1", "kind": "test_run", "result": "pass",
                "note": "pytest passed"}], goals_all=[g])
    assert "测试是否通过" in ci.prompt and "目标是否达成" in ci.prompt
    assert "test_run" in ci.prompt and "pass" in ci.prompt
    assert "修复并给出根因" in ci.prompt
    assert ci.evidence_results == ["pass"]


# ── 场景 2：声称完成但无产物 ────────────────────────────────────────────

def test_claim_without_evidence_yields_no_evidence_and_reminder(tmp_path):
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("no_evidence"))
    _goal(ledger)
    rep = loop.process_event(_claim_event("全部搞定，已经解决。"))
    js = ledger.all_judgments()
    assert js[0].verdict == "no_evidence" and js[0].evidence_ids == []
    assert rep.reminders_emitted and \
        rep.reminders_emitted[0].verdict == "no_evidence"


def test_no_evidence_verdict_requires_claim_binding():
    """no_evidence 是"查过但没证据"的结论：它必须绑定被对账的宣称，
    但**不要求** evidence_ids 非空（否则该判定无法表达）。"""
    j = Judgment(judgment_id="j", jtype="goal_coverage",
                 subject_ids=["c1"], reference_ids=["g1"], evidence_ids=[],
                 verdict="no_evidence", model_id="m")
    assert j.verdict == "no_evidence"
    # 但 supported_full/partial 必须绑定证据
    with pytest.raises(ValueError, match="evidence_ids"):
        Judgment(judgment_id="j2", jtype="goal_coverage",
                 subject_ids=["c1"], reference_ids=["g1"], evidence_ids=[],
                 verdict="supported_full", model_id="m")
    with pytest.raises(ValueError, match="evidence_ids"):
        Judgment(judgment_id="j3", jtype="goal_coverage",
                 subject_ids=["c1"], reference_ids=["g1"], evidence_ids=[],
                 verdict="supported_partial", model_id="m")
    # 且必须绑定主体与目标
    with pytest.raises(ValueError, match="subject and reference"):
        Judgment(judgment_id="j4", jtype="goal_coverage",
                 subject_ids=[], reference_ids=["g1"], evidence_ids=[],
                 verdict="no_evidence", model_id="m")


# ── 场景 3：后台承诺尚未兑现 ────────────────────────────────────────────

def test_pending_promise_not_treated_as_completed(tmp_path):
    """延后承诺：登记为 pending_promise，不产生完成结论。"""
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("no_evidence"))
    _goal(ledger)
    p = Entity(entity_id="p1", kind="goal", quote="稍后给出验收报告",
               span=None, event_id="m0", source_role="assistant_claim",
               scope="s", status="pending_promise")
    ledger.add_entity(p)
    rep = loop.process_event(_claim_event("报告稍后补上。"))
    # 有效目标只含 active 的 g1，pending 承诺不参与"已覆盖"的判定
    covered = [j for j in ledger.all_judgments()
               if j.verdict == "supported_full"]
    assert not covered
    assert rep.reminders_emitted  # 仍按 no_evidence 提醒


# ── 场景 4：部分完成 ────────────────────────────────────────────────────

def test_partial_completion_emits_partial_reminder(tmp_path):
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("supported_partial"))
    _goal(ledger)
    loop.register_evidence("tr1", "test_run", "pass", note="单测通过")
    rep = loop.process_event(_claim_event("主路径已修好。"))
    assert rep.reminders_emitted[0].verdict == "supported_partial"
    assert rep.reminders_emitted[0].jtype == "goal_coverage"


def test_full_completion_emits_no_reminder(tmp_path):
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("supported_full"))
    _goal(ledger)
    loop.register_evidence("ar1", "artifact", "pass", note="修复产物已提交")
    rep = loop.process_event(_claim_event())
    assert rep.judgments == 1 and not rep.reminders_emitted


# ── 场景 5：目标被用户修订后按新版本对账 ────────────────────────────────

def test_target_revised_then_reconciled_against_new_version(tmp_path):
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("supported_partial"))
    old = _goal(ledger, quote="修复贴底加载", eid="g1")
    new = Entity(entity_id="g2", kind="goal",
                 quote="修复贴底加载，并且给出根因分析", span=None,
                 event_id="u2", source_role="user_requirement", scope="s",
                 revises=old.entity_id)
    ledger.add_entity(new)
    loop.register_evidence("tr1", "test_run", "pass", note="主要用例通过")
    rep = loop.process_event(_claim_event("加载问题已修复。"))
    js = [j for j in ledger.all_judgments() if j.jtype == "goal_coverage"]
    assert len(js) == 1
    assert js[0].reference_ids == ["g2"]        # 对账的是修订后的新目标
    assert rep.reminders_emitted


def test_assistant_cannot_self_crown_target(tmp_path):
    """助手不能把自述当成用户目标（否则可以自我降低验收标准）。"""
    with pytest.raises(ValueError, match="forbids source_role"):
        Entity(entity_id="c1", kind="constraint", quote="只做表面修复",
               span=None, event_id="a1", source_role="assistant_claim",
               scope="s")


# ── 证据登记的机械校验 ──────────────────────────────────────────────────

def test_register_evidence_validates_kind_and_result(tmp_path):
    ledger, loop = _mk(tmp_path)
    with pytest.raises(ValueError, match="illegal evidence kind"):
        loop.register_evidence("e1", "vibes", "pass")
    with pytest.raises(ValueError, match="illegal evidence result"):
        loop.register_evidence("e2", "test_run", "mostly-fine")


def test_claim_detector_failure_is_nonblocking(tmp_path):
    def boom(text):
        raise RuntimeError("detector down")

    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("no_evidence"),
                       detector=boom)
    _goal(ledger)
    rep = loop.process_event(_claim_event())
    assert rep.failures and rep.failures[0]["stage"] == "claim_detect"


def test_coverage_backend_failure_registers_abstain_not_pass(tmp_path):
    class _Boom:
        model_id = "boom"

        def judge(self, *a, **k):
            raise RuntimeError("down")

    ledger, loop = _mk(tmp_path, coverage_judge=_Boom())
    _goal(ledger)
    rep = loop.process_event(_claim_event())
    js = ledger.all_judgments()
    assert js[0].verdict == "insufficient_info" and js[0].abstain
    assert rep.failures and rep.failures[0]["stage"] == "coverage_judge"
    assert not rep.reminders_emitted


# ── 端到端：证据 → 宣称 → 目标 → 提醒（完整对账链）──────────────────────

def test_end_to_end_reconciliation_chain(tmp_path):
    """完整链路：用户目标 → 机械证据 → 助手宣称 → 覆盖裁决 → 提醒。

    断言的核心不是"模型判对了"，而是**每一环都可追溯**：证明确实是
    那条机械证据、针对那个目标版本、由那个后端得出那条结论。
    """
    ledger, loop = _mk(tmp_path, coverage_judge=_FixedCoverage("supported_partial"),
                       detector=lambda t: "加载问题已修复，根因稍后补")
    goal = _goal(ledger, quote="修复偶发贴底加载，并给出根因分析")
    loop.register_evidence("tr1", "test_run", "pass", note="pytest 687 passed")
    loop.register_evidence("ar1", "artifact", "partial", note="仅改了加载分支")

    rep = loop.process_event(_claim_event("加载问题已修复，根因稍后补"))

    assert rep.claims_registered == 1
    claims = [e for e in ledger.all_entities() if e.kind == "completion_claim"]
    assert len(claims) == 1
    assert claims[0].quote == "加载问题已修复，根因稍后补"
    assert claims[0].source_role == "assistant_claim"

    js = [j for j in ledger.all_judgments() if j.jtype == "goal_coverage"]
    assert len(js) == 1
    j = js[0]
    assert j.subject_ids == [claims[0].entity_id]     # 绑到那条宣称
    assert j.reference_ids == [goal.entity_id]        # 绑到那个目标版本
    assert set(j.evidence_ids) == {
        e.entity_id for e in ledger.effective("evidence_ref", "s")}
    assert j.evidence_results == ["pass", "partial"]  # 机械事实如实带出

    assert len(rep.reminders_emitted) == 1
    r = rep.reminders_emitted[0]
    assert r.jtype == "goal_coverage" and r.verdict == "supported_partial"
    assert r.subject_ids == [claims[0].entity_id]
    assert r.policy == "remind-only"                  # 提醒，不阻断

    # 同一声称重放 → 去重，不刷屏
    rep2 = loop.process_event(_claim_event("加载问题已修复，根因稍后补"))
    assert not rep2.reminders_emitted and rep2.reminders_deduped == 1

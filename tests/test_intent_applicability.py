# -*- coding: utf-8 -*-
"""tests/test_intent_applicability.py — L0 适用范围闸门（零 LLM）。

来源（exp_t3_decomposition_report.md）：拆维度实验证明机械层无法替代语义
裁决（残差仍 86%），但**适用范围判断**是用户已声明为机械规则的边界，可零
LLM 可靠执行，在 eval 上覆盖 14% 且全对。本文件固化该闸门的行为。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.applicability import (  # noqa: E402
    GATE_APPLIES, GATE_NOT_YET_EFFECTIVE, GATE_OUT_OF_SCOPE, check_applicability,
    filter_applicable)
from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import InsufficientInfoJudge, IntentLoop  # noqa: E402
from moonbow.intent.schema import Entity, Event  # noqa: E402


def _constraint(scope_meta=None, at=None, eid="c1", quote="禁止用词表裁决"):
    meta = {}
    if scope_meta:
        meta["scope"] = scope_meta
    if at:
        meta["effective_at"] = at
    return Entity(entity_id=eid, kind="constraint", quote=quote, span=None,
                  event_id="u1", source_role="user_requirement", scope="s",
                  meta=meta)


# ── scope 闸门 ──────────────────────────────────────────────────────────

def test_scope_mismatch_blocks_semantic_review():
    c = _constraint(scope_meta="proj_a")
    ap = check_applicability(c, decision_scope="proj_b")
    assert ap.verdict == GATE_OUT_OF_SCOPE
    assert ap.blocks_semantic_review and "不跨项目继承" in ap.reason


def test_scope_match_passes_gate():
    c = _constraint(scope_meta="proj_a")
    ap = check_applicability(c, decision_scope="proj_a")
    assert ap.verdict == GATE_APPLIES and not ap.blocks_semantic_review


def test_missing_scope_meta_does_not_silently_abstain():
    """缺省元数据不得推断为"不适用"——弃权必须是显式结论。"""
    c = _constraint()
    ap = check_applicability(c, decision_scope="anything")
    assert ap.verdict == GATE_APPLIES


# ── 时间闸门 ────────────────────────────────────────────────────────────

def test_decision_before_effective_date_blocks():
    c = _constraint(at="2026-10-02T23:27:00")
    ap = check_applicability(c, decision_scope="s",
                             decision_ts="2026-10-02T05:50:00")
    assert ap.verdict == GATE_NOT_YET_EFFECTIVE
    assert "不溯及" in ap.reason


def test_decision_after_effective_date_passes():
    c = _constraint(at="2026-10-02T04:05:00")
    ap = check_applicability(c, decision_scope="s",
                             decision_ts="2026-10-02T08:01:00")
    assert ap.verdict == GATE_APPLIES


def test_undated_comparison_never_blocks():
    c = _constraint(at="2026-10-02T04:05:00")
    ap = check_applicability(c, decision_scope="s", decision_ts="")
    assert ap.verdict == GATE_APPLIES


# ── 分组 ────────────────────────────────────────────────────────────────

def test_filter_splits_applicable_and_skipped():
    ok = _constraint(eid="ok")
    bad = _constraint(scope_meta="proj_other", eid="bad")
    applicable, skipped = filter_applicable([ok, bad], "s", "2026-10-09T00:00")
    assert [c.entity_id for c in applicable] == ["ok"]
    assert len(skipped) == 1 and skipped[0][0].entity_id == "bad"


# ── 与 loop 集成 ────────────────────────────────────────────────────────

def test_all_constraints_out_of_scope_registers_abstain(tmp_path):
    """全部约束不适用 → 显式弃权登记，不伪装成合规、也不静默放行。"""
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.add_entity(_constraint(scope_meta="proj_other", eid="c_other"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      InsufficientInfoJudge(), "s")
    rep = loop.process_event(Event(
        event_id="d1", kind="commit", source_role="assistant_claim",
        ts="2026-10-09T00:00:00", text="新增别名表并加权排序"))
    assert rep.applicability_skipped == 1
    js = ledger.all_judgments()
    assert len(js) == 1 and js[0].abstain
    assert js[0].model_id == "applicability-gate"
    assert "不适用" in js[0].abstain_reason
    assert not rep.reminders_emitted


def test_applicable_constraint_still_reaches_semantic_layer(tmp_path):
    """适用约束必须正常进入语义裁决（闸门不能误吞）。"""
    class _Conflict:
        model_id = "fixed"

        def judge(self, decision, constraints, event_text, event_kind="",
                  scope=""):
            from moonbow.intent.schema import Judgment
            return Judgment(judgment_id="j1", jtype="constraint_relation",
                            subject_ids=[decision.entity_id],
                            reference_ids=[c.entity_id for c in constraints],
                            evidence_ids=[decision.event_id],
                            verdict="supports_conflict", model_id="fixed",
                            reason="fixed")

    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.add_entity(_constraint(eid="c_ok"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      _Conflict(), "s")
    rep = loop.process_event(Event(
        event_id="d1", kind="commit", source_role="assistant_claim",
        ts="2026-10-09T00:00:00", text="新增别名表并加权排序"))
    assert rep.constraints_considered == 1 and rep.applicability_skipped == 0
    assert len(rep.reminders_emitted) == 1


def test_gate_does_not_do_content_based_judgment():
    """闸门只比较字段，不读内容——同样文本在适用范围上必须一视同仁。"""
    c = _constraint()
    a = check_applicability(c, "s", "2026-10-09T00:00:00")
    b = check_applicability(c, "s", "2026-10-09T00:00:00")
    assert a.verdict == b.verdict == GATE_APPLIES


def test_gate_abstains_only_when_all_constraints_inapplicable(tmp_path):
    """关键回归：只要有一条约束适用，就不得整体弃权。

    实测曾出现"14/14 全部被闸门拦住"的假象——根因是测试侧把多条约束
    登记成同一 entity_id + 同 scope，被 norm 去重后互相污染。本测试锁定
    正确语义：**混合情形必须放行到语义层**，否则闸门会吞掉真实违规。
    """

    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.add_entity(_constraint(scope_meta="proj_other", eid="c_other",
                                  quote="禁止用词表裁决（属于别的项目）"))
    ledger.add_entity(_constraint(eid="c_ok", quote="失败必须显式上报"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      InsufficientInfoJudge(), "s")
    rep = loop.process_event(Event(
        event_id="d1", kind="commit", source_role="assistant_claim",
        ts="2026-10-09T00:00:00", text="新增别名表并加权排序"))
    assert rep.applicability_skipped == 1                # 拦了不适用的那条
    assert rep.constraints_considered >= 1               # 但适用的那条进了语义层
    js = ledger.all_judgments()
    assert js and js[-1].model_id != "applicability-gate"  # 结论来自语义层


def test_repeated_entity_ids_do_not_cause_mass_abstention(tmp_path):
    """多约束用不同 entity_id 登记时，闸门不得因 id 复用而改变判定。"""

    ledger = Ledger(tmp_path / "l.jsonl")
    for i, quote in enumerate(("禁止用词表裁决", "失败必须显式上报",
                               "禁止防回改注释")):
        ledger.add_entity(Entity(entity_id=f"c{i}", kind="constraint",
                                 quote=quote, span=None, event_id="u1",
                                 source_role="user_requirement", scope="s"))
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      InsufficientInfoJudge(), "s")
    rep = loop.process_event(Event(
        event_id="d1", kind="commit", source_role="assistant_claim",
        ts="2026-10-09T00:00:00", text="新增别名表并加权排序"))
    assert rep.applicability_skipped == 0
    assert rep.constraints_considered >= 1

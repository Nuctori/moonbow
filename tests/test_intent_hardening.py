# -*- coding: utf-8 -*-
"""tests/test_intent_hardening.py — 第二轮审计缺陷的回归测试（纯逻辑，无模型）。

覆盖：D1 约束进入裁决输入 / D2 三分类 / D3 判断归属 / D4 决策事件面 /
D5 非阻塞 / D6 修订权限 / D7 abstain 契约 / 引用校验 / 截断显式。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent import protocol  # noqa: E402
from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import (  # noqa: E402
    InsufficientInfoJudge, IntentLoop)
from moonbow.intent.schema import Entity, Event, Judgment  # noqa: E402


def _cons(tmp_path, quote="禁止用关键词表解决问题", eid="c1"):
    return Entity(entity_id=eid, kind="constraint", quote=quote, span=None,
                  event_id="u1", source_role="user_requirement", scope="s")


def _commit(text, eid="d1"):
    return Event(event_id=eid, kind="commit", source_role="assistant_claim",
                 ts="2026-10-09T00:00:00", text=text)


# ── D1：约束原文必须进入裁决输入 ────────────────────────────────────────

def test_judgment_input_contains_constraint_text():
    c = Entity(entity_id="c1", kind="constraint",
               quote="禁止用关键词表解决问题", span=None, event_id="u1",
               source_role="user_requirement", scope="s")
    ji = protocol.build_judgment_input("用别名表加权排序", [c],
                                       event_kind="commit", scope="s")
    assert "禁止用关键词表解决问题" in ji.prompt
    assert "[c1]" in ji.prompt
    assert "用别名表加权排序" in ji.prompt
    assert "commit" in ji.prompt


def test_different_constraints_produce_different_input():
    """同一决策在相反约束下必须产生不同输入（否则等于没读到约束）。"""
    d = "新增 alias 表"
    c1 = Entity(entity_id="c1", kind="constraint", quote="禁止词表", span=None,
                event_id="u1", source_role="user_requirement", scope="s")
    c2 = Entity(entity_id="c2", kind="constraint", quote="允许词表", span=None,
                event_id="u2", source_role="user_requirement", scope="s")
    p1 = protocol.build_judgment_input(d, [c1]).prompt
    p2 = protocol.build_judgment_input(d, [c2]).prompt
    assert p1 != p2


def test_long_decision_marked_truncated_not_silently_cut():
    ji = protocol.build_judgment_input("行\n" * 5000, [])
    assert ji.truncated and ji.truncation_note
    assert len(ji.decision_chunks) > 1
    assert ji.prompt  # 仍产出可判输入，但状态显式


# ── D2：三分类 ──────────────────────────────────────────────────────────

def test_protocol_labels_are_three():
    assert protocol.LABELS == ("violation", "compliant", "insufficient")
    assert protocol.LABEL_OF_VERDICT["insufficient_info"] == "insufficient"
    assert protocol.VERDICT_OF_LABEL["insufficient"] == "insufficient_info"


# ── D5/D7：非阻塞与 abstain 契约 ────────────────────────────────────────

class _BoomJudge:
    model_id = "boom"

    def judge(self, *a, **k):
        raise RuntimeError("backend outage")


def _loop(tmp_path, judge=None, name="ledger.jsonl"):
    ledger = Ledger(tmp_path / name)
    stage = CaptureStage(FakeCaptureBackend([]))
    return ledger, IntentLoop(ledger, stage, judge or InsufficientInfoJudge(),
                              "s")


def test_backend_failure_never_raises_nonblocking(tmp_path):
    ledger, loop = _loop(tmp_path, judge=_BoomJudge())
    ledger.add_entity(_cons(tmp_path))
    rep = loop.process_event(_commit("用别名表加权"))
    assert rep.failures and rep.failures[0]["stage"] == "judge"
    # 失败如实登记为弃权判断，不伪装成 compliance
    js = ledger.all_judgments()
    assert len(js) == 1 and js[0].abstain and js[0].abstain_reason
    assert not rep.reminders_emitted


def test_abstain_requires_reason():
    with pytest.raises(ValueError, match="abstain_reason"):
        Judgment(judgment_id="j", jtype="constraint_relation",
                 subject_ids=["s"], reference_ids=["r"], evidence_ids=[],
                 verdict="insufficient_info", model_id="m", abstain=True)


def test_abstain_verdict_must_be_insufficient():
    with pytest.raises(ValueError, match="insufficient_info"):
        Judgment(judgment_id="j", jtype="constraint_relation",
                 subject_ids=["s"], reference_ids=["r"],
                 evidence_ids=["e"], verdict="supports_compat",
                 model_id="m", abstain=True, abstain_reason="?")


def test_capture_failure_is_nonblocking(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")

    class _BoomExtract:
        name = "boom"
        last_backend_failed = False
        last_backend_abstain = False
        last_backend_truncated = False

        def extract(self, text):
            raise RuntimeError("capture down")

    # CaptureStage 自身把后端异常吞成 backend_failed（capture 层契约），
    # 这里验证的是**更外层**：即便 stage 本身抛错，loop 也不得抛出。
    class _BoomStage:
        def capture(self, *a, **k):
            raise RuntimeError("stage down")

    loop = IntentLoop(ledger, _BoomStage(), InsufficientInfoJudge(), "s")
    rep = loop.process_event(Event(event_id="m1", kind="message",
                                   source_role="user_requirement",
                                   ts="t", text="任意文本"))
    assert rep.failures and rep.failures[0]["stage"] == "capture"

    # 后端异常被 CaptureStage 转为显式失败（不冒充"无实体"）
    stage = CaptureStage(_BoomExtract())
    res = stage.capture("m2", "任意文本", "s")
    assert res.backend_failed and res.abstain and not res.ok


# ── D6：修订权限 ────────────────────────────────────────────────────────

def test_assistant_cannot_revise_user_entity_via_ledger(tmp_path):
    """助手无法自我加冕约束（schema 层已挡），也无法借修订绕过用户约束。"""
    ledger = Ledger(tmp_path / "l.jsonl")
    user = _cons(tmp_path)
    ledger.add_entity(user)

    # 路径一：助手直接创建 constraint 实体 —— schema 层拒绝
    with pytest.raises(ValueError, match="forbids source_role"):
        Entity(entity_id="bad1", kind="constraint", quote="可以加词表",
               span=None, event_id="a1", source_role="assistant_claim",
               scope="s")

    # 路径二：助手借"修订用户目标"绕过 —— ledger 写入点拒绝
    asst_goal = Entity(entity_id="g1", kind="goal", quote="修复加载问题",
                       span=None, event_id="a1",
                       source_role="assistant_claim", scope="s")
    ledger.add_entity(asst_goal)
    user_goal = Entity(entity_id="g2", kind="goal", quote="修复并给根因",
                       span=None, event_id="u2",
                       source_role="user_requirement", scope="s")
    ledger.add_entity(user_goal)
    asst_revise = Entity(entity_id="g3", kind="goal", quote="改回只修表面",
                         span=None, event_id="a3",
                         source_role="assistant_claim", scope="s",
                         revises=user_goal.entity_id)
    with pytest.raises(ValueError, match="user_requirement"):
        ledger.add_entity(asst_revise)

    # 用户约束仍有效
    assert user.entity_id in [e.entity_id
                              for e in ledger.effective("constraint", "s")]


def test_revision_rejects_dangling_cross_scope_and_kind_change(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    user = _cons(tmp_path)
    ledger.add_entity(user)

    dangling = Entity(entity_id="x1", kind="constraint", quote="q1", span=None,
                      event_id="u2", source_role="user_requirement", scope="s",
                      revises="ent_missing")
    with pytest.raises(ValueError, match="悬空修订"):
        ledger.add_entity(dangling)

    cross = Entity(entity_id="x2", kind="constraint", quote="q2", span=None,
                   event_id="u3", source_role="user_requirement", scope="other",
                   revises=user.entity_id)
    with pytest.raises(ValueError, match="跨 scope"):
        ledger.add_entity(cross)

    kindchg = Entity(entity_id="x3", kind="goal", quote="q3", span=None,
                     event_id="u4", source_role="user_requirement", scope="s",
                     revises=user.entity_id)
    with pytest.raises(ValueError, match="改变 kind"):
        ledger.add_entity(kindchg)


def test_user_revision_deactivates_old_constraint(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    old = _cons(tmp_path)
    ledger.add_entity(old)
    new = Entity(entity_id="c2", kind="constraint", quote="改为允许白名单校验",
                 span=None, event_id="u2", source_role="user_requirement",
                 scope="s", revises=old.entity_id)
    ledger.add_entity(new)
    eff = ledger.effective("constraint", "s")
    assert [e.entity_id for e in eff] == ["c2"]


def test_double_revision_rejected(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    a = _cons(tmp_path, eid="c1")
    ledger.add_entity(a)
    b = Entity(entity_id="c2", kind="constraint", quote="q2", span=None,
               event_id="u2", source_role="user_requirement", scope="s",
               revises=a.entity_id)
    ledger.add_entity(b)
    c = Entity(entity_id="c3", kind="constraint", quote="q3", span=None,
               event_id="u3", source_role="user_requirement", scope="s",
               revises=a.entity_id)
    with pytest.raises(ValueError, match="已被修订"):
        ledger.add_entity(c)


def test_entity_status_validated():
    with pytest.raises(ValueError, match="illegal entity status"):
        Entity(entity_id="e", kind="constraint", quote="q", span=None,
               event_id="u", source_role="user_requirement", scope="s",
               status="bogus")


def test_entity_cannot_revise_itself():
    with pytest.raises(ValueError, match="itself"):
        Entity(entity_id="e1", kind="constraint", quote="q", span=None,
               event_id="u", source_role="user_requirement", scope="s",
               revises="e1")


# ── 引用校验 ────────────────────────────────────────────────────────────

def test_judgment_references_must_exist(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    j = Judgment(judgment_id="j1", jtype="constraint_relation",
                 subject_ids=["ent_ghost"], reference_ids=["ent_ghost2"],
                 evidence_ids=["ev_ghost"], verdict="supports_conflict",
                 model_id="m")
    with pytest.raises(ValueError, match="不可解析"):
        ledger.add_judgment(j)


# ── D4：叙述消息不裁决 ──────────────────────────────────────────────────

def test_narration_message_not_adjudicated(tmp_path):
    ledger, loop = _loop(tmp_path, judge=_BoomJudge())
    ledger.add_entity(_cons(tmp_path))
    rep = loop.process_event(Event(
        event_id="m1", kind="message", source_role="assistant_claim",
        ts="t", text="我先说明一下整体思路和进度。"))
    assert rep.judgments == 0 and not rep.failures


def test_decision_without_constraints_no_judgment(tmp_path):
    ledger, loop = _loop(tmp_path, judge=_BoomJudge())
    rep = loop.process_event(_commit("用别名表加权"))
    assert rep.judgments == 0 and not rep.failures


# ── 提醒去重（真断言，替换恒真断言）─────────────────────────────────────

class _AlwaysConflict:
    model_id = "fixed"

    def judge(self, decision, constraints, event_text, event_kind="", scope=""):
        return Judgment(judgment_id=f"j_{decision.entity_id[:6]}",
                        jtype="constraint_relation",
                        subject_ids=[decision.entity_id],
                        reference_ids=[c.entity_id for c in constraints],
                        evidence_ids=[decision.event_id],
                        verdict="supports_conflict", model_id=self.model_id,
                        reason="fixed margin=2.5")


def test_identical_decision_deduped_distinct_emitted(tmp_path):
    """同一决策文本跨事件 → 去重（内容基键，不随 event_id 变化）；
    不同决策文本 → 各自提醒。真实会话重放抓出的缺陷回归。"""
    ledger, loop = _loop(tmp_path, judge=_AlwaysConflict())
    ledger.add_entity(_cons(tmp_path))
    txt = "用别名表加权解决该问题"
    r1 = loop.process_event(_commit(txt, eid="d1"))
    assert len(r1.reminders_emitted) == 1
    # 同一文本的第二个事件：内容基去重 → 不再发
    r2 = loop.process_event(_commit(txt, eid="d2"))
    assert not r2.reminders_emitted and r2.reminders_deduped == 1
    # 明确不同决策 → 新提醒
    r3 = loop.process_event(_commit("改用白名单校验", eid="d3"))
    assert len(r3.reminders_emitted) == 1
    # 约束集变化 → 同文本也算新问题
    ledger.add_entity(_cons(tmp_path, quote="失败必须显式上报", eid="c2"))
    r4 = loop.process_event(_commit(txt, eid="d4"))
    assert len(r4.reminders_emitted) == 1

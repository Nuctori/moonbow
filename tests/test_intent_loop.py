# -*- coding: utf-8 -*-
"""tests/test_intent_loop.py — intent 模块阶段一契约测试（纯逻辑，无模型依赖）。

覆盖：引文校验、角色规则、修订权限、账本往返、去重、生效查询、
scope 隔离、判断绑定、abstain 显式性、提醒非阻塞、后台承诺 pending。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.schema import (  # noqa: E402
    Entity, Event, Judgment, Reminder,
    CONSTRAINT_VERDICTS, COVERAGE_VERDICTS)
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.verify import verify_quote, store_event_text  # noqa: E402
from moonbow.intent.capture import (  # noqa: E402
    CaptureStage, FakeCaptureBackend)


TEXT = ("用户：修复偶发贴底加载问题。禁止用关键词表解决问题。\n"
        "助手：我已用别名表加权解决了该问题，全部测试通过。")

CONSTRAINT_QUOTE = "禁止用关键词表解决问题"
CLAIM_QUOTE = "我已用别名表加权解决了该问题，全部测试通过"


# ── 引文校验 ─────────────────────────────────────────────────────────────

def test_verify_quote_verbatim_ok():
    ok, span = verify_quote(TEXT, CONSTRAINT_QUOTE)
    assert ok and TEXT[span[0]:span[1]] == CONSTRAINT_QUOTE


def test_verify_quote_non_verbatim_rejected():
    ok, span = verify_quote(TEXT, "禁止用关键词表解决问题呀")
    assert not ok and span is None          # 原文无此句 → 拒绝


def test_verify_quote_span_mismatch_falls_back_to_index():
    ok, span = verify_quote(TEXT, CONSTRAINT_QUOTE, span=(0, 5))
    assert ok and TEXT[span[0]:span[1]] == CONSTRAINT_QUOTE


def test_store_event_text_roundtrip(tmp_path):
    rel, sha = store_event_text(tmp_path, "ev_test_1", TEXT)
    assert rel == "ev_test_1.txt"
    assert (tmp_path / rel).read_text(encoding="utf-8") == TEXT
    import hashlib
    assert sha == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()


# ── 角色规则 ─────────────────────────────────────────────────────────────

def _entity(kind, role, quote, scope="proj_d-react-app", **kw):
    return Entity(entity_id=kw.pop("entity_id"), kind=kind, quote=quote,
                  span=None, event_id=kw.pop("event_id"), source_role=role,
                  scope=scope, **kw)


def test_completion_claim_forbids_tool_observation():
    with pytest.raises(ValueError, match="forbids source_role"):
        _entity("completion_claim", "tool_observation", CLAIM_QUOTE,
                entity_id="e1", event_id="ev1")


def test_constraint_requires_user_or_assistant():
    with pytest.raises(ValueError, match="forbids source_role"):
        _entity("constraint", "tool_observation", CONSTRAINT_QUOTE,
                entity_id="e2", event_id="ev1")


def test_event_kind_validated():
    with pytest.raises(ValueError):
        Event(event_id="ev1", kind="not_a_kind",
              source_role="user_requirement", ts="2026-10-09T00:00:00",
              text=TEXT)


# ── 修订权限 ─────────────────────────────────────────────────────────────

def test_assistant_cannot_revise_user_entity():
    user = _entity("constraint", "user_requirement", CONSTRAINT_QUOTE,
                   entity_id="c_user", event_id="ev1")
    with pytest.raises(ValueError, match="forbids source_role"):
        _entity("constraint", "assistant_claim", "禁止新增词条", entity_id="c_a",
                event_id="ev2", revises=user.entity_id)


def test_user_can_revise_assistant_entity():
    asst = _entity("goal", "assistant_claim", "修复贴底问题", entity_id="g_a",
                   event_id="ev2")
    user_rev = _entity("goal", "user_requirement", "修复贴底问题，并且要给出根因",
                       entity_id="g_user", event_id="ev3", revises=g_id(asst))
    assert user_rev.revises == asst.entity_id


def g_id(e):
    return e.entity_id


# ── 账本 ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


def test_ledger_roundtrip(ledger):
    c = _entity("constraint", "user_requirement", CONSTRAINT_QUOTE,
                entity_id="c1", event_id="ev1")
    ledger.add_entity(c)
    dec = _entity("decision", "assistant_claim", "用别名表加权排序",
                  entity_id="d1", event_id="ev1")
    ledger.add_entity(dec)          # 主体须先登记（引用校验）
    ledger.add_event_record({"event_id": "ev1", "kind": "commit",
                             "source_role": "assistant_claim", "ts": "t"})
    j = Judgment(judgment_id="j1", jtype="constraint_relation",
                 subject_ids=["d1"], reference_ids=["c1"],
                 evidence_ids=["ev1"], verdict="supports_conflict",
                 model_id="test", reason="别名表加权与禁令冲突")
    ledger.add_judgment(j)
    assert [e.entity_id for e in ledger.all_entities()] == ["c1", "d1"]
    js = ledger.all_judgments()
    assert len(js) == 1 and js[0].verdict == "supports_conflict"


def test_ledger_rejects_unresolvable_judgment_refs(ledger):
    """引用不可解析的判断必须被拒（否则回放无法归属证据）。"""
    ledger.add_entity(_entity("constraint", "user_requirement", CONSTRAINT_QUOTE,
                              entity_id="c1", event_id="ev1"))
    ledger.add_event_record({"event_id": "ev1", "kind": "commit",
                             "source_role": "assistant_claim", "ts": "t"})
    bad = Judgment(judgment_id="j9", jtype="constraint_relation",
                   subject_ids=["ghost_decision"], reference_ids=["c1"],
                   evidence_ids=["ev1"], verdict="supports_conflict",
                   model_id="test")
    with pytest.raises(ValueError, match="不可解析"):
        ledger.add_judgment(bad)


def test_ledger_dedup_by_norm(ledger):
    a = _entity("constraint", "user_requirement", CONSTRAINT_QUOTE,
                entity_id="c1", event_id="ev1")
    b = _entity("constraint", "user_requirement", "禁止用关键词表解决问题",
                entity_id="c2", event_id="ev1")
    ledger.add_entity(a)
    ledger.add_entity(b)
    ents = ledger.effective("constraint", "proj_d-react-app")
    assert len(ents) == 1 and a.norm == b.norm


def test_effective_excludes_retracted_and_other_scope(ledger):
    c1 = _entity("constraint", "user_requirement", CONSTRAINT_QUOTE,
                 entity_id="c1", event_id="ev1")
    c2 = _entity("constraint", "user_requirement", "禁止用正则做语义判断",
                 entity_id="c2", event_id="ev1",
                 scope="proj_m-ai-spark-4b")
    c3 = _entity("constraint", "user_requirement", "禁止改文案模板",
                 entity_id="c3", event_id="ev1", status="retracted")
    for c in (c1, c2, c3):
        ledger.add_entity(c)
    got = ledger.effective("constraint", "proj_d-react-app")
    assert [e.entity_id for e in got] == ["c1"]


def test_revision_supersedes_old_entity(ledger):
    old = _entity("constraint", "user_requirement", "禁止用关键词表解决问题",
                  entity_id="c_old", event_id="ev1")
    new = _entity("constraint", "user_requirement",
                  "禁止用关键词表或词表解决问题", entity_id="c_new",
                  event_id="ev2", revises="c_old")
    ledger.add_entity(old)
    ledger.add_entity(new)
    got = ledger.effective("constraint", "proj_d-react-app")
    assert [e.entity_id for e in got] == ["c_new"]


def test_background_promise_stays_pending(ledger):
    p = _entity("decision", "assistant_claim",
                "后台修复完成后我会给验收报告", entity_id="d_bg",
                event_id="ev1", status="pending_promise")
    ledger.add_entity(p)
    got = ledger.effective("decision", "proj_d-react-app")
    assert len(got) == 1 and got[0].status == "pending_promise"


# ── 判断绑定与 abstain 显式性 ────────────────────────────────────────────

def test_judgment_requires_evidence_for_support_verdicts():
    with pytest.raises(ValueError, match="requires evidence_ids"):
        Judgment(judgment_id="j1", jtype="constraint_relation",
                 subject_ids=["d1"], reference_ids=["c1"],
                 evidence_ids=[], verdict="supports_conflict",
                 model_id="test")


def test_judgment_verdict_enum_enforced():
    with pytest.raises(ValueError, match="illegal verdict"):
        Judgment(judgment_id="j1", jtype="goal_coverage",
                 subject_ids=["d1"], reference_ids=["g1"],
                 evidence_ids=["e1"], verdict="supports_conflict",
                 model_id="test")


def test_insufficient_info_allows_empty_evidence_and_is_explicit():
    j = Judgment(judgment_id="j1", jtype="constraint_relation",
                 subject_ids=["d1"], reference_ids=["c1"],
                 evidence_ids=[], verdict="insufficient_info",
                 model_id="none", abstain=True,
                 abstain_reason="no semantic backend configured")
    assert j.abstain and j.verdict == "insufficient_info"


def test_reminder_never_blocks_and_dedup_stable():
    kw = dict(jtype="constraint_relation", subject_ids=["d1"],
              reference_ids=["c1"], reason="别名表加权与禁令冲突")
    r1 = Reminder(verdict="supports_conflict", model_id="test", **kw)
    r2 = Reminder(verdict="supports_conflict", model_id="test", **kw)
    d = r1.to_dict()
    assert d["policy"] == "remind-only" and d["type"] == "constraint-reminder"
    assert r1.dedup_key == r2.dedup_key


# ── 捕获层 ───────────────────────────────────────────────────────────────

class _Cap:
    def __init__(self, kind, quote, span=None):
        self.kind, self.quote, self.span = kind, quote, span


def test_capture_maps_and_verifies():
    backend = FakeCaptureBackend([_Cap("constraint", CONSTRAINT_QUOTE),
                                  _Cap("constraint", "不存在的引文")])
    stage = CaptureStage(backend)
    result = stage.capture("ev1", TEXT, "proj_d-react-app",
                           source_role="user_requirement")
    assert result.ok and not result.abstain
    assert len(result.entities) == 1 and result.dropped_bad_quote == 1
    assert result.entities[0].quote == CONSTRAINT_QUOTE
    assert result.entities[0].source_role == "user_requirement"


def test_capture_drops_assistant_self_crowned_constraint():
    """助手文本断言 constraint → 角色不匹配丢弃（助手转述不能自我加冕为约束）"""
    backend = FakeCaptureBackend([_Cap("constraint", CONSTRAINT_QUOTE)])
    result = CaptureStage(backend).capture("ev1", TEXT, "proj_d-react-app")
    assert result.entities == [] and result.dropped_role_mismatch == 1


def test_capture_abstain_is_explicit_not_empty_ok():
    stage = CaptureStage(FakeCaptureBackend([], abstain=True))
    result = stage.capture("ev1", TEXT, "proj_d-react-app")
    assert not result.ok and result.abstain and result.backend_abstain


def test_capture_failure_is_explicit():
    stage = CaptureStage(FakeCaptureBackend([], failed=True))
    result = stage.capture("ev1", TEXT, "proj_d-react-app")
    assert not result.ok and result.backend_failed


def test_capture_requires_explicit_backend():
    with pytest.raises(ValueError, match="显式注入"):
        CaptureStage(None)

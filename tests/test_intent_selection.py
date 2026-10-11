# -*- coding: utf-8 -*-
"""tests/test_intent_selection.py — 两阶段第一级（约束选择）接线回归。

run13 之后的架构：卡数 >5（捕获漂移）时先选卡再义务判定。
选卡只决定"用哪张约束"，结论仍由义务判官给出；选卡失败 → 显式弃权。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent.capture import CaptureStage, FakeCaptureBackend  # noqa: E402
from moonbow.intent.ledger import Ledger  # noqa: E402
from moonbow.intent.loop import IntentLoop  # noqa: E402
from moonbow.intent.schema import Entity, Event  # noqa: E402


class _ConflictCapture:
    """义务判官 fake：记录每次收到的约束条数。"""

    model_id = "pass1"
    received = []

    def judge(self, decision, constraints, event_text, event_kind="", scope=""):
        type(self).received.append([c.quote for c in constraints])
        from moonbow.intent.schema import Judgment
        return Judgment(judgment_id=f"j_{len(type(self).received)}",
                        jtype="constraint_relation",
                        subject_ids=[decision.entity_id],
                        reference_ids=[c.entity_id for c in constraints],
                        evidence_ids=[decision.event_id],
                        verdict="supports_conflict", model_id=self.model_id,
                        reason="pass1 conflict")


class _SelectPick:
    """选卡 fake：恒选第 2 张（index 1，即干扰卡）。"""

    model_id = "select"

    def select(self, decision_text, card_quotes):
        return 1 if len(card_quotes) > 1 else 0


def _cards(n):
    qs = ["禁止用关键词表做语义裁决", "失败必须显式上报", "禁止防回改注释",
          "测试断言断言政策", "用户可见文案必须来自模型",
          "禁止用正则做语义归类", "不许静默重试超过一次", "文档必须随接口更新"]
    return [Entity(entity_id=f"c{i}", kind="constraint", quote=q, span=None,
                   event_id="u", source_role="user_requirement", scope="s")
            for i, q in enumerate(qs[:n])]


def _loop(tmp_path, cards, verify=None, select=None):
    ledger = Ledger(tmp_path / "l.jsonl")
    for c in cards:
        ledger.add_entity(c)
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                      _ConflictCapture(), "s", verify_judge=verify,
                      select_judge=select)
    return ledger, loop


def _commit(eid="d1"):
    return Event(event_id=eid, kind="tool_call", source_role="assistant_claim",
                 ts="t", text="新增别名表并加权排序")


def test_within_five_cards_no_selection(tmp_path):
    """≤5 张（训练分布内）：不触发选卡，全部约束进义务判官。"""
    cards = _cards(5)
    ledger, loop = _loop(tmp_path, cards)
    _ConflictCapture.received.clear()
    loop.process_event(_commit())
    assert len(_ConflictCapture.received[-1]) == 5   # 全量进判定


def test_over_five_cards_selection_narrows_to_one(tmp_path):
    """>5 张（捕获漂移）：选卡判官选 1 张进义务判官——不拿全卡硬判。"""
    cards = _cards(8)
    ledger, loop = _loop(tmp_path, cards, select=_SelectPick())
    _ConflictCapture.received.clear()
    loop.process_event(_commit())
    got = _ConflictCapture.received[-1]
    assert len(got) == 1                              # 选卡收窄为 1 张
    assert got[0] == "失败必须显式上报"                # fake 选了 index 1


def test_select_crash_fails_closed_to_abstain(tmp_path):
    """选卡判官崩溃 → 显式弃权（不拿全卡硬判，也不静默放行）。"""
    class _Boom:
        model_id = "select"

        def select(self, decision_text, card_quotes):
            raise RuntimeError("select down")

    ledger, loop = _loop(tmp_path, _cards(8), select=_Boom())
    rep = loop.process_event(_commit())
    assert rep.failures and rep.failures[0]["stage"] == "select"
    js = ledger.all_judgments()
    assert js[-1].abstain and "选卡" in (js[-1].abstain_reason or "")


def test_five_cards_untouched_by_gate_drift(tmp_path):
    """L0 闸门过滤后 ≤5 张 → 判官收到的是适用集（回归：曾误传全量）。"""
    from moonbow.intent.schema import Event as _E
    cards = _cards(6)
    # 第 6 张跨 scope（L0 应滤掉）
    extra = Entity(entity_id="c_far", kind="constraint", quote="别项目约束",
                   span=None, event_id="u2",
                   source_role="user_requirement", scope="other",
                   meta={"scope": "other"})
    ledger, loop = _loop(tmp_path, cards)
    ledger.add_entity(extra)
    _ConflictCapture.received.clear()
    loop.process_event(_commit())
    got = _ConflictCapture.received[-1]
    assert all(q != "别项目约束" for q in got)        # 跨 scope 未进判官

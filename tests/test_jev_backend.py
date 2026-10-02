# -*- coding: utf-8 -*-
"""JevBackend 契约测试（注入 fake engine，零权重零 XPU 依赖）。

覆盖：4 模式模板路由与概率语义、缺 context.task 弃权、ts.capture
unsupported、概率非法按 invalid_output 拒绝（不编造分数）、出口必过
validate_response、provenance（backend=jev / template 版本 / revision 回填）。
真实权重路径由 run_jev_eval.py 端到端覆盖（XPU 断言在脚本内）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from moonbow.semantic.backends.base import BackendInvalidOutput
from moonbow.semantic.backends.jev import (
    DEFAULT_MODEL_ID, JEV_TEMPLATE_VERSION, JevBackend,
)
from moonbow.semantic.schema import MatchRequest, validate_response


class FakeEngine:
    """按问题类型回固定概率分布的 duck-typed engine。"""

    def __init__(self, noul_p=0.9, choice_probs=None, bad=False):
        self.noul_p = noul_p
        self.choice_probs = choice_probs or {
            "assert": 0.6, "promise": 0.3, "question": 0.1}
        self.bad = bad
        self.calls = []

    def predict(self, payload):
        self.calls.append(payload)
        q = payload["questions"]["q"]
        if self.bad:
            return {"answers": {"q": {"type": q["type"]}}}
        if q["type"] == "noul":
            p = self.noul_p
            return {"answers": {"q": {
                "type": "noul", "noul": p,
                "probabilities": {"false": 1 - p, "true": p}}}}
        return {"answers": {"q": {
            "type": "choice", "choice": "assert",
            "probabilities": dict(self.choice_probs)}}}


def _backend(**kw):
    return JevBackend(engine=FakeEngine(**kw))


def test_load_count_and_identity():
    b = _backend()
    assert b.load_count == 1
    b.initialize()
    assert b.load_count == 1  # 幂等
    assert b.name == "jev"
    assert b.model_revision  # 注入路径回填 refs/main 解析值


def test_binary_noul_score_and_matched():
    b = _backend(noul_p=0.9)
    r = b.match(MatchRequest(text="完成了一切", pattern="completion.asserted@1"))
    assert (r.status, r.matched, r.score) == ("ok", True, 0.9)
    assert r.score_type == "noul_probability"
    assert r.provenance.backend == "jev"
    assert r.provenance.calibration_id == f"template:{JEV_TEMPLATE_VERSION}"
    validate_response(MatchRequest(text="完成了一切",
                                   pattern="completion.asserted@1"), r)


def test_binary_threshold_respected():
    b = _backend(noul_p=0.55)
    r = b.match(MatchRequest(text="x", pattern="process.unresolved@1"))
    assert r.matched is True  # 0.55 >= 0.5 默认
    r2 = b.match(MatchRequest(text="x", pattern="process.unresolved@1",
                              threshold=0.6))
    assert r2.matched is False


def test_modality_choice_semantics():
    b = _backend(choice_probs={"assert": 0.6, "promise": 0.3, "question": 0.1})
    r = b.match(MatchRequest(text="x", pattern="modality.assertive@1"))
    assert r.score_type == "choice_probability"
    assert r.score == 0.6 and r.matched is True
    # argmax 非 assert → matched=false 即便 prob 高
    b2 = _backend(choice_probs={"assert": 0.2, "promise": 0.7, "question": 0.1})
    r2 = b2.match(MatchRequest(text="x", pattern="modality.assertive@1"))
    assert r2.matched is False
    # argmax=assert 但低于阈值 → false
    b3 = _backend(choice_probs={"assert": 0.45, "promise": 0.35, "question": 0.2})
    r3 = b3.match(MatchRequest(text="x", pattern="modality.assertive@1",
                               threshold=0.5))
    assert r3.matched is False


def test_alignment_abstain_without_task():
    b = _backend()
    r = b.match(MatchRequest(text="x", pattern="task.object.alignment@1"))
    assert (r.status, r.reason_code, r.matched) == (
        "abstain", "insufficient_context", None)


def test_alignment_with_task_goes_noul():
    b = _backend(noul_p=0.9)
    r = b.match(MatchRequest(text="cand", pattern="task.object.alignment@1",
                             context={"task": "task text"}))
    assert r.status == "ok" and r.matched is True and r.score == 0.9
    # state 拼装模板：task 与 text 都进 state
    state = b.engine.calls[-1]["state"]
    assert "task text" in state and "cand" in state


def test_user_text_only_in_state():
    b = _backend()
    b.match(MatchRequest(text="用户原文", pattern="completion.asserted@1"))
    payload = b.engine.calls[-1]
    assert payload["state"] == "用户原文"
    q = payload["questions"]["q"]
    assert "用户原文" not in q["instructions"]
    assert all("用户原文" not in str(v) for v in q["criteria"].values())


def test_ts_capture_unsupported():
    b = _backend()
    r = b.match(MatchRequest(text="x", pattern="ts.capture@1",
                             operation="find_all"))
    assert (r.status, r.reason_code, r.matched) == ("abstain", "unsupported",
                                                    None)


def test_bad_probability_is_invalid_output():
    b = _backend(bad=True)
    with pytest.raises(BackendInvalidOutput):
        b.match(MatchRequest(text="x", pattern="completion.asserted@1"))


def test_capabilities_declare_template_and_patterns():
    caps = _backend().capabilities()
    assert caps["backend"] == "jev"
    assert caps["template_version"] == JEV_TEMPLATE_VERSION
    assert caps["calibrated"] is False
    refs = {p["ref"] for p in caps["patterns"]}
    assert refs == {"completion.asserted@1", "modality.assertive@1",
                    "task.object.alignment@1", "process.unresolved@1"}


def test_all_ok_responses_pass_validate():
    b = _backend()
    for p in ("completion.asserted@1", "modality.assertive@1",
              "task.object.alignment@1", "process.unresolved@1"):
        req = MatchRequest(text="t", pattern=p,
                           context={"task": "tt"}
                           if p == "task.object.alignment@1" else {})
        r = b.match(req)
        validate_response(req, r)


def test_model_id_default():
    assert DEFAULT_MODEL_ID == "TokenRhythm/NeoHorse-Jev-4B"

# -*- coding: utf-8 -*-
"""JevBackend 契约测试（注入 fake engine，零权重零 XPU 依赖）。

覆盖：4 模式模板路由与概率语义、缺 context.task 弃权、ts.capture
unsupported、概率非法按 invalid_output 拒绝（不编造分数）、出口必过
validate_response、provenance（backend=jev / template 版本 / revision 回填）、
模板版本按模式选择（completion=v2，其余=v1；JEV_TEMPLATE_PIN=v1 全模式
回退）与 v2 措辞对 experiments/jev-template-v2.json 的逐字一致性。
真实权重路径由 run_jev_eval.py 端到端覆盖（XPU 断言在脚本内）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from moonbow.semantic.backends.base import BackendInvalidOutput
from moonbow.semantic.backends.jev import (
    DEFAULT_MODEL_ID, JEV_TEMPLATE_PIN_ENV, JEV_TEMPLATE_VERSION, JevBackend,
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


# --- 模板版本化（jev-template-v2 落地，2026-10-04）-------------------------

_V1_PATTERNS = [("process.unresolved@1", {}),
                ("modality.assertive@1", {}),
                ("task.object.alignment@1", {"task": "tt"})]


def test_completion_provenance_template_v2(monkeypatch):
    """completion 模式缺省用 v2：provenance 与 diagnostics 均回填 v2。"""
    monkeypatch.delenv(JEV_TEMPLATE_PIN_ENV, raising=False)
    b = _backend()
    r = b.match(MatchRequest(text="x", pattern="completion.asserted@1"))
    assert r.provenance.calibration_id == "template:jev-template-v2"
    assert b.diagnostics[-1]["template_version"] == "jev-template-v2"
    # payload 用 v2 措辞（"wrap-up statement" v2 独有；v1 是 "fully complete"）
    q = b.engine.calls[-1]["questions"]["q"]
    assert "wrap-up statement" in q["instructions"]
    assert "fully complete" not in q["instructions"]


def test_other_patterns_provenance_template_v1():
    """其余 3 模式维持各自身版本 v1（同卷对照措辞不动）。"""
    b = _backend(noul_p=0.9)
    for p, ctx in _V1_PATTERNS:
        r = b.match(MatchRequest(text="x", pattern=p, context=ctx))
        assert r.provenance.calibration_id == "template:jev-template-v1", p
        assert b.diagnostics[-1]["template_version"] == \
            "jev-template-v1", p


def test_v2_completion_template_verbatim_from_experiment_json():
    """落地的 v2 三段措辞与 experiments/jev-template-v2.json 逐字一致，
    且 v2 集内其余模式仍为 v1 措辞（同卷对照）。"""
    import json
    from pathlib import Path
    spec = json.loads((Path(__file__).resolve().parent.parent /
                       "experiments" / "jev-template-v2.json")
                      .read_text(encoding="utf-8"))
    from moonbow.semantic.backends import jev as jev_mod
    v2 = jev_mod._JEV_TEMPLATE_SETS["jev-template-v2"]
    assert v2["completion.asserted"] == spec["templates"]["completion.asserted"]
    for name in ("modality.assertive", "task.object.alignment",
                 "process.unresolved"):
        assert v2[name] == spec["templates"][name], name


def test_pin_v1_rolls_back_all_patterns(monkeypatch):
    """env JEV_TEMPLATE_PIN=v1：全模式（含 completion）回退 v1 措辞与版本。"""
    monkeypatch.setenv(JEV_TEMPLATE_PIN_ENV, "v1")
    b = _backend(noul_p=0.9)
    r = b.match(MatchRequest(text="x", pattern="completion.asserted@1"))
    assert r.provenance.calibration_id == "template:jev-template-v1"
    q = b.engine.calls[-1]["questions"]["q"]
    assert "fully complete" in q["instructions"]  # 回到 v1 字面问法
    for p, ctx in _V1_PATTERNS:
        r2 = b.match(MatchRequest(text="x", pattern=p, context=ctx))
        assert r2.provenance.calibration_id == "template:jev-template-v1", p


def test_pin_unset_defaults_to_new_version(monkeypatch):
    """缺省（env 未设）= 新版本：completion=v2，其余模式=v1。"""
    monkeypatch.delenv(JEV_TEMPLATE_PIN_ENV, raising=False)
    caps = _backend().capabilities()
    assert caps["template_version"] == JEV_TEMPLATE_VERSION == \
        "jev-template-v2"
    tv = caps["template_versions"]
    assert tv["completion.asserted"] == "jev-template-v2"
    assert all(v == "jev-template-v1" for k, v in tv.items()
               if k != "completion.asserted")
    assert set(tv) == {"completion.asserted", "process.unresolved",
                       "modality.assertive", "task.object.alignment"}


def test_pin_unknown_value_raises(monkeypatch):
    """未知 pin 值显式报错，不静默忽略（配置错误必须可见）。"""
    monkeypatch.setenv(JEV_TEMPLATE_PIN_ENV, "v9")
    with pytest.raises(ValueError):
        _backend().match(
            MatchRequest(text="x", pattern="completion.asserted@1"))

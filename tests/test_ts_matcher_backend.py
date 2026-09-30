# -*- coding: utf-8 -*-
"""P7 MatcherExtractor 适配层与失败隔离测试（全部 fake，不加载模型）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moonbow.semantic.schema import Evidence, MatchResponse, Provenance
from moonbow.task_structure.extractor import RuleBackend, analyze_text
from moonbow.task_structure.matcher_backend import MatcherExtractor

PROV = Provenance(backend="fake", model_revision="r1",
                  pattern_version="ts.capture@1", calibrated=False)

TEXT = "修复登录接口，保持旧客户端兼容。"


def ok_resp(evidence, truncated=False):
    return MatchResponse(status="ok", matched=True, evidence=evidence,
                         truncated=truncated, provenance=PROV)


def ev(quote, start, end, relation="supports"):
    return Evidence(quote=quote, start=start, end=end, relation=relation)


class FakeMatcher:
    """按 kind 返回预设响应的 matcher 替身（记录 find_all 调用）。"""

    def __init__(self, responses=None, error=None):
        # responses: {kind: MatchResponse | Exception}
        self.responses = responses or {}
        self.error = error
        self.calls = []

    def find_all(self, text, pattern=None, context=None, **kw):
        kind = (context or {}).get("kind")
        self.calls.append((text, pattern, kind))
        if self.error is not None:
            raise self.error
        return self.responses.get(kind, MatchResponse(
            status="ok", matched=False, provenance=PROV))


# ---- 构造契约 ----

def test_no_matcher_and_no_url_raises():
    with pytest.raises(ValueError):
        MatcherExtractor()


# ---- 正常映射 ----

def test_ok_response_maps_evidence_to_captures():
    idx = TEXT.index("修复登录接口")
    m = FakeMatcher({"goal": ok_resp([ev("修复登录接口", idx, idx + 6)])})
    caps = MatcherExtractor(matcher=m).extract(TEXT)
    assert len(caps) == 1
    c = caps[0]
    assert c.kind == "goal"
    assert c.quote == "修复登录接口"
    assert c.span == (idx, idx + 6)
    # 每次 find_all 以 context.kind 圈定类别
    assert all(p == "ts.capture@1" for _t, p, _k in m.calls)
    assert {k for _t, _p, k in m.calls} >= {"goal"}


def test_offset_fallback_when_evidence_span_mismatched():
    idx = TEXT.index("保持旧客户端兼容")
    # 证据偏移与 quote 不对齐 → 回退 text.index
    m = FakeMatcher({"constraint": ok_resp([ev("保持旧客户端兼容", 0, 3)])})
    caps = MatcherExtractor(matcher=m).extract(TEXT)
    assert len(caps) == 1
    assert caps[0].span == (idx, idx + len("保持旧客户端兼容"))


def test_bad_quote_dropped():
    x = MatcherExtractor(matcher=FakeMatcher({"goal": ok_resp([ev("不存在的引文", 0, 6)])}))
    caps = x.extract(TEXT)
    assert caps == []
    assert x.last_status == "ok" and x.last_backend_failed is False


def test_captures_sorted_by_position():
    g0 = TEXT.index("修复登录接口")
    c0 = TEXT.index("保持旧客户端兼容")
    m = FakeMatcher({
        "constraint": ok_resp([ev("保持旧客户端兼容", c0, c0 + 8)]),
        "goal": ok_resp([ev("修复登录接口", g0, g0 + 6)]),
    })
    caps = MatcherExtractor(matcher=m).extract(TEXT)
    assert [c.kind for c in caps] == ["goal", "constraint"]


# ---- 契约状态贯通 ----

def test_error_status_marks_backend_failed_and_empty():
    m = FakeMatcher({"goal": MatchResponse(status="error",
                                           reason_code="unavailable")})
    x = MatcherExtractor(matcher=m)
    caps = x.extract(TEXT)
    assert caps == []
    assert x.last_backend_failed is True
    assert x.last_backend_abstain is False
    assert x.last_status == "error"
    assert x.last_reason_code == "unavailable"


def test_matcher_exception_marks_backend_failed():
    m = FakeMatcher(error=RuntimeError("boom"))
    x = MatcherExtractor(matcher=m)
    caps = x.extract(TEXT)
    assert caps == []
    assert x.last_backend_failed is True
    assert x.last_reason_code == "RuntimeError"


def test_abstain_marks_explicit_flag():
    m = FakeMatcher({"goal": MatchResponse(status="abstain",
                                           reason_code="insufficient_context")})
    x = MatcherExtractor(matcher=m)
    caps = x.extract(TEXT)
    assert caps == []
    assert x.last_backend_abstain is True
    assert x.last_backend_failed is False


def test_truncated_marked_even_on_ok():
    idx = TEXT.index("修复登录接口")
    m = FakeMatcher({"goal": ok_resp([ev("修复登录接口", idx, idx + 6)],
                                     truncated=True)})
    x = MatcherExtractor(matcher=m)
    caps = x.extract(TEXT)
    assert caps and x.last_backend_truncated is True
    assert x.last_backend_failed is False


# ---- analyze_text 集成（新字段语义） ----

def test_structure_analysis_new_fields_default_false():
    a = analyze_text(TEXT)
    assert a.backend_failed is False and a.backend_truncated is False
    d = a.to_dict()
    assert d["backend_failed"] is False and d["backend_truncated"] is False


def test_rule_backend_path_unaffected():
    a = analyze_text("修复登录接口，保持旧客户端兼容。", backend=RuleBackend())
    assert a.backend_failed is False and a.backend_truncated is False
    assert a.captures, "rule 基线在该文本上应有捕获"


def test_analyze_text_backend_failed_via_matcher():
    m = FakeMatcher(error=RuntimeError("down"))
    a = analyze_text(TEXT, backend=MatcherExtractor(matcher=m))
    assert a.backend_failed is True
    assert a.backend_truncated is False
    assert a.captures == []
    # 失败空捕获不得解读为简单：仍是 abstain/unknown 路径
    assert a.level == "unknown"


def test_analyze_text_backend_abstain_reason():
    m = FakeMatcher({"goal": MatchResponse(status="abstain",
                                           reason_code="insufficient_context")})
    a = analyze_text(TEXT, backend=MatcherExtractor(matcher=m))
    assert a.abstain is True
    assert a.abstain_reason == "backend_abstain"


def test_analyze_text_backend_truncated_via_matcher():
    idx = TEXT.index("修复登录接口")
    m = FakeMatcher({"goal": ok_resp([ev("修复登录接口", idx, idx + 6)],
                                     truncated=True)})
    a = analyze_text(TEXT, backend=MatcherExtractor(matcher=m))
    assert a.backend_truncated is True
    # 输入本身未截断，两个字段语义独立
    assert a.truncated is False


# ---- service 透传 ----

def test_service_passes_backend_status_through(monkeypatch):
    from moonbow.task_structure import service
    m = FakeMatcher(error=RuntimeError("down"))
    extractor = MatcherExtractor(matcher=m)
    monkeypatch.setattr(service, "get_backend", lambda: extractor)
    out = service.analyze_payload({"text": TEXT})
    assert out["backend_failed"] is True
    assert out["backend_truncated"] is False


def test_service_rule_path_fields_false(monkeypatch):
    from moonbow.task_structure import service
    monkeypatch.setattr(service, "get_backend", RuleBackend)
    out = service.analyze_payload({"text": TEXT})
    assert out["backend_failed"] is False and out["backend_truncated"] is False

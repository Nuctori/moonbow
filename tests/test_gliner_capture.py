# -*- coding: utf-8 -*-
"""R2 GlinerCaptureBackend 单测（fake/monkeypatch，无真实推理）。"""
import pytest

from moonbow.semantic.backends.base import BackendUnavailable
from moonbow.semantic.backends.gliner_capture import (
    LABELS_V1,
    LABELS_VERSION,
    SUPPORTED_PATTERN,
    GlinerCaptureBackend,
)
from moonbow.semantic.matcher import (
    SemanticMatcher,
    register_pattern_backend,
    unregister_pattern_backend,
    reset_registry,
)
from moonbow.semantic.patterns import CAPTURE_KINDS
from moonbow.semantic.schema import MatchRequest, MatchResponse


class FakeExtractor:
    """替身：返回与真实 gliner2 extract 相同形状的输出。"""

    def __init__(self, entities_per_call):
        self.entities_per_call = entities_per_call
        self.calls = []

    def extract(self, text, schema, include_spans=False,
                include_confidence=False):
        self.calls.append(text)
        return {"entities": self.entities_per_call(text)}


def make_backend(entities_fn, threshold=0.05):
    backend = GlinerCaptureBackend(model_path="fake/model", device="cpu",
                                   threshold=threshold)
    backend._extractor = FakeExtractor(entities_fn)
    backend._schema = object()  # fake 路径不触达
    # 绕过 initialize（真实加载需 XPU/gliner2）；直接标记就绪。
    backend._initialized = True
    backend.load_count += 1
    return backend


def find_all_req(text):
    return MatchRequest(text=text, pattern=SUPPORTED_PATTERN,
                        operation="find_all", request_id="r1")


TEXT = "请先完成登录模块。"


def test_labels_v1_covers_eight_kinds():
    assert sorted(LABELS_V1) == sorted(CAPTURE_KINDS)
    assert len(set(LABELS_V1.values())) == len(CAPTURE_KINDS)
    assert LABELS_VERSION == "gliner:labels-v1"


def test_find_all_ok_and_evidence_offsets():
    ent = {"text": "登录模块", "start": 4, "end": 8,
           "confidence": 0.42, "label_kind": "object"}
    backend = make_backend(lambda t: [ent])
    resp = backend.match(find_all_req(TEXT))
    assert resp.status == "ok"
    assert resp.matched is True
    assert len(resp.evidence) == 1
    ev = resp.evidence[0]
    assert (ev.quote, ev.start, ev.end) == ("登录模块", 4, 8)
    assert ev.relation == "supports"
    assert resp.provenance.calibration_id == LABELS_VERSION
    assert resp.provenance.backend == "gliner"
    assert resp.provenance.pattern_version == SUPPORTED_PATTERN
    assert backend.dropped_quotes == 0


def test_bad_quotes_dropped_and_counted():
    ents = [
        {"text": "错位引文", "start": 4, "end": 8, "confidence": 0.5,
         "label_kind": "object"},            # text[4:8] != quote → 丢
        {"text": "越界", "start": 999, "end": 1002, "confidence": 0.5,
         "label_kind": "goal"},               # 越界 → 丢
        {"text": "", "start": 0, "end": 1, "confidence": 0.5,
         "label_kind": "goal"},               # 空引文 → 丢
        {"text": "登录模块", "start": 4, "end": 8, "confidence": 0.5,
         "label_kind": "未知类"},             # 未知 kind → 丢
        {"text": "登录模块", "start": 4, "end": 8, "confidence": 0.5,
         "label_kind": "goal"},               # 合法
    ]
    backend = make_backend(lambda t: ents)
    resp = backend.match(find_all_req(TEXT))
    assert resp.status == "ok"
    assert len(resp.evidence) == 1
    assert backend.dropped_quotes == 4
    assert backend.total_spans == 5


def test_empty_capture_is_ok_negative_not_abstain():
    backend = make_backend(lambda t: [])
    resp = backend.match(find_all_req(TEXT))
    assert resp.status == "ok"
    assert resp.matched is False
    assert resp.evidence == []
    assert resp.reason_code is None
    assert resp.provenance is not None


def test_unsupported_operations_and_patterns():
    backend = make_backend(lambda t: [])
    # match 操作 → unsupported
    resp = backend.match(MatchRequest(text=TEXT, pattern="ts.capture@1",
                                      operation="match"))
    assert resp.status == "error"
    assert resp.reason_code == "unsupported"
    # find_all 但别的 pattern → unsupported
    resp = backend.match(MatchRequest(text=TEXT, pattern="completion.asserted@1",
                                      operation="find_all"))
    assert resp.status == "error"
    assert resp.reason_code == "unsupported"


def test_xpu_required_no_cpu_fallback():
    backend = GlinerCaptureBackend(model_path="fake/model", device="xpu")
    # 默认解释器通常无 XPU torch；如有 XPU 则跳过该断言分支。
    import torch
    if torch.xpu.is_available():  # pragma: no cover
        pytest.skip("XPU host: absence-branch not testable")
    with pytest.raises(BackendUnavailable):
        backend.initialize()


def test_matcher_route_find_all_to_gliner_match_untouched(monkeypatch):
    reset_registry()
    calls = {"capture": 0, "default": 0}

    capture = make_backend(lambda t: [
        {"text": "登录模块", "start": 4, "end": 8, "confidence": 0.5,
         "label_kind": "object"}])

    class DefaultBackend:
        name = "fake-default"
        model_revision = "r0"

        def capabilities(self):
            return {"name": self.name}

        @property
        def is_initialized(self):
            return True

        def initialize(self):
            calls["default"] += 1

        def match(self, request):
            if request.operation == "match":
                return MatchResponse(status="ok", matched=True,
                                     provenance=__import__(
                                         "moonbow.semantic.schema",
                                         fromlist=["Provenance"]).Provenance(
                                         backend="fake",
                                         model_revision="r0",
                                         pattern_version=request.pattern,
                                         calibrated=False),
                                     request_id=request.request_id)
            return MatchResponse(status="error", reason_code="unsupported",
                                 request_id=request.request_id)

    def capture_factory():
        calls["capture"] += 1
        return capture

    register_pattern_backend("find_all", SUPPORTED_PATTERN, capture_factory)
    try:
        matcher = SemanticMatcher.shared(backend=DefaultBackend())
        # find_all + ts.capture@1 → 路由到 gliner 捕获 backend
        resp = matcher.find_all(TEXT, pattern=SUPPORTED_PATTERN)
        assert resp.status == "ok" and resp.matched is True
        assert calls["capture"] == 1
        # match → 默认 runtime，不受路由影响
        resp = matcher.match(TEXT, pattern=SUPPORTED_PATTERN)
        assert resp.status == "ok" and resp.matched is True
        assert calls["capture"] == 1  # 未再触发捕获工厂
        # 第二次 find_all 命中共享路由 runtime，工厂不再调用（load once）
        matcher.find_all(TEXT, pattern=SUPPORTED_PATTERN)
        assert calls["capture"] == 1
    finally:
        unregister_pattern_backend("find_all", SUPPORTED_PATTERN)
        reset_registry()


def test_labels_mismatch_rejected():
    bad = {k: "x" for k in CAPTURE_KINDS}
    del bad["acceptance"]
    with pytest.raises(ValueError):
        GlinerCaptureBackend(model_path="fake/model", device="cpu",
                             labels=bad)

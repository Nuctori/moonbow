# -*- coding: utf-8 -*-
"""tests/test_semantic_contract.py

语义匹配运行时 P1 契约冻结测试（计划 §4-P1 全部用例）。
纯逻辑：不依赖 torch / 模型权重 / 服务。
"""
import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from moonbow.semantic.schema import (  # noqa: E402
    ValidationError, MatchRequest, MatchResponse, Evidence, Provenance,
    PatternSpec, validate_response,
    OPERATIONS, STATUSES, REASON_CODES, RELATIONS, MAX_TEXT_LENGTH,
)
from moonbow.semantic.patterns import (  # noqa: E402
    PatternRegistry, load_builtin_patterns, parse_pattern_ref,
    check_request_context, CAPTURE_KINDS,
)

PATTERN = "completion.asserted@1"


def prov(pattern_version=PATTERN, **kw):
    base = dict(backend="fake", model_revision="test-rev",
                pattern_version=pattern_version, calibrated=False)
    base.update(kw)
    return Provenance(**base)


def ok_resp(text="应该修好了。", pattern=PATTERN, **kw):
    base = dict(
        status="ok", matched=False, score=0.08, score_type="label_probability",
        calibrated=False, evidence=[], truncated=False, reason_code=None,
        provenance=prov(pattern_version=pattern), request_id=None,
    )
    base.update(kw)
    return MatchResponse(**base)


def req(text="应该修好了。", **kw):
    base = dict(pattern=PATTERN, operation="match")
    base.update(kw)
    return MatchRequest(text=text, **base)


# ---------------------------------------------------------------------------
# 1. 二选一字段：pattern 与 requirement
# ---------------------------------------------------------------------------

class TestExclusivity:
    def test_both_present_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            MatchRequest(text="x", pattern=PATTERN, requirement="判断是否完成")

    def test_both_missing_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            MatchRequest(text="x")

    def test_pattern_only_ok(self):
        r = req()
        assert r.pattern == PATTERN and r.requirement is None

    def test_requirement_only_ok(self):
        r = MatchRequest(text="x", requirement="判断是否完成")
        assert r.requirement == "判断是否完成" and r.pattern is None

    def test_no_implicit_priority_in_roundtrip(self):
        d = req().to_dict()
        assert d["requirement"] is None


# ---------------------------------------------------------------------------
# 2. 空 / 超长文本
# ---------------------------------------------------------------------------

class TestTextLimits:
    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            MatchRequest(text="", pattern=PATTERN)

    def test_non_string_text_rejected(self):
        with pytest.raises(ValidationError, match="text must be str"):
            MatchRequest(text=123, pattern=PATTERN)

    def test_overlong_text_rejected(self):
        with pytest.raises(ValidationError, match="exceeds limit"):
            MatchRequest(text="a" * (MAX_TEXT_LENGTH + 1), pattern=PATTERN)

    def test_text_at_limit_ok(self):
        r = MatchRequest(text="a" * MAX_TEXT_LENGTH, pattern=PATTERN)
        assert len(r.text) == MAX_TEXT_LENGTH


# ---------------------------------------------------------------------------
# 3. operation / threshold / request_id
# ---------------------------------------------------------------------------

class TestRequestFields:
    @pytest.mark.parametrize("op", ["match", "find_all"])
    def test_valid_operations(self, op):
        assert req(operation=op).operation == op

    def test_invalid_operation(self):
        with pytest.raises(ValidationError, match="operation"):
            req(operation="batch")

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "x", True, float("nan")])
    def test_invalid_threshold(self, bad):
        with pytest.raises(ValidationError):
            req(threshold=bad)

    def test_threshold_none_ok(self):
        assert req(threshold=None).threshold is None

    def test_threshold_boundary_ok(self):
        assert req(threshold=0.0).threshold == 0.0
        assert req(threshold=1.0).threshold == 1.0


# ---------------------------------------------------------------------------
# 4. Unicode / emoji / 重复引文偏移
# ---------------------------------------------------------------------------

class TestEvidenceOffsets:
    def test_emoji_is_single_code_point(self):
        text = "a😀b"
        assert len(text) == 3  # code point 语义，不是 UTF-16 字节

    def test_evidence_offsets_after_emoji(self):
        text = "状态：😀已修复"
        e = Evidence(quote="已修复", start=4, end=7, relation="supports")
        assert text[e.start:e.end] == e.quote

    def test_repeated_quote_second_occurrence(self):
        # 重复引文必须能定位第二个 occurrence，不能总用首次命中。
        text = "same then again same"
        first = text.index("same")
        e = Evidence(quote="same", start=16, end=20, relation="supports")
        assert e.start != first
        assert text[e.start:e.end] == "same"
        r = req(text)
        resp = ok_resp(evidence=[e])
        validate_response(r, resp)  # 不误报为越界或错位

    def test_negative_offset_rejected(self):
        with pytest.raises(ValidationError, match=">= 0"):
            Evidence(quote="x", start=-1, end=0, relation="supports")

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match=">= start"):
            Evidence(quote="xy", start=2, end=1, relation="supports")

    def test_empty_quote_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            Evidence(quote="", start=0, end=0, relation="supports")

    def test_invalid_relation_rejected(self):
        with pytest.raises(ValidationError, match="relation"):
            Evidence(quote="x", start=0, end=1, relation="refutes")

    @pytest.mark.parametrize("rel", ["supports", "contradicts"])
    def test_valid_relations(self, rel):
        assert Evidence(quote="x", start=0, end=1, relation=rel).relation == rel

    def test_span_out_of_text_rejected_by_validate(self):
        r = req("abc")
        resp = ok_resp(evidence=[Evidence("abc", 2, 9, "supports")])
        with pytest.raises(ValidationError, match="out of range"):
            validate_response(r, resp)

    def test_quote_mismatch_rejected_by_validate(self):
        r = req("abcdef")
        resp = ok_resp(evidence=[Evidence("bCd", 1, 4, "supports")])
        with pytest.raises(ValidationError, match="not text"):
            validate_response(r, resp)

    def test_quote_must_be_exact_slice(self):
        r = req("abcdef")
        # 偏移正确但 quote 被模型"修饰"过 → 拒绝
        resp = ok_resp(evidence=[Evidence("BCD", 1, 4, "supports")])
        with pytest.raises(ValidationError):
            validate_response(r, resp)


# ---------------------------------------------------------------------------
# 5. 非法 score：NaN / 越界 / 非数
# ---------------------------------------------------------------------------

class TestScoreValidation:
    @pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.01, "x", True, [0.5]])
    def test_invalid_score_rejected(self, bad):
        with pytest.raises(ValidationError):
            ok_resp(score=bad)

    def test_null_score_ok(self):
        assert ok_resp(score=None).score is None

    def test_score_boundaries_ok(self):
        assert ok_resp(score=0.0).score == 0.0
        assert ok_resp(score=1.0).score == 1.0

    def test_nan_via_json_rejected(self):
        # JSON 允许 NaN 字面量，但契约拒绝
        raw = '{"status": "ok", "matched": true, "score": NaN, ' \
              '"provenance": {"backend": "b", "model_revision": "r", ' \
              '"pattern_version": "p@1", "calibrated": false}}'
        with pytest.raises(ValidationError):
            MatchResponse.from_json(raw)


# ---------------------------------------------------------------------------
# 6. 状态互斥：status / matched / reason_code / provenance / evidence / score
# ---------------------------------------------------------------------------

class TestStatusMutex:
    def test_ok_with_matched_null_rejected(self):
        with pytest.raises(ValidationError, match="matched"):
            ok_resp(matched=None)

    def test_abstain_with_matched_false_rejected(self):
        # unknown 不转 false：abstain 响应不得携带布尔 matched
        with pytest.raises(ValidationError, match="matched=null"):
            MatchResponse(status="abstain", matched=False,
                          reason_code="insufficient_context")

    def test_error_with_matched_true_rejected(self):
        with pytest.raises(ValidationError, match="matched=null"):
            MatchResponse(status="error", matched=True,
                          reason_code="timeout")

    def test_ok_without_provenance_rejected(self):
        with pytest.raises(ValidationError, match="provenance"):
            ok_resp(provenance=None)

    def test_ok_with_reason_code_rejected(self):
        with pytest.raises(ValidationError, match="reason_code"):
            ok_resp(reason_code="timeout")

    def test_abstain_without_reason_code_rejected(self):
        with pytest.raises(ValidationError, match="reason_code"):
            MatchResponse(status="abstain", matched=None)

    def test_abstain_with_provenance_rejected(self):
        with pytest.raises(ValidationError, match="provenance"):
            MatchResponse(status="abstain", matched=None,
                          reason_code="insufficient_context",
                          provenance=prov())

    def test_abstain_with_evidence_rejected(self):
        with pytest.raises(ValidationError, match="evidence"):
            MatchResponse(status="abstain", matched=None,
                          reason_code="insufficient_context",
                          evidence=[Evidence("x", 0, 1, "supports")])

    def test_error_with_score_rejected(self):
        with pytest.raises(ValidationError, match="score"):
            MatchResponse(status="error", matched=None,
                          reason_code="timeout", score=0.9)

    def test_unknown_is_not_false(self):
        # 契约表达：无法判断时 matched=null，而不是 false 冒充正常负例
        r = MatchResponse(status="abstain", matched=None,
                          reason_code="insufficient_context")
        d = r.to_dict()
        assert d["matched"] is None and "matched" in d

    def test_invalid_status(self):
        with pytest.raises(ValidationError, match="status"):
            MatchResponse(status="unknown")

    def test_invalid_reason_code(self):
        with pytest.raises(ValidationError, match="reason_code"):
            MatchResponse(status="abstain", matched=None, reason_code="whatever")

    def test_all_reason_codes_accepted(self):
        for rc in REASON_CODES:
            MatchResponse(status="abstain", matched=None, reason_code=rc)

    def test_invalid_calibrated_type(self):
        with pytest.raises(ValidationError, match="calibrated"):
            ok_resp(calibrated="yes")


# ---------------------------------------------------------------------------
# 7. 严格模式：未知顶层字段拒绝
# ---------------------------------------------------------------------------

class TestStrictUnknownFields:
    def test_request_unknown_field(self):
        d = req().to_dict()
        d["temperature"] = 0.7
        with pytest.raises(ValidationError, match="unknown field"):
            MatchRequest.from_dict(d)

    def test_response_unknown_field(self):
        d = ok_resp().to_dict()
        d["confidence"] = 0.99
        with pytest.raises(ValidationError, match="unknown field"):
            MatchResponse.from_dict(d)

    def test_evidence_unknown_field(self):
        d = Evidence("x", 0, 1, "supports").to_dict()
        d["confidence"] = 0.9
        with pytest.raises(ValidationError, match="unknown field"):
            Evidence.from_dict(d)

    def test_provenance_unknown_field(self):
        d = prov().to_dict()
        d["model_name"] = "whatever"
        with pytest.raises(ValidationError, match="unknown field"):
            Provenance.from_dict(d)

    def test_matched_must_be_bool_or_null(self):
        with pytest.raises(ValidationError):
            ok_resp(matched="true")


# ---------------------------------------------------------------------------
# 8. PatternSpec 与注册表：版本不匹配
# ---------------------------------------------------------------------------

class TestPatternRegistry:
    @pytest.fixture(scope="class")
    def registry(self):
        return load_builtin_patterns()

    def test_five_builtin_patterns(self, registry):
        refs = registry.list_refs()
        assert refs == [
            "completion.asserted@1", "modality.assertive@1",
            "process.unresolved@1", "task.object.alignment@1", "ts.capture@1",
        ]

    def test_version_mismatch_rejected(self, registry):
        with pytest.raises(ValidationError, match="unknown pattern version"):
            registry.get("completion.asserted@2")

    def test_unknown_pattern_rejected(self, registry):
        with pytest.raises(ValidationError):
            registry.get("nonexistent@1")

    def test_no_silent_fallback_to_other_version(self, registry):
        # 同 id 高版本存在时也不静默回退
        with pytest.raises(ValidationError):
            registry.get("completion.asserted@99")

    def test_malformed_ref_rejected(self, registry):
        for bad in ["completion.asserted", "completion.asserted@x", "@1", "A@1"]:
            with pytest.raises(ValidationError):
                parse_pattern_ref(bad)

    def test_every_pattern_has_min_examples(self, registry):
        for ref in registry.list_refs():
            spec = registry.get(ref)
            assert len(spec.positive_examples) >= 3
            assert len(spec.negative_examples) >= 3
            assert spec.definition.strip()
            assert isinstance(spec.exclusions, list)
            assert spec.operation in OPERATIONS
            assert isinstance(spec.output_schema, dict)
            assert spec.scoring.strip()

    def test_ts_capture_kind_enum_matches_task_structure(self, registry):
        spec = registry.get("ts.capture@1")
        assert spec.operation == "find_all"
        captures = spec.output_schema["captures"]
        kind_enum = next(c["kind_enum"] for c in captures
                         if isinstance(c, dict) and "kind_enum" in c)
        assert kind_enum == list(CAPTURE_KINDS)
        assert len(CAPTURE_KINDS) == 8

    def test_task_object_alignment_requires_context_task(self, registry):
        spec = registry.get("task.object.alignment@1")
        assert spec.requires_context == ["task"]
        r = req(pattern="task.object.alignment@1")
        with pytest.raises(ValidationError, match="context"):
            check_request_context(r, spec)

    def test_task_object_alignment_with_context_ok(self, registry):
        spec = registry.get("task.object.alignment@1")
        r = req(pattern="task.object.alignment@1", context={"task": "修复登录"})
        check_request_context(r, spec)  # 不抛

    def test_duplicate_registration_rejected(self, registry):
        spec = registry.get(PATTERN)
        with pytest.raises(ValidationError, match="already registered"):
            registry.register(spec)

    def test_pattern_spec_min_examples_enforced(self):
        from moonbow.semantic.patterns import spec_from_payload
        base = dict(id="x.y", version=1, definition="d",
                    positive_examples=["a", "b"], negative_examples=["c", "d", "e"],
                    exclusions=[], operation="match",
                    output_schema={}, scoring="s")
        with pytest.raises(ValidationError, match="positive"):
            spec_from_payload(dict(base))
        base["positive_examples"] = ["a", "b", "c"]
        base["negative_examples"] = ["c", "d"]
        with pytest.raises(ValidationError, match="negative"):
            spec_from_payload(dict(base))

    def test_builtin_json_matches_file_name(self, registry):
        import pathlib
        pdir = pathlib.Path(
            os.path.join(os.path.dirname(__file__), "..", "src", "moonbow", "patterns"))
        for p in pdir.glob("*.json"):
            payload = json.loads(p.read_text(encoding="utf-8"))
            assert p.stem == f"{payload['id']}@{payload['version']}"


# ---------------------------------------------------------------------------
# 9. validate_response：跨对象自洽性
# ---------------------------------------------------------------------------

class TestValidateResponse:
    def test_ok_negative_roundtrip(self):
        validate_response(req(), ok_resp())

    def test_ok_positive_with_evidence(self):
        r = req("应该修好了")
        e = Evidence("修好", 2, 4, "supports")
        validate_response(r, ok_resp(matched=True, evidence=[e]))

    def test_request_id_mismatch_rejected(self):
        resp = ok_resp(request_id="other")
        with pytest.raises(ValidationError, match="request_id"):
            validate_response(req(request_id="mine"), resp)

    def test_request_id_match_ok(self):
        validate_response(req(request_id="rid1"), ok_resp(request_id="rid1"))

    def test_pattern_version_mismatch_rejected(self):
        resp = ok_resp()
        resp.provenance = prov(pattern_version="completion.asserted@2")
        with pytest.raises(ValidationError, match="pattern_version"):
            validate_response(req(), resp)

    def test_find_all_allows_multiple_evidence(self):
        r = req("先跑迁移再更新 API", operation="find_all",
                pattern="ts.capture@1")
        evs = [Evidence("先跑迁移", 0, 4, "supports"),
               Evidence("再更新 API", 4, 11, "supports")]
        validate_response(r, ok_resp(pattern="ts.capture@1", matched=True,
                                     evidence=evs))

    def test_match_rejects_multiple_evidence(self):
        r = req("abc abc")
        evs = [Evidence("abc", 0, 3, "supports"),
               Evidence("abc", 4, 7, "supports")]
        resp = ok_resp(matched=True, evidence=evs)
        with pytest.raises(ValidationError, match="at most 1"):
            validate_response(r, resp)

    def test_truncated_find_all_cannot_claim_clean_negative(self):
        r = req("文本", operation="find_all", pattern="ts.capture@1")
        resp = ok_resp(pattern="ts.capture@1", matched=False, truncated=True)
        with pytest.raises(ValidationError, match="truncated"):
            validate_response(r, resp)

    def test_truncated_find_all_abstain_ok(self):
        r = req("文本", operation="find_all", pattern="ts.capture@1")
        resp = MatchResponse(status="abstain", matched=None,
                             reason_code="truncated")
        validate_response(r, resp)

    def test_ok_truncated_positive_ok(self):
        # 截断 + 正向命中仍可成立（不能宣称的只是"全量无捕获"）
        r = req("修好了", operation="find_all", pattern="ts.capture@1")
        validate_response(r, ok_resp(pattern="ts.capture@1", matched=True,
                                     truncated=True))


# ---------------------------------------------------------------------------
# 10. round-trip：所有类型 JSON 序列化/反序列化
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_request_roundtrip_dict(self):
        r = req(context={"task": "修复登录"}, threshold=0.85, request_id="r1")
        assert MatchRequest.from_dict(r.to_dict()) == r

    def test_request_roundtrip_json(self):
        r = req(context={"task": "修复登录"}, threshold=0.85, request_id="r1")
        assert MatchRequest.from_json(r.to_json()) == r

    def test_request_json_chinese_not_escaped(self):
        r = req(text="修复登录")
        assert "修复登录" in r.to_json()

    def test_request_from_json_invalid(self):
        with pytest.raises(ValidationError, match="invalid JSON"):
            MatchRequest.from_json("{not json")

    def test_response_ok_roundtrip(self):
        r = req("abc def")
        resp = ok_resp(matched=True, evidence=[Evidence("def", 4, 7, "supports")],
                       request_id="r9")
        d = resp.to_dict()
        assert MatchResponse.from_dict(d) == resp
        assert MatchResponse.from_json(resp.to_json()) == resp

    def test_response_abstain_roundtrip(self):
        resp = MatchResponse(status="abstain", matched=None,
                             reason_code="insufficient_context")
        assert MatchResponse.from_dict(resp.to_dict()) == resp

    def test_response_error_roundtrip(self):
        resp = MatchResponse(status="error", matched=None, reason_code="timeout")
        assert MatchResponse.from_json(resp.to_json()) == resp

    def test_provenance_roundtrip(self):
        p = prov(calibrated=True, calibration_id="cal-2026-09")
        assert Provenance.from_dict(p.to_dict()) == p

    def test_evidence_roundtrip(self):
        e = Evidence("quote", 3, 8, "contradicts")
        assert Evidence.from_dict(e.to_dict()) == e

    def test_pattern_spec_roundtrip(self):
        registry = load_builtin_patterns()
        for ref in registry.list_refs():
            spec = registry.get(ref)
            clone = PatternSpec(**spec.to_dict())
            assert clone == spec
            assert clone.ref == ref

    def test_from_dict_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            MatchRequest.from_dict({"pattern": PATTERN})
        with pytest.raises(ValidationError):
            MatchResponse.from_dict({"status": "ok"})

    def test_from_dict_non_object_rejected(self):
        with pytest.raises(ValidationError):
            MatchRequest.from_dict([1, 2])
        with pytest.raises(ValidationError):
            MatchResponse.from_dict("ok")

    def test_constants_frozen(self):
        assert OPERATIONS == ("match", "find_all")
        assert STATUSES == ("ok", "abstain", "error")
        assert RELATIONS == ("supports", "contradicts")
        assert set(REASON_CODES) == {
            "insufficient_context", "truncated", "unsupported", "timeout",
            "overloaded", "invalid_output", "unavailable"}


# ---------------------------------------------------------------------------
# 11. 深拷贝 / 可哈希性等杂项防回归
# ---------------------------------------------------------------------------

class TestMisc:
    def test_default_context_not_shared(self):
        a, b = req(), req()
        a.context["task"] = "x"
        assert b.context == {}

    def test_copy_roundtrip(self):
        resp = ok_resp()
        assert copy.deepcopy(resp) == resp

    def test_score_int_accepted_as_number(self):
        assert ok_resp(score=0).score == 0.0
        assert isinstance(ok_resp(score=1).score, float)



# ---------------------------------------------------------------------------
# 12. adapter 字段（R1，schema v2）：校验 / unsupported 语义 / provenance 回填
# ---------------------------------------------------------------------------

class TestAdapterField:
    # -- 请求字段校验 -------------------------------------------------------

    def test_adapter_none_is_default(self):
        r = req()
        assert r.adapter is None
        assert req(adapter=None).adapter is None

    def test_adapter_valid_names(self):
        for name in ["lora-a", "ft_v2", "LoRA.1", "a" * 128]:
            assert req(adapter=name).adapter == name

    @pytest.mark.parametrize("bad", ["", "  ", "lora/a", "中文", "a b",
                                     "a+b", 123, True, "a" * 129])
    def test_adapter_invalid_rejected(self, bad):
        with pytest.raises(ValidationError, match="adapter"):
            req(adapter=bad)

    def test_adapter_roundtrip(self):
        r = req(adapter="lora-a")
        assert MatchRequest.from_dict(r.to_dict()) == r
        assert MatchRequest.from_json(r.to_json()) == r

    def test_adapter_strict_unknown_still_rejected(self):
        d = req(adapter="lora-a").to_dict()
        d["adapter_name"] = "x"
        with pytest.raises(ValidationError, match="unknown field"):
            MatchRequest.from_dict(d)

    # -- 旧请求（不带 adapter 字段）逐字段等价 ------------------------------

    def test_legacy_request_unchanged(self):
        legacy = {"text": "x", "pattern": PATTERN, "requirement": None,
                  "context": {}, "operation": "match", "threshold": None,
                  "request_id": None}
        r = MatchRequest.from_dict(copy.deepcopy(legacy))
        assert r.adapter is None
        d = r.to_dict()
        for k, v in legacy.items():
            assert d[k] == v, k          # 旧字段值逐字段不变
        assert d["adapter"] is None      # 新字段仅以 null 出现
        assert MatchRequest.from_dict(d) == r

    def test_legacy_response_provenance_has_no_adapter(self):
        resp = ok_resp()
        assert resp.provenance.adapter is None
        d = resp.to_dict()
        assert d["provenance"]["adapter"] is None
        assert MatchResponse.from_dict(d) == resp

    # -- validate_response：provenance.adapter 一致性 ------------------------

    def test_ok_with_adapter_must_backfill_provenance(self):
        r = req(adapter="lora-a")
        resp = ok_resp()                       # provenance.adapter=None
        with pytest.raises(ValidationError, match="adapter mismatch"):
            validate_response(r, resp)
        validate_response(r, ok_resp(provenance=prov(adapter="lora-a")))  # 不抛

    def test_ok_without_adapter_must_not_claim_adapter(self):
        r = req()
        resp = ok_resp(provenance=prov(adapter="lora-a"))
        with pytest.raises(ValidationError, match="do not claim"):
            validate_response(r, resp)

    def test_adapter_mismatch_rejected(self):
        r = req(adapter="lora-a")
        resp = ok_resp(provenance=prov(adapter="lora-b"))
        with pytest.raises(ValidationError, match="adapter mismatch"):
            validate_response(r, resp)

    def test_error_response_with_adapter_ok(self):
        # unsupported error 响应不带 provenance（契约错误态互斥），仍过校验
        r = req(adapter="lora-a")
        resp = MatchResponse(status="error", matched=None,
                             reason_code="unsupported",
                             request_id=r.request_id)
        validate_response(r, resp)

    def test_provenance_adapter_invalid_name_rejected(self):
        with pytest.raises(ValidationError, match="provenance.adapter"):
            prov(adapter="bad name!")

    def test_provenance_adapter_roundtrip(self):
        p = prov(adapter="lora-a")
        assert Provenance.from_dict(p.to_dict()) == p

    # -- backend 语义：unknown adapter 必须 unsupported，禁止回退 base -------

    def test_fake_unknown_adapter_is_unsupported_not_fallback(self):
        from moonbow.semantic.backends.fake import FakeBackend, FakeRule
        from moonbow.semantic.runtime import SemanticRuntime
        fb = FakeBackend(rules={PATTERN: FakeRule(matched=True, score=0.9)})
        rt = SemanticRuntime(fb, queue_capacity=4)
        r = req(text="应该修好了。", adapter="nope")
        resp = rt.match(r)
        validate_response(r, resp)             # 契约内 error
        assert resp.status == "error"
        assert resp.reason_code == "unsupported"
        assert resp.matched is None
        assert resp.provenance is None          # 绝不冒充任何执行头
        assert fb.call_count == 1               # 走到了 backend 门禁

    def test_fake_registered_adapter_uses_own_rules_and_backfills(self):
        from moonbow.semantic.backends.fake import FakeBackend, FakeRule
        from moonbow.semantic.runtime import SemanticRuntime
        fb = FakeBackend(rules={PATTERN: FakeRule(matched=False)})
        fb.register_adapter("lora-a", {PATTERN: FakeRule(matched=True,
                                                         score=0.9,
                                                         quotes=["修好了"])})
        rt = SemanticRuntime(fb, queue_capacity=4)
        r = req(text="应该修好了。", adapter="lora-a")
        resp = rt.match(r)
        validate_response(r, resp)
        assert resp.status == "ok" and resp.matched is True
        assert resp.provenance.adapter == "lora-a"

    def test_fake_capabilities_list_adapters(self):
        from moonbow.semantic.backends.fake import FakeBackend
        fb = FakeBackend()
        fb.register_adapter("lora-a")
        assert fb.capabilities()["adapters"] == ["lora-a"]

    # -- matcher / runtime 透传 ---------------------------------------------

    def test_matcher_passes_adapter_through(self):
        from moonbow.semantic.backends.fake import FakeBackend, FakeRule
        fb = FakeBackend(rules={PATTERN: FakeRule(matched=False)})
        fb.register_adapter("lora-a", {PATTERN: FakeRule(matched=True,
                                                         score=0.9)})
        from moonbow.semantic.matcher import SemanticMatcher
        from moonbow.semantic.runtime import SemanticRuntime
        m = SemanticMatcher(SemanticRuntime(fb, queue_capacity=4))
        ok = m.match(text="x", pattern=PATTERN, adapter="lora-a")
        assert ok.status == "ok" and ok.matched is True
        bad = m.match(text="x", pattern=PATTERN, adapter="unknown-ft")
        assert bad.status == "error" and bad.reason_code == "unsupported"
        plain = m.match(text="x", pattern=PATTERN)   # 不带 = 原行为
        assert plain.status == "ok" and plain.matched is False
        fa = m.find_all(text="x", pattern=PATTERN, adapter="unknown-ft")
        assert fa.status == "error" and fa.reason_code == "unsupported"

    def test_batch_carries_adapter(self):
        from moonbow.semantic.backends.fake import FakeBackend, FakeRule
        from moonbow.semantic.matcher import SemanticMatcher
        from moonbow.semantic.runtime import SemanticRuntime
        fb = FakeBackend(rules={PATTERN: FakeRule(matched=False)})
        fb.register_adapter("lora-a", {PATTERN: FakeRule(matched=True)})
        m = SemanticMatcher(SemanticRuntime(fb, queue_capacity=8))
        results = m.batch([
            {"text": "x", "pattern": PATTERN, "adapter": "lora-a"},
            {"text": "x", "pattern": PATTERN, "adapter": "ghost"},
            {"text": "x", "pattern": PATTERN},
        ])
        assert [r.status for r in results] == ["ok", "error", "ok"]
        assert results[1].reason_code == "unsupported"
        assert results[0].provenance.adapter == "lora-a"

    # -- HTTP 层透传（进程内起 ThreadingHTTPServer，端口 0，纯 fake） --------

    def test_http_adapter_passthrough(self):
        import threading
        import urllib.request
        from moonbow.semantic.http import SemanticHTTPServer
        cfg = {
            "server": {"host": "127.0.0.1", "port": 0},
            "runtime": {"warmup": False},
            "backend": {
                "type": "fake", "model_revision": "rev-adapter",
                "rules": {PATTERN: {"matched": False}},
                "adapters": {"lora-a": {"rules": {
                    PATTERN: {"matched": True, "score": 0.9,
                              "quotes": ["修好了"]}}}},
            },
        }
        server = SemanticHTTPServer(("127.0.0.1", 0), cfg)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}))

            def post(payload):
                http_req = urllib.request.Request(
                    "http://127.0.0.1:%d/v1/match" % port,
                    data=json.dumps(payload).encode("utf-8"), method="POST",
                    headers={"Content-Type": "application/json"})
                with opener.open(http_req, timeout=5) as r:
                    return json.loads(r.read())

            ok = post({"text": "修好了", "pattern": PATTERN,
                       "adapter": "lora-a", "request_id": "h1"})
            assert ok["status"] == "ok" and ok["matched"] is True
            assert ok["provenance"]["adapter"] == "lora-a"

            bad = post({"text": "修好了", "pattern": PATTERN,
                        "adapter": "ghost", "request_id": "h2"})
            assert bad["status"] == "error"
            assert bad["reason_code"] == "unsupported"
            assert bad["matched"] is None and bad["provenance"] is None

            caps = json.loads(opener.open(
                "http://127.0.0.1:%d/capabilities" % port,
                timeout=5).read())
            assert caps["backend"]["adapters"] == ["lora-a"]
        finally:
            server.shutdown()
            server.server_close()

    def test_http_invalid_adapter_is_400(self):
        import threading
        import urllib.error
        import urllib.request
        from moonbow.semantic.http import SemanticHTTPServer
        cfg = {
            "server": {"host": "127.0.0.1", "port": 0},
            "runtime": {"warmup": False},
            "backend": {"type": "fake", "rules": {
                PATTERN: {"matched": False}}},
        }
        server = SemanticHTTPServer(("127.0.0.1", 0), cfg)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}))
            http_req = urllib.request.Request(
                "http://127.0.0.1:%d/v1/match" % port,
                data=json.dumps({"text": "x", "pattern": PATTERN,
                                 "adapter": "bad name!"}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as ei:
                opener.open(http_req, timeout=5)
            assert ei.value.code == 400
            assert "adapter" in ei.value.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

    # -- SCHEMA_VERSION ------------------------------------------------------

    def test_schema_version_bumped(self):
        from moonbow.semantic.schema import MAX_ADAPTER_LENGTH, SCHEMA_VERSION
        assert SCHEMA_VERSION == 2
        assert MAX_ADAPTER_LENGTH == 128

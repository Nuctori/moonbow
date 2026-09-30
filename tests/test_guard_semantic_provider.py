# -*- coding: utf-8 -*-
"""tests/test_guard_semantic_provider.py

P6 Guard 迁移测试：语义 provider 抽象（默认 legacy 零行为变化）。
纯逻辑：fake registry / FakeBackend，不加载真实权重，不碰 CPU/GPU 推理。
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from moonbow import Decision, ProgressGuard  # noqa: E402
from moonbow.guard.server import GuardHTTPRequestHandler  # noqa: E402
from moonbow.guard.semantic_provider import (  # noqa: E402
    LegacyProvider,
    SemanticProviderAdapter,
    SLM_CAPTURE_SCORE_KEY,
)
from moonbow.semantic.backends.fake import FakeBackend, FakeRule  # noqa: E402
from moonbow.semantic.matcher import SemanticMatcher  # noqa: E402

from test_guard_regressions import manifest, post  # noqa: E402


class FakeModels:
    """固定输出的 fake registry：与 test_guard_regressions 相同形状。"""

    def __init__(self, similarity=0.9, capture=0.9, modality="assert"):
        self.similarity = similarity
        self.capture = capture
        self.modality = modality

    def get_modality(self, text):
        return self.modality

    def get_similarity(self, req, text):
        return self.similarity

    def get_capture_prob(self, text):
        return self.capture


class FakeFactoryModels(FakeModels):
    """可被 ModelRegistry(models_dir=..., device=...) 构造的 fake 类。"""

    def __init__(self, models_dir=None, device="cpu", **kwargs):
        super().__init__(**kwargs)


def legacy_guard(similarity=0.9, capture=0.9, modality="assert", **kwargs):
    guard = ProgressGuard(lazy_load=True, **kwargs)
    guard._registry = FakeModels(similarity=similarity, capture=capture, modality=modality)
    return guard


def semantic_guard(similarity=0.9, capture=0.9, modality="assert",
                   modality_matched=None, capture_matched=True, backend=None, **kwargs):
    """SemanticProviderAdapter + FakeBackend，语义判定与 legacy fake 对齐。"""
    if modality_matched is None:
        modality_matched = (modality == "assert")
    if backend is None:
        backend = FakeBackend(rules={
            "completion.modality@1": FakeRule(matched=modality_matched),
            "completion.capture@1": FakeRule(matched=capture_matched,
                                             score=float(capture)),
        })
    guard = ProgressGuard(lazy_load=True, **kwargs)
    guard._registry = FakeModels(similarity=similarity, capture=capture, modality=modality)
    guard._provider = SemanticProviderAdapter(SemanticMatcher(guard_runtime(backend)))
    return guard


def guard_runtime(backend):
    from moonbow.semantic.runtime import SemanticRuntime
    return SemanticRuntime(backend)


DECISION_FIELDS = ("decision", "is_closed", "disputed", "allow_stop", "acceptance")


def assert_same_ruling(a: dict, b: dict):
    """两个裁决在决策层逐字段一致（新 provider_kind 键除外）。"""
    for key in DECISION_FIELDS:
        assert a[key] == b[key], f"{key}: legacy={a[key]} semantic={b[key]}"


# ---------------------------------------------------------------------------
# 1. 默认构造（无 provider 参数）与迁移前逐字段一致
# ---------------------------------------------------------------------------

def test_default_construction_matches_pre_migration(monkeypatch):
    """默认构造：lazy 后经 monkeypatch 的 ModelRegistry 构建 LegacyProvider，
    输出与直接注入 fake registry 的迁移前用法完全一致（新增键仅 provider_kind）。"""
    monkeypatch.setattr("moonbow.guard.models.ModelRegistry", FakeFactoryModels)
    default_guard = ProgressGuard(lazy_load=True)
    assert default_guard._provider is None
    v_default = default_guard.check("任务", manifest(), rounds=2).to_dict()

    pre_guard = legacy_guard()
    v_pre = pre_guard.check("任务", manifest(), rounds=2).to_dict()

    assert v_pre["scores"]["provider_kind"] == "legacy"
    assert v_default["scores"]["provider_kind"] == "legacy"
    # 除 provider_kind 外 scores 完全一致（modality/similarity/capture_prob/
    # semantic_signals 逐字段相同）
    s_def = {k: v for k, v in v_default["scores"].items() if k != "provider_kind"}
    s_pre = {k: v for k, v in v_pre["scores"].items() if k != "provider_kind"}
    assert s_def == s_pre
    assert s_pre["modality"] == "assert"
    assert s_pre["similarity"] == pytest.approx(0.9, abs=1e-3)
    assert s_pre["capture_prob"] == pytest.approx(0.9, abs=1e-3)
    assert_same_ruling(v_default, v_pre)


def test_default_construction_lazy_skeleton(monkeypatch):
    """默认构造 + lazy + 权重缺失：骨架降级语义与迁移前一致，provider_kind=unavailable。"""
    monkeypatch.setattr("moonbow.guard.models.ModelRegistry", FakeFactoryModels)
    guard = ProgressGuard(lazy_load=True)

    def unavailable():
        raise FileNotFoundError("no weights")

    monkeypatch.setattr(guard, "_ensure_models", unavailable)
    v = guard.check("任务", manifest(evidence="无"))
    assert v.decision == Decision.CLARIFY
    assert v.scores["skeleton_only"] is True
    assert v.scores["provider_kind"] == "unavailable"
    assert "semantic_unavailable" not in v.scores


def test_legacy_provider_signals_shape():
    """LegacyProvider 输出结构与迁移前模型调用一一对应。"""
    provider = LegacyProvider(FakeModels(similarity=0.42, capture=0.77, modality="promise"))
    obs = provider.observe(req="任务", statement="做完了", want_capture=True)
    assert obs.modality == "promise"
    assert obs.similarity == pytest.approx(0.42)
    assert obs.capture_prob == pytest.approx(0.77)
    assert obs.extra_scores == {}
    assert obs.unavailable_reason is None
    obs2 = provider.observe(req="任务", statement="做完了", want_capture=False)
    assert obs2.capture_prob is None
    assert provider.describe()["provider_kind"] == "legacy"


# ---------------------------------------------------------------------------
# 2. Legacy vs Semantic 裁决对比矩阵（同一语义判定 -> 决策零差异）
# ---------------------------------------------------------------------------

def semantic_adapter(matched_modality, matched_capture, score=0.9, backend=None):
    if backend is None:
        backend = FakeBackend(rules={
            "completion.modality@1": FakeRule(matched=matched_modality),
            "completion.capture@1": FakeRule(matched=matched_capture, score=score),
        })
    return SemanticProviderAdapter(SemanticMatcher(guard_runtime(backend)))


@pytest.mark.parametrize("status", ["A", "B", "C", "D"])
@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize("mode", ["advisory", "strict"])
def test_matrix_self_reported_status(status, rounds, mode):
    """自报 B/C/D（以及 A 基线）：同一语义判定下新旧裁决决策零差异。"""
    legacy = legacy_guard(similarity=0.9, capture=0.9)
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(True, True))
    args = dict(rounds=rounds, mode=mode, external_tool_success=None)
    a = legacy.check("任务", manifest(status=status), **args).to_dict()
    b = semantic.check("任务", manifest(status=status), **args).to_dict()
    assert_same_ruling(a, b)
    assert a["scores"]["provider_kind"] == "legacy"
    assert b["scores"]["provider_kind"] == "semantic"


@pytest.mark.parametrize("remaining, success", [("还缺单测", True), ("还缺单测", None), ("无", True)])
def test_matrix_remaining_and_tool_success(remaining, success):
    """有 remaining（硬信号）与工具成功豁免：决策零差异。"""
    legacy = legacy_guard(similarity=0.1, capture=0.1, modality="promise")
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(False, False, score=0.1))
    a = legacy.check("任务", manifest(remaining=remaining),
                     external_tool_success=success).to_dict()
    b = semantic.check("任务", manifest(remaining=remaining),
                       external_tool_success=success).to_dict()
    assert_same_ruling(a, b)


def test_matrix_missing_protocol():
    """缺协议（REQUIRE_MANIFEST）：provider 不参与，零差异。"""
    legacy = legacy_guard(similarity=0.1, capture=0.1)
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(False, False))
    a = legacy.check("任务", "done").to_dict()
    b = semantic.check("任务", "done").to_dict()
    assert a["decision"] == Decision.REQUIRE_MANIFEST.value
    assert_same_ruling(a, b)


@pytest.mark.parametrize("rounds", [1, 2])
def test_matrix_a_without_evidence(rounds):
    """A 无证据：软信号 EVIDENCE 路径决策零差异。"""
    legacy = legacy_guard()
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(True, True))
    a = legacy.check("任务", manifest(evidence="无"), rounds=rounds).to_dict()
    b = semantic.check("任务", manifest(evidence="无"), rounds=rounds).to_dict()
    assert_same_ruling(a, b)
    assert b["decision"] == Decision.CLARIFY.value


@pytest.mark.parametrize("mode", ["advisory", "strict"])
def test_matrix_semantic_dispute(mode):
    """语义争议（SLM 判未完全闭合 + 模态非断言）：advisory CLARIFY /
    strict 第 2 轮 disputed CLOSE，与 legacy 争议路径决策一致
    （n08 实验后保留原契约：闸门方案净退化已回退）。"""
    legacy = legacy_guard(similarity=0.1, capture=0.1, modality="promise")
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(False, False, score=0.1))
    first_a = legacy.check("任务", manifest(), rounds=1, mode=mode).to_dict()
    first_b = semantic.check("任务", manifest(), rounds=1, mode=mode).to_dict()
    assert_same_ruling(first_a, first_b)
    second_a = legacy.check("任务", manifest(), rounds=2, mode=mode).to_dict()
    second_b = semantic.check("任务", manifest(), rounds=2, mode=mode).to_dict()
    assert_same_ruling(second_a, second_b)
    if mode == "strict":
        assert second_b["decision"] == Decision.CLOSE.value
        assert second_b["disputed"] is True
    else:
        assert second_b["decision"] == Decision.CLARIFY.value
        # 语义信号：模态与捕获信号文案逐字一致；SLM 不产余弦相似度，
        # 无 similarity 信号（新分数走新命名键，绝不塞旧键）。
        assert second_b["soft_signals"][0] == second_a["soft_signals"][0]
        assert second_b["soft_signals"][1] == second_a["soft_signals"][2]
        assert len(second_b["soft_signals"]) == 2
        assert len(second_a["soft_signals"]) == 3


def test_matrix_tool_success_exempts_semantic_conflict():
    """工具成功豁免语义争议：CLOSE 且 undisputed，零差异。"""
    legacy = legacy_guard(similarity=0.1, capture=0.1)
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(True, False, score=0.1))
    a = legacy.check("任务", manifest(), external_tool_success=True).to_dict()
    b = semantic.check("任务", manifest(), external_tool_success=True).to_dict()
    assert_same_ruling(a, b)
    assert b["decision"] == Decision.CLOSE.value


def test_semantic_scores_use_new_names_only():
    """SLM 分数只写新命名键，绝不写旧 similarity / capture_prob 键。"""
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(True, True, score=0.93))
    v = semantic.check("任务", manifest())
    assert v.scores[SLM_CAPTURE_SCORE_KEY] == pytest.approx(0.93, abs=1e-3)
    assert "similarity" not in v.scores
    assert "capture_prob" not in v.scores
    assert "modality" in v.scores  # 模态类别（非分数）仍写旧键，与 legacy 键位对齐
    assert v.scores["provider_kind"] == "semantic"


def test_semantic_capture_ok_no_signal_while_legacy_ok_same():
    """同一「闭合」判定下两者都不产出捕获异议。"""
    legacy = legacy_guard(capture=0.9)
    semantic = ProgressGuard(lazy_load=True, provider=semantic_adapter(True, True, score=0.9))
    a = legacy.check("任务", manifest()).to_dict()
    b = semantic.check("任务", manifest()).to_dict()
    assert_same_ruling(a, b)
    assert b["decision"] == Decision.CLOSE.value


# ---------------------------------------------------------------------------
# 3. provider 失败/不可用 -> skeleton 降级 + semantic_unavailable 标记
# ---------------------------------------------------------------------------

def _degraded_verdict(backend):
    guard = ProgressGuard(lazy_load=True,
                          provider=SemanticProviderAdapter(SemanticMatcher(guard_runtime(backend))))
    return guard.check("任务", manifest())


@pytest.mark.parametrize("fault", ["unavailable", "overloaded", "timeout", "invalid_output"])
def test_backend_fault_degrades_to_skeleton(fault):
    """backend 契约内故障：降级进骨架路径，标记 semantic_unavailable，不伪造通过。"""
    backend = FakeBackend(rules={
        "completion.modality@1": FakeRule(matched=True),
        "completion.capture@1": FakeRule(matched=True),
    }, fault=fault)
    v = _degraded_verdict(backend)
    assert v.scores["semantic_unavailable"] is True
    assert v.scores["provider_kind"] == "semantic"
    assert v.scores["semantic_signals"] == []
    assert "modality" not in v.scores and SLM_CAPTURE_SCORE_KEY not in v.scores
    assert v.decision == Decision.CLOSE  # 有证据的 A：骨架层允许放行
    assert v.acceptance == "unverified"  # 但绝不宣称 checked/verified


def test_backend_fault_with_empty_evidence_never_closes():
    """降级后骨架语义保持：A 无证据仍 CLARIFY，不因 provider 故障放行。"""
    backend = FakeBackend(fault="unavailable")
    guard = ProgressGuard(lazy_load=True,
                          provider=SemanticProviderAdapter(SemanticMatcher(guard_runtime(backend))))
    v = guard.check("任务", manifest(evidence="无"))
    assert v.decision == Decision.CLARIFY
    assert v.scores["semantic_unavailable"] is True


def test_missing_pattern_rule_degrades():
    """未注册 pattern 规则 -> 默认负例 ok；modality -> promise 信号，不降级不崩溃。"""
    v = _degraded_verdict(FakeBackend())  # 无规则：确定性负例
    assert v.scores.get("semantic_unavailable") is not True
    assert v.scores["modality"] == "promise"
    assert v.decision == Decision.CLARIFY


def test_semantic_provider_raise_degrades_not_crashes():
    """matcher 抛异常（非契约错误）：同样降级，不崩溃。"""
    class BoomMatcher:
        def match(self, **kwargs):
            raise RuntimeError("boom")

    guard = ProgressGuard(lazy_load=True, provider=SemanticProviderAdapter(BoomMatcher()))
    v = guard.check("任务", manifest(evidence="无"))
    assert v.decision == Decision.CLARIFY
    assert v.scores["semantic_unavailable"] is True
    assert v.scores["provider_kind"] == "semantic"


def test_legacy_provider_error_propagates():
    """legacy provider 推理抛错：保持迁移前行为——向上抛（eager 即 500），不静默降级。"""
    class BadModels(FakeModels):
        def get_modality(self, text):
            raise RuntimeError("inference failed")

    guard = ProgressGuard(lazy_load=True)
    guard._registry = BadModels()
    with pytest.raises(RuntimeError):
        guard.check("任务", manifest())


# ---------------------------------------------------------------------------
# 4. HTTP 层：/check semantic_provider 键、/v1/semantic-status
# ---------------------------------------------------------------------------

def get(path, handler):
    h = handler.__new__(handler)
    h.path = path
    responses = []
    h._send_json = lambda status, data: responses.append((status, data))
    h.do_GET()
    assert len(responses) == 1
    return responses[0]


def post_sem(payload, guard, provider):
    """post 变体：给 handler 实例注入 semantic_provider。"""
    h = GuardHTTPRequestHandler.__new__(GuardHTTPRequestHandler)
    body = json.dumps(payload).encode("utf-8")
    h.path = "/check"
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h.guard = guard
    h.semantic_provider = provider
    responses = []
    h._send_json = lambda status, data: responses.append((status, data))
    h.do_POST()
    assert len(responses) == 1
    return responses[0]


def test_http_semantic_provider_key_defaults_to_legacy():
    guard = legacy_guard()
    status, data = post({"req": "任务", "resp": manifest()}, guard)
    assert status == 200
    assert data["scores"]["provider_kind"] == "legacy"


def test_http_semantic_provider_explicit_legacy_same():
    guard = legacy_guard()
    a = post({"req": "任务", "resp": manifest()}, guard)
    b = post({"req": "任务", "resp": manifest(), "semantic_provider": "legacy"}, guard)
    assert a[0] == b[0] == 200
    assert a[1]["scores"] == b[1]["scores"]


@pytest.mark.parametrize("value", ["slm", "auto", 1, True, [], {}, None])
def test_http_semantic_provider_invalid_value_400(value):
    status, data = post({"req": "任务", "resp": manifest(), "semantic_provider": value}, legacy_guard())
    assert status == 400
    assert "semantic_provider" in data["error"]


def test_http_semantic_provider_without_runtime_400():
    status, data = post({"req": "任务", "resp": manifest(), "semantic_provider": "semantic"}, legacy_guard())
    assert status == 400
    assert "not configured" in data["error"]


def test_http_semantic_provider_with_runtime_uses_semantic():
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    adapter = semantic_adapter(True, True, score=0.93)
    status, data = post_sem({"req": "任务", "resp": manifest(), "semantic_provider": "semantic"}, guard, adapter)
    assert status == 200
    assert data["scores"]["provider_kind"] == "semantic"
    assert data["scores"][SLM_CAPTURE_SCORE_KEY] == pytest.approx(0.93, abs=1e-3)
    assert "similarity" not in data["scores"]


def test_http_semantic_status_default_legacy():
    h = GuardHTTPRequestHandler.__new__(GuardHTTPRequestHandler)
    h.path = "/v1/semantic-status"
    h.guard = legacy_guard()
    h.semantic_provider = None
    responses = []
    h._send_json = lambda s, d: responses.append((s, d))
    h.do_GET()
    status, data = responses[0]
    assert status == 200
    assert data["provider_kind"] == "legacy"
    assert data["calibrated"] is False
    assert "backend" in data


def test_http_semantic_status_with_semantic_provider():
    h = GuardHTTPRequestHandler.__new__(GuardHTTPRequestHandler)
    h.path = "/v1/semantic-status"
    h.guard = legacy_guard()
    h.semantic_provider = semantic_adapter(True, True)
    responses = []
    h._send_json = lambda s, d: responses.append((s, d))
    h.do_GET()
    status, data = responses[0]
    assert status == 200
    assert data["provider_kind"] == "semantic"
    assert data["backend"] == "fake"
    assert data["calibrated"] is False


def test_http_unknown_route_still_404():
    status, _ = get("/v1/nope", GuardHTTPRequestHandler)
    assert status == 404


def test_post_helper_with_provider():
    """post() 辅助支持注入 semantic_provider（类属性覆盖用实例属性）。"""
    assert post({"req": "任务", "resp": manifest()}, legacy_guard())[0] == 200

# ---------------------------------------------------------------------------
# 5. R5 白名单路由（final-test 验收后）：PASS 参与判定 / process 仅 shadow /
#    modality uncalibrated（纯逻辑，FakeBackend）
# ---------------------------------------------------------------------------

from moonbow.guard.semantic_provider import (  # noqa: E402
    MODALITY_UNCALIBRATED_KEY,
    R4_CAPABILITY_STATUS,
    R5_PASS_SIGNALS,
    R5_SHADOW_SIGNALS,
    SHADOW_SIGNALS_KEY,
    build_whitelist_adapter,
    env_provider_key,
)
from moonbow.semantic.schema import MatchResponse  # noqa: E402


def whitelist_adapter(modality_matched=True, capture_matched=True,
                      alignment_matched=True, process_matched=False,
                      process_error=False):
    """R5 白名单 adapter + FakeBackend（base 规则供各 adapter 回落）。"""

    def process_rule(request):
        if process_error:
            return MatchResponse(status="error", reason_code="invalid_output",
                                 request_id=request.request_id)
        return MatchResponse(
            status="ok", matched=process_matched, score=0.9 if process_matched else 0.1,
            provenance=__import__("moonbow.semantic.schema", fromlist=["Provenance"]).Provenance(
                backend="fake", model_revision="fake-rev-1",
                pattern_version="process.unresolved@1", calibrated=False,
                adapter=request.adapter),
            request_id=request.request_id)

    backend = FakeBackend(rules={
        "modality.assertive@1": FakeRule(matched=modality_matched),
        "completion.asserted@1": FakeRule(matched=capture_matched, score=0.9),
        "task.object.alignment@1": FakeRule(matched=alignment_matched),
        "process.unresolved@1": process_rule,
    })
    for name in ("completion-lora-v2", "alignment-lora-v1", "process-lora-v2"):
        backend.register_adapter(name)
    return build_whitelist_adapter(SemanticMatcher(guard_runtime(backend)))


def test_r5_capability_status_table():
    """R4_CAPABILITY_STATUS：R6b 起 process=pass-preliminary（holdout_v2 口径
    P=0.8571 R=0.9474），3 PASS + modality no_go（如实，不写成达标）。"""
    assert R4_CAPABILITY_STATUS["process.unresolved"]["status"] == "pass-preliminary"
    assert "0.9474" in R4_CAPABILITY_STATUS["process.unresolved"]["calibration"]
    assert R4_CAPABILITY_STATUS["completion.asserted"]["status"] == "pass"
    assert R4_CAPABILITY_STATUS["task.object.alignment"]["status"] == "pass"
    assert R4_CAPABILITY_STATUS["modality.assertive"]["status"] == "no_go"
    assert R5_PASS_SIGNALS["completion.asserted"]["adapter"] == "completion-lora-v2"
    assert R5_PASS_SIGNALS["process.unresolved"]["adapter"] == "process-lora-v2"
    assert R5_SHADOW_SIGNALS == {}  # R6b：shadow 清空，process 参与判定


def test_r5_pass_signals_participate_clean_close():
    """全 PASS 命中负例（陈述干净）：无语义信号，CLOSE；三个 PASS 信号结果
    均落 scores（R6b 起 process 参与判定，shadow_signals 为空）。"""
    guard = ProgressGuard(lazy_load=True,
                          provider=whitelist_adapter(process_matched=False))
    v = guard.check("任务", manifest())
    assert v.decision == Decision.CLOSE
    assert v.scores["semantic_signals"] == []
    assert v.scores["completion.asserted@1"]["matched"] is True
    assert v.scores["task.object.alignment@1"]["matched"] is True
    assert v.scores["process.unresolved@1"]["matched"] is False
    assert v.scores.get("shadow_signals") is None  # R6b：shadow 清空
    assert v.scores["unresolved_detected"] is False


def test_r5_pass_signals_dispute_clarify():
    """PASS 信号判负（capture=False + alignment=False + 模态非断言）：
    参与判定，产生语义信号 -> CLARIFY。"""
    guard = ProgressGuard(lazy_load=True, provider=whitelist_adapter(
        modality_matched=False, capture_matched=False, alignment_matched=False))
    v = guard.check("任务", manifest(), rounds=1)
    assert v.decision == Decision.CLARIFY
    assert v.scores["completion.asserted@1"]["matched"] is False
    assert v.scores["task.object.alignment@1"]["matched"] is False
    assert any("未完全闭合" in s for s in v.scores["semantic_signals"])
    assert any("客体偏离" in s for s in v.scores["semantic_signals"])


def test_r5_process_unresolved_judges_when_hit():
    """R6b：process.unresolved 参与判定——命中=True 产出未决软信号 → CLARIFY。"""
    guard = ProgressGuard(lazy_load=True,
                          provider=whitelist_adapter(process_matched=True))
    v = guard.check("任务", manifest())
    assert v.decision == Decision.CLARIFY
    assert any("未解决的疑点" in s for s in v.scores["semantic_signals"])
    assert v.scores["process.unresolved@1"]["matched"] is True
    assert v.scores["unresolved_detected"] is True
    assert "semantic_unavailable" not in v.scores


def test_r5_process_signal_failure_degrades():
    """process 判定契约内失败：作为参与判定信号，整体降级不出半套。"""
    guard = ProgressGuard(lazy_load=True,
                          provider=whitelist_adapter(process_error=True))
    v = guard.check("任务", manifest())
    assert v.scores["semantic_unavailable"] is True
    assert v.decision == Decision.CLOSE  # 骨架裁决：干净清单放行


def test_r5_participating_signal_failure_degrades():
    """PASS 信号失败（未注册 adapter -> unsupported）：整体降级，不出半套。"""
    backend = FakeBackend(rules={
        "modality.assertive@1": FakeRule(matched=True),
    })
    provider = build_whitelist_adapter(SemanticMatcher(guard_runtime(backend)))
    guard = ProgressGuard(lazy_load=True, provider=provider)
    v = guard.check("任务", manifest())
    assert v.scores["semantic_unavailable"] is True
    assert v.scores["semantic_signals"] == []


def test_r5_modality_uncalibrated_marked():
    """modality 底座 zero-shot：uncalibrated 标记进 scores 与 describe；
    信号文案语义不变；不计入达标声明。"""
    guard = ProgressGuard(lazy_load=True, provider=whitelist_adapter(
        modality_matched=False))
    v = guard.check("任务", manifest())
    assert v.scores[MODALITY_UNCALIBRATED_KEY] is True
    assert any("「promise」" in s for s in v.scores["semantic_signals"])
    status = guard._provider.describe()
    assert status["modality_judgment"] == {
        "mode": "base zero-shot", "uncalibrated": True, "attestation": False}
    assert status["capabilities"]["signals"]["process.unresolved"]["status"] == "pass-preliminary"
    assert status["capabilities"]["shadow_signals"] == []
    assert sorted(status["capabilities"]["pass_signals"]) == [
        "completion.asserted", "process.unresolved", "task.object.alignment"]


# ---------------------------------------------------------------------------
# 6. R5 默认切换（MOONBOW_GUARD_PROVIDER）：配置激活 / 未配置零变化 /
#    非法值回退 legacy
# ---------------------------------------------------------------------------

def test_env_provider_key_values(monkeypatch):
    monkeypatch.delenv("MOONBOW_GUARD_PROVIDER", raising=False)
    assert env_provider_key() is None
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "semantic")
    assert env_provider_key() == "semantic"
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "legacy")
    assert env_provider_key() == "legacy"
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "  SEMANTIC  ")
    assert env_provider_key() == "semantic"
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "yes")
    assert env_provider_key() is None  # 非法值：记 warning，按缺省 legacy


def test_env_semantic_default_switches_sdk(monkeypatch):
    """MOONBOW_GUARD_PROVIDER=semantic：SDK 默认走 semantic（构造自配置工厂）。"""
    monkeypatch.setattr("moonbow.guard.semantic_provider.semantic_provider_from_config",
                        lambda cfg=None: semantic_adapter(True, True, score=0.93))
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "semantic")
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    v = guard.check("任务", manifest())
    assert v.scores["provider_kind"] == "semantic"
    assert v.scores[SLM_CAPTURE_SCORE_KEY] == pytest.approx(0.93, abs=1e-3)


def test_env_unset_defaults_legacy_zero_change(monkeypatch):
    """环境变量未设置：默认 legacy，与既有行为逐字段一致（零变化）。"""
    monkeypatch.delenv("MOONBOW_GUARD_PROVIDER", raising=False)
    monkeypatch.setattr(
        "moonbow.guard.semantic_provider.semantic_provider_from_config",
        lambda cfg=None: (_ for _ in ()).throw(
            AssertionError("未配置环境不得构建 semantic provider")))
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    v = guard.check("任务", manifest()).to_dict()
    pre = legacy_guard().check("任务", manifest(), rounds=1).to_dict()
    assert v["scores"]["provider_kind"] == "legacy"
    assert v["scores"] == pre["scores"]
    assert_same_ruling(v, pre)


def test_env_invalid_value_falls_back_legacy(monkeypatch):
    monkeypatch.setattr(
        "moonbow.guard.semantic_provider.semantic_provider_from_config",
        lambda cfg=None: (_ for _ in ()).throw(
            AssertionError("非法值不得构建 semantic provider")))
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "yes")
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    v = guard.check("任务", manifest())
    assert v.scores["provider_kind"] == "legacy"


def test_env_semantic_build_failure_falls_back_legacy(monkeypatch):
    """配置缺失/工厂失败：记 warning 并回退 legacy，不崩溃、不阻塞裁决。"""
    def boom(cfg=None):
        raise FileNotFoundError(cfg or "config/semantic_runtime_lora.json")

    monkeypatch.setattr(
        "moonbow.guard.semantic_provider.semantic_provider_from_config", boom)
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "semantic")
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    v = guard.check("任务", manifest())
    assert v.scores["provider_kind"] == "legacy"


def test_env_semantic_explicit_legacy_override(monkeypatch):
    """默认 semantic 时 provider 覆盖仍生效（override 优先于环境默认）。"""
    monkeypatch.setattr(
        "moonbow.guard.semantic_provider.semantic_provider_from_config",
        lambda cfg=None: semantic_adapter(True, True))
    monkeypatch.setenv("MOONBOW_GUARD_PROVIDER", "semantic")
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    v = guard.check("任务", manifest())
    assert v.scores["provider_kind"] == "semantic"
    v2 = guard._check("任务", manifest(), 1, None, "advisory",
                      provider_override=LegacyProvider(FakeModels()))
    assert v2.scores["provider_kind"] == "legacy"


# ---------------------------------------------------------------------------
# 7. R5 HTTP 默认切换：default_provider_key 配置驱动
# ---------------------------------------------------------------------------

def test_http_default_provider_key_semantic(monkeypatch):
    monkeypatch.setattr(GuardHTTPRequestHandler, "default_provider_key", "semantic")
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels()
    adapter = semantic_adapter(True, True, score=0.93)
    status, data = post_sem({"req": "任务", "resp": manifest()}, guard, adapter)
    assert status == 200
    assert data["scores"]["provider_kind"] == "semantic"
    # 显式 legacy 覆盖缺省
    status, data = post_sem({"req": "任务", "resp": manifest(),
                             "semantic_provider": "legacy"}, guard, adapter)
    assert status == 200
    assert data["scores"]["provider_kind"] == "legacy"


def test_http_default_semantic_without_runtime_400(monkeypatch):
    monkeypatch.setattr(GuardHTTPRequestHandler, "default_provider_key", "semantic")
    status, data = post({"req": "任务", "resp": manifest()}, legacy_guard())
    assert status == 400
    assert "not configured" in data["error"]


def test_http_default_provider_key_legacy_unchanged():
    """缺省（未配置）：default_provider_key=legacy，既有行为零变化。"""
    assert GuardHTTPRequestHandler.default_provider_key == "legacy"
    status, data = post({"req": "任务", "resp": manifest()}, legacy_guard())
    assert status == 200
    assert data["scores"]["provider_kind"] == "legacy"

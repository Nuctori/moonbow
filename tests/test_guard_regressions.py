# -*- coding: utf-8 -*-
"""Deterministic protocol, decision and HTTP boundary regressions."""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("src"))

from moonbow import Decision, ProgressGuard, StatusCode, parse_manifest
from moonbow.guard.server import GuardHTTPRequestHandler


def manifest(status="A 全部完成", evidence="pytest: 5 passed", remaining="无"):
    return f"STATUS: {status}\nREMAINING: {remaining}\nEVIDENCE: {evidence}"


class FakeModels:
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


def model_guard(**kwargs):
    guard = ProgressGuard(lazy_load=True)
    guard._registry = FakeModels(**kwargs)
    return guard


@pytest.fixture
def skeleton(monkeypatch):
    guard = ProgressGuard(lazy_load=True)

    def unavailable():
        raise FileNotFoundError("test weights unavailable")

    monkeypatch.setattr(guard, "_ensure_models", unavailable)
    return guard


@pytest.mark.parametrize("status", ["Z", "ABC", "Approved", "Done", "不是全部完成", "Z 全部完成", "B 全部完成", "A 部分完成", "A / B", ""])
def test_invalid_status_requires_manifest(status):
    text = manifest(status=status)
    parsed = parse_manifest(text)
    assert parsed.status is None
    assert not parsed.is_valid_format
    verdict = model_guard().check("任务", text, rounds=2, external_tool_success=True)
    assert verdict.decision == Decision.REQUIRE_MANIFEST
    assert not verdict.is_closed


@pytest.mark.parametrize("status, expected", [
    ("A", StatusCode.A), ("a 全部完成", StatusCode.A), ("全部完成", StatusCode.A),
    ("B 部分完成", StatusCode.B), ("部分完成", StatusCode.B),
    ("C 进行中/受阻", StatusCode.C), ("C 进行中或受阻", StatusCode.C),
    ("进行中", StatusCode.C), ("受阻", StatusCode.C),
    ("D 失败/已回滚", StatusCode.D), ("D 失败或已回滚", StatusCode.D),
    ("失败", StatusCode.D), ("已回滚", StatusCode.D),
])
def test_status_enum_compatible_forms(status, expected):
    parsed = parse_manifest(manifest(status=status))
    assert parsed.is_valid_format
    assert parsed.status == expected


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, [], {}, 0.0])
def test_sdk_rejects_non_boolean_tool_success(value):
    with pytest.raises(TypeError, match="external_tool_success"):
        model_guard().check("任务", manifest(), external_tool_success=value)


@pytest.mark.parametrize("evidence", ["", "无", "没有", "沒有", "none", "NONE", "NULL", "n/a"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_skeleton_empty_evidence_never_closes(skeleton, evidence, rounds):
    verdict = skeleton.check("任务", manifest(evidence=evidence), rounds=rounds)
    assert verdict.decision == Decision.CLARIFY
    assert not verdict.is_closed
    assert not verdict.disputed
    assert verdict.scores["skeleton_only"] is True
    assert any("EVIDENCE" in signal for signal in verdict.soft_signals)


@pytest.mark.parametrize("success", [False, None])
def test_tool_without_success_does_not_replace_evidence(skeleton, success):
    verdict = skeleton.check("任务", manifest(evidence="无"), external_tool_success=success)
    assert verdict.decision == Decision.CLARIFY


def test_real_boolean_success_can_replace_empty_evidence(skeleton):
    verdict = skeleton.check("任务", manifest(evidence="无"), external_tool_success=True)
    assert verdict.decision == Decision.CLOSE
    assert not verdict.disputed


def test_skeleton_with_evidence_closes_without_fabricated_dispute(skeleton):
    verdict = skeleton.check("任务", manifest(), rounds=2)
    assert verdict.decision == Decision.CLOSE
    assert not verdict.disputed
    assert verdict.scores["skeleton_only"] is True


@pytest.mark.parametrize("signals", [{"similarity": 0.1}, {"capture": 0.1}, {"similarity": 0.1, "capture": 0.1}])
def test_semantic_conflict_clarifies_then_disputed_close(signals):
    guard = model_guard(**signals)
    first = guard.check("任务", manifest(), rounds=1, mode="strict")
    assert first.decision == Decision.CLARIFY
    assert first.soft_signals
    assert not first.disputed
    second = guard.check("任务", manifest(), rounds=2, mode="strict")
    assert second.decision == Decision.CLOSE
    assert second.is_closed
    assert second.disputed
    assert second.soft_signals == first.soft_signals
    assert second.to_dict()["disputed"] is True


def test_empty_evidence_cannot_disputed_close():
    verdict = model_guard(similarity=0.1, capture=0.1).check("任务", manifest(evidence="无"), rounds=2)
    assert verdict.decision == Decision.CLARIFY
    assert not verdict.disputed
    assert any("EVIDENCE" in signal for signal in verdict.soft_signals)


def test_no_conflict_is_normal_close():
    verdict = model_guard().check("任务", manifest(), rounds=2)
    assert verdict.decision == Decision.CLOSE
    assert not verdict.disputed


def test_tool_success_overrides_only_semantic_conflicts():
    verdict = model_guard(similarity=0.1, capture=0.1).check("任务", manifest(), external_tool_success=True)
    assert verdict.decision == Decision.CLOSE
    assert not verdict.disputed


@pytest.mark.parametrize("status, remaining", [("B", "无"), ("C", "无"), ("D", "无"), ("A", "还缺单测")])
@pytest.mark.parametrize("success", [True, False, None])
def test_hard_signals_cannot_be_disputed_or_overridden(status, remaining, success):
    verdict = model_guard(similarity=0.1, capture=0.1).check(
        "任务", manifest(status=status, remaining=remaining), rounds=2,
        external_tool_success=success,
    )
    assert verdict.decision == Decision.BLOCK
    assert not verdict.is_closed
    assert not verdict.disputed
    assert verdict.hard_signals


def post(payload, guard):
    handler = GuardHTTPRequestHandler.__new__(GuardHTTPRequestHandler)
    body = json.dumps(payload).encode("utf-8")
    handler.path = "/check"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.guard = guard
    responses = []
    handler._send_json = lambda status, data: responses.append((status, data))
    handler.do_POST()
    assert len(responses) == 1
    return responses[0]


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, [], {}, 0.0])
def test_http_rejects_non_boolean_tool_success(value):
    status, data = post({"req": "任务", "resp": manifest(), "external_tool_success": value}, model_guard())
    assert status == 400
    assert "external_tool_success" in data["error"]


@pytest.mark.parametrize("success, decision", [(True, "CLOSE"), (False, "CLARIFY"), (None, "CLARIFY")])
def test_http_preserves_boolean_tool_success(skeleton, success, decision):
    status, data = post({"req": "任务", "resp": manifest(evidence="无"), "external_tool_success": success}, skeleton)
    assert status == 200
    assert data["decision"] == decision


def test_http_missing_tool_success_defaults_to_none(skeleton):
    status, data = post({"req": "任务", "resp": manifest(evidence="无")}, skeleton)
    assert status == 200
    assert data["decision"] == "CLARIFY"


@pytest.mark.parametrize("payload", [[], None, "false", 1])
def test_http_requires_json_object(payload):
    status, data = post(payload, model_guard())
    assert status == 400
    assert "error" in data


@pytest.mark.parametrize("rounds", [None, "invalid", [], {}])
def test_http_invalid_rounds_returns_400(rounds):
    status, data = post({"req": "任务", "resp": manifest(), "rounds": rounds}, model_guard())
    assert status == 400
    assert "rounds" in data["error"]


def test_http_internal_error_remains_500(monkeypatch):
    guard = model_guard()

    def broken(**kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(guard, "check", broken)
    status, data = post({"req": "任务", "resp": manifest()}, guard)
    assert status == 500
    assert "Internal Guard Error" in data["error"]

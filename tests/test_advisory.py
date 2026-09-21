import importlib.util
import io
import json
from pathlib import Path

import pytest

from test_guard_regressions import manifest, model_guard, post, skeleton
from moonbow import Decision


@pytest.mark.parametrize("signals", [{"similarity": .1}, {"capture": .1}, {"modality": "promise"}])
def test_one_semantic_review_without_promoting_acceptance(signals):
    guard = model_guard(**signals)
    first = guard.check("任务", manifest(), rounds=99)
    assert first.review_requested and not first.allow_stop
    assert first.acceptance == "disputed"
    second = guard.check("任务", manifest(), semantic_review_used=True)
    assert second.allow_stop and not second.review_requested
    assert not second.is_closed and second.disputed
    assert second.decision == Decision.CLARIFY
    assert second.acceptance == "disputed"
    assert second.prompt is None


def test_combined_signals_request_one_review():
    verdict = model_guard(similarity=.1, capture=.1, modality="promise").check("任务", manifest())
    assert len(verdict.scores["semantic_signals"]) == 3
    assert verdict.review_requested
    assert verdict.prompt.count("【进度守卫复核建议】") == 1


@pytest.mark.parametrize("status", ["B", "C", "D"])
def test_incomplete_can_stop_without_becoming_complete(status):
    verdict = model_guard(similarity=.1).check("任务", manifest(status=status))
    assert verdict.allow_stop and not verdict.is_closed
    assert not verdict.review_requested
    assert verdict.acceptance == "incomplete"
    strict = model_guard().check("任务", manifest(status=status), mode="strict")
    assert not strict.allow_stop


def test_missing_evidence_stops_unverified_not_closed(skeleton):
    verdict = skeleton.check("任务", manifest(evidence="无"), semantic_review_used=True)
    assert verdict.allow_stop and not verdict.is_closed
    assert verdict.acceptance == "unverified"
    strict = skeleton.check("任务", manifest(evidence="无"), mode="strict")
    assert not strict.allow_stop


def test_model_unavailable_is_not_acceptance(skeleton):
    verdict = skeleton.check("任务", manifest())
    assert verdict.allow_stop and verdict.acceptance == "unverified"


def test_no_signals_is_not_independent_acceptance():
    verdict = model_guard().check("任务", manifest())
    assert verdict.allow_stop and verdict.acceptance == "unchecked"


def test_host_verified_signal_is_explicit():
    verdict = model_guard().check("任务", manifest(), external_tool_success=True)
    assert verdict.allow_stop and verdict.acceptance == "verified"


def test_format_not_waived_by_semantic_budget():
    verdict = model_guard().check("任务", "done", semantic_review_used=True)
    assert not verdict.allow_stop and verdict.acceptance == "invalid"


@pytest.mark.parametrize("fields", [{"mode": "unknown"}, {"mode": []}, {"semantic_review_used": "false"}, {"semantic_review_used": 1}])
def test_http_rejects_policy_types(fields):
    code, _ = post({"req": "任务", "resp": manifest(), **fields}, model_guard())
    assert code == 400


def test_http_returns_policy():
    code, body = post({"req": "任务", "resp": manifest(), "semantic_review_used": True}, model_guard(similarity=.1))
    assert code == 200
    assert body["allow_stop"] and not body["is_closed"]
    assert body["acceptance"] == "disputed"


@pytest.fixture
def hook(tmp_path, monkeypatch):
    path = Path("src/moonbow/guard/extensions/guard_stop_hook.py")
    spec = importlib.util.spec_from_file_location("stop_hook_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(module, "LOG", str(tmp_path / "hook.log"))
    monkeypatch.setattr(module, "MODE", "advisory")
    return module


def test_hook_persists_budget_and_resets_new_user(hook, tmp_path, monkeypatch, capsys):
    transcript = tmp_path / "session.jsonl"
    def write_task(identifier):
        with transcript.open("a", encoding="utf-8") as stream:
            for role, text in [("user", "same task"), ("assistant", manifest())]:
                stream.write(json.dumps({"uuid": identifier + role, "message": {"role": role, "content": text}}) + "\n")
    write_task("one")
    calls = []
    class Opener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            verdict = model_guard(similarity=.1).check(**payload)
            return io.BytesIO(json.dumps(verdict.to_dict()).encode())
    monkeypatch.setattr(hook.urllib.request, "build_opener", lambda *_: Opener())
    def run(active=False):
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps({"session_id": "session", "transcript_path": str(transcript), "stop_hook_active": active})))
        assert hook.main() == 0
        return capsys.readouterr().out
    assert json.loads(run())["decision"] == "block"
    assert run(True) == ""
    assert run() == ""
    assert calls[-1]["semantic_review_used"] is True
    write_task("two")
    assert json.loads(run())["decision"] == "block"
    assert calls[-1]["semantic_review_used"] is False


@pytest.mark.parametrize("text", ["<task>修复接口</task>", "请解释【进度守卫提示】这段输出"])
def test_hook_preserves_real_markup_tasks(hook, tmp_path, text):
    path = tmp_path / "markup.jsonl"
    rows = [
        {"uuid": "old", "message": {"role": "user", "content": "old"}},
        {"uuid": "new", "message": {"role": "user", "content": text}},
        {"message": {"role": "assistant", "content": "done"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert hook._last_texts(str(path)) == (text, "done", "new")


def test_hook_ignores_own_feedback_and_clears_old_assistant(hook, tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [
        {"uuid": "u1", "message": {"role": "user", "content": "task"}},
        {"message": {"role": "assistant", "content": "done"}},
        {"message": {"role": "user", "content": "【进度守卫提示】review"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert hook._last_texts(str(path)) == ("task", "done", "u1")
    rows.append({"uuid": "u2", "message": {"role": "user", "content": "next"}})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert hook._last_texts(str(path)) == ("next", "", "u2")

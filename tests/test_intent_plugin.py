import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = load("zcode_hook", ROOT / "tools/intent_plugin/zcode_hook.py")
service_mod = load("intent_service", ROOT / "src/moonbow/intent/service.py")


def test_hook_endpoint_is_pinned_loopback():
    assert hook.JUDGE_ENDPOINT == "http://127.0.0.1:18502/judge"
    assert "environ" not in __import__("inspect").getsource(hook.call_service)


def test_decision_text_prefers_raw_fields():
    assert hook.decision_text("Bash", {"command": "git status"}).startswith("[Bash] git status")
    assert hook.decision_text("Write", {"file_path": "a.ts",
                                        "content": "x=1"}).startswith("[Write] x=1")
    assert hook.decision_text("Foo", "raw-string") == "raw-string"
    assert "[Foo] {}" in hook.decision_text("Foo", {})


def test_judge_payload_is_capped_and_kind_tool_call():
    payload = hook.judge_payload({"tool_name": "Bash", "tool_input": {"command": "x" * 9000}}, "s1")
    assert payload["kind"] == "tool_call"
    assert payload["session"] == "s1"
    assert len(payload["text"]) <= 4000


def test_format_reminders_none_when_empty():
    assert hook.format_reminders({"reminders": []}) is None
    message = hook.format_reminders({"reminders": [{"policy": "strong", "reason": "r",
                                                    "quote": "q"}]})
    assert "仅提醒" in message and "strong" in message


def test_hook_main_fail_open_when_service_down(monkeypatch, capsys):
    def boom(*a, **kw):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(hook, "call_service", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"tool_name": "Bash", "tool_input": {"command": "ls"}})))
    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_hook_main_emits_additional_context(monkeypatch, capsys):
    monkeypatch.setattr(hook, "call_service", lambda p: {
        "reminders": [{"policy": "manual-review", "reason": "r", "quote": "q"}]})
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert hook.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert "additionalContext" in out and "manual-review" in out["additionalContext"]


def test_hook_main_ignores_bad_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("not-json"))
    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_service_config_contract():
    cfg = service_mod.load_config(ROOT / "config/intent_service.json")
    assert set(cfg["adapters"]) == {"pass1", "verify", "select"}
    assert len(cfg["cards"]) == 5
    assert cfg["port"] == 18502
    for path in cfg["adapters"].values():
        assert Path(path).exists(), path


def test_adapter_bank_switches_active_adapter():
    class FakePeft:
        def __init__(self):
            self.active = None
        def set_adapter(self, name):
            self.active = name
    bank = service_mod.LoRAAdapterBank.__new__(service_mod.LoRAAdapterBank)
    bank._model = FakePeft()
    bank._active = None
    bank.set("pass1")
    assert bank._model.active == "pass1" and bank._active == "pass1"
    bank.set("pass1")                       # 幂等：不重复切换
    bank.set("verify")
    assert bank._model.active == "verify"


def test_service_judge_rejects_blank_text_without_model():
    svc = service_mod.IntentService.__new__(service_mod.IntentService)
    result = svc.judge({"session": "s", "text": "  "})
    assert result["error"] and result["reminders"] == []

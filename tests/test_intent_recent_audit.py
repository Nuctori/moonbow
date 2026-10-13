import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

import os

def _local(*names):
    root = ROOT / "data" / "intent_eval"
    return all((root / n).exists() for n in names)


requires_local_data = pytest.mark.skipif(
    not os.environ.get("MOONBOW_HAS_PRIVATE_DATA") and not _local(
        "train_t3_v13.jsonl"), reason="本地真实数据不在社区树中，跳过")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "intent_eval"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "intent_eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tr = load("train_intent_lora")
real = load("exp_real_recall")
autopsy = load("exp_error_autopsy")


def _has_local_gold():
    return bool(real.GOLD)


requires_local_gold = pytest.mark.skipif(
    not _has_local_gold(), reason="本地 gold 锚定文件不在社区树中，跳过")


def test_real_recall_adapter_alias_and_unknown_flag():
    args = real.parse_args(["--adapter", "new", "--verify", "v", "--select", "s", "--train", "t", "--out", "o"])
    assert args.pass1 == "new"
    with pytest.raises(SystemExit):
        real.parse_args(["--adaptre", "new"])


@requires_local_gold
def test_real_recall_detects_in_sample():
    anchor = real.GOLD[0][0]
    assert real.evaluation_partition([{"gold_note": f"anchored {anchor}"}]) == "in-sample"
    assert real.evaluation_partition([]) == "heldout-unverified"


def test_missing_sensitivity_does_not_improve_checkpoint_score():
    assert tr.checkpoint_score(0.8, -1) == 0.8
    assert tr.checkpoint_score(0.7, -1) < tr.checkpoint_score(0.8, -1)


def test_best_adapter_weights_are_restored(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        set_peft_model_state_dict=lambda model, weights, adapter_name:
        (calls.append((model, weights, adapter_name)) or SimpleNamespace(missing_keys=[], unexpected_keys=[]))))
    model = SimpleNamespace(state_dict=lambda: {"layer.lora_A.default.weight": 1})
    def loader(path, device):
        assert path == "best" and device == "xpu"
        return {"weight": 42}
    assert tr.restore_best_adapter(model, "best", 5, loader) is model
    assert calls == [(model, {"weight": 42}, "default")]


@pytest.mark.parametrize("rows", [[], [{"event_text": "x", "label": "bad"}],
                                  [{"event_text": " ", "label": "violation"}]])
def test_training_invalid_rows_fail_explicitly(rows):
    with pytest.raises(ValueError):
        tr.validated_rows(rows, "t3")


def test_normalized_exact_leak_check():
    exact, near = tr.leak_check([{"event_text": "a b c"}], [{"case_id": "e", "event_text": "abc"}])
    assert exact == ["e"] and not near


def test_autopsy_requires_explicit_case_constraints():
    with pytest.raises(ValueError, match="constraint"):
        autopsy.validate_case({"event_text": "通知文案轮换", "gold": "violation"})


def test_autopsy_antonym_uses_m4_not_m3():
    modes = autopsy.classify_results("v", "v", "c", "c", "v")
    assert "antonym-no-flip" in modes
    assert "constraint-identity-sensitive" in modes
    assert "antonym-no-flip" not in autopsy.classify_results("v", "v", "c", "v", "c")


def test_v12_audit_gold_is_not_weak_labels(tmp_path, monkeypatch):
    import json
    builder = load("build_v12_data")
    gen = tmp_path / "gen"
    gen.mkdir()
    def dump(path, rows):
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    fake_anchors = {f"sha{i:04d}": ("violation" if i < 5 else "compliant")
                    for i in range(12)}
    dump(gen / "real_commits_domain.jsonl", [
        {"sha": sha, "subject": sha, "diff_head": "diff"} for sha in fake_anchors])
    dump(gen / "real_commits_domain_labels.jsonl", [
        {"sha": sha, "polarity": "none"} for sha in fake_anchors])
    dump(tmp_path / "train_t3_v9.jsonl", [{"case_id": "base", "event_text": "base", "label": "compliant"}])
    monkeypatch.setattr(builder, "AUDIT_HELD_OUT", set(fake_anchors))
    monkeypatch.setattr(builder, "AUDIT_GOLD", fake_anchors)
    monkeypatch.setattr(builder, "OUT", tmp_path)
    monkeypatch.setattr(builder, "GEN", gen)
    assert builder.main() == 0
    rows = tr.load_jsonl(tmp_path / "heldout_real_recall.jsonl")
    assert len(rows) == 12
    assert sum(r["label"] == "violation" for r in rows) == 5
    assert sum(r["label"] == "compliant" for r in rows) == 7


def test_v13_exact_pairing_and_unique_ids(tmp_path, monkeypatch):
    import json
    builder = load("build_v13_data")
    prefix = "x" * 80
    rows = [
        {"case_id": "dup", "event_text": prefix + "a", "label": "violation", "constraints": [{"quote": "ban"}]},
        {"case_id": "dup", "event_text": prefix + "b", "label": "compliant", "constraints": [{"quote": "allow"}]},
        {"case_id": "p1", "event_text": "same", "label": "violation", "constraints": [{"quote": "ban"}]},
        {"case_id": "p2", "event_text": "same", "label": "compliant", "constraints": [{"quote": "allow"}]},
    ]
    (tmp_path / "train_t3_v11.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(builder, "OUT", tmp_path)
    monkeypatch.setattr(builder, "REPO", tmp_path)
    assert builder.main() == 0
    out = tr.load_jsonl(tmp_path / "train_t3_v13.jsonl")
    assert len(out) == 6
    assert len({r["case_id"] for r in out}) == 6
    assert sum(r["event_text"] == "same" for r in out) == 4


def test_best_adapter_incomplete_restore_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        set_peft_model_state_dict=lambda *a, **kw: SimpleNamespace(
            missing_keys=["layer.lora_A.default.weight"], unexpected_keys=[])))
    model = SimpleNamespace(state_dict=lambda: {"layer.lora_A.default.weight": 1})
    with pytest.raises(ValueError, match="恢复不完整"):
        tr.restore_best_adapter(model, "best", 5, lambda *a, **kw: {})


def test_autopsy_adapter_alias():
    assert autopsy.parse_args(["--adapter", "new"]).pass1 == "new"
    with pytest.raises(SystemExit):
        autopsy.parse_args(["--adaptre", "new"])


@requires_local_data
def test_rebuilt_v13_ids_and_sources():
    rows = tr.load_jsonl(ROOT / "data/intent_eval/train_t3_v13.jsonl")
    assert len(rows) == len({r["case_id"] for r in rows})
    assert all(r.get("source_case_id") for r in rows)


@requires_local_data
def test_rebuilt_real_holdout_is_complete_and_leak_free():
    held = tr.load_jsonl(ROOT / "data/intent_eval/heldout_real_recall.jsonl")
    train = tr.load_jsonl(ROOT / "data/intent_eval/train_t3_v12.jsonl")
    assert len(held) == 12
    assert sum(r["label"] == "violation" for r in held) == 5
    assert sum(r["label"] == "compliant" for r in held) == 7
    assert tr.leak_check(train, held) == ([], [])


@requires_local_gold
def test_full_sha_marks_in_sample():
    assert real.evaluation_partition(
        [{"sha": real.GOLD[0][0] + "abcdef0123"}]) == "in-sample"


def test_experiment_resume_rejects_changed_inputs():
    evidence = load("experiment_evidence")
    with pytest.raises(ValueError):
        evidence.require_matching_evidence({"hash": "a"}, {"hash": "b"})


@requires_local_data
def test_autopsy_all_cases_have_m5():
    rows = tr.load_jsonl(ROOT / "data/intent_eval/autopsy_cases.jsonl")
    assert all(autopsy.validate_case(r)["paraphrase"] != r["event_text"] for r in rows)


@requires_local_data
def test_v12_unique_ids():
    rows = tr.load_jsonl(ROOT / "data/intent_eval/train_t3_v12.jsonl")
    assert len(rows) == len({r["case_id"] for r in rows})
    assert all(r.get("source_case_id") for r in rows)


def test_legacy_provenance_requires_explicit_opt_in(tmp_path):
    evidence = load("experiment_evidence")
    train = tmp_path / "train.jsonl"
    train.write_text("sample", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        evidence.training_provenance(tmp_path, train)
    assert evidence.training_provenance(tmp_path, train, True)["status"] == "unverified-legacy"


def test_atomic_json_retries_transient_windows_lock(tmp_path, monkeypatch):
    evidence = load("experiment_evidence")
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("temporary lock")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    target = tmp_path / "state.json"
    evidence.atomic_json(target, {"complete": True})
    assert len(attempts) == 2
    assert 'true' in target.read_text(encoding="utf-8")

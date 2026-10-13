import pytest
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/intent_eval"))
sys.path.insert(0, str(ROOT / "src"))
def _requires_local(*names):
    return pytest.mark.skipif(
        not all((ROOT / "data/intent_eval" / n).exists() for n in names),
        reason="本地真实数据不在社区树中，跳过")


spec = importlib.util.spec_from_file_location("v14", ROOT / "tools/intent_eval/build_v14_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@_requires_local("train_t3_v14.jsonl", "eval_t3_v14_dev.jsonl")
def test_v14_actual_replay_polarity_and_isolation():
    out = ROOT / "data/intent_eval"
    rows = module.load_jsonl(out / "train_t3_v14.jsonl")
    evals = module.load_jsonl(out / "eval_t3_v14_dev.jsonl")
    assert len(rows) == 234
    assert len(evals) == 206
    assert len({r["case_id"] for r in rows}) == len(rows)
    assert sum("repeat" in r for r in rows) == 144
    assert sum(module.label_of(r, "t3") == "insufficient" for r in rows) == 30
    assert module.leak_check(rows, evals) == ([], [])


def test_parent_adapter_ancestry_never_promoted_to_verified(tmp_path):
    import json
    from experiment_evidence import adapter_evidence, file_hash, training_provenance
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test")
    train = tmp_path / "train.jsonl"
    train.write_text("row", encoding="utf-8")
    manifest = {"train_sha256": file_hash(train), "adapter": adapter_evidence(tmp_path),
                "parent_adapter": {"path": "historical-parent"}}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert training_provenance(tmp_path, train)["status"] == "verified-current-stage-unverified-ancestry"

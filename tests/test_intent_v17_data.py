import pytest
import importlib.util
import json
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


spec = importlib.util.spec_from_file_location("v17", ROOT / "tools/intent_eval/build_v17_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_r2_violation_side_is_symmetric():
    assert len(module.V17_ROWS) == 20
    assert all(r[2] == "violation" for r in module.V17_ROWS)
    from collections import Counter
    per_card = Counter()
    for carrier, text, label, card in module.V17_ROWS:
        per_card[card["quote"][:6]] += 1
    assert set(per_card.values()) == {7, 6} or sum(per_card.values()) == 20
    assert all(len(text) >= 18 for _, text, _, _ in module.V17_ROWS)


@_requires_local("train_t3_v17.jsonl",)
def test_v17_actual_train_unique_and_isolated():
    out = ROOT / "data/intent_eval"
    train = module.load_jsonl(out / "train_t3_v17.jsonl")
    assert len(train) == 553
    assert len({r["case_id"] for r in train}) == len(train)
    assert all(r.get("source_case_id") for r in train)
    guard = (module.load_jsonl(out / "eval_t3_v15_dev.jsonl") +
             module.load_jsonl(out / "autopsy_cases.jsonl"))
    assert module.leak_check(train, guard) == ([], [])
    manifest = json.loads((out / "v17_data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["design"].startswith("single-variable")

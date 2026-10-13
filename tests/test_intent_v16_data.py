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


spec = importlib.util.spec_from_file_location("v16", ROOT / "tools/intent_eval/build_v16_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_v16_boundary_rows_are_paired_and_clean():
    assert len(module.NEW_ROWS) >= 24
    compliant = [r for r in module.NEW_ROWS if r[2] == "compliant"]
    violation = [r for r in module.NEW_ROWS if r[2] == "violation"]
    assert len(compliant) >= 2 * len(violation)
    assert all(len(r[1]) >= 20 for r in module.NEW_ROWS)
    assert all(card["quote"] for *_, card in module.NEW_ROWS)


@_requires_local("train_t3_v16.jsonl",)
def test_v16_actual_train_unique_and_isolated():
    out = ROOT / "data/intent_eval"
    train = module.load_jsonl(out / "train_t3_v16.jsonl")
    assert len(train) == 528
    assert len({r["case_id"] for r in train}) == len(train)
    assert all(r.get("source_case_id") for r in train)
    guard = (module.load_jsonl(out / "eval_t3_v15_dev.jsonl") +
             module.load_jsonl(out / "autopsy_cases.jsonl"))
    assert module.leak_check(train, guard) == ([], [])

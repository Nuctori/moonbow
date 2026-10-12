import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/intent_eval"))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("roles", ROOT / "tools/intent_eval/build_v15_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_roles_have_both_exclusive_policies_and_actions():
    for split, size in (("train", 72), ("dev", 24)):
        rows = module.controlled_rows(split)
        assert len(rows) == size
        assert len({r["case_id"] for r in rows}) == size
        for row in rows:
            assert row["label"] == ("violation" if row["policy"] == row["action"] else "compliant")
        assert sum(r["label"] == "violation" for r in rows) == size // 2


def test_v15_inputs_are_unique_and_isolated():
    out = ROOT / "data/intent_eval"
    train = module.load_jsonl(out / "train_t3_v15.jsonl")
    dev = module.load_jsonl(out / "eval_t3_v15_dev.jsonl")
    assert len(train) == 450
    assert len(dev) == 230
    assert len({r["case_id"] for r in train}) == 450
    assert module.leak_check(train, dev + module.load_jsonl(out / "autopsy_cases.jsonl")) == ([], [])

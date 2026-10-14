import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/intent_eval"))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("v19", ROOT / "tools/intent_eval/build_v19_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_v19_carriers_are_paired_and_independent_from_gate():
    assert len(module.V19_ROWS) >= 24
    comp = [r for r in module.V19_ROWS if r[2] == "compliant"]
    viol = [r for r in module.V19_ROWS if r[2] == "violation"]
    assert len(comp) >= len(viol) >= 10
    train_texts = {t for _, t, _, _ in module.V19_ROWS}
    for gate in module.GATE_VIOL_ROWS:
        assert gate["event_text"] not in train_texts


def test_v19_actual_train_unique_and_isolated():
    out = ROOT / "data/intent_eval"
    train = module.load_jsonl(out / "train_t3_v19.jsonl")
    assert len(train) == 609
    assert len({r["case_id"] for r in train}) == len(train)
    guard = (module.load_jsonl(out / "eval_t3_v15_dev.jsonl") +
             module.load_jsonl(out / "gate_real_fp.jsonl") +
             module.load_jsonl(out / "gate_carrier_viol.jsonl"))
    assert module.leak_check(train, guard) == ([], [])

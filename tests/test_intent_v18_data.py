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


spec = importlib.util.spec_from_file_location("v18", ROOT / "tools/intent_eval/build_v18_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_r3_rows_are_symmetric_and_disjoint_domains():
    rows = module.r3_rows()
    assert len(rows) == 24
    assert len({r["case_id"] for r in rows}) == 24
    assert sum(r["label"] == "violation" for r in rows) == 12
    for r in rows:
        assert "不允许" in r["constraints"][0]["quote"]
    topics = {t for t, _ in module.R3_DOMAINS}
    assert topics.isdisjoint({"缓存", "排序", "确认", "记录", "压缩", "备份"})
    assert topics.isdisjoint({"校验", "审核", "加密", "重试", "审批", "清理"})


@_requires_local("train_t3_v18.jsonl",)
def test_v18_actual_train_builds_on_v17_and_isolated():
    out = ROOT / "data/intent_eval"
    train = module.load_jsonl(out / "train_t3_v18.jsonl")
    v17 = module.load_jsonl(out / "train_t3_v17.jsonl")
    assert len(train) == 601
    assert len(v17) == 553
    assert len({r["case_id"] for r in train}) == len(train)
    guard = (module.load_jsonl(out / "eval_t3_v15_dev.jsonl") +
             module.load_jsonl(out / "autopsy_cases.jsonl"))
    assert module.leak_check(train, guard) == ([], [])

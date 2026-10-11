# -*- coding: utf-8 -*-
"""tests/test_intent_training_contract.py — 训练/评测契约的机械守卫。

来源（第五轮：把两个后端都真训了再测）：旧训练脚本把 dev 自动并入训练集
再用同一条 dev 评测，"16/16"是泄漏产出的自测。本文件锁定新训练器的硬规则，
使这类错误无法再次静默发生。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from moonbow.intent import protocol  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_intent_lora", REPO / "tools" / "intent_eval" / "train_intent_lora.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)


# ── 泄漏检测 ────────────────────────────────────────────────────────────

def test_leak_check_catches_exact_duplicate():
    train = [{"event_text": "新增别名表并加权排序"}]
    evals = [{"case_id": "e1", "event_text": "新增别名表并加权排序"}]
    exact, near = tr.leak_check(train, evals)
    assert exact == ["e1"] and not near


def test_leak_check_catches_near_duplicate():
    train = [{"event_text": "新增 DEST_ALIAS 表，按别名命中数加权排序"}]
    evals = [{"case_id": "e2",
              "event_text": "新增 DEST_ALIAS 表，按别名命中数加权排序。"}]
    exact, near = tr.leak_check(train, evals)
    assert not exact
    assert near and near[0][0] == "e2" and near[0][1] >= 0.85


def test_leak_check_passes_for_disjoint_sets():
    train = [{"event_text": "新增别名表并加权排序"}]
    evals = [{"case_id": "e3", "event_text": "把首屏耗时降到 1s 以内"}]
    exact, near = tr.leak_check(train, evals)
    assert not exact and not near


def test_shipped_train_eval_sets_are_leak_free():
    """仓库内实际交付的数据集（v4）必须无泄漏（回归护栏）。"""
    for task in ("t3", "t4"):
        train = tr.load_jsonl(REPO / "data" / "intent_eval" / f"train_{task}_v4.jsonl")
        evals = tr.load_jsonl(REPO / "data" / "intent_eval" / f"eval_{task}_v4.jsonl")
        exact, near = tr.leak_check(train, evals)
        assert not exact, f"{task} 存在精确重复：{exact}"
        assert not near, f"{task} 存在近重：{near}"


def test_shipped_sets_cover_all_labels():
    """eval 必须覆盖全部标签；train 覆盖除 L0 闸门接管类之外的标签。

    T3 的 insufficient_info 由适用范围闸门（机械）产生，训练集刻意不含。
    """
    exempt = {"t3": {"insufficient"}, "t4": set()}
    for task, spec in ((t, tr.TASK_SPECS[t]) for t in ("t3", "t4")):
        evals = tr.load_jsonl(
            REPO / "data" / "intent_eval" / f"eval_{task}_v4.jsonl")
        got = {tr.label_of(r, task) for r in evals}
        assert got == set(spec["labels"]), f"{task}/eval 缺标签：{got}"
        train = tr.load_jsonl(
            REPO / "data" / "intent_eval" / f"train_{task}_v4.jsonl")
        got_tr = {tr.label_of(r, task) for r in train}
        need = set(spec["labels"]) - exempt[task]
        assert need <= got_tr, f"{task}/train 缺标签：{need - got_tr}"


# ── 标签别名与首 token 互异 ─────────────────────────────────────────────

def test_t4_runtime_alias_matches_training_alias():
    """运行时别名必须与训练别名逐字一致（T4 的原始字面量不能用）。"""
    assert protocol.COVERAGE_ALIASES == tr.TASK_SPECS["t4"]["alias"]
    assert protocol.COVERAGE_ALIASES["supported_full"] == "FULL"
    assert protocol.COVERAGE_ALIASES["supported_partial"] == "PART"


def test_t4_raw_literals_are_known_broken():
    """固化原因：T4 原始字面量共享首 token，说明为何必须用别名。"""
    assert protocol.COVERAGE_LITERALS == protocol.COVERAGE_ALIASES
    assert protocol.COVERAGE_LITERALS["supported_full"] != \
        protocol.COVERAGE_LITERALS["supported_partial"]


def test_t3_aliases_are_the_raw_literals():
    assert tr.TASK_SPECS["t3"]["alias"] == {
        "violation": "violation", "compliant": "compliant",
        "insufficient": "insufficient"}


# ── 标签映射的健壮性 ───────────────────────────────────────────────────

def test_label_of_accepts_ledger_and_audit_vocabularies():
    assert tr.label_of({"label": "supports_conflict"}, "t3") == "violation"
    assert tr.label_of({"label": "violation"}, "t3") == "violation"
    assert tr.label_of({"label": "supported_partial"}, "t4") == \
        "supported_partial"
    assert tr.label_of({"label": "PART"}, "t4") == "supported_partial"
    assert tr.label_of({"label": "unknown-thing"}, "t3") is None


# ── 评测集规模守卫（第六轮教训）─────────────────────────────────────────

MIN_EVAL_N = 50


def test_eval_sets_are_large_enough_for_accuracy_claims():
    """v4 起：评测集 ≥50 条（第六轮教训的修复已落地，此为回归护栏）。"""
    for task in ("t3", "t4"):
        evals = tr.load_jsonl(
            REPO / "data" / "intent_eval" / f"eval_{task}_v4.jsonl")
        assert len(evals) >= MIN_EVAL_N, (
            f"{task} 评测集仅 {len(evals)} 条（<{MIN_EVAL_N}），"
            f"不足以支撑精度声明")


def test_single_sample_flip_swing_is_bounded():
    """单样本翻转摆动 ≤5pp（对应 ≥20 条；v4 为 57/205 条）。"""
    for task in ("t3", "t4"):
        n = len(tr.load_jsonl(
            REPO / "data" / "intent_eval" / f"eval_{task}_v4.jsonl"))
        swing = 100.0 / n
        assert swing <= 5.0, (
            f"{task} 单样本摆动 {swing:.1f}pp（{n} 条），超过 5pp 噪声线")

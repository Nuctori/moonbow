# -*- coding: utf-8 -*-
"""tests/test_intent_label_protocol.py — 受限输出协议的标签性质守卫。

来源（T4 真实实验，.mimosa/intent-loop/exp_t4_coverage_report.md）：
覆盖四分类里 supported_full 与 supported_partial 共享首 token
（都是 'supported'），首 token 比较退化到无法区分；且按长度归一化会让
较长标签系统性占优（3-token 的 insufficient_info 压倒 2-token 的
supported_*，与输入内容无关）。这两点都会静默污染实验结论，故加机械守卫。

本文件用**假 tokenizer/模型**验证协议逻辑，不依赖真实权重（可在 CI 跑）。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moonbow.intent import protocol  # noqa: E402


class _FakeTok:
    """最简 tokenizer：按字符切，等价于"每个字符一个 token"。"""

    def __init__(self, mapping):
        self._m = mapping

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._m[text]}


class _FakeModel:
    """返回固定 logits 的假模型：最后一个位置的 logits 由构造给定。"""

    def __init__(self, vocab_logits):
        self._v = torch.tensor(vocab_logits, dtype=torch.float32)

    def __call__(self, ids):
        n = ids.shape[1]

        class _Out:
            def __init__(self, v):
                self.logits = v

        # 每个位置都给同样的 logits，便于按位置取用
        return _Out(self._v.unsqueeze(0).unsqueeze(0).repeat(1, n, 1))


# ── 首 token 互异守卫 ───────────────────────────────────────────────────

def test_distinct_first_tokens_passes_for_t3_labels():
    """T3 三标签首 token 互异 → 历史首 token 比较结论有效。"""
    ids = {"violation": 9676, "compliant": 848, "insufficient": 1283}
    protocol.assert_distinct_first_tokens(ids)   # 不抛即通过


def test_distinct_first_tokens_rejects_shared_prefix():
    """覆盖四标签里 supported_full/partial 共享首 token → 必须报错。"""
    ids = {"supported_full": 17675, "supported_partial": 17675,
           "no_evidence": 2083, "insufficient_info": 1283}
    with pytest.raises(ValueError, match="首 token 相同"):
        protocol.assert_distinct_first_tokens(ids)
    # 报错信息须点名可用替代方案
    with pytest.raises(ValueError, match="label_sequence_scores"):
        protocol.assert_distinct_first_tokens(ids)


def test_distinct_first_tokens_can_scope_to_subset():
    ids = {"violation": 1, "compliant": 2, "supported_full": 3,
           "supported_partial": 3}
    protocol.assert_distinct_first_tokens(ids, labels=["violation", "compliant"])
    with pytest.raises(ValueError):
        protocol.assert_distinct_first_tokens(
            ids, labels=["supported_full", "supported_partial"])


# ── 整序列打分：能区分共享前缀标签 ──────────────────────────────────────

def test_sequence_scores_separate_shared_prefix_labels():
    """共享首 token 的两个标签，在整序列似然下必须能被区分。"""
    toks = _FakeTok({"prompt": [1, 2, 3],
                     "supported_full": [10, 11],
                     "supported_partial": [10, 12]})
    # 第 4 个位置偏好 11（full），第 5 个偏好 12（partial）→ 分数应不同
    vocab = [-1.0] * 20
    vocab[10] = -0.1
    vocab[11] = -0.2
    vocab[12] = -5.0
    model = _FakeModel(vocab)
    scores = protocol.sequence_scores(
        model, toks, torch.tensor([[1, 2, 3]]),
        ["supported_full", "supported_partial"],
        literals={"supported_full": "supported_full",
                  "supported_partial": "supported_partial"},
        normalize=False)
    assert scores["supported_full"] != scores["supported_partial"]


def test_normalize_amplifies_length_effect():
    """长度归一化会让长标签占优：本用例固化这一已知偏置，提醒勿盲用。"""
    toks = _FakeTok({"p": [1], "ab": [10, 11], "abcd": [10, 11, 12, 13]})
    vocab = [-1.0] * 20
    for t in (10, 11, 12, 13):
        vocab[t] = -1.0
    model = _FakeModel(vocab)
    pid = torch.tensor([[1]])
    raw = protocol.sequence_scores(model, toks, pid, ["ab", "abcd"],
                                   literals={"ab": "ab", "abcd": "abcd"},
                                   normalize=False)
    avg = protocol.sequence_scores(model, toks, pid, ["ab", "abcd"],
                                   literals={"ab": "ab", "abcd": "abcd"},
                                   normalize=True)
    # 原始总似然：长标签自然更低（累计更多负项）
    assert raw["abcd"] < raw["ab"]
    # 长度归一后：两者接近（每 token 平均似然相同）——偏置来自长度差
    assert abs(avg["abcd"] - avg["ab"]) < 1e-6


# ── 标签集与判据常量 ────────────────────────────────────────────────────

def test_coverage_labels_and_literals_consistent():
    assert protocol.COVERAGE_LABELS == ("supported_full", "supported_partial",
                                        "no_evidence", "insufficient_info")
    assert set(protocol.COVERAGE_LITERALS) == set(protocol.COVERAGE_LABELS)


def test_coverage_prompt_carries_mechanical_vs_semantic_boundary():
    """边界条款必须进 prompt：否则"测试绿"会被当成"目标达成"。"""
    assert "测试是否通过" in protocol.COVERAGE_BOUNDARY
    assert "目标是否达成" in protocol.COVERAGE_BOUNDARY
    assert "supported_full" in protocol.COVERAGE_BOUNDARY


def test_labels_three_way_contract_unchanged():
    assert protocol.LABELS == ("violation", "compliant", "insufficient")
    assert protocol.VERDICT_OF_LABEL["insufficient"] == "insufficient_info"

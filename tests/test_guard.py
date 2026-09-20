# -*- coding: utf-8 -*-
"""tests/test_guard.py

Moonbow / Progress Guard 核心协议与裁决引擎单元测试套件。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("src"))
from moonbow import (
    ProgressGuard,
    Decision,
    StatusCode,
    parse_manifest,
)


def test_parse_manifest_standard():
    text = (
        "STATUS: A 全部完成\n"
        "REMAINING: 无\n"
        "EVIDENCE: pytest tests/test_blueprints.py 49 passed"
    )
    m = parse_manifest(text)
    assert m.is_valid_format is True
    assert m.status == StatusCode.A
    assert m.has_remaining is False
    assert m.has_evidence is True
    assert "49 passed" in m.evidence


def test_parse_manifest_cjk_colons_and_markdown():
    text = (
        "**STATUS**：B 部分完成\n"
        "- REMAINING: 还差文档更新\n"
        "EVIDENCE：无"
    )
    m = parse_manifest(text)
    assert m.is_valid_format is True
    assert m.status == StatusCode.B
    assert m.has_remaining is True
    assert m.remaining == "还差文档更新"
    assert m.has_evidence is False


def test_parse_manifest_missing_fields():
    text = "我搞定啦，代码都写好了！"
    m = parse_manifest(text)
    assert m.is_valid_format is False
    assert m.status is None


def test_guard_require_manifest():
    guard = ProgressGuard(models_dir="models", device="cpu")
    verdict = guard.check(
        req="修复超时重试 bug",
        resp="我改好代码了，应该没问题了。",
        rounds=1,
    )
    assert verdict.decision == Decision.REQUIRE_MANIFEST
    assert verdict.is_closed is False
    assert verdict.prompt is not None
    assert "STATUS:" in verdict.prompt


def test_guard_self_reported_partial_blocked():
    guard = ProgressGuard(models_dir="models", device="cpu")
    resp = (
        "STATUS: B 部分完成\n"
        "REMAINING: 超时异常抛出逻辑还未联调\n"
        "EVIDENCE: 无"
    )
    verdict = guard.check(
        req="为请求增加超时和重试并抛出自定义异常",
        resp=resp,
        rounds=1,
    )
    assert verdict.decision == Decision.BLOCK
    assert verdict.is_closed is False
    assert any("B 部分完成" in s for s in verdict.hard_signals)


def test_guard_unverified_assert_clarify():
    guard = ProgressGuard(models_dir="models", device="cpu")
    # 自报全部完成，但证据为空
    resp = (
        "STATUS: A 全部完成\n"
        "REMAINING: 无\n"
        "EVIDENCE: 无"
    )
    verdict = guard.check(
        req="把接口改为 GraphQL 游标分页",
        resp=resp,
        rounds=1,
    )
    assert verdict.decision == Decision.CLARIFY
    assert verdict.is_closed is False
    assert verdict.prompt is not None


def test_guard_disputed_close_round2():
    guard = ProgressGuard(models_dir="models", device="cpu")
    # 第 2 轮主模型在看到提示后，重申 A 并补充了测试结果证据
    resp = (
        "STATUS: A 全部完成\n"
        "REMAINING: 无\n"
        "EVIDENCE: python -m pytest tests/test_graphql.py 返回码 0，5 passed"
    )
    verdict = guard.check(
        req="把接口改为 GraphQL 游标分页",
        resp=resp,
        rounds=2,
    )
    assert verdict.decision == Decision.CLOSE
    assert verdict.is_closed is True


def test_guard_external_tool_success_override():
    guard = ProgressGuard(models_dir="models", device="cpu")
    resp = (
        "STATUS: A 全部完成\n"
        "REMAINING: 无\n"
        "EVIDENCE: 验证通过"
    )
    # 外部真实单测已通过
    verdict = guard.check(
        req="修复某些潜在边界 bug",
        resp=resp,
        rounds=1,
        external_tool_success=True,
    )
    assert verdict.decision == Decision.CLOSE
    assert verdict.is_closed is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])

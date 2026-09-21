# -*- coding: utf-8 -*-
"""tests/test_guard.py

Moonbow / Progress Guard 核心协议与裁决引擎单元测试套件。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

# 生产权重不随仓库发布（~930MB，经 GitHub Release 分发）。
# 无权重环境（如 CI core job）自动跳过依赖微模型的用例，仅跑协议/路由子集。
MODELS_READY = os.path.isdir(os.environ.get("MOONBOW_MODELS_DIR", "models"))
needs_models = pytest.mark.skipif(not MODELS_READY, reason="需要生产权重目录 models/（经 Release 分发，仓库不含）")

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


@needs_models
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


@needs_models
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


@needs_models
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


@needs_models
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
        mode="strict",
    )
    assert verdict.decision == Decision.CLOSE
    assert verdict.is_closed is True


@needs_models
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


# ---------------- 骨架模式（无权重可跑）----------------
# models_dir 指向不存在目录 + lazy_load=True：强制骨架裁决（仅定量层），
# 本地与 CI（无权重）行为一致，验证状态机的确定性部分。
def _skeleton_guard():
    return ProgressGuard(models_dir="models/__nonexistent__", lazy_load=True)


def test_skeleton_require_manifest():
    guard = _skeleton_guard()
    v = guard.check(req="任意任务", resp="我做完了，都挺好的")
    assert v.decision == Decision.REQUIRE_MANIFEST
    assert not v.is_closed


def test_skeleton_status_b_blocked():
    guard = _skeleton_guard()
    resp = "STATUS: B 部分完成{N}REMAINING: 未运行测试{N}EVIDENCE: 无".replace("{N}", chr(10))
    v = guard.check(req="任务", resp=resp)
    assert v.decision == Decision.BLOCK
    assert any("B 部分完成" in s for s in v.hard_signals)


def test_skeleton_status_d_blocked():
    guard = _skeleton_guard()
    resp = "STATUS: D 失败或已回滚{N}REMAINING: 无{N}EVIDENCE: 已回滚改动".replace("{N}", chr(10))
    v = guard.check(req="任务", resp=resp)
    assert v.decision == Decision.BLOCK


def test_skeleton_remaining_blocked():
    guard = _skeleton_guard()
    resp = "STATUS: A 全部完成{N}REMAINING: 文档还没写{N}EVIDENCE: pytest 5 passed".replace("{N}", chr(10))
    v = guard.check(req="任务", resp=resp)
    assert v.decision == Decision.BLOCK
    assert any("文档还没写" in s for s in v.hard_signals)


def test_skeleton_tool_success_close():
    guard = _skeleton_guard()
    resp = "STATUS: A 全部完成{N}REMAINING: 无{N}EVIDENCE: pytest tests/test_utils.py 返回码 0, 5 passed".replace("{N}", chr(10))
    v = guard.check(req="实现 slugify 并补充单测", resp=resp, external_tool_success=True)
    assert v.decision == Decision.CLOSE and v.is_closed
    assert v.scores.get("skeleton_only") is True  # 走到模型层且成功降级


def test_skeleton_tool_success_cannot_override_bad_format():
    guard = _skeleton_guard()
    v = guard.check(req="任务", resp="都做完了", external_tool_success=True)
    assert v.decision == Decision.REQUIRE_MANIFEST


def test_protocol_markdown_bold_and_fullwidth_colon():
    m1 = parse_manifest("**STATUS**: A 全部完成{N}**REMAINING**: 无{N}**EVIDENCE**: pytest 5 passed".replace("{N}", chr(10)))
    assert m1.is_valid_format and m1.status.value == "A"
    m2 = parse_manifest("STATUS：B 部分完成{N}REMAINING：差文档{N}EVIDENCE：无".replace("{N}", chr(10)))
    assert m2.is_valid_format and m2.status.value == "B"

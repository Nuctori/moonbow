# -*- coding: utf-8 -*-
"""阶段审计引擎（process_audit）确定性规则测试。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moonbow.guard.process_audit import StageAuditor, audit_stage_payload
from moonbow.guard.protocol import StageBlock, StageFinding


def A(blocks, prior=None, ver=1, req="修复登录页崩溃"):
    return StageAuditor().audit(req, blocks, prior, ver)


def claim(seq, text="已经修复了，测试也通过了"):
    return StageBlock(seq=seq, kind="text", text=text)


def think(seq, text):
    return StageBlock(seq=seq, kind="thinking", text=text)


def tool_ok(seq, name="pytest", cid="t1"):
    return StageBlock(seq=seq, kind="toolResult", text="1 passed", tool_call_id=cid,
                      tool_name=name, is_error=False)


def tool_err(seq, name="pytest", cid="t2"):
    return StageBlock(seq=seq, kind="toolResult", text="1 failed", tool_call_id=cid,
                      tool_name=name, is_error=True)


# ---- 思考疑点 ----

def test_thinking_concern_is_candidate_when_resolved_in_same_block():
    r = A([think(1, "这里可能遗漏了权限检查。所以我接下来补一个检查并加测试。")])
    fs = [f for f in r["findings"] if f["kind"] == "thinking-concern"]
    assert fs and fs[0]["status"] == "candidate"


def test_thinking_concern_without_resolution_is_unverified_not_actionable():
    r = A([think(1, "不确定这个假设是否成立。")])
    fs = [f for f in r["findings"] if f["kind"] == "thinking-concern"]
    assert fs and fs[0]["status"] == "unverified"


def test_hidden_cot_no_thinking_blocks_no_findings():
    r = A([claim(1, "问题定位在缓存层，准备调整失效逻辑。")])
    assert not [f for f in r["findings"] if f["kind"] == "thinking-concern"]


# ---- 完成声明 vs 工具证据 ----

def test_claim_without_any_tool_evidence_is_unverified_not_failure():
    r = A([tool_ok(1), claim(2)])
    fs = [f for f in r["findings"] if f["kind"] == "unverified-claim"]
    assert not fs  # 有成功工具证据（虽早于声明且无意图），不产生 unverified
    assert not [f for f in r["findings"] if f["status"] == "actionable"]


def test_claim_with_zero_evidence_stays_unverified():
    r = A([claim(1, "已经完成了全部工作")])
    fs = [f for f in r["findings"] if f["kind"] == "unverified-claim"]
    assert fs and fs[0]["status"] == "unverified"
    assert not r["reminder"]  # 未知不提醒


def test_evidence_before_intent_cannot_support_later_claim():
    blocks = [
        tool_ok(1),                                   # 修改前的测试
        StageBlock(seq=2, kind="text", text="我接下来修改登录逻辑，准备重写校验。"),
        claim(3),                                     # 修改后的完成声明
    ]
    r = A(blocks)
    fs = [f for f in r["findings"] if f["kind"] == "evidence-order"]
    assert fs and fs[0]["status"] == "actionable"
    assert "修改前" in fs[0]["summary"] or "早于" in fs[0]["summary"]


def test_evidence_after_intent_supports_claim_no_finding():
    blocks = [
        StageBlock(seq=1, kind="text", text="我接下来修改登录逻辑。"),
        tool_ok(2, cid="t9"),
        claim(3),
    ]
    r = A(blocks)
    assert not [f for f in r["findings"] if f["kind"] == "evidence-order"]


def test_failed_tool_after_claim_is_contradiction():
    r = A([claim(1), tool_err(2)])
    fs = [f for f in r["findings"] if f["kind"] == "contradiction"]
    assert fs and fs[0]["status"] == "actionable"
    assert r["reminder"] and "失败" in r["reminder"]["summary"]


def test_claim_before_intent_not_flagged_as_evidence_order():
    # 复核 P1 回归：声明(2)先于意图(3)时，早于二者的证据(1)可有效支持声明，
    # 不得误报 evidence-order（否则会烧掉语义预算）
    blocks = [
        tool_ok(1),
        claim(2, "已经修复了，测试也通过了"),
        StageBlock(seq=3, kind="text", text="我接下来修改登录逻辑，准备重写校验。"),
    ]
    r = A(blocks)
    assert not [f for f in r["findings"] if f["kind"] == "evidence-order"]
    assert r["reminder"] is None


def test_thinking_fingerprint_stable_across_processes():
    # 复核 P3：fingerprint 不得依赖盐化 hash
    import subprocess, sys, json, os
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from moonbow.guard.process_audit import StageAuditor;"
        "from moonbow.guard.protocol import StageBlock;"
        "r = StageAuditor().audit('x', [StageBlock(seq=1, kind='thinking', "
        "text='不确定这个假设是否成立。')], [], 1);"
        "print(json.dumps([f['fingerprint'] for f in r['findings']]))"
    )
    outs = set()
    for _ in range(2):
        env = dict(os.environ, PYTHONHASHSEED="random")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        outs.add(out.stdout.strip())
    assert len(outs) == 1, outs


def test_interim_report_without_claim_words_triggers_nothing():
    r = A([claim(1, "问题定位在缓存层，接下来调整失效逻辑。")])
    assert r["findings"] == [] and r["reminder"] is None


# ---- prior 复核 ----

def test_later_tool_evidence_resolves_evidence_order_finding():
    first = A([
        tool_ok(1), claim(2, "已经修复了"),
    ])
    # 无意图块时不产生 evidence-order；手工构造 prior 场景
    prior = StageFinding(
        fingerprint="evidence-order:2:1", kind="evidence-order", status="actionable",
        summary="完成声明仅有早于修改意图的证据",
        evidence=[{"seq": 2, "kind": "text", "excerpt": "claim"},
                  {"seq": 1, "kind": "text", "excerpt": "intent"}],
        snapshot_version=1,
    )
    r = A([tool_ok(5, cid="t5")], prior=[prior], ver=2)
    fs = {f["fingerprint"]: f for f in r["findings"]}
    assert fs["evidence-order:2:1"]["status"] == "resolved"


# ---- 提醒文案 ----

def test_reminder_contains_both_legal_responses():
    r = A([claim(1), tool_err(2)])
    s = r["reminder"]["suggestion"]
    assert "补做" in s and ("如实" in s or "修正" in s)


# ---- 幂等 / fingerprint ----

def test_same_blocks_produce_same_fingerprints():
    blocks = [tool_ok(1), StageBlock(seq=2, kind="text", text="准备重构。"), claim(3)]
    r1 = A(blocks, ver=1)
    r2 = A(blocks, ver=1)
    f1 = sorted(f["fingerprint"] for f in r1["findings"])
    f2 = sorted(f["fingerprint"] for f in r2["findings"])
    assert f1 == f2 and f1


# ---- HTTP payload 入口 ----

def test_payload_validation():
    with pytest.raises(ValueError):
        audit_stage_payload({"blocks": "not-a-list"})
    with pytest.raises(ValueError):
        audit_stage_payload({"blocks": [], "findings": 5})
    r = audit_stage_payload({"blocks": [{"seq": 1, "kind": "text", "text": "已经修复完成"}],
                             "req": "x", "snapshot_version": 3})
    assert r["based_on"] == 3


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        StageBlock(seq=1, kind="bogus")

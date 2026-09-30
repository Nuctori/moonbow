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

def test_reminder_instructs_continuation_not_just_restatement():
    """提醒必须指向"继续执行"，而非给模型"申报未完成即可收工"的出口。

    2026-09-22 实测：旧文案（"请二选一……或如实修正完成声明"）让弱主动性
    模型直接申报"部分完成"并终止，把本可完成的任务变成提前投降。
    新契约：明确要求补做验证/继续调用工具，并显式否掉"仅以文字代替执行"。
    """
    r = A([claim(1), tool_err(2)])
    s = r["reminder"]["suggestion"]
    assert "补做" in s, "应要求补做验证"
    assert "继续" in s, "应要求继续执行"
    assert "不要仅以文字" in s, "应否掉仅表态收工"


def test_reminder_does_not_offer_restatement_only_exit():
    """回归保护：文案不得把"如实申报未完成"当作可接受的终局选项。"""
    r = A([claim(1), tool_err(2)])
    s = r["reminder"]["suggestion"]
    assert "请二选一" not in s, "不应提供二选一式的退出选项"


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


# ---- 写入后未验证（2026-09-22 修复的规则缝隙）----
# 旧实现把写入类工具的成功结果也计入 last_success_tool，于是"改完没验证就
# 宣布完成"（真实 SWE 运行里最普遍的失败模式）恰好落在 2.1/2.2/2.3 三条分支
# 之间的缝隙里 → 零发现。下面四条锁住修复后的行为边界。

def _edit_ok(seq, cid="w1"):
    return StageBlock(seq=seq, kind="toolResult", text="Successfully replaced 1 block",
                      tool_call_id=cid, tool_name="edit", is_error=False)


def test_write_without_verification_is_actionable():
    """改完直接宣布完成、全程无验证 -> actionable（核心修复点）。"""
    r = A([
        think(1, "我接下来要修改 login 函数"),
        StageBlock(seq=2, kind="toolCall", text='{"name":"edit"}', tool_name="edit"),
        _edit_ok(3),
        claim(4),
    ])
    fs = [f for f in r["findings"] if f["kind"] == "unverified-claim"
          and f["status"] == "actionable"]
    assert fs, f"应报'改了但没验证'，实际: {r['findings']}"
    assert "验证" in fs[0]["summary"]


def test_write_then_verified_is_clean():
    """写后补了验证 -> 不误报（合法路径必须放行）。"""
    r = A([
        think(1, "我接下来要修改 login 函数"),
        StageBlock(seq=2, kind="toolCall", text='{"name":"edit"}', tool_name="edit"),
        _edit_ok(3),
        StageBlock(seq=4, kind="toolCall", text='{"name":"bash"}', tool_name="bash"),
        tool_ok(5, name="bash"),
        claim(6),
    ])
    assert not [f for f in r["findings"] if f["status"] == "actionable"], \
        f"写后已验证不该报 actionable: {r['findings']}"


def test_read_only_then_claim_is_not_actionable():
    """只读未改就宣布完成 -> 不报 unverified-write（那是另一种缺口）。"""
    r = A([
        StageBlock(seq=1, kind="toolCall", text='{"name":"read"}', tool_name="read"),
        StageBlock(seq=2, kind="toolResult", text="def login()", tool_name="read"),
        claim(3),
    ])
    assert not [f for f in r["findings"] if f["kind"] == "unverified-claim"
                and f["status"] == "actionable"], f"只读不该报: {r['findings']}"


def test_verification_before_write_does_not_count():
    """改动前的验证不能支持改动后的完成声明（与 2.2 时序规则一致）。"""
    r = A([
        StageBlock(seq=1, kind="toolCall", text='{"name":"bash"}', tool_name="bash"),
        tool_ok(2, name="bash"),
        think(3, "我接下来要修改 login 函数"),
        StageBlock(seq=4, kind="toolCall", text='{"name":"edit"}', tool_name="edit"),
        _edit_ok(5),
        claim(6),
    ])
    assert [f for f in r["findings"] if f["status"] == "actionable"], \
        f"改动前验证不得放行: {r['findings']}"


# ---- 终止确认：stream_ended 门控（2026-09-22）----
# 实测 S2 场景：write 刚落盘就报"未经验证"，而模型下一个动作正是 cat 验证，
# 连锁触发 3 次误报注入。修法是让"运行已结束"成为显式信号，而非从
# 增量批次里猜终止。

def _write_only():
    return [StageBlock(seq=1, kind="toolCall", text='{"name":"edit"}', tool_name="edit"),
            StageBlock(seq=2, kind="toolResult", text="Successfully replaced 1 block",
                       tool_name="edit", is_error=False)]


def test_write_without_verification_reports_only_when_stream_ended():
    r = StageAuditor().audit("x", _write_only(), [], 1, stream_ended=True)
    act = [f for f in r["findings"] if f["status"] == "actionable"]
    assert act, "运行已结束且改动未验证 -> 应报"


def test_write_without_verification_silent_while_stream_alive():
    """验证尚未到达时不得误报（这是 S2 误报注入的根因）。"""
    r = StageAuditor().audit("x", _write_only(), [], 1, stream_ended=False)
    act = [f for f in r["findings"] if f["status"] == "actionable"]
    assert not act, "流未静止时不得判'跑了没验证'"


def test_write_then_verified_silent_even_when_ended():
    blocks = _write_only() + [
        StageBlock(seq=3, kind="toolCall", text='{"name":"bash"}', tool_name="bash"),
        StageBlock(seq=4, kind="toolResult", text="1 passed", tool_name="bash", is_error=False)]
    r = StageAuditor().audit("x", blocks, [], 1, stream_ended=True)
    act = [f for f in r["findings"] if f["status"] == "actionable"]
    assert not act, "写后有验证 -> 闭环，不应报"


def test_stream_ended_defaults_false_in_http_layer():
    """HTTP 层缺省 stream_ended=False：老客户端不传时行为保守（不误报）。"""
    r = audit_stage_payload({"blocks": [
        {"seq": 1, "kind": "toolCall", "text": "{}", "tool_name": "edit"},
        {"seq": 2, "kind": "toolResult", "text": "ok", "tool_name": "edit"}],
        "req": "x", "snapshot_version": 1})
    assert not [f for f in r["findings"] if f["status"] == "actionable"]


# ---- P7：语义声明检测 shadow 通道（默认 None 零行为变化） ----

from moonbow.guard.audit_semantic import (  # noqa: E402
    ProcessSemanticAdapter, ShadowObservation)


class FakeShadow:
    """shadow provider 替身：按文本关键词返回观察。"""

    kind = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def observe(self, text):
        self.seen.append(text)
        if self.fail:
            raise RuntimeError("boom")
        if "还不确定" in text:
            return ShadowObservation(matched=True, score=0.91)
        return ShadowObservation(matched=False)


def _base_blocks():
    return [
        tool_ok(1),
        StageBlock(seq=2, kind="text", text="我接下来修改登录逻辑。"),
        claim(3),
    ]


def test_shadow_channel_disabled_by_default():
    r = StageAuditor().audit("修复登录页崩溃", _base_blocks(), None, 1)
    assert "shadow_semantic" not in r  # 零行为变化：键都不出现


def test_shadow_findings_recorded_without_changing_verdicts():
    shadow = FakeShadow()
    r = StageAuditor(semantic_shadow=shadow).audit(
        "修复登录页崩溃", _base_blocks(), None, 1)
    base = StageAuditor().audit("修复登录页崩溃", _base_blocks(), None, 1)
    # 主判定逐字段不变
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["semantic"] == base["semantic"]
    assert r["based_on"] == base["based_on"]
    # shadow 结果单独成键
    assert "shadow_semantic" in r
    # 只观察中间汇报文本（text 块，含完成声明块）；thinking/tool 不观察
    assert shadow.seen == ["我接下来修改登录逻辑。", "已经修复了，测试也通过了"]
    entries = r["shadow_semantic"]
    assert len(entries) == 2
    assert [e["seq"] for e in entries] == [2, 3]
    assert all(e["matched"] is False for e in entries)
    assert all(e["pattern"] == "process.unresolved@1" for e in entries)


def test_shadow_matched_does_not_create_findings_or_reminder():
    blocks = [StageBlock(seq=1, kind="text", text="超时原因还不确定，待确认。")]
    r = StageAuditor(semantic_shadow=FakeShadow()).audit("x", blocks, None, 1)
    assert r["shadow_semantic"][0]["matched"] is True
    assert r["findings"] == []
    assert r["reminder"] is None
    assert r["semantic"] is False


def test_shadow_provider_failure_is_isolated():
    shadow = FakeShadow(fail=True)
    r = StageAuditor(semantic_shadow=shadow).audit(
        "修复登录页崩溃", _base_blocks(), None, 1)
    base = StageAuditor().audit("修复登录页崩溃", _base_blocks(), None, 1)
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["shadow_semantic"][0]["unavailable_reason"] == "error:RuntimeError"


def test_shadow_thinking_and_tool_blocks_not_observed():
    shadow = FakeShadow()
    blocks = [think(1, "不确定这个假设是否成立。"), tool_ok(2)]
    r = StageAuditor(semantic_shadow=shadow).audit("x", blocks, None, 1)
    assert shadow.seen == []
    assert r["shadow_semantic"] == []


def test_process_semantic_adapter_wraps_matcher():
    from moonbow.semantic.schema import MatchResponse, Provenance
    prov = Provenance(backend="fake", model_revision="r",
                      pattern_version="process.unresolved@1", calibrated=False)

    class M:
        def __init__(self, resp=None, err=None):
            self.resp, self.err = resp, err

        def match(self, text, pattern=None, **kw):
            assert pattern == "process.unresolved@1"
            if self.err:
                raise self.err
            return self.resp

    a = ProcessSemanticAdapter(M(MatchResponse(
        status="ok", matched=True, score=0.87, provenance=prov)))
    o = a.observe("还不确定")
    assert o.matched is True and o.score == 0.87
    assert o.unavailable_reason is None

    o = ProcessSemanticAdapter(M(MatchResponse(
        status="abstain", reason_code="insufficient_context"))).observe("x")
    assert o.matched is None and o.unavailable_reason == "abstain:insufficient_context"

    o = ProcessSemanticAdapter(M(err=ValueError("bad"))).observe("x")
    assert o.unavailable_reason == "error:ValueError"

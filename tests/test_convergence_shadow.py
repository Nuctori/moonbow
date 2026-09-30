# -*- coding: utf-8 -*-
"""收敛进度追踪 Phase 1：ConvergenceShadow 单元 + stage audit shadow 通道 +
/v1/stage-check 透传集成测试。

硬约束（全部有用例锁定）：
- shadow 通道零投递：结果只进 shadow_convergence 键，绝不进
  reminder/findings/semantic；
- 默认关闭零变化：不注入/不传 flag 时响应与现状逐字节一致；
- 幂等：同一块流重复计算/重复审计同结果（与 audit() 幂等一致）。
"""
import json
import os
import sys
import threading
import urllib.request
from http.server import HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moonbow.guard.convergence import (  # noqa: E402
    ConvergenceShadow, ShadowEntry, unavailable_entry,
    _pytest_events, coverage_trajectory, is_full_pass, is_fail_result,
    summary_counts,
)
from moonbow.guard.process_audit import StageAuditor, audit_stage_payload  # noqa: E402
from moonbow.guard.protocol import StageBlock  # noqa: E402


def _tr(seq, text, cid=None, name="bash", err=False):
    return StageBlock(seq=seq, kind="toolResult", text=text, tool_call_id=cid,
                      tool_name=name, is_error=err)


def _call(seq, command, cid):
    return StageBlock(seq=seq, kind="toolCall", tool_call_id=cid, tool_name="bash",
                      text=json.dumps({"name": "bash", "arguments": {"command": command}}))


def _shadow(auditor=None, **kw):
    return StageAuditor(convergence_shadow=auditor or ConvergenceShadow(**kw))


# ---- 幂等性：同流两次审计同结果 ----

def _mixed_stream():
    return [
        _call(1, "pytest -q tests/", "c1"), _tr(2, "1 failed in 0.01s", "c1"),
        StageBlock(seq=3, kind="text", text="准备修改 login 逻辑。"),
        _tr(4, "read 120 lines", name="read"),
        _call(5, "pytest -q tests/", "c2"), _tr(6, "1 failed in 0.02s", "c2"),
        _call(7, "pytest -q tests/", "c3"), _tr(8, "1 passed in 0.03s", "c3"),
        StageBlock(seq=9, kind="text", text="已经修复完成。"),
    ]


def test_compute_is_pure_function_same_stream_same_result():
    shadow = ConvergenceShadow()
    b1 = shadow.compute("x", _mixed_stream())
    b2 = shadow.compute("x", _mixed_stream())
    assert b1 == b2
    assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)


def test_audit_idempotent_same_stream_same_full_result():
    r1 = _shadow().audit("修复登录页崩溃", _mixed_stream(), None, 1)
    r2 = _shadow().audit("修复登录页崩溃", _mixed_stream(), None, 1)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_audit_idempotent_regardless_of_block_order():
    """块乱序到达（seq 保序）时结果不变：compute 内部按 seq 排序。"""
    stream = _mixed_stream()
    r1 = _shadow().audit("x", stream, None, 1)
    r2 = _shadow().audit("x", list(reversed(stream)), None, 1)
    assert r1["shadow_convergence"] == r2["shadow_convergence"]


# ---- 默认关闭：零行为变化 ----

def test_disabled_by_default_no_key_byte_identical():
    r = StageAuditor().audit("修复登录页崩溃", _mixed_stream(), None, 1)
    assert "shadow_convergence" not in r
    # 与现状逐字节一致：响应恰为既有四键
    assert sorted(r.keys()) == ["based_on", "findings", "reminder", "semantic"]
    assert json.dumps(r, sort_keys=True) == json.dumps({
        "findings": r["findings"], "reminder": r["reminder"],
        "semantic": r["semantic"], "based_on": r["based_on"]},
        sort_keys=True)


def test_injection_does_not_change_main_verdicts():
    base = StageAuditor().audit("修复登录页崩溃", _mixed_stream(), None, 1)
    r = _shadow().audit("修复登录页崩溃", _mixed_stream(), None, 1)
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["semantic"] == base["semantic"]
    assert r["based_on"] == base["based_on"]
    assert len(r["shadow_convergence"]) == 2


def test_audit_stage_payload_default_unchanged():
    payload = {"req": "x", "blocks": [b.to_dict() for b in _mixed_stream()]}
    assert "shadow_convergence" not in audit_stage_payload(payload)
    assert "shadow_convergence" in audit_stage_payload(
        payload, convergence_shadow=ConvergenceShadow())


# ---- 轮定义：tool_result 事件计数近似 ----

def test_round_definition_counts_tool_results_only():
    blocks = [
        StageBlock(seq=1, kind="thinking", text="想想从哪入手。"),
        StageBlock(seq=2, kind="text", text="先看代码。"),
        _tr(3, "def login(): ...", name="read"),
        _tr(4, "ok", name="edit"),
        _tr(5, "readme 内容", name="read"),
        _tr(6, "grep 无结果", name="grep"),
        _tr(7, "over", name="write"),
    ]
    entries = ConvergenceShadow().compute("x", blocks)
    stall = entries[0]
    assert stall["signal"] == "converge.stall"
    assert stall["detail"]["round"] == 5          # 7 块中 5 个 toolResult
    assert stall["abstain_reason"] == "no_pytest_evidence"
    assert stall["matched"] is None


def test_round_count_at_least_four_gates_stall():
    blocks = []
    for i, (seq, txt) in enumerate([(1, "1 failed"), (3, "1 failed"), (5, "1 failed")], start=1):
        blocks.append(_call(seq, "pytest -q", "c%d" % i))
        blocks.append(_tr(seq + 1, txt, "c%d" % i))
    entries = ConvergenceShadow().compute("x", blocks)
    assert entries[0]["detail"]["round"] == 3
    assert entries[0]["matched"] is None
    assert entries[0]["abstain_reason"] == "rounds<4"
    # fail_streak 无轮次门槛：连续失败即时可见（shadow 只记录）
    assert entries[1]["detail"]["streak"] == 3
    assert entries[1]["matched"] is True


# ---- pytest 证据识别：调用配对优先，文本特征回退 ----

def test_pytest_detected_via_call_pairing_even_if_text_plain():
    blocks = [
        _call(1, "pytest -q tests/", "c1"), _tr(2, "all good", "c1"),
    ]
    entries = ConvergenceShadow().compute("x", blocks)
    assert entries[0]["detail"]["pytest_rounds"] == 1


def test_non_pytest_call_not_evidence_even_if_text_looks_like_summary():
    """配对存在时由调用侧决定：`ls` 的输出再像 pytest 摘要也不算。"""
    blocks = [
        _call(1, "ls -la", "c1"), _tr(2, "3 passed in 0.01s", "c1"),
        _tr(3, "def login()", name="read"),
        _tr(4, "ok", name="edit"),
        _tr(5, "more", name="read"),
    ]
    entries = ConvergenceShadow().compute("x", blocks)
    assert entries[0]["detail"]["pytest_rounds"] == 0
    assert entries[0]["abstain_reason"] == "no_pytest_evidence"
    assert entries[1]["detail"]["streak"] == 0


def test_pytest_detected_by_text_fallback_without_call_block():
    """toolCall 块被环形缓冲截掉（窗口 400）时按结果文本特征识别。"""
    blocks = [_tr(1, "================= 2 failed in 0.10s =================")]
    entries = ConvergenceShadow().compute("x", blocks)
    assert entries[0]["detail"]["pytest_rounds"] == 1


# ---- pytest 摘要解析（移植自 goal_coverage_study.py 的口径） ----

def test_parse_summary_counts_and_helpers():
    assert summary_counts("1 failed, 2 passed in 0.02s") == (2, 1)
    assert summary_counts("no tests ran in 0.01s") == (None, None)
    assert is_full_pass("3 passed in 0.01s") is True
    assert is_full_pass("1 failed, 2 passed") is False
    assert is_full_pass("1 passed, 1 error") is False
    assert is_fail_result("INTERNALERROR x") is True
    assert is_fail_result("no tests ran in 0.01s") is False


def test_pass_count_updates_coverage_then_fail_regresses():
    """通过数变化：全绿 → coverage 1.0；随后失败摘要按 grounded=total-M 回撤。"""
    shadow = ConvergenceShadow(goals_total=3)
    blocks = [
        _call(1, "pytest -q", "c1"), _tr(2, "3 passed in 0.01s", "c1"),
    ]
    e = shadow.compute("x", blocks)
    assert e[0]["detail"]["goals_passed"] == 3
    assert e[0]["detail"]["coverage"] == 1.0
    blocks.append(_call(3, "pytest -q", "c2"))
    blocks.append(_tr(4, "1 failed, 2 passed in 0.02s", "c2"))   # 回归 1 个
    e = shadow.compute("x", blocks)
    assert e[0]["detail"]["goals_passed"] == 2                    # 3 - 1 failed
    assert e[0]["detail"]["coverage"] == round(2 / 3, 4)


def test_named_targets_fail_list_removal_and_partial_score():
    """失败名单：点名 test_half 失败 + 摘要 1 failed/2 passed 一致 →
    其余目标按部分得分视为通过（goal_coverage_study named 分支口径）。"""
    targets = ["test_half", "test_sort_words", "test_add_tag"]
    shadow = ConvergenceShadow(goals_total=3, target_tests=targets)
    txt = ("FAILED tests/test_words.py::test_half - AssertionError\n"
           "1 failed, 2 passed in 0.02s")
    blocks = [_call(1, "pytest -q", "c1"), _tr(2, txt, "c1")]
    e = shadow.compute("x", blocks)
    assert e[0]["detail"]["goals_passed"] == 2
    assert e[0]["detail"]["coverage"] == round(2 / 3, 4)
    # 同一流重复计算不变（幂等）
    assert shadow.compute("x", blocks)[0] == e[0]


def test_unparseable_output_leaves_passed_untouched():
    """收集期错误（无摘要行）→ "无证据"而非"失败证据"：passed 不动。"""
    shadow = ConvergenceShadow(goals_total=2)
    blocks = [
        _call(1, "pytest -q", "c1"), _tr(2, "2 passed in 0.01s", "c1"),
        _call(3, "pytest -q", "c2"),
        _tr(4, "ERROR collecting tests/test_x.py\nImportError while loading", "c2"),
    ]
    e = shadow.compute("x", blocks)
    assert e[0]["detail"]["goals_passed"] == 2
    traj = coverage_trajectory(_pytest_events(blocks), 2, None)
    assert traj[-1]["event"] == "fail_unattributable"


# ---- G2 / converge.stall ----

def _failing_rounds(n, goals_total=1, start_seq=1, text="1 failed"):
    """n 轮失败 pytest（bash 退出码 1 → is_error=True；convergence 只看文本，
    主审计的矛盾规则才看 is_error）。"""
    blocks = []
    for i in range(n):
        cid = "c%d" % (start_seq + i)
        blocks.append(_call(start_seq + 2 * i, "pytest -q", cid))
        blocks.append(_tr(start_seq + 2 * i + 1, text, cid, err=True))
    return blocks


def test_g2_fires_at_round4_zero_coverage():
    shadow = ConvergenceShadow()   # goals_total=1：套件过 = 目标落地
    e = shadow.compute("x", _failing_rounds(4))
    stall = e[0]
    assert stall["signal"] == "converge.stall"
    assert stall["matched"] is True
    assert stall["detail"] == {"round": 4, "goals_passed": 0, "goals_total": 1,
                               "coverage": 0.0, "pytest_rounds": 4}
    assert stall["abstain_reason"] is None


def test_g2_not_fires_when_round4_has_coverage():
    blocks = _failing_rounds(3) + [
        _call(7, "pytest -q", "c7"), _tr(8, "1 passed in 0.01s", "c7"),
    ]
    e = ConvergenceShadow().compute("x", blocks)
    assert e[0]["matched"] is False
    assert e[0]["detail"]["coverage"] == 1.0
    assert e[0]["abstain_reason"] is None


def test_g2_not_fires_for_partial_coverage_goals_total_3():
    """goals_total=3 部分覆盖：grounded=1/3 → coverage>0 → 不触发 stall。"""
    shadow = ConvergenceShadow(goals_total=3)
    e = shadow.compute("x", _failing_rounds(4, text="2 failed in 0.02s"))
    stall = e[0]
    assert stall["matched"] is False
    assert stall["detail"]["goals_passed"] == 1
    assert stall["detail"]["coverage"] == round(1 / 3, 4)   # 1 grounded = 3 - 2 failed
    # 同一流上 fail_streak 照常触发（部分得分输出在严格二分口径下非全绿）
    assert e[1]["matched"] is True
    assert e[1]["detail"]["streak"] == 4


def test_g2_abstains_for_tasks_without_pytest():
    blocks = [
        _tr(1, "def login(): ...", name="read"),
        _tr(2, "patched", name="edit"),
        _tr(3, "ok", name="bash"),
        _tr(4, "done", name="write"),
    ]
    e = ConvergenceShadow().compute("x", blocks)
    assert e[0]["matched"] is None
    assert e[0]["abstain_reason"] == "no_pytest_evidence"
    # fail_streak 可判：无 pytest 失败 → streak 0 / matched False
    assert e[1]["matched"] is False
    assert e[1]["detail"]["streak"] == 0


# ---- converge.fail_streak ----

def test_fail_streak_fires_at_two_consecutive_failures():
    e = ConvergenceShadow().compute("x", _failing_rounds(2))
    assert e[1]["signal"] == "converge.fail_streak"
    assert e[1]["matched"] is True
    assert e[1]["detail"]["streak"] == 2


def test_fail_streak_resets_after_full_pass_and_skips_non_pytest():
    """全绿打断连败；中间夹非 pytest 结果不打断（原口径跳过无验证轮）。"""
    blocks = _failing_rounds(2) + [
        _call(5, "pytest -q", "c5"), _tr(6, "1 passed in 0.01s", "c5"),
        _tr(7, "cat 配置文件", name="read"),
        _call(8, "pytest -q", "c8"), _tr(9, "1 failed", "c8"),
    ]
    e = ConvergenceShadow().compute("x", blocks)
    assert e[1]["detail"]["streak"] == 1
    assert e[1]["matched"] is False


def test_fail_streak_strict_dichotomy_counts_partial_and_unparsed():
    """严格二分口径（convergence_study 移植）：部分得分输出与无摘要
    输出都非全绿 → 均计失败。"""
    blocks = [
        _call(1, "pytest -q", "c1"), _tr(2, "1 failed, 2 passed in 0.02s", "c1"),
        _call(3, "pytest -q", "c2"), _tr(4, "no tests ran in 0.01s", "c2"),
    ]
    e = ConvergenceShadow(goals_total=3).compute("x", blocks)
    assert e[1]["detail"]["streak"] == 2
    assert e[1]["matched"] is True


# ---- shadow 零投递：绝不进 reminder / findings / semantic ----

def test_firing_shadow_never_creates_findings_or_reminder():
    """G2 已触发的流：shadow 标记 matched=True，但主审计零变化。"""
    blocks = _failing_rounds(5)
    base = StageAuditor().audit("修复登录崩溃", blocks, None, 1)
    r = _shadow().audit("修复登录崩溃", blocks, None, 1)
    stall = [e for e in r["shadow_convergence"] if e["signal"] == "converge.stall"][0]
    assert stall["matched"] is True
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["semantic"] == base["semantic"]
    assert r["reminder"] is None


def test_firing_shadow_with_claim_stream_does_not_alter_verdicts():
    """带完成声明+声明后失败结果（主审计有发现/提醒）的流：shadow 零改变。"""
    blocks = (_failing_rounds(3)
              + [StageBlock(seq=7, kind="text", text="已经修复完成，测试通过")]
              + _failing_rounds(1, start_seq=8))
    base = StageAuditor().audit("x", blocks, None, 1)
    r = _shadow().audit("x", blocks, None, 1)
    assert base["findings"] and base["reminder"]   # 主审计确有发现（矛盾）
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["semantic"] == base["semantic"]
    assert "shadow_convergence" in r and r["shadow_convergence"]


def test_shadow_entry_schema():
    d = ShadowEntry("converge.stall", None, {"round": 1}, "rounds<4").to_dict()
    assert d == {"signal": "converge.stall", "matched": None,
                 "detail": {"round": 1}, "abstain_reason": "rounds<4"}
    assert unavailable_entry("error:X")["abstain_reason"] == "error:X"


# ---- 失败隔离：shadow 自身异常不影响主审计 ----

class _BrokenShadow:
    kind = "broken"

    def compute(self, req, blocks):
        raise RuntimeError("boom")


def test_shadow_failure_is_isolated():
    blocks = _mixed_stream()
    base = StageAuditor().audit("x", blocks, None, 1)
    r = StageAuditor(convergence_shadow=_BrokenShadow()).audit("x", blocks, None, 1)
    assert r["findings"] == base["findings"]
    assert r["reminder"] == base["reminder"]
    assert r["shadow_convergence"] == [unavailable_entry("error:RuntimeError")]


# ---- goals_total 注入 ----

def test_goals_total_validation():
    with pytest.raises(ValueError):
        ConvergenceShadow(goals_total=0)
    with pytest.raises(ValueError):
        ConvergenceShadow(target_tests=["a", "a"])
    assert ConvergenceShadow(goals_total=3, target_tests=["a", "b", "c"]).goals_total == 3
    assert ConvergenceShadow(goals_total=1).goals_total == 1


# ---- /v1/stage-check 透传（HTTP 集成） ----

from moonbow.guard.server import GuardHTTPRequestHandler  # noqa: E402
from moonbow.guard.verifier import ProgressGuard  # noqa: E402


@pytest.fixture(scope="module")
def base_url():
    GuardHTTPRequestHandler.guard = ProgressGuard(lazy_load=True)
    GuardHTTPRequestHandler.convergence_shadow = None   # 隔离：测试前重置
    server = HTTPServer(("127.0.0.1", 0), GuardHTTPRequestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    GuardHTTPRequestHandler.convergence_shadow = None


def _post(base, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as w:
        return json.loads(w.read().decode("utf-8"))


def _stage_payload(flag=None):
    p = {
        "req": "修复崩溃", "snapshot_version": 1,
        "blocks": [
            {"seq": s, "kind": "toolResult", "text": "1 failed in 0.01s",
             "tool_call_id": "c%d" % s, "tool_name": "bash", "is_error": True}
            for s in (1, 2, 3, 4)
        ],
    }
    if flag is not None:
        p["enable_convergence_shadow"] = flag
    return p


def test_stage_check_without_flag_has_no_shadow_key(base_url):
    """默认关闭零变化：响应与直接调用 audit_stage_payload 逐字节一致。"""
    payload = _stage_payload()
    resp = _post(base_url, "/v1/stage-check", payload)
    assert "shadow_convergence" not in resp
    direct = audit_stage_payload(payload)
    assert json.dumps(resp, sort_keys=True) == json.dumps(direct, sort_keys=True)


def test_stage_check_flag_false_explicit_no_shadow_key(base_url):
    resp = _post(base_url, "/v1/stage-check", _stage_payload(flag=False))
    assert "shadow_convergence" not in resp


def test_stage_check_flag_true_returns_shadow_entries(base_url):
    resp = _post(base_url, "/v1/stage-check", _stage_payload(flag=True))
    entries = resp["shadow_convergence"]
    assert [e["signal"] for e in entries] == ["converge.stall", "converge.fail_streak"]
    stall = entries[0]
    assert stall["matched"] is True and stall["detail"]["coverage"] == 0.0
    assert entries[1]["matched"] is True and entries[1]["detail"]["streak"] == 4
    # 主判定四键不受影响
    assert sorted(k for k in resp if k != "shadow_convergence") == \
        ["based_on", "findings", "reminder", "semantic"]
    # 零投递：触发也不产生新 findings/reminder
    assert resp["findings"] == [] and resp["reminder"] is None


def test_stage_check_flag_must_be_bool(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url, "/v1/stage-check", _stage_payload(flag="yes"))
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url, "/v1/stage-check", _stage_payload(flag=1))
    assert e.value.code == 400

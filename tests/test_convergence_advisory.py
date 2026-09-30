# -*- coding: utf-8 -*-
"""收敛进度追踪 Phase 2：advisory 投递包装测试。

硬约束（全部有用例锁定）：
- 触发规则本体零改动：命中判定完全来自 Phase 1 shadow 条目；
- off / shadow 与 Phase 1 逐字节一致（无 delivered 键、无收敛 reminder）；
- advisory：措辞固定、可忽略标注、独立预算（每任务最多 2 次、同信号一次）、
  主审计 reminder 优先、task_id 独立于 req 哈希回退；
- 投递状态写进 shadow_convergence 条目（delivered: true）。
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
    ADVISORY_MARK, ADVISORY_TEXT, CONVERGENCE_ENV, CONVERGENCE_MODES,
    AdvisoryBudget, ConvergenceShadow, advisory_fingerprint,
    build_advisory_reminder, convergence_mode_from_env, req_hash_of,
)
from moonbow.guard.process_audit import (  # noqa: E402
    StageAuditor, audit_stage_payload,
)
from moonbow.guard.protocol import StageBlock  # noqa: E402


def _failing_rounds(n, start_seq=1, text="1 failed"):
    """n 轮失败 pytest（每轮 2 块：toolCall + toolResult）。"""
    blocks = []
    for i in range(n):
        cid = "c%d" % (start_seq + i)
        blocks.append(StageBlock(seq=start_seq + 2 * i, kind="toolCall",
                                 tool_call_id=cid, tool_name="bash",
                                 text=json.dumps({"command": "pytest -q"})))
        blocks.append(StageBlock(seq=start_seq + 2 * i + 1, kind="toolResult",
                                 text=text, tool_call_id=cid,
                                 tool_name="bash", is_error=True))
    return blocks


def _payload(rounds=4, task_id=None, targets=None, goals=None):
    p = {
        "req": "修复三个 bug", "snapshot_version": 1,
        "blocks": [b.to_dict() for b in _failing_rounds(rounds)],
    }
    if task_id is not None:
        p["task_id"] = task_id
    if targets is not None:
        p["convergence_target_tests"] = targets
    if goals is not None:
        p["convergence_goals_total"] = goals
    return p


# ---- 措辞与指纹 ----

def test_advisory_wording_is_fixed():
    assert ADVISORY_TEXT["converge.stall"] == (
        "你已连续多轮修改但没有一次验证成功。建议收窄范围："
        "先把其中一个问题修到测试通过，其余如实列入 REMAINING。")
    assert ADVISORY_TEXT["converge.fail_streak"] == (
        "同一验证反复失败，建议先聚焦让单个测试通过，再扩展。")
    assert ADVISORY_MARK == "【进度守卫（收敛提示）】"
    assert set(("off", "shadow", "advisory")) == set(CONVERGENCE_MODES)


def test_build_advisory_reminder_schema_and_ignore_note():
    entry = {"signal": "converge.stall",
             "detail": {"round": 4, "goals_passed": 0, "goals_total": 3,
                        "coverage": 0.0, "pytest_rounds": 4}}
    r = build_advisory_reminder("converge.stall", "k1", entry)
    assert sorted(r.keys()) == ["evidence", "fingerprint", "suggestion", "summary"]
    assert r["fingerprint"] == "convergence:converge.stall:k1"
    assert r["suggestion"].startswith(ADVISORY_MARK)
    assert ADVISORY_TEXT["converge.stall"] in r["suggestion"]
    assert "可忽略" in r["suggestion"]          # 措辞纪律：必须可忽略
    assert "已落地目标 0/3" in "".join(r["evidence"])
    # 守卫标记前缀：客户端采集层据此跳过（防自我审计循环）
    assert "【进度守卫" in r["suggestion"]


def test_req_hash_stable_and_input_free():
    h1 = req_hash_of("同一任务文本")
    h2 = req_hash_of("同一任务文本")
    assert h1 == h2 and len(h1) == 16
    assert req_hash_of("") == req_hash_of(None)  # type: ignore[arg-type]


# ---- env 门控解析 ----

def test_env_mode_parsing_default_off_and_invalid_off():
    assert convergence_mode_from_env({}) == "off"
    assert convergence_mode_from_env({CONVERGENCE_ENV: "advisory"}) == "advisory"
    assert convergence_mode_from_env({CONVERGENCE_ENV: "shadow"}) == "shadow"
    assert convergence_mode_from_env({CONVERGENCE_ENV: "off"}) == "off"
    assert convergence_mode_from_env({CONVERGENCE_ENV: "ON"}) == "off"
    assert convergence_mode_from_env({CONVERGENCE_ENV: "always"}) == "off"
    assert convergence_mode_from_env({CONVERGENCE_ENV: " advisory "}) == "advisory"


# ---- AdvisoryBudget：独立预算 ----

def test_budget_same_signal_once_per_task_and_max_two():
    b = AdvisoryBudget()
    assert b.allow("t1", "converge.fail_streak")
    b.record("t1", "converge.fail_streak")
    assert not b.allow("t1", "converge.fail_streak")   # 同信号不重投
    assert b.allow("t1", "converge.stall")
    b.record("t1", "converge.stall")
    assert not b.allow("t1", "converge.stall")
    assert b.used("t1") == 2                            # 每任务最多 2 次
    assert not b.allow("t1", "converge.stall")


def test_budget_independent_across_task_keys():
    b = AdvisoryBudget()
    b.record("t1", "converge.stall"); b.record("t1", "converge.fail_streak")
    assert b.allow("t2", "converge.stall")              # 其他任务不受影响
    assert b.used("t2") == 0
    h = req_hash_of("req 文本")
    b.record(h, "converge.stall")
    assert not b.allow(h, "converge.stall")


# ---- off / shadow 等价：Phase 1 逐字节一致 ----

def test_no_advisory_no_delivered_key_and_byte_identical():
    p = _payload(rounds=4)
    r = audit_stage_payload(p, convergence_shadow=ConvergenceShadow())
    entries = r["shadow_convergence"]
    assert [e["signal"] for e in entries] == ["converge.stall", "converge.fail_streak"]
    assert all("delivered" not in e for e in entries)
    assert r["reminder"] is None                        # 零投递
    # 与不传 shadow 的 Phase 1 主四键一致
    assert sorted(k for k in r if k != "shadow_convergence") == \
        ["based_on", "findings", "reminder", "semantic"]


def test_advisory_none_flag_false_paths_untouched():
    p = _payload(rounds=4, task_id="t1")
    assert "shadow_convergence" not in audit_stage_payload(p)  # flag 未开
    # advisory 注入但 flag 未开：无 shadow 键 → 包装器空转
    r = audit_stage_payload(p, convergence_advisory=AdvisoryBudget())
    assert "shadow_convergence" not in r and r["reminder"] is None


def test_shadow_wrapper_not_mutating_stage_auditor_path():
    """StageAuditor.audit 直调（不经 payload 层）绝不产生 delivered/收敛 reminder。"""
    r = StageAuditor(convergence_shadow=ConvergenceShadow()).audit(
        "修复三个 bug", _failing_rounds(4), None, 1)
    assert all("delivered" not in e for e in r["shadow_convergence"])
    assert r["reminder"] is None


# ---- advisory 投递 ----

def test_advisory_delivers_stall_at_round4():
    budget = AdvisoryBudget()
    r = audit_stage_payload(_payload(rounds=4, task_id="t1"),
                            convergence_shadow=ConvergenceShadow(),
                            convergence_advisory=budget)
    rem = r["reminder"]
    assert rem and rem["suggestion"].startswith(ADVISORY_MARK)
    assert ADVISORY_TEXT["converge.stall"] in rem["suggestion"]
    assert rem["fingerprint"] == advisory_fingerprint("converge.stall", "t1")
    stall = [e for e in r["shadow_convergence"] if e["signal"] == "converge.stall"][0]
    fs = [e for e in r["shadow_convergence"] if e["signal"] == "converge.fail_streak"][0]
    assert stall["delivered"] is True and fs["delivered"] is False
    # 主审计四键不变（findings/semantic 未被收敛提示污染）
    assert r["findings"] == [] and r["semantic"] is False


def test_advisory_delivers_fail_streak_before_round4():
    budget = AdvisoryBudget()
    r = audit_stage_payload(_payload(rounds=2, task_id="t2"),
                            convergence_shadow=ConvergenceShadow(),
                            convergence_advisory=budget)
    rem = r["reminder"]
    assert rem and ADVISORY_TEXT["converge.fail_streak"] in rem["suggestion"]
    assert rem["fingerprint"] == advisory_fingerprint("converge.fail_streak", "t2")
    fs = [e for e in r["shadow_convergence"] if e["signal"] == "converge.fail_streak"][0]
    assert fs["delivered"] is True


def test_advisory_stall_takes_priority_when_both_fire():
    r = audit_stage_payload(_payload(rounds=4, task_id="t3"),
                            convergence_shadow=ConvergenceShadow(),
                            convergence_advisory=AdvisoryBudget())
    assert ADVISORY_TEXT["converge.stall"] in r["reminder"]["suggestion"]
    stall = [e for e in r["shadow_convergence"] if e["signal"] == "converge.stall"][0]
    assert stall["delivered"] is True


def test_advisory_silent_when_no_trigger():
    budget = AdvisoryBudget()
    r = audit_stage_payload(_payload(rounds=1, task_id="t4"),
                            convergence_shadow=ConvergenceShadow(),
                            convergence_advisory=budget)
    assert r["reminder"] is None
    assert all(e["delivered"] is False for e in r["shadow_convergence"])
    assert budget.used("t4") == 0


def test_advisory_yields_to_main_audit_reminder():
    """主审计已有 reminder（声明后失败矛盾）→ 收敛提示不覆盖不投递。"""
    blocks = (_failing_rounds(2)
              + [StageBlock(seq=99, kind="text", text="已经修复完成，测试通过")]
              + _failing_rounds(1, start_seq=200))
    p = {"req": "x", "snapshot_version": 1, "task_id": "t5",
         "blocks": [b.to_dict() for b in blocks]}
    budget = AdvisoryBudget()
    r = audit_stage_payload(p, convergence_shadow=ConvergenceShadow(),
                            convergence_advisory=budget)
    # 主审计确有 actionable（contradiction）提醒，收敛触发同时命中
    assert r["reminder"] and r["reminder"]["fingerprint"].startswith("contradiction:")
    fs = [e for e in r["shadow_convergence"] if e["signal"] == "converge.fail_streak"][0]
    assert fs["matched"] is True and fs["delivered"] is False
    assert all(e["delivered"] is False for e in r["shadow_convergence"])
    assert budget.used("t5") == 0


def test_advisory_budget_exhausts_after_two_signals():
    budget = AdvisoryBudget()
    p2 = _payload(rounds=2, task_id="t6")
    r1 = audit_stage_payload(p2, convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    assert r1["reminder"] and "converge.fail_streak" in r1["reminder"]["fingerprint"]
    # 同一流重审：fail_streak 已投过 → 不重投（预算按信号去重）
    r2 = audit_stage_payload(p2, convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    assert r2["reminder"] is None
    # 流增长后 stall 触发 → 第二个信号可投（每任务第 2 次）
    p4 = _payload(rounds=4, task_id="t6")
    r3 = audit_stage_payload(p4, convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    assert r3["reminder"] and "converge.stall" in r3["reminder"]["fingerprint"]
    # 预算用满（2/2）：后续触发（理论新流）不再投
    assert budget.used("t6") == AdvisoryBudget.MAX_PER_TASK
    r4 = audit_stage_payload(_payload(rounds=4, task_id="t7", goals=1),
                             convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    assert r4["reminder"] is not None                   # 别的任务不受影响


def test_budget_keyed_by_task_id_not_req_text():
    """批量实验关键性质：同 req 文本、不同 task_id 的任务预算各自独立。"""
    budget = AdvisoryBudget()
    for tid in ("run-a", "run-b", "run-c"):
        r = audit_stage_payload(_payload(rounds=4, task_id=tid),
                                convergence_shadow=ConvergenceShadow(),
                                convergence_advisory=budget)
        assert r["reminder"], tid
        assert r["reminder"]["fingerprint"].endswith(":" + tid)


def test_budget_falls_back_to_req_hash_without_task_id():
    budget = AdvisoryBudget()
    p = _payload(rounds=4)
    r1 = audit_stage_payload(p, convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    h = req_hash_of("修复三个 bug")
    assert r1["reminder"]["fingerprint"] == advisory_fingerprint("converge.stall", h)
    r2 = audit_stage_payload(p, convergence_shadow=ConvergenceShadow(),
                             convergence_advisory=budget)
    assert r2["reminder"] is None                       # 同 req 哈希共享预算


# ---- 可选目标注入（goals_total / target_tests） ----

def test_target_tests_partial_coverage_no_stall_but_fail_streak():
    """命名目标 1/3 落地：coverage>0 → stall 不触发；严格二分 fail_streak 触发。"""
    targets = ["test_half", "test_sort_words", "test_add_tag"]
    txt = ("FAILED test_app.py::test_half - AssertionError\n"
           "1 failed, 2 passed in 0.02s")
    p = {"req": "x", "snapshot_version": 1, "task_id": "t8",
         "blocks": [b.to_dict() for b in _failing_rounds(4, text=txt)]}
    r = audit_stage_payload(p, convergence_shadow=ConvergenceShadow(target_tests=targets),
                            convergence_advisory=AdvisoryBudget())
    stall = [e for e in r["shadow_convergence"] if e["signal"] == "converge.stall"][0]
    assert stall["matched"] is False
    assert stall["detail"]["goals_passed"] == 2
    assert r["reminder"]["fingerprint"].startswith("convergence:converge.fail_streak:")


def test_invalid_convergence_config_rejected():
    with pytest.raises(ValueError):
        audit_stage_payload({"req": "x", "blocks": [],
                             "convergence_goals_total": 0})
    with pytest.raises(ValueError):
        audit_stage_payload({"req": "x", "blocks": [],
                             "convergence_goals_total": True})
    with pytest.raises(ValueError):
        audit_stage_payload({"req": "x", "blocks": [],
                             "convergence_target_tests": ["a", "a"]})
    with pytest.raises(ValueError):
        audit_stage_payload({"req": "x", "blocks": [],
                             "convergence_target_tests": "test_a"})


# ---- HTTP 集成：server 门控 ----

from moonbow.guard.server import GuardHTTPRequestHandler  # noqa: E402
from moonbow.guard.verifier import ProgressGuard  # noqa: E402


@pytest.fixture()
def advisory_url():
    GuardHTTPRequestHandler.guard = ProgressGuard(lazy_load=True)
    GuardHTTPRequestHandler.convergence_shadow = None
    GuardHTTPRequestHandler.convergence_mode = "advisory"
    GuardHTTPRequestHandler.convergence_advisory = AdvisoryBudget()
    server = HTTPServer(("127.0.0.1", 0), GuardHTTPRequestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    GuardHTTPRequestHandler.convergence_mode = "off"
    GuardHTTPRequestHandler.convergence_advisory = None
    GuardHTTPRequestHandler.convergence_shadow = None


@pytest.fixture()
def shadow_url():
    GuardHTTPRequestHandler.guard = ProgressGuard(lazy_load=True)
    GuardHTTPRequestHandler.convergence_shadow = None
    GuardHTTPRequestHandler.convergence_mode = "shadow"
    GuardHTTPRequestHandler.convergence_advisory = None
    server = HTTPServer(("127.0.0.1", 0), GuardHTTPRequestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    GuardHTTPRequestHandler.convergence_mode = "off"
    GuardHTTPRequestHandler.convergence_shadow = None


def _post(base, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as w:
        return json.loads(w.read().decode("utf-8"))


def test_http_advisory_delivers_convergence_reminder(advisory_url):
    p = _payload(rounds=4, task_id="http-1")
    p["enable_convergence_shadow"] = True
    r = _post(advisory_url, "/v1/stage-check", p)
    assert r["reminder"] and r["reminder"]["suggestion"].startswith(ADVISORY_MARK)
    stall = [e for e in r["shadow_convergence"] if e["signal"] == "converge.stall"][0]
    assert stall["delivered"] is True
    # 状态端点可见预算计数
    with urllib.request.urlopen(advisory_url + "/v1/convergence-status", timeout=10) as w:
        st = json.loads(w.read().decode("utf-8"))
    assert st["mode"] == "advisory" and st["tasks_with_delivery"] == 1


def test_http_shadow_mode_never_delivers(shadow_url):
    p = _payload(rounds=4, task_id="http-2")
    p["enable_convergence_shadow"] = True
    r = _post(shadow_url, "/v1/stage-check", p)
    assert r["reminder"] is None
    assert all("delivered" not in e for e in r["shadow_convergence"])
    # 与 Phase 1 直调逐字节一致（shadow 模式零变化）
    direct = audit_stage_payload(
        {k: v for k, v in p.items() if k != "enable_convergence_shadow"},
        convergence_shadow=GuardHTTPRequestHandler.convergence_shadow)
    assert json.dumps(r, sort_keys=True) == json.dumps(direct, sort_keys=True)


def test_http_advisory_budget_survives_across_requests(advisory_url):
    p = _payload(rounds=2, task_id="http-3")
    p["enable_convergence_shadow"] = True
    r1 = _post(advisory_url, "/v1/stage-check", dict(p))
    r2 = _post(advisory_url, "/v1/stage-check", dict(p))
    assert r1["reminder"] is not None and r2["reminder"] is None


def test_http_flag_off_no_shadow_even_in_advisory_mode(advisory_url):
    r = _post(advisory_url, "/v1/stage-check", _payload(rounds=4, task_id="http-4"))
    assert "shadow_convergence" not in r and r["reminder"] is None


def test_http_bad_convergence_config_400(advisory_url):
    p = _payload(rounds=2, task_id="http-5", targets=["a", "a"])
    p["enable_convergence_shadow"] = True
    req = urllib.request.Request(advisory_url + "/v1/stage-check",
                                 data=json.dumps(p).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 400

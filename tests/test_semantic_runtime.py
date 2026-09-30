# -*- coding: utf-8 -*-
"""tests/test_semantic_runtime.py

语义匹配运行时 P2 测试（计划 §4-P2 门禁全部用例）。
纯逻辑：FakeBackend，不依赖 torch / 模型权重 / 服务。
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from moonbow.semantic.backends.base import (  # noqa: E402
    Backend, BackendInfo, BackendInvalidOutput,
)
from moonbow.semantic.backends.fake import FakeBackend, FakeRule  # noqa: E402
from moonbow.semantic.matcher import (  # noqa: E402
    SemanticMatcher, get_or_create, reset_registry,
)
from moonbow.semantic.runtime import (  # noqa: E402
    SemanticRuntime, SemanticRuntimeRejected,
    DEFAULT_QUEUE_CAPACITY,
)
from moonbow.semantic.schema import (  # noqa: E402
    MatchRequest, MatchResponse, Provenance, ValidationError, validate_response,
)

PATTERN = "completion.asserted@1"
TEXT = "应该修好了，但还没运行测试。"


def req(text=TEXT, pattern=PATTERN, **kw):
    return MatchRequest(text=text, pattern=pattern, **kw)


def all_valid(request, response):
    """每个 runtime 出口响应都必须过 P1 契约校验。"""
    validate_response(request, response)
    return response


def wait_until(cond, timeout=3.0, interval=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# ---------------------------------------------------------------------------
# 1. 并发首用只加载一次（load_count 可验证）
# ---------------------------------------------------------------------------

def test_concurrent_first_use_loads_once():
    fb = FakeBackend(rules={PATTERN: FakeRule(matched=True, score=0.9)})
    rt = SemanticRuntime(fb, queue_capacity=16)  # 队列足以容纳全部并发首用
    n = 12
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = rt.match(req())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.status == "ok" and r.matched is True for r in results)
    assert fb.load_count == 1
    assert rt.metrics()["load_count"] == 1
    assert rt.state == "ready"


def test_backend_initialize_threadsafe_once():
    fb = FakeBackend()
    barrier = threading.Barrier(8)

    def call():
        barrier.wait()
        fb.initialize()

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fb.load_count == 1


# ---------------------------------------------------------------------------
# 2. 失败重试有上限，不无限重试
# ---------------------------------------------------------------------------

def test_transient_failure_retries_capped():
    fb = FakeBackend(fault="unavailable")
    rt = SemanticRuntime(fb, max_retries=2)
    resp = all_valid(req(), rt.match(req()))
    assert resp.status == "error" and resp.reason_code == "unavailable"
    # 1 次原始调用 + 2 次上限内重试，绝不无限。
    assert fb.call_count == 3
    assert rt.metrics()["retries"] == 2
    assert rt.metrics()["errors"] == 1


def test_non_transient_failure_not_retried():
    fb = FakeBackend(fault="invalid_output")
    rt = SemanticRuntime(fb, max_retries=5)
    resp = all_valid(req(), rt.match(req()))
    assert resp.reason_code == "invalid_output"
    assert fb.call_count == 1  # 非瞬态不重试


# ---------------------------------------------------------------------------
# 3. 取消排队任务不执行
# ---------------------------------------------------------------------------

def test_cancel_queued_task_not_executed():
    fb = FakeBackend(delay=0.4, rules={PATTERN: FakeRule(matched=True)})
    rt = SemanticRuntime(fb)
    first = rt.submit(req(request_id="r1"))
    second = rt.submit(req(request_id="r2"))
    assert first is not second
    time.sleep(0.05)  # first 已开始执行，second 仍在排队
    assert second.cancel() is True
    assert second.cancel() is False  # 重复取消返回 False
    resp = all_valid(req(request_id="r2"), second.wait(2.0))
    assert resp.status == "error" and resp.reason_code == "unavailable"
    assert wait_until(lambda: fb.call_count == 1)
    time.sleep(0.2)
    assert fb.call_count == 1  # 被取消的任务从未进入推理


def test_cancel_completed_task_fails():
    fb = FakeBackend(rules={PATTERN: FakeRule(matched=True)})
    rt = SemanticRuntime(fb)
    h = rt.submit(req())
    h.wait(2.0)
    assert h.cancel() is False


# ---------------------------------------------------------------------------
# 4. close()：排队任务 unavailable，运行中任务等待完成后结束
# ---------------------------------------------------------------------------

def test_close_waits_for_running_and_rejects_queued():
    fb = FakeBackend(delay=0.3, rules={PATTERN: FakeRule(matched=True, score=0.8)})
    rt = SemanticRuntime(fb)
    running = rt.submit(req(request_id="run"))
    queued = rt.submit(req(request_id="queued"))
    time.sleep(0.05)
    rt.close(drain_timeout=3.0)
    r_run = all_valid(req(request_id="run"), running.wait(1.0))
    assert r_run.status == "ok" and r_run.matched is True
    r_q = all_valid(req(request_id="queued"), queued.wait(1.0))
    assert r_q.status == "error" and r_q.reason_code == "unavailable"
    assert wait_until(lambda: not rt._worker.is_alive())
    # 关闭后的新请求也返回 unavailable，不复活 worker。
    resp = all_valid(req(), rt.match(req()))
    assert resp.status == "error" and resp.reason_code == "unavailable"
    assert fb.call_count == 1


def test_close_on_idle_runtime_is_safe():
    rt = SemanticRuntime(FakeBackend())
    rt.close()
    assert rt.match(req()).reason_code == "unavailable"


# ---------------------------------------------------------------------------
# 5. 过载 429 语义：立即返回 overloaded，不阻塞排队
# ---------------------------------------------------------------------------

def test_queue_full_returns_overloaded_immediately():
    gate = threading.Event()
    fb = FakeBackend(gate=gate, rules={PATTERN: FakeRule(matched=True)})
    rt = SemanticRuntime(fb, queue_capacity=1)
    rt.submit(req(request_id="a"))       # 占住 worker（gate 阻塞中）
    assert wait_until(lambda: fb.call_count == 1)  # worker 已取走 a
    rt.submit(req(request_id="b"))       # 占满队列（容量 1）
    t0 = time.monotonic()
    resp = all_valid(req(request_id="c"), rt.match(req(request_id="c")))
    elapsed = time.monotonic() - t0
    assert resp.status == "error" and resp.reason_code == "overloaded"
    assert elapsed < 0.2  # 立即拒绝，而非排队等待
    assert rt.metrics()["overloaded"] >= 1
    gate.set()  # 释放，让 worker 收尾
    assert wait_until(lambda: fb.call_count == 2)


# ---------------------------------------------------------------------------
# 6. 超时可区分于其他错误
# ---------------------------------------------------------------------------

def test_timeout_distinguishable_from_other_errors():
    slow = FakeBackend(delay=1.0, rules={PATTERN: FakeRule(matched=True)})
    rt = SemanticRuntime(slow)
    resp = all_valid(req(), rt.match(req(), deadline=0.15))
    assert resp.status == "error" and resp.reason_code == "timeout"

    gate = threading.Event()
    full = FakeBackend(gate=gate)
    rt2 = SemanticRuntime(full, queue_capacity=1)
    rt2.submit(req())
    assert wait_until(lambda: full.call_count == 1)
    rt2.submit(req())
    resp2 = all_valid(req(), rt2.match(req()))
    assert resp2.reason_code == "overloaded"

    down = FakeBackend(fault="unavailable")
    rt3 = SemanticRuntime(down, max_retries=0)
    resp3 = all_valid(req(), rt3.match(req()))
    assert resp3.reason_code == "unavailable"
    codes = {resp.reason_code, resp2.reason_code, resp3.reason_code}
    assert codes == {"timeout", "overloaded", "unavailable"}
    assert rt.metrics()["timeout"] == 1


def test_timeout_cancels_queued_task():
    # deadline 到期后，排队中的任务被标记取消，worker 不再执行。
    gate = threading.Event()
    fb = FakeBackend(gate=gate)
    rt = SemanticRuntime(fb)
    rt.submit(req(request_id="first"))       # worker 阻塞在 gate
    second = rt.submit(req(request_id="second"))
    assert second.cancel() is True           # 显式取消排队任务
    resp = all_valid(req(request_id="third"),
                     rt.match(req(request_id="third"), deadline=0.1))
    assert resp.reason_code == "timeout"
    gate.set()
    assert wait_until(lambda: fb.call_count == 1)
    time.sleep(0.2)
    assert fb.call_count == 1  # 取消的排队任务未执行


# ---------------------------------------------------------------------------
# 7. 健康检查不被慢推理阻塞
# ---------------------------------------------------------------------------

def test_health_not_blocked_by_slow_inference():
    fb = FakeBackend(delay=1.0, rules={PATTERN: FakeRule(matched=True)})
    rt = SemanticRuntime(fb)
    rt.submit(req())
    time.sleep(0.05)  # 推理进行中
    t0 = time.monotonic()
    h = rt.health()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, "health must not wait for in-flight inference"
    assert h["state"] == "ready"
    assert h["backend"]["fake"] is True
    assert wait_until(lambda: rt.metrics()["requests"] >= 1)


# ---------------------------------------------------------------------------
# 8. 两个 Matcher 会话间无状态串扰
# ---------------------------------------------------------------------------

def test_two_matchers_no_state_crosstalk():
    reset_registry()
    fb = FakeBackend(rules={
        PATTERN: FakeRule(matched=True, score=0.9),
        "process.unresolved@1": FakeRule(matched=False, score=0.1),
    })
    m1 = SemanticMatcher.shared(fb, key="shared-test")
    m2 = SemanticMatcher.shared(fb, key="shared-test")
    assert m1.runtime is m2.runtime  # 同键共享同一 runtime

    t1 = "应该修好了，但还没运行测试。"
    t2 = "计划下一步先运行测试。"
    r1a = m1.match(t1, pattern=PATTERN, request_id="s1")
    r2a = m2.match(t1, pattern=PATTERN, request_id="s2")
    r1b = m1.match(t2, pattern="process.unresolved@1", request_id="s3")
    r2b = m2.match(t2, pattern="process.unresolved@1", request_id="s4")
    # 同一文本在两个 Matcher 下结果一致，互不污染。
    assert (r1a.matched, r1a.score) == (r2a.matched, r2a.score)
    assert (r1b.matched, r1b.score) == (r2b.matched, r2b.score)
    # 不同 pattern/文本各自确定性，没有被上一个请求污染。
    assert r1a.matched is True and r1b.matched is False
    assert all(r.request_id == rid
               for r, rid in [(r1a, "s1"), (r2a, "s2"), (r1b, "s3"), (r2b, "s4")])
    # 证据按各自请求独立构造（见 test_matcher_match_find_all_and_contract）。
    assert r1a.evidence == [] and r1b.evidence == []


# ---------------------------------------------------------------------------
# 9. 全部响应过 validate_response + Matcher 门面/batch
# ---------------------------------------------------------------------------

def test_matcher_match_find_all_and_contract():
    fb = FakeBackend(rules={
        PATTERN: FakeRule(matched=True, score=0.9, quotes=("应该修好了",)),
        "ts.capture@1": FakeRule(
            matched=True, score=0.7,
            quotes=("运行测试", "还没运行测试")),
    })
    m = SemanticMatcher(SemanticRuntime(fb))
    r = m.match(TEXT, pattern=PATTERN, request_id="m1")
    all_valid(req(request_id="m1"), r)
    assert r.matched is True
    assert len(r.evidence) == 1  # match 单一判定 ≤1 条证据
    assert r.provenance.backend == "fake"

    fa = m.find_all(TEXT, pattern="ts.capture@1", request_id="m2")
    all_valid(MatchRequest(text=TEXT, pattern="ts.capture@1",
                           operation="find_all", request_id="m2"), fa)
    assert len(fa.evidence) == 2
    assert fa.provenance.pattern_version == "ts.capture@1"


def test_batch_items_aligned_and_failures_marked():
    fb = FakeBackend(rules={PATTERN: FakeRule(matched=True, score=0.9)})
    m = SemanticMatcher(SemanticRuntime(fb))
    items = [
        {"text": TEXT, "pattern": PATTERN, "request_id": "b1"},
        {"text": "x", "pattern": "nope@1", "request_id": "b2"},  # 后端默认负例
        {"text": "", "pattern": PATTERN, "request_id": "b3"},    # 非法请求
    ]
    results = m.batch(items)
    assert len(results) == 3  # 逐项对齐，不丢结果
    assert results[0].status == "ok" and results[0].request_id == "b1"
    assert results[1].status == "ok" and results[1].matched is False
    assert results[2].status == "error" and results[2].request_id == "b3"


def test_batch_size_cap():
    m = SemanticMatcher(SemanticRuntime(FakeBackend()))
    with pytest.raises(ValueError):
        m.batch([{"text": TEXT, "pattern": PATTERN}] * 17)
    assert len(m.batch([{"text": TEXT, "pattern": PATTERN}] * 16)) == 16


# ---------------------------------------------------------------------------
# 10. registry：不同键不共享；工厂只调用一次
# ---------------------------------------------------------------------------

def test_registry_different_keys_separate_runtimes():
    reset_registry()
    fb1 = FakeBackend(rules={PATTERN: FakeRule(matched=True)})
    fb2 = FakeBackend(rules={PATTERN: FakeRule(matched=False)},
                      model_revision="fake-rev-2")
    r1 = get_or_create(fb1, key="a")
    r2 = get_or_create(fb2, key="b")
    r1b = get_or_create(fb1, key="a")
    assert r1 is r1b and r1 is not r2
    calls = []

    def factory():
        calls.append(1)
        return FakeBackend()

    ra = get_or_create(backend_factory=factory, key="factory-x", queue_capacity=4)
    rb = get_or_create(backend_factory=factory, key="factory-x", queue_capacity=4)
    assert ra is rb
    assert len(calls) == 1  # 工厂只调用一次
    assert ra._queue_capacity == 4
    reset_registry()


# ---------------------------------------------------------------------------
# 11. FakeBackend 故障注入矩阵 + 非 Backend 异常按 invalid_output 处理
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault,code", [
    ("timeout", "timeout"),
    ("invalid_output", "invalid_output"),
    ("unavailable", "unavailable"),
    ("overloaded", "overloaded"),
])
def test_fake_fault_modes(fault, code):
    fb = FakeBackend(fault=fault)
    rt = SemanticRuntime(fb, max_retries=0)
    resp = all_valid(req(), rt.match(req()))
    assert resp.status == "error" and resp.reason_code == code
    assert resp.matched is None and resp.provenance is None
    assert resp.evidence == []


def test_rule_with_nonexistent_quote_is_invalid_output():
    fb = FakeBackend(rules={PATTERN: FakeRule(matched=True, quotes=("不在原文",))})
    rt = SemanticRuntime(fb, max_retries=0)
    resp = all_valid(req(), rt.match(req()))
    assert resp.reason_code == "invalid_output"


def test_custom_rule_fn_bad_output_is_invalid_output():
    def bad(_request):
        return "not a response"
    fb = FakeBackend(rules={PATTERN: bad})
    rt = SemanticRuntime(fb, max_retries=0)
    resp = all_valid(req(), rt.match(req()))
    assert resp.reason_code == "invalid_output"


def test_backend_load_failure_marks_failed():
    class Broken(Backend):
        def __init__(self):
            super().__init__(BackendInfo("broken", "rev-0"))

        def _load(self):
            raise RuntimeError("weights missing")

        def match(self, request):
            raise BackendInvalidOutput("unreachable")

    rt = SemanticRuntime(Broken())
    resp = all_valid(req(), rt.match(req()))
    assert resp.status == "error" and resp.reason_code == "unavailable"
    assert rt.state == "failed"
    resp2 = all_valid(req(), rt.match(req()))
    assert resp2.reason_code == "unavailable"  # failed 状态不再接推理请求


# ---------------------------------------------------------------------------
# 12. 自由 requirement 路径 + 延迟注入 deadline 内完成
# ---------------------------------------------------------------------------

def test_requirement_path_and_delay_within_deadline():
    fb = FakeBackend(delay=0.05,
                     rules={"requirement:是否声称完成": FakeRule(matched=True,
                                                                score=0.8)})
    rt = SemanticRuntime(fb)
    r = rt.match(MatchRequest(text=TEXT, requirement="是否声称完成"), deadline=2.0)
    all_valid(MatchRequest(text=TEXT, requirement="是否声称完成"), r)
    assert r.status == "ok" and r.matched is True

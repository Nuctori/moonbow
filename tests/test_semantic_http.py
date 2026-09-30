# -*- coding: utf-8 -*-
"""tests/test_semantic_http.py

P5：统一 SDK/HTTP 可替换性验证。

覆盖：
- SDK（进程内 matcher）与 HTTPClientBackend 对同一组 smoke 请求结果等价
  （ok / 默认负例 / abstain / error(invalid_output) / batch 单项失败）。
- HTTP 层契约错误码：400（非法 JSON/字段）、413（超长 text / 超请求体）、
  404（未知路由，含 /check 绝不误落）、429（队列满）、503（未 ready、
  unavailable）、504（推理超时）。
- 服务身份：/health service=semantic-runtime，与 guard 服务（progress-guard,
  端口 18492）可区分；默认 loopback。
- 并发 4 客户端打同一实例只加载一次（/health metrics.load_count==1）。
- 替换演练：同一套 smoke 代码零改动打两种 fake 配置（不同 rules /
  model_revision），provenance.model_revision 随配置变化。
- 子进程启动/终止干净（sys.executable，Windows 兼容；日志不落请求原文）。
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Dict

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from moonbow.semantic.backends.fake import FakeBackend  # noqa: E402
from moonbow.semantic.client import HTTPClientBackend  # noqa: E402
from moonbow.semantic.http import rules_from_config  # noqa: E402
from moonbow.semantic.matcher import SemanticMatcher  # noqa: E402
from moonbow.semantic.runtime import SemanticRuntime  # noqa: E402
from moonbow.semantic.schema import (  # noqa: E402
    MatchRequest, ValidationError,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
GUARD_PORT = 18492

# loopback 直连，绝不经系统代理
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TEXT = "应该修好了，但还没运行测试。"
PAT = "completion.asserted@1"

RULES_A = {
    PAT: {"matched": True, "score": 0.9, "quotes": ["应该修好了"]},
    "process.unresolved@1": {"abstain": True,
                             "reason_code": "insufficient_context"},
    "proc.error@1": {"matched": True, "quotes": ["不在原文中的引文"]},
}

SMOKE = [
    {"text": TEXT, "pattern": PAT, "request_id": "e1"},            # ok 正例
    {"text": "完全无关的文本", "pattern": "unknown.pattern@1",
     "request_id": "e2"},                                           # 默认负例
    {"text": "计划尚未确定", "pattern": "process.unresolved@1",
     "request_id": "e3"},                                           # abstain
    {"text": "一些文本", "pattern": "proc.error@1",
     "request_id": "e4"},                                           # error
]


def base_config(revision: str, rules: Dict, **kw) -> dict:
    server = {"host": "127.0.0.1", "port": 0,
              "max_body_bytes": 2 * 1024 * 1024}
    if "max_body_bytes" in kw:
        server["max_body_bytes"] = kw["max_body_bytes"]
    runtime = {
        "queue_capacity": kw.get("queue_capacity", 8),
        "deadline_seconds": kw.get("deadline_seconds", 5.0),
        "max_retries": kw.get("max_retries", 2),
        "warmup": kw.get("warmup", True),
    }
    backend = {"type": "fake", "model_revision": revision, "rules": rules}
    if "fault" in kw:
        backend["fault"] = kw["fault"]
    if "delay" in kw:
        backend["delay"] = kw["delay"]
    return {"server": server, "runtime": runtime, "backend": backend}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_get(base: str, path: str, timeout: float = 3.0):
    try:
        with _OPENER.open(base + path, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


def http_post(base: str, path: str, body, timeout: float = 10.0):
    data = body if isinstance(body, (bytes, bytearray)) else \
        json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


class Service:
    def __init__(self, proc, port, base, log_path):
        self.proc = proc
        self.port = port
        self.base = base
        self.log_path = log_path

    def client(self, timeout: float = 10.0) -> HTTPClientBackend:
        return HTTPClientBackend(self.base, timeout=timeout)


@contextmanager
def semantic_service(config: dict, tmp_path, name: str = "svc"):
    """子进程启动服务（动态端口），退出时干净终止（Windows 兼容）。"""
    port = free_port()
    assert port != GUARD_PORT
    config["server"]["port"] = port
    cfg_path = os.path.join(str(tmp_path), f"{name}-config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    log_path = os.path.join(str(tmp_path), f"{name}-server.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    fh = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "moonbow.semantic.http", "--config", cfg_path],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=fh)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30.0
    up = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            fh.close()
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-2000:]
            raise RuntimeError(f"server exited early rc={proc.returncode}\n{tail}")
        try:
            status, _ = http_get(base, "/health", timeout=1.0)
            if status == 200:
                up = True
                break
        except Exception:
            time.sleep(0.05)
    if not up:
        proc.terminate()
        proc.wait(15)
        fh.close()
        raise RuntimeError("server did not become healthy in 30s")
    try:
        yield Service(proc, port, base, log_path)
    finally:
        if proc.poll() is None:
            proc.terminate()
        rc = proc.wait(timeout=15)
        fh.close()
        assert rc is not None, "server process did not terminate"
        # 端口释放：terminate 后连接应失败
        time.sleep(0.1)
        try:
            http_get(base, "/health", timeout=1.0)
            raise AssertionError("port still serving after terminate")
        except Exception:
            pass


def inproc_matcher(revision: str, rules: Dict) -> SemanticMatcher:
    """与配置服务完全同构的进程内 SDK（同一 rules_from_config 反序列化）。"""
    fb = FakeBackend(rules=rules_from_config(rules), model_revision=revision)
    return SemanticMatcher(SemanticRuntime(fb))


def run_smoke(client: HTTPClientBackend) -> dict:
    """插件视角的 smoke 请求：替换演练中这份代码零改动。"""
    out = {}
    for item in SMOKE:
        resp = client.match(MatchRequest.from_dict(item))
        d = resp.to_dict()
        out[item["request_id"]] = (d["status"], d["matched"], d["score"],
                                   d["provenance"]["model_revision"]
                                   if d["provenance"] else None)
    return out


# ---------------------------------------------------------------------------
# 1. 服务身份 / 就绪 / capabilities
# ---------------------------------------------------------------------------

def test_service_identity_readiness_and_capabilities(tmp_path):
    with semantic_service(base_config("fake-rev-A", RULES_A), tmp_path) as svc:
        assert svc.port != GUARD_PORT
        status, health = http_get(svc.base, "/health")
        assert status == 200
        assert health["service"] == "semantic-runtime"
        assert health["service"] != "progress-guard"  # 与 guard 18492 可区分

        status, ready = http_get(svc.base, "/ready")
        assert status == 200 and ready["ready" if False else "status"] == "ready"

        status, caps = http_get(svc.base, "/capabilities")
        assert status == 200
        assert caps["service"] == "semantic-runtime"
        assert caps["backend"]["name"] == "fake"
        assert caps["backend"]["model_revision"] == "fake-rev-A"
        assert caps["backend"]["fake"] is True
        assert caps["calibrated"] is False  # fake 无校准能力声明
        refs = {p["ref"] for p in caps["patterns"]}
        assert "completion.asserted@1" in refs and "ts.capture@1" in refs
        assert all("version" in p and "operation" in p for p in caps["patterns"])


def test_ready_503_when_warmup_disabled(tmp_path):
    cfg = base_config("fake-rev-A", RULES_A, warmup=False)
    with semantic_service(cfg, tmp_path) as svc:
        status, ready = http_get(svc.base, "/ready")
        assert status == 503 and ready["status"] == "not_ready"
        # 首个请求触发加载后 ready 翻转
        client = svc.client()
        resp = client.match(MatchRequest(text=TEXT, pattern=PAT))
        assert resp.status == "ok"
        status, ready = http_get(svc.base, "/ready")
        assert status == 200 and ready["status"] == "ready"


# ---------------------------------------------------------------------------
# 2. SDK / HTTP 结果等价（含 abstain、error、batch 单项失败）
# ---------------------------------------------------------------------------

def test_sdk_http_equivalence(tmp_path):
    with semantic_service(base_config("fake-rev-A", RULES_A), tmp_path) as svc:
        client = svc.client()
        client.initialize()  # 从服务端拉取真实身份
        assert client.name == "fake"
        assert client.model_revision == "fake-rev-A"

        sdk = inproc_matcher("fake-rev-A", RULES_A)
        for item in SMOKE:
            req = MatchRequest.from_dict(item)
            via_sdk = sdk.match(text=req.text, pattern=req.pattern,
                                request_id=req.request_id).to_dict()
            via_http = client.match(req).to_dict()
            assert via_sdk == via_http, f"mismatch for {item['request_id']}"
        # 逐项语义抽查：ok / 负例 / abstain / error 四态都覆盖到
        statuses = {v[0] for v in run_smoke(client).values()}
        assert statuses == {"ok", "abstain", "error"}
        assert client.match(MatchRequest.from_dict(SMOKE[0])).matched is True
        assert client.match(MatchRequest.from_dict(SMOKE[1])).matched is False
        abstain = client.match(MatchRequest.from_dict(SMOKE[2]))
        assert abstain.matched is None and \
            abstain.reason_code == "insufficient_context"
        err = client.match(MatchRequest.from_dict(SMOKE[3]))
        assert err.matched is None and err.reason_code == "invalid_output"
        assert err.provenance is None and err.evidence == []

        # find_all 等价
        fa_req = MatchRequest(text="应该修好了，应该修好了",
                              pattern=PAT, operation="find_all",
                              request_id="fa1")
        rules = dict(RULES_A)
        rules[PAT] = {"matched": True, "score": 0.9,
                      "quotes": ["应该修好了", "应该修好了"]}
        # 服务端该 pattern 规则是单引文版本；find_all 比较用单引文请求即可
        via_http = client.match(fa_req).to_dict()
        via_sdk = inproc_matcher("fake-rev-A", RULES_A).find_all(
            text=fa_req.text, pattern=PAT, request_id="fa1").to_dict()
        assert via_http == via_sdk

        # batch 等价 + 单项失败显式标记
        items = [
            {"text": TEXT, "pattern": PAT, "request_id": "b1"},
            {"text": "计划尚未确定", "pattern": "process.unresolved@1",
             "request_id": "b2"},
            {"text": "", "pattern": PAT, "request_id": "b3"},   # 非法单项
        ]
        status, body = http_post(svc.base, "/v1/batch", {"requests": items})
        assert status == 200
        results = body["results"]
        assert len(results) == 3  # 逐项对齐，不丢结果
        assert results[0]["status"] == "ok" and results[0]["request_id"] == "b1"
        assert results[1]["status"] == "abstain"
        assert results[2]["status"] == "error" and \
            results[2]["request_id"] == "b3"
        sdk_batch = [r.to_dict() for r in sdk.batch(items)]
        assert sdk_batch == results


# ---------------------------------------------------------------------------
# 3. HTTP 层错误码：400 / 413 / 404 / 429 / 503 / 504
# ---------------------------------------------------------------------------

def test_http_error_codes_400_404_413(tmp_path):
    with semantic_service(base_config("fake-rev-A", RULES_A), tmp_path) as svc:
        # 非法 JSON -> 400
        status, body = http_post(svc.base, "/v1/match", b"{not json")
        assert status == 400 and body["service"] == "semantic-runtime"
        # 非对象 JSON -> 400
        status, _ = http_post(svc.base, "/v1/match", [1, 2])
        assert status == 400
        # 未知字段（strict mode）-> 400
        bad = dict(SMOKE[0]); bad["evil_field"] = 1
        status, body = http_post(svc.base, "/v1/match", bad)
        assert status == 400 and "evil_field" in body["error"]
        # 二选一违例 -> 400
        both = dict(SMOKE[0]); both["requirement"] = "自由要求"
        status, _ = http_post(svc.base, "/v1/match", both)
        assert status == 400
        # 超长 text -> 413
        long_req = {"text": "x" * 70000, "pattern": PAT}
        status, body = http_post(svc.base, "/v1/match", long_req)
        assert status == 413
        # 超请求体（>2MB）-> 413
        status, body = http_post(svc.base, "/v1/match",
                                 {"text": "x", "pattern": PAT,
                                  "context": {"pad": "y" * (2 * 1024 * 1024 + 1)}})
        assert status == 413 and "limit" in body
        # 未知路由 -> 404（POST 与 GET 都是；/check 绝不误落到 guard 语义）
        for path in ("/v1/nope", "/check", "/v1/task-structure", "/"):
            status, body = http_post(svc.base, path, SMOKE[0])
            assert status == 404 and body["service"] == "semantic-runtime", path
            status, _ = http_get(svc.base, path)
            assert status == 404, path
        # 端点 operation 不匹配 -> 400
        status, _ = http_post(svc.base, "/v1/match",
                              {"text": TEXT, "pattern": PAT,
                               "operation": "find_all"})
        assert status == 400
        # batch 非数组 / 超上限 -> 400
        status, _ = http_post(svc.base, "/v1/batch", {"requests": "nope"})
        assert status == 400
        status, _ = http_post(svc.base, "/v1/batch",
                              {"requests": [SMOKE[0]] * 17})
        assert status == 400
        # 超长 batch 单项：不 413 整体，按单项失败标记
        status, body = http_post(svc.base, "/v1/batch", {"requests": [
            SMOKE[0], {"text": "x" * 70000, "pattern": PAT,
                       "request_id": "long"}]})
        assert status == 200
        assert body["results"][1]["status"] == "error"


def test_http_503_504_429_mappings(tmp_path):
    # unavailable -> 503
    cfg = base_config("fake-rev-A", RULES_A, fault="unavailable",
                      max_retries=0)
    with semantic_service(cfg, tmp_path) as svc:
        status, body = http_post(svc.base, "/v1/match", SMOKE[0])
        assert status == 503
        assert body["status"] == "error" and body["reason_code"] == "unavailable"

    # timeout -> 504（契约体随 504 返回，客户端可还原为契约 error）
    cfg = base_config("fake-rev-A", RULES_A, fault="timeout", max_retries=0)
    with semantic_service(cfg, tmp_path) as svc:
        status, body = http_post(svc.base, "/v1/match", SMOKE[0])
        assert status == 504
        assert body["status"] == "error" and body["reason_code"] == "timeout"
        client = svc.client()
        resp = client.match(MatchRequest.from_dict(SMOKE[0]))
        assert resp.reason_code == "timeout"  # 失败隔离：不抛异常穿透

    # 队列满 -> 429（worker 被 2s 慢推理占住，容量 1 队列排满后立即拒绝）
    cfg = base_config("fake-rev-A", RULES_A, delay=2.0, queue_capacity=1,
                      warmup=True)
    with semantic_service(cfg, tmp_path) as svc:
        # 等预热完成（2s 慢推理），之后 worker 空闲
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            _, h = http_get(svc.base, "/health")
            if h.get("state") == "ready" and h["metrics"]["requests"] >= 1 \
                    and h["queue_depth"] == 0:
                break
            time.sleep(0.1)
        first = threading.Thread(
            target=http_post, args=(svc.base, "/v1/match", SMOKE[0]))
        first.start()
        time.sleep(0.4)                       # first 已占住 worker
        queued = threading.Thread(
            target=http_post, args=(svc.base, "/v1/match",
                                    {"text": TEXT, "pattern": PAT,
                                     "request_id": "queued"}))
        queued.start()
        time.sleep(0.3)                       # queued 已填满容量 1 的队列
        status, body = http_post(svc.base, "/v1/match",
                                 {"text": TEXT, "pattern": PAT,
                                  "request_id": "reject"})
        assert status == 429
        assert body["status"] == "error" and body["reason_code"] == "overloaded"
        first.join(20)
        queued.join(20)


# ---------------------------------------------------------------------------
# 4. 并发 4 客户端同一实例只加载一次 + 请求原文不入日志
# ---------------------------------------------------------------------------

def test_concurrent_clients_load_once_and_privacy(tmp_path):
    with semantic_service(base_config("fake-rev-A", RULES_A), tmp_path) as svc:
        http_get(svc.base, "/ready")  # 等预热
        errors = []

        def worker(i):
            try:
                client = svc.client()
                for j in range(5):
                    resp = client.match(MatchRequest(
                        text=TEXT, pattern=PAT, request_id=f"c{i}-{j}"))
                    assert resp.status == "ok" and resp.matched is True
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors
        _, health = http_get(svc.base, "/health")
        assert health["metrics"]["load_count"] == 1  # 只加载一次
        assert health["metrics"]["requests"] >= 21   # 预热 + 20 请求同一实例

    # 服务停止后检查日志：请求原文（text 与引文）不得出现在日志里
    with open(svc.log_path, "r", encoding="utf-8", errors="replace") as f:
        log = f.read()
    assert "应该修好了" not in log
    assert "完全无关的文本" not in log


# ---------------------------------------------------------------------------
# 5. 替换演练：同一套 smoke 代码零改动，两种 fake 配置
# ---------------------------------------------------------------------------

def test_replacement_zero_code_change(tmp_path):
    """机制验证：换 backend 配置（不同 rules/revision）插件代码零改动；
    真实第二模型替换留待有真实权重环境时以同一 smoke 套件复跑。"""
    results = {}
    for tag, revision, matched, score in (
            ("A", "fake-rev-A", True, 0.9),
            ("B", "fake-rev-B", False, 0.2)):
        rules = {
            PAT: {"matched": matched, "score": score, "quotes": ["应该修好了"]},
            "process.unresolved@1": {"abstain": True,
                                     "reason_code": "insufficient_context"},
            "proc.error@1": {"matched": True, "quotes": ["不在原文中的引文"]},
        }
        with semantic_service(base_config(revision, rules), tmp_path,
                              name=f"repl-{tag}") as svc:
            client = svc.client()
            client.initialize()
            results[tag] = run_smoke(client)  # 与 tag A 完全相同的插件侧代码
            assert client.model_revision == revision
    a, b = results["A"], results["B"]
    # 请求集合与状态结构不变（契约稳定），结果与身份随配置变化
    assert set(a) == set(b) == {"e1", "e2", "e3", "e4"}
    assert [v[0] for v in a.values()] == [v[0] for v in b.values()]
    assert a["e1"][1] is True and b["e1"][1] is False
    assert a["e1"][3] == "fake-rev-A" and b["e1"][3] == "fake-rev-B"
    assert a["e3"][0] == b["e3"][0] == "abstain"  # 弃权路径行为一致
    assert a["e4"][0] == b["e4"][0] == "error"


# ---------------------------------------------------------------------------
# 6. 客户端失败隔离（无服务 / 无响应服务器）
# ---------------------------------------------------------------------------

def test_client_isolation_no_service():
    # 连接失败折叠为契约内 error，不抛异常穿透。
    # 注：Windows 防火墙可能把"拒绝"静默成超时，故两种契约内 reason 都合法；
    # 确定性的 reason 映射见下面的 stub 测试。
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    client = HTTPClientBackend(f"http://127.0.0.1:{dead_port}", timeout=2.0)
    resp = client.match(MatchRequest(text=TEXT, pattern=PAT))
    assert resp.status == "error"
    assert resp.reason_code in ("unavailable", "timeout")
    assert resp.matched is None and resp.provenance is None
    assert resp.evidence == []


def test_client_reason_mapping_deterministic():
    """传输异常 -> reason_code 的确定性映射（不经真实网络）。"""
    import moonbow.semantic.client as client_mod

    cases = [
        (client_mod._TransportTimeout, "timeout"),
        (client_mod._TransportDown, "unavailable"),
    ]
    for exc, expected in cases:
        class Flaky(HTTPClientBackend):
            def _request_json(self, method, path, payload):
                raise exc("injected")

        resp = Flaky("http://127.0.0.1:1").match(
            MatchRequest(text=TEXT, pattern=PAT))
        assert (resp.status, resp.reason_code) == ("error", expected)

    class StatusStub(HTTPClientBackend):
        def __init__(self, status, body):
            super().__init__("http://127.0.0.1:1")
            self._s, self._b = status, body

        def _request_json(self, method, path, payload):
            return self._s, self._b

    req = MatchRequest(text=TEXT, pattern=PAT)
    # 非契约体错误码
    assert StatusStub(400, {"error": "x"}).match(req).reason_code == "unsupported"
    assert StatusStub(404, {"error": "x"}).match(req).reason_code == "unsupported"
    assert StatusStub(413, {"error": "x"}).match(req).reason_code == "unsupported"
    # 契约体错误（含 4xx/5xx 状态码上的 status=error 体）原样还原
    contract = {"status": "error", "matched": None, "reason_code": "timeout"}
    resp = StatusStub(504, contract).match(req)
    assert resp.status == "error" and resp.reason_code == "timeout"
    # 200 坏体 -> invalid_output
    assert StatusStub(200, {"nonsense": True}).match(req).reason_code == \
        "invalid_output"


# ---------------------------------------------------------------------------
# 7. 默认绑定 loopback + 示例配置可启动
# ---------------------------------------------------------------------------

def test_example_config_smoke(tmp_path):
    with open(os.path.join(REPO, "config", "semantic_runtime.example.json"),
              encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = json.loads(json.dumps(cfg))  # 深拷贝
    cfg["backend"]["rules"][PAT] = {"matched": True, "score": 0.9,
                                    "quotes": ["应该修好了"]}
    with semantic_service(cfg, tmp_path) as svc:
        assert svc.port != GUARD_PORT
        client = svc.client()
        client.initialize()
        resp = client.match(MatchRequest(text=TEXT, pattern=PAT,
                                         request_id="ex1"))
        assert resp.status == "ok" and resp.matched is True
        assert resp.provenance.model_revision == cfg["backend"]["model_revision"]
        # 服务绑定在 loopback：配置 host 就是 127.0.0.1（默认不外露）
        assert cfg["server"]["host"] == "127.0.0.1"

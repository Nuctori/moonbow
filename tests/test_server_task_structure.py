# -*- coding: utf-8 -*-
"""/v1/task-structure 端点 HTTP 集成测试。

验收项（Goal 阶段 C）：
- 端点独立工作（分析、校验错误 400、未知路径 404 不受影响）；
- /check 与 /v1/stage-check 的既有语义不受本端点影响；
- task_structure 模块缺失时端点返回 503，守卫端点照常工作。
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moonbow.guard.server import GuardHTTPRequestHandler
from moonbow.guard.verifier import ProgressGuard


@pytest.fixture(scope="module")
def base_url():
    GuardHTTPRequestHandler.guard = ProgressGuard(lazy_load=True)
    server = HTTPServer(("127.0.0.1", 0), GuardHTTPRequestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _post(base, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as w:
        return json.loads(w.read().decode("utf-8"))


def test_task_structure_endpoint_analyzes(base_url):
    r = _post(base_url, "/v1/task-structure", {
        "text": "修复登录接口，保持旧客户端兼容，增加限流，并补充测试和文档。",
    })
    assert r["level"] == "high"
    assert r["vector"]["goals"] == 3
    assert r["fingerprint"]
    assert "message" in r and "非要求" in r["message"]
    assert "text" not in r                      # 默认不回显原文


def test_task_structure_endpoint_rejects_empty(base_url):
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(base_url, "/v1/task-structure", {"text": ""})
    assert ei.value.code == 400


def test_task_structure_endpoint_get_returns_404(base_url):
    # 端点仅支持 POST；GET 走 do_GET 的既有 404 分支
    req = urllib.request.Request(base_url + "/v1/task-structure")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 404


def test_task_structure_endpoint_rejects_bad_source(base_url):
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(base_url, "/v1/task-structure",
              {"text": "修复登录接口。", "source": "tool"})
    assert ei.value.code == 400


def test_guard_check_semantics_unaffected_by_new_endpoint(base_url):
    # 守卫 /check 在同一服务上照常工作（骨架模式：skeleton 判定路径）
    r = _post(base_url, "/check", {
        "req": "修复崩溃", "resp": "已经全部完成", "rounds": 1,
        "mode": "advisory", "semantic_review_used": False,
    })
    assert "decision" in r and "allow_stop" in r


def test_stage_check_semantics_unaffected_by_new_endpoint(base_url):
    r = _post(base_url, "/v1/stage-check", {
        "req": "修复崩溃", "snapshot_version": 2, "mode": "shadow",
        "blocks": [
            {"seq": 1, "kind": "toolResult", "text": "1 passed", "tool_call_id": "t1",
             "tool_name": "pytest", "is_error": False},
            {"seq": 2, "kind": "text", "text": "已经修复完成，测试通过"},
        ],
    })
    assert "findings" in r and "based_on" in r

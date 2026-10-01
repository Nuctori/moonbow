# -*- coding: utf-8 -*-
"""/v1/stage-check 端点 HTTP 集成测试（不影响 /check 语义）。"""
import json
import os
import sys
import threading
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
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as w:  # 首次加载权重受内存压力可达分钟级
        return json.loads(w.read().decode("utf-8"))


def test_stage_check_returns_findings(base_url):
    r = _post(base_url, "/v1/stage-check", {
        "req": "修复崩溃", "snapshot_version": 2, "mode": "shadow",
        "blocks": [
            {"seq": 1, "kind": "toolResult", "text": "1 passed", "tool_call_id": "t1",
             "tool_name": "pytest", "is_error": False},
            {"seq": 2, "kind": "text", "text": "我接下来修改登录逻辑。"},
            {"seq": 3, "kind": "text", "text": "已经修复完成，测试通过"},
        ],
    })
    kinds = [f["kind"] for f in r["findings"]]
    assert "evidence-order" in kinds
    assert r["reminder"] and "补做" in r["reminder"]["suggestion"]
    assert r["based_on"] == 2


def test_stage_check_rejects_bad_payload(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url, "/v1/stage-check", {"blocks": "bad"})
    assert e.value.code == 400


def test_check_endpoint_unchanged(base_url):
    """收尾 /check 行为保持：B 状态 -> BLOCK 硬信号。"""
    r = _post(base_url, "/check", {
        "req": "任务", "resp": "STATUS: B 部分完成\nREMAINING: 还有\nEVIDENCE: 无",
        "rounds": 1, "mode": "advisory", "semantic_review_used": False,
    })
    assert r["decision"] == "BLOCK"
    # advisory 语义：incomplete 不强制继续（allow_stop=true），但验收状态如实标注
    assert r["allow_stop"] is True
    assert r["acceptance"] == "incomplete"

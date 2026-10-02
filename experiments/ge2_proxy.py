# -*- coding: utf-8 -*-
"""experiments/ge2_proxy.py — guard-effect-v2 模型网关（OpenAI 兼容透传）。

把本地 127.0.0.1:8901 的标准 OpenAI /v1 请求原样透传到上游
http://117.21.200.87:8317/v1，并注入 API key（key 只存在本进程环境，
不落 pi 配置）。上游本身就是 OpenAI 兼容契约（2026-10-03 实测：/v1/models、
非流式与流式 chat/completions 均通），因此本代理**不做任何协议翻译**——
只做透传 + 请求日志（模型身份审计用）。与 8899 的 cc_proxy_server
（commandcode 私有契约翻译）互不相干，不要混用。

设计要点：
- 错误显式透传：上游 4xx/5xx 原状态码 + 原 body 返回，不吞不改
  （预登记门禁 3：环境错误显式透传并从完成率分母排除）；
- 流式：逐行转发 SSE，不缓冲整个响应（agent 循环需要低延迟）；
- 请求日志：每请求一行 JSONL 追加到
  results/guard-effect-v2/logs/proxy_requests.jsonl
  （ts / path / model / stream / status / elapsed_s / resp_model），
  供模型身份断言（pi_eval/model_identity.py 之外的网关侧旁证）；
- 模型锁定：不拦截非白名单模型（透传优先），但日志记录 model 字段，
  跑批后用 runs_smoke.jsonl 的 model_identity 字段做臂级断言。

用法：
  GE2_API_KEY=sk-xxx python -u experiments/ge2_proxy.py
  （默认端口 8901，上游 http://117.21.200.87:8317；env 可覆盖：
   GE2_PORT / GE2_UPSTREAM / GE2_LOG）
健康检查：curl http://127.0.0.1:8901/health
"""
import json
import os
import sys
import time
import urllib.request

from flask import Flask, request, Response, jsonify

UPSTREAM = os.environ.get("GE2_UPSTREAM", "http://117.21.200.87:8317").rstrip("/")
API_KEY = os.environ.get("GE2_API_KEY", "")
PORT = int(os.environ.get("GE2_PORT", "8901"))
DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "guard-effect-v2", "logs",
                           "proxy_requests.jsonl")
REQ_LOG = os.environ.get("GE2_LOG", DEFAULT_LOG)
# 期望锁定模型（仅用于 /health 展示与日志标注 warning，不拦截——身份断言
# 在 runner 侧按会话 responseModel 做，这里透传优先）
EXPECTED_MODEL = os.environ.get("GE2_EXPECTED_MODEL", "gemini-3.5-flash-lite")

assert API_KEY, "GE2_API_KEY 未设置 —— 显式失败，不静默（用法见文件头）"

os.makedirs(os.path.dirname(os.path.abspath(REQ_LOG)), exist_ok=True)
app = Flask(__name__)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 直连，绕过宿主代理

_LOG_LOCK = None  # 单进程 flask threaded；append 单行 <4KB 原子性足够，不加锁


def _log(entry):
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with open(REQ_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # 日志失败不影响透传主路径，但打印到 stdout 留痕
        print(f"[ge2-proxy] log write failed: {e}", flush=True)


def _open_with_retry(req, attempts=3):
    """带显式日志的连接级重试（仅限尚未开始消费响应体的失败）。

    上游 2026-10-03 实测会间歇性返回 400 FAILED_PRECONDITION
    ("User location is not supported for the API use.")，pi 收到即死（rc=1），
    12.5% run 因此夭折。此类瞬时 geo/容量错误在网关层重试是标准做法：
    每次重试都落 proxy_requests.jsonl（retry 字段），绝不静默；
    重试耗尽后原样透传最后一次错误。
    """
    last = None
    for i in range(1, attempts + 1):
        try:
            return OPENER.open(req, timeout=900), i
        except urllib.error.HTTPError as e:
            raw = e.read()
            msg = raw.decode("utf-8", "replace")
            retryable = (e.code in (429, 500, 502, 503, 504)
                         or (e.code == 400 and ("FAILED_PRECONDITION" in msg
                                                or "location is not supported" in msg)))
            last = (e.code, msg)
            _log({"path": getattr(req, 'full_url', '?'), "status": e.code,
                  "attempt": i, "retryable": retryable,
                  "error": msg[:300]})
            if not retryable or i == attempts:
                raise urllib.error.HTTPError(e.url, e.code, msg,
                                             e.headers, None) from None
            time.sleep(2 * i)
        except Exception as e:
            last = (None, str(e))
            _log({"path": getattr(req, 'full_url', '?'), "status": None,
                  "attempt": i, "retryable": True, "error": str(e)[:300]})
            if i == attempts:
                raise
            time.sleep(2 * i)
    raise RuntimeError(f"unreachable retry state: {last}")


def _forward(subpath, method):
    """透传 /v1/<subpath> 到上游。返回 (status, content_type, 迭代器或 bytes)。"""
    url = f"{UPSTREAM}/v1/{subpath}"
    data = request.get_data() if method == "POST" else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    if request.headers.get("Content-Type"):
        req.add_header("Content-Type", request.headers["Content-Type"])
    req.add_header("Accept", request.headers.get("Accept", "application/json"))
    t0 = time.time()
    model = stream = None
    if data:
        try:
            body = json.loads(data)
            model = body.get("model")
            stream = bool(body.get("stream"))
        except Exception:
            pass
    try:
        resp, attempts = _open_with_retry(req)
    except urllib.error.HTTPError as e:
        # 上游 4xx/5xx：原样透传状态码与 body（错误显式，不静默）
        raw = e.read()
        _log({"path": subpath, "model": model, "stream": stream,
              "status": e.code, "elapsed_s": round(time.time() - t0, 2),
              "error": raw.decode("utf-8", "replace")[:500]})
        return e.code, e.headers.get("Content-Type", "application/json"), raw
    except Exception as e:
        _log({"path": subpath, "model": model, "stream": stream,
              "status": 502, "elapsed_s": round(time.time() - t0, 2),
              "error": f"upstream unreachable: {e}"})
        return 502, "application/json", json.dumps(
            {"error": {"message": f"upstream unreachable: {e}",
                       "type": "upstream_error"}}).encode("utf-8")
    ctype = resp.headers.get("Content-Type", "application/json")
    status = resp.status
    if stream:
        def gen():
            resp_model = None
            n_chunks = 0
            try:
                for raw_line in resp:
                    n_chunks += 1
                    if resp_model is None and raw_line.startswith(b"data:"):
                        try:
                            obj = json.loads(raw_line[5:].strip())
                            resp_model = (obj.get("model") or None)
                        except Exception:
                            pass
                    yield raw_line
            finally:
                resp.close()
                _log({"path": subpath, "model": model, "stream": True,
                      "status": status, "chunks": n_chunks,
                      "resp_model": resp_model,
                      "model_mismatch": bool(resp_model and model
                                             and resp_model != model),
                      "elapsed_s": round(time.time() - t0, 2)})
        return status, ctype, gen()
    raw = resp.read()
    resp.close()
    resp_model = None
    try:
        resp_model = json.loads(raw).get("model")
    except Exception:
        pass
    _log({"path": subpath, "model": model, "stream": False, "status": status,
          "resp_model": resp_model,
          "model_mismatch": bool(resp_model and model and resp_model != model),
          "elapsed_s": round(time.time() - t0, 2)})
    return status, ctype, raw


@app.route("/v1", methods=["GET"])
@app.route("/v1/", methods=["GET"])
def v1_root():
    st, ctype, payload = _forward("models", "GET")
    return Response(payload, status=st, content_type=ctype)


@app.route("/v1/<path:subpath>", methods=["GET", "POST"])
def passthrough(subpath):
    st, ctype, payload = _forward(subpath, request.method)
    if isinstance(payload, bytes):
        return Response(payload, status=st, content_type=ctype)
    return Response(payload, status=st, content_type=ctype)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ge2-proxy",
                    "upstream": UPSTREAM, "port": PORT,
                    "expected_model": EXPECTED_MODEL})


if __name__ == "__main__":
    print(f"[ge2-proxy] upstream={UPSTREAM} port={PORT} log={REQ_LOG} "
          f"expected_model={EXPECTED_MODEL}", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=True)

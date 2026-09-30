# -*- coding: utf-8 -*-
"""moonbow.semantic.http

统一语义匹配运行时的独立 HTTP 服务（P5）。

风格与 guard/server.py 一致：stdlib http.server，无新第三方依赖。
差异（按 P1 契约 §3.2）：
- 端点：POST /v1/match、/v1/find-all、/v1/batch；GET /health、/ready、/capabilities。
- 错误码：非法 JSON/字段 -> 400；超长（请求体或 text）-> 413；
  队列满 -> 429；未 ready -> 503（/ready）或 reason=unavailable -> 503；
  推理超时（reason=timeout）-> 504；未知路由一律 404，绝不误落到任何处理器
  （包括 guard 的 /check 兼容路由——本服务没有该路由）。
- 默认绑定 127.0.0.1；请求体上限默认 2MB；日志默认不记录请求原文
  （行内容只在 DEBUG 及以下才会出现，默认级别 INFO 不输出）。
- 契约响应体（status=ok/abstain/error 的 MatchResponse）即使在 4xx/5xx
  时也原样返回，SDK 与 HTTP 才能逐字段等价。

启动：python -m moonbow.semantic.http --config config/semantic_runtime.example.json
backend 由配置注入（type=fake|slm），服务代码不硬编码任何模型。
"""
import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from moonbow.semantic.backends.fake import FakeBackend, FakeRule, FakeRuleFn
from moonbow.semantic.matcher import SemanticMatcher
from moonbow.semantic.patterns import load_builtin_patterns
from moonbow.semantic.runtime import SemanticRuntime
from moonbow.semantic.schema import (
    MAX_TEXT_LENGTH,
    MatchRequest,
    MatchResponse,
    ValidationError,
)

logger = logging.getLogger("moonbow.semantic.http")

SERVICE_NAME = "semantic-runtime"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18500
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024   # 2MB
DEFAULT_MAX_BATCH = 16
DEFAULT_QUEUE_CAPACITY = 8
DEFAULT_DEADLINE_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 2

# reason_code -> HTTP 状态码（P1 契约 §3.2）。其余 reason（invalid_output、
# unsupported）属于"契约内失败结果"，返回 200 + status=error 响应体。
REASON_HTTP_STATUS = {
    "timeout": 504,
    "overloaded": 429,
    "unavailable": 503,
}


# ---------------------------------------------------------------------------
# 配置 -> backend
# ---------------------------------------------------------------------------

def rule_from_config(cfg: Any) -> "FakeRule | FakeRuleFn":
    """单条 fake 规则的 JSON 反序列化。

    两种形态：
    - {"matched": bool, "score": ..., "quotes": [...], "relation": ...,
       "truncated": bool}  -> FakeRule
    - {"abstain": true, "reason_code": "insufficient_context"} -> 规则函数，
      返回契约内 abstain 响应（用于等价性验证中的弃权路径）。
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"fake rule must be an object, got {type(cfg).__name__}")
    if cfg.get("abstain"):
        reason = cfg.get("reason_code", "insufficient_context")

        def abstain_rule(request: MatchRequest) -> MatchResponse:
            return MatchResponse(
                status="abstain",
                reason_code=reason,
                request_id=request.request_id,
            )

        return abstain_rule
    return FakeRule(
        matched=bool(cfg.get("matched", False)),
        score=cfg.get("score"),
        quotes=tuple(cfg.get("quotes") or ()),
        relation=cfg.get("relation", "supports"),
        truncated=bool(cfg.get("truncated", False)),
    )


def rules_from_config(cfg: Optional[dict]) -> Dict[str, Any]:
    """整表规则反序列化。测试用同一函数在进程内重建等价 backend。"""
    return {ref: rule_from_config(rule) for ref, rule in (cfg or {}).items()}


def register_adapters_from_config(backend, cfg: Optional[dict]) -> None:
    """backend.adapters 配置 -> 注册表。fake：{"rules": {...}}；slm：{"path": ...}。
    未知 backend 类型不做任何事（调用方已校验类型）。"""
    for name, acfg in (cfg or {}).items():
        if backend.name == "slm":
            backend.register_adapter(name, str((acfg or {}).get("path", "")))
        else:
            backend.register_adapter(name, rules_from_config(
                (acfg or {}).get("rules")))


def build_backend(cfg: dict):
    """由配置构造 backend。服务代码不硬编码模型，只认 type。"""
    btype = (cfg or {}).get("type", "fake")
    if btype == "fake":
        backend = FakeBackend(
            rules=rules_from_config(cfg.get("rules")),
            fault=cfg.get("fault"),
            delay=float(cfg.get("delay", 0.0)),
            name=str(cfg.get("name", "fake")),
            model_revision=str(cfg.get("model_revision", "fake-rev-1")),
        )
        register_adapters_from_config(backend, cfg.get("adapters"))
        return backend
    if btype == "slm":
        # 延迟导入：fake 配置不承担 torch 导入成本。
        from moonbow.semantic.backends.slm import (
            DEFAULT_MODEL_ID, PROMPT_TEMPLATE_VERSION, SLMBackend,
        )
        backend = SLMBackend(
            model_id=str(cfg.get("model_id", DEFAULT_MODEL_ID)),
            model_revision=cfg.get("revision"),
            dtype=cfg.get("dtype"),
            device=str(cfg.get("device", "cpu")),
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
            # R4：model_kind=sequence 选 openjev NLI 交叉编码器底座
            # （LoRA 适配器挂载目标）；默认 causal 不变（向后兼容）。
            model_kind=str(cfg.get("model_kind", "causal")),
            prompt_version=str(cfg.get("prompt_version",
                                       PROMPT_TEMPLATE_VERSION)),
        )
        register_adapters_from_config(backend, cfg.get("adapters"))
        return backend
    raise ValueError(f"unknown backend type {btype!r}; expected 'fake' or 'slm'")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    return cfg


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class SemanticHTTPServer(ThreadingHTTPServer):
    """每请求一线程；健康检查与推理互不阻塞（runtime.health 不走队列）。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, config: dict):
        self.config = config
        server_cfg = config.get("server", {})
        runtime_cfg = config.get("runtime", {})
        self.max_body_bytes = int(
            server_cfg.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self.max_batch = int(runtime_cfg.get("max_batch", DEFAULT_MAX_BATCH))
        self._backend = build_backend(config.get("backend", {}))
        self.runtime = SemanticRuntime(
            self._backend,
            queue_capacity=int(runtime_cfg.get("queue_capacity",
                                               DEFAULT_QUEUE_CAPACITY)),
            default_deadline=float(runtime_cfg.get("deadline_seconds",
                                                   DEFAULT_DEADLINE_SECONDS)),
            max_retries=int(runtime_cfg.get("max_retries", DEFAULT_MAX_RETRIES)),
        )
        self.matcher = SemanticMatcher(self.runtime)
        self.registry = load_builtin_patterns()
        super().__init__(address, SemanticRequestHandler)

    # -- 观测 ----------------------------------------------------------------

    def capabilities_payload(self) -> dict:
        caps = self.runtime.backend.capabilities()
        patterns = []
        for ref in self.registry.list_refs():
            spec = self.registry.get(ref)
            patterns.append({
                "ref": spec.ref,
                "id": spec.id,
                "version": spec.version,
                "operation": spec.operation,
                "threshold": spec.threshold,
            })
        return {
            "service": SERVICE_NAME,
            "backend": caps,
            # 校准能力声明：只有 backend 显式声明已校准时才为 True；
            # fake/未校准 SLM 一律 False（raw_score 语义）。
            "calibrated": bool(caps.get("calibrated", False)),
            "patterns": patterns,
            "limits": {
                "max_body_bytes": self.max_body_bytes,
                "max_batch": self.max_batch,
                "max_text_code_points": MAX_TEXT_LENGTH,
            },
        }


class SemanticRequestHandler(BaseHTTPRequestHandler):
    server: SemanticHTTPServer

    # -- 输出 ----------------------------------------------------------------

    def _send_json(self, status_code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_contract(self, response: MatchResponse) -> None:
        """契约响应：错误 reason 映射 HTTP 码，但响应体始终是 MatchResponse。"""
        status = 200
        if response.status == "error":
            status = REASON_HTTP_STATUS.get(response.reason_code, 200)
        self._send_json(status, response.to_dict())

    def log_message(self, format, *args):  # noqa: A002 - stdlib 签名
        # 默认级别（INFO）下不输出任何请求行/请求内容；原文永不入日志。
        logger.debug("%s - - [%s] %s", self.client_address[0],
                     self.log_date_time_string(), format % args)

    # -- 输入 ----------------------------------------------------------------

    def _read_json_object(self) -> Optional[dict]:
        """读取并解析请求体。失败时已发送错误响应并返回 None。"""
        srv = self.server
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length",
                                  "service": SERVICE_NAME})
            return None
        if length > srv.max_body_bytes:
            # 先排空已声明的请求体再应答，否则客户端在发送中途被重置，
            # 收不到 413 响应本身（上限外最多再读 64MB 防御性钳制）。
            remaining = min(length, 64 * 1024 * 1024)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            self._send_json(413, {"error": "request body too large",
                                  "limit": srv.max_body_bytes,
                                  "service": SERVICE_NAME})
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": f"invalid JSON: {e}",
                                  "service": SERVICE_NAME})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request must be a JSON object",
                                  "service": SERVICE_NAME})
            return None
        return payload

    def _parse_match_request(self, payload: dict, operation: str
                             ) -> Optional[MatchRequest]:
        # 超长 text 属于 413（limits 语义），先于 schema 校验区分。
        text = payload.get("text")
        if isinstance(text, str) and len(text) > MAX_TEXT_LENGTH:
            self._send_json(413, {
                "error": f"text exceeds limit: {len(text)} > "
                         f"{MAX_TEXT_LENGTH} code points",
                "service": SERVICE_NAME})
            return None
        requested = payload.get("operation")
        if requested is not None and requested != operation:
            self._send_json(400, {
                "error": f"operation {requested!r} does not match endpoint "
                         f"({operation})",
                "service": SERVICE_NAME})
            return None
        payload = dict(payload)
        payload["operation"] = operation     # 端点决定 operation
        try:
            return MatchRequest.from_dict(payload)
        except ValidationError as e:
            self._send_json(400, {"error": str(e), "service": SERVICE_NAME})
            return None

    # -- GET -----------------------------------------------------------------

    def do_GET(self):
        srv = self.server
        path = self.path.split("?", 1)[0]
        if path == "/health":
            health = srv.runtime.health()
            self._send_json(200, {"status": "ok", "service": SERVICE_NAME,
                                  "state": health["state"],
                                  "queue_depth": health["queue_depth"],
                                  "metrics": health["metrics"]})
            return
        if path == "/ready":
            health = srv.runtime.health()
            if health["ready"]:
                self._send_json(200, {"status": "ready",
                                      "service": SERVICE_NAME,
                                      "state": health["state"]})
            else:
                self._send_json(503, {"status": "not_ready",
                                      "service": SERVICE_NAME,
                                      "state": health["state"]})
            return
        if path == "/capabilities":
            self._send_json(200, srv.capabilities_payload())
            return
        # 未知路由（含 /、/check、/v1/* 之外的一切）一律 404。
        self._send_json(404, {"error": "not found", "service": SERVICE_NAME})

    # -- POST ----------------------------------------------------------------

    def do_POST(self):
        srv = self.server
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/match", "/v1/find-all", "/v1/batch"):
            # 未知路由 404：不读请求体语义、不误落到任何处理器。
            self._send_json(404, {"error": "not found", "service": SERVICE_NAME})
            return

        payload = self._read_json_object()
        if payload is None:
            return

        if path == "/v1/batch":
            self._handle_batch(payload)
            return

        operation = "find_all" if path == "/v1/find-all" else "match"
        request = self._parse_match_request(payload, operation)
        if request is None:
            return
        try:
            response = srv.runtime.match(request)
        except Exception as e:  # 防御：服务端 bug 不泄漏栈，按契约 error 返回
            logger.exception("internal error on %s", path)
            self._send_json(500, {"error": f"internal error: {e}",
                                  "service": SERVICE_NAME})
            return
        self._send_contract(response)

    def _handle_batch(self, payload: dict) -> None:
        srv = self.server
        requests = payload.get("requests")
        if not isinstance(requests, list):
            self._send_json(400, {"error": "batch requires a 'requests' array",
                                  "service": SERVICE_NAME})
            return
        if len(requests) > srv.max_batch:
            self._send_json(400, {"error": f"batch size {len(requests)} "
                                           f"exceeds limit {srv.max_batch}",
                                  "service": SERVICE_NAME})
            return
        try:
            results = srv.matcher.batch(requests, max_batch=srv.max_batch)
        except Exception as e:  # 防御
            logger.exception("internal error on /v1/batch")
            self._send_json(500, {"error": f"internal error: {e}",
                                  "service": SERVICE_NAME})
            return
        # 逐项对齐返回；单项失败已是契约内 status=error 条目，不丢结果。
        self._send_json(200, {
            "service": SERVICE_NAME,
            "results": [r.to_dict() for r in results],
        })


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def _warmup(server: SemanticHTTPServer) -> None:
    """预热：先加载再 ready，不让首个宿主请求承担加载耗时（计划 §2.1）。"""
    try:
        server.runtime.match(
            MatchRequest(text="__warmup__", pattern="completion.asserted@1"),
            deadline=120.0)
    except Exception:  # noqa: BLE001 - 预热失败不影响服务启动；/ready 如实反映
        logger.warning("warmup request failed; service stays observable")


def serve(config: dict, host: Optional[str] = None,
          port: Optional[int] = None) -> None:
    server_cfg = config.get("server", {})
    host = host if host is not None else str(server_cfg.get("host", DEFAULT_HOST))
    port = int(port) if port is not None else int(server_cfg.get("port", DEFAULT_PORT))
    server = SemanticHTTPServer((host, port), config)
    actual_port = server.server_address[1]
    if config.get("runtime", {}).get("warmup", True):
        threading.Thread(target=_warmup, args=(server,),
                         name="semantic-warmup", daemon=True).start()
    # 启动证据行（stdout；不含任何请求原文）。
    print(f"SEMANTIC_RUNTIME_LISTENING {host} {actual_port} "
          f"service={SERVICE_NAME}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
        server.runtime.close(drain_timeout=5.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Moonbow unified semantic matching runtime HTTP service")
    parser.add_argument("--config", required=True,
                        help="path to semantic runtime config JSON")
    parser.add_argument("--host", default=None,
                        help=f"bind address (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=None,
                        help="bind port (overrides config)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config)
    serve(config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

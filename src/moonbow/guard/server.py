# -*- coding: utf-8 -*-
"""progress_guard.server

本地轻量进度守卫 HTTP 微服务。
供 Pi 扩展插件、跨语言 Harness、CI 脚本等在 Agent 运行中毫秒级并发调用。
"""
import json
import logging
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from .verifier import ProgressGuard, Decision

logger = logging.getLogger("progress_guard.server")


class GuardHTTPRequestHandler(BaseHTTPRequestHandler):
    guard: ProgressGuard = None  # 类属性，在启动时注入

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/", "/ping"):
            self._send_json(200, {"status": "ok", "service": "progress-guard"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path in ("/health", "/ping"):
            self._send_json(200, {"status": "ok", "service": "progress-guard"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8", errors="replace")
            payload = json.loads(raw_body) if raw_body else {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
            return

        req = payload.get("req", "")
        resp = payload.get("resp", "")
        rounds = int(payload.get("rounds", 1))
        ext_success = payload.get("external_tool_success", None)

        try:
            verdict = self.guard.check(
                req=req,
                resp=resp,
                rounds=rounds,
                external_tool_success=ext_success,
            )
            self._send_json(200, verdict.to_dict())
        except Exception as e:
            logger.exception("守卫裁决发生内部异常")
            self._send_json(500, {"error": f"Internal Guard Error: {e}"})

    def log_message(self, format, *args):
        # 覆写安静模式，避免刷屏
        logger.debug("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def start_server(
    host: str = "127.0.0.1",
    port: int = 18492,
    models_dir: Optional[str] = None,
    device: str = "cpu",
):
    """启动本地守卫微服务并常驻监听。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("正在初始化 Progress Guard 模型 (%s @ %s)...", models_dir or "default", device)

    guard = ProgressGuard(models_dir=models_dir, device=device)
    GuardHTTPRequestHandler.guard = guard

    server = HTTPServer((host, port), GuardHTTPRequestHandler)
    logger.info("Progress Guard 服务已启动: http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("正在停止服务...")
    finally:
        server.server_close()

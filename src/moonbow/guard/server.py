# -*- coding: utf-8 -*-
"""progress_guard.server

本地轻量进度守卫 HTTP 微服务。
供 Pi 扩展插件、跨语言 Harness、CI 脚本等在 Agent 运行中毫秒级并发调用。
"""
import json
import logging
import os
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from .verifier import ProgressGuard, Decision
from .process_audit import audit_stage_payload
from .convergence import ConvergenceShadow
from .semantic_provider import (
    PROVIDER_ENV,
    SEMANTIC_CONFIG_ENV,
    env_provider_key,
)

logger = logging.getLogger("progress_guard.server")


class GuardHTTPRequestHandler(BaseHTTPRequestHandler):
    guard: ProgressGuard = None  # 类属性，在启动时注入
    # 可选：语义 provider（P2 SemanticMatcher 经 adapter）。None = 未配置，
    # /check 请求 semantic_provider="semantic" 时明确 400，不静默回退。
    semantic_provider = None
    # R5：/check 未显式指定 semantic_provider 时的缺省值。默认 "legacy"
    # （既有行为零变化）；MOONBOW_GUARD_PROVIDER=semantic 启动时置为
    # "semantic"（配置激活，代码就绪）。
    default_provider_key = "legacy"
    # Phase 1：收敛信号 shadow 观察者（/v1/stage-check 的
    # enable_convergence_shadow=true 时惰性构造、类级复用。
    # ConvergenceShadow.compute 是块流纯函数、无跨请求状态，复用安全。
    # 缺省 None：不传 flag 的请求行为与之前逐字节一致）。
    convergence_shadow = None

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
        elif self.path == "/v1/semantic-status":
            # 只读状态：当前语义 provider kind / backend / calibrated 声明。
            if self.semantic_provider is not None:
                status = dict(self.semantic_provider.describe())
            else:
                status = self.guard.semantic_status()
            self._send_json(200, status)
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

        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON payload must be an object"})
            return

        # ---- 阶段审计端点（过程审计；与收尾 /check 分离，不影响其语义）----
        if self.path in ("/v1/stage-check", "/stage-check"):
            try:
                # Phase 1：可选收敛 shadow 通道（只记录、零投递）。
                # payload 缺省不带 enable_convergence_shadow → 走原路径，
                # 响应不含 shadow_convergence 键（零行为变化）。
                conv_flag = payload.get("enable_convergence_shadow", False)
                if not isinstance(conv_flag, bool):
                    raise ValueError("enable_convergence_shadow must be a boolean")
                if conv_flag:
                    if GuardHTTPRequestHandler.convergence_shadow is None:
                        GuardHTTPRequestHandler.convergence_shadow = ConvergenceShadow()
                    result = audit_stage_payload(
                        payload,
                        convergence_shadow=GuardHTTPRequestHandler.convergence_shadow)
                else:
                    result = audit_stage_payload(payload)
                logger.info("/v1/stage-check findings=%d reminder=%s",
                            len(result.get("findings", [])),
                            bool(result.get("reminder")))
                self._send_json(200, result)
            except (ValueError, TypeError) as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                logger.exception("阶段审计发生内部异常")
                self._send_json(500, {"error": f"Internal Stage Audit Error: {e}"})
            return

        # ---- 任务结构顾问端点（独立能力；lazy import + 异常完全隔离，
        #      不触达守卫的任何状态与判定；/check 语义不变）----
        if self.path in ("/v1/task-structure", "/task-structure"):
            try:
                from ..task_structure.service import analyze_payload
                result = analyze_payload(payload)
                logger.info("/v1/task-structure level=%s goals=%s abstain=%s",
                            result.get("level"),
                            (result.get("vector") or {}).get("goals"),
                            result.get("abstain"))
                self._send_json(200, result)
            except (ValueError, TypeError) as e:
                self._send_json(400, {"error": str(e)})
            except ImportError as e:
                logger.warning("task_structure 模块不可用: %s", e)
                self._send_json(503, {"error": "task_structure unavailable"})
            except Exception as e:
                logger.exception("任务结构分析发生内部异常")
                self._send_json(500, {"error": f"Internal Task Structure Error: {e}"})
            return

        ext_success = payload.get("external_tool_success", None)
        if ext_success is not None and type(ext_success) is not bool:
            self._send_json(400, {"error": "external_tool_success must be a boolean or null"})
            return

        mode = payload.get("mode", "advisory")
        review_used = payload.get("semantic_review_used", False)
        if mode not in ("advisory", "strict") or type(review_used) is not bool:
            self._send_json(400, {"error": "mode must be advisory/strict and semantic_review_used must be boolean"})
            return

        req = payload.get("req", "")
        resp = payload.get("resp", "")
        try:
            rounds = int(payload.get("rounds", 1))
        except (TypeError, ValueError, OverflowError):
            self._send_json(400, {"error": "rounds must be an integer"})
            return

        # P6/R5：可选语义 provider 选择。缺省值由部署配置决定
        # （MOONBOW_GUARD_PROVIDER=semantic 时默认 semantic；缺省 legacy，
        # 与既有 /check 行为一致）。请求显式指定时覆盖缺省。
        provider_key = payload.get("semantic_provider", self.default_provider_key)
        if provider_key not in ("legacy", "semantic"):
            self._send_json(400, {"error": "semantic_provider must be 'legacy' or 'semantic'"})
            return
        if provider_key == "semantic" and self.semantic_provider is None:
            self._send_json(400, {"error": "semantic_provider='semantic' requested but semantic runtime is not configured on this server"})
            return
        check_provider = self.semantic_provider if provider_key == "semantic" else None

        try:
            verdict = self.guard.check(
                req=req,
                resp=resp,
                rounds=rounds,
                external_tool_success=ext_success,
                mode=mode,
                semantic_review_used=review_used,
                provider=check_provider,
            )
            # 裁决遥测：stdout 可见（CI / 安装验证以这行 log 为投递证据）
            logger.info("/check verdict=%s is_closed=%s skeleton=%s",
                        verdict.decision.value, verdict.is_closed,
                        bool(verdict.scores.get("skeleton_only")))
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
    lazy: bool = False,
    semantic_provider=None,
):
    """启动本地守卫微服务并常驻监听。

    lazy=True：骨架模式，无权重可运行（协议/硬信号/工具证据层；
    权重缺失时定性信号自动降级，见 verifier）。
    semantic_provider：可选 SemanticProviderAdapter（P6）。None = 仅 legacy，
    /check 请求 semantic_provider="semantic" 时明确 400。

    R5 默认切换（配置驱动，缺省 legacy 零变化）：
    - MOONBOW_GUARD_PROVIDER=semantic 启动时按
      MOONBOW_GUARD_SEMANTIC_CONFIG（缺省 config/semantic_runtime_lora.json）
      构建白名单 semantic provider，/check 与 SDK 默认走 semantic；
      构建失败记 warning 并保持 legacy。
    - 一键回滚 = 移除该环境变量 + 重启（verdict 与 legacy 基线逐字段一致，
      见 P10 回滚演练 results/semantic-runtime/p9-host/rollback_drill.md）。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("正在初始化 Progress Guard 模型 (%s @ %s, lazy=%s)...", models_dir or "default", device, lazy)

    env_key = env_provider_key()
    if semantic_provider is None and env_key == "semantic":
        try:
            from .semantic_provider import semantic_provider_from_config
            semantic_provider = semantic_provider_from_config(
                os.environ.get(SEMANTIC_CONFIG_ENV))
            GuardHTTPRequestHandler.default_provider_key = "semantic"
            logger.info("MOONBOW_GUARD_PROVIDER=semantic: /check 默认走 "
                        "semantic provider（白名单；回滚 = 移除该环境变量并重启）")
        except Exception as e:
            logger.warning("semantic provider 构建失败（%s: %s），保持 legacy 默认",
                           type(e).__name__, e)
    elif env_key is None:
        GuardHTTPRequestHandler.default_provider_key = "legacy"
        logger.info("%s 未设置或非法：/check 默认 legacy（既有行为零变化）", PROVIDER_ENV)

    guard = ProgressGuard(models_dir=models_dir, device=device, lazy_load=lazy)
    GuardHTTPRequestHandler.guard = guard
    GuardHTTPRequestHandler.semantic_provider = semantic_provider

    server = HTTPServer((host, port), GuardHTTPRequestHandler)
    logger.info("Progress Guard 服务已启动: http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("正在停止服务...")
    finally:
        server.server_close()

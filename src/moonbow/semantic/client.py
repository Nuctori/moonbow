# -*- coding: utf-8 -*-
"""moonbow.semantic.client

HTTPClientBackend：把统一语义匹配 HTTP 服务包装成 P2 的 Backend 协议
（P5）。供插件/测试跨进程调用同一个常驻服务，而不是每插件私有加载模型。

失败隔离立场：
- 传输失败（连接拒绝/超时）、非 200、坏 JSON 都不抛异常穿透，
  一律折叠为契约内 status=error 响应（reason_code 区分）。
- 服务端返回的契约响应体（含 4xx/5xx 时的 status=error 体）原样经
  MatchResponse.from_dict 还原，provenance 由服务端产生并透传，
  客户端不改写 matched/score/evidence/provenance。

映射：
- 客户端等待超时 / 服务端 504 -> reason_code=timeout
- 连接失败 / 服务端 5xx（非契约体）-> unavailable
- 429 -> overloaded；400/404/413 -> unsupported；200 坏体 -> invalid_output
"""
import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from moonbow.semantic.backends.base import Backend, BackendInfo, error_response
from moonbow.semantic.schema import MatchRequest, MatchResponse, ValidationError

DEFAULT_TIMEOUT_SECONDS = 10.0

# loopback 服务直连：绝不经系统代理（环境代理会把 127.0.0.1 请求路由出去，
# 把连接失败伪装成 502）。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _TransportTimeout(Exception):
    """客户端等待超时（含连接与读）。"""


class _TransportDown(Exception):
    """连接失败/拒绝/重置。"""


class HTTPClientBackend(Backend):
    """Backend 协议的 HTTP 传输实现。

    initialize()（可由 runtime 触发）拉取 /capabilities，用服务端真实
    backend 身份（name、model_revision、fake 标记）回填本实例 info；
    未 initialize 时也可直接 match（identity 显示为构造值）。
    """

    def __init__(self, base_url: str,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 name: str = "http",
                 model_revision: str = "unknown"):
        super().__init__(BackendInfo(
            name=name, model_revision=model_revision, fake=False))
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._remote_capabilities: Optional[Dict[str, Any]] = None

    # -- Backend 协议 ---------------------------------------------------------

    def _load(self) -> None:
        caps = self._request_json("GET", "/capabilities", None)[1]
        backend = caps.get("backend") if isinstance(caps, dict) else None
        if not isinstance(backend, dict):
            raise ValueError("capabilities response missing backend object")
        self._info = BackendInfo(
            name=str(backend.get("name", self._info.name)),
            model_revision=str(backend.get("model_revision",
                                           self._info.model_revision)),
            fake=bool(backend.get("fake", False)),
            extra={"base_url": self._base_url,
                   "service": caps.get("service"),
                   "remote_capabilities": caps},
        )
        self._remote_capabilities = caps

    def match(self, request: MatchRequest) -> MatchResponse:
        path = "/v1/find-all" if request.operation == "find_all" else "/v1/match"
        try:
            status, body = self._request_json("POST", path, request.to_dict())
        except _TransportTimeout:
            return error_response(request, "timeout")
        except _TransportDown:
            return error_response(request, "unavailable")

        if isinstance(body, dict) and "status" in body:
            # 服务端契约响应体（含 4xx/5xx 上的 status=error 体）：原样还原。
            try:
                return MatchResponse.from_dict(body)
            except ValidationError:
                return error_response(request, "invalid_output")
        # 非契约体（400/404/413 的 {"error": ...}、空体、坏 JSON）：
        code = {400: "unsupported", 404: "unsupported", 413: "unsupported",
                429: "overloaded", 503: "unavailable", 504: "timeout",
                }.get(status, "unavailable" if status >= 500
                      else "invalid_output")
        return error_response(request, code)

    # -- 观测 -----------------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        caps = dict(super().capabilities())
        if self._remote_capabilities is not None:
            caps["remote"] = self._remote_capabilities
        return caps

    @property
    def base_url(self) -> str:
        return self._base_url

    # -- 传输 -----------------------------------------------------------------

    def _request_json(self, method: str, path: str,
                      payload: Optional[dict]) -> Tuple[int, Any]:
        url = self._base_url + path
        data = None
        headers: Dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=headers)
        try:
            with _OPENER.open(req, timeout=self._timeout) as resp:
                return resp.status, self._decode(resp.read())
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
            except Exception:  # noqa: BLE001
                raw = b""
            return e.code, self._decode(raw)
        except (socket.timeout, TimeoutError) as e:
            raise _TransportTimeout(str(e)) from e
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), (socket.timeout,
                                                       TimeoutError)):
                raise _TransportTimeout(str(e)) from e
            raise _TransportDown(str(e)) from e
        except (ConnectionError, OSError) as e:
            raise _TransportDown(str(e)) from e

    @staticmethod
    def _decode(raw: bytes) -> Any:
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

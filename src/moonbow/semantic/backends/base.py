# -*- coding: utf-8 -*-
"""moonbow.semantic.backends.base

Backend 协议（P2）。任何 backend 只做推理，不做决策：
文本 + 匹配要求 -> MatchResponse；证据、协议、预算由程序校验。

约定：
- backend 返回的 MatchResponse 必须能通过 P1 的 validate_response(req, resp)；
  runtime 在接收到结果后强制校验，未通过按 invalid_output 处理。
- backend 不自行加载模型多次：initialize() 幂等，runtime 保证 load once。
- backend 的故障用 BackendError 家族表达，由 runtime 统一映射为
  契约内的 error 响应（reason_code 区分），不向插件泄漏异常栈。
"""
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from moonbow.semantic.schema import MatchRequest, MatchResponse

# 可重试的瞬态 reason_code（runtime 重试上限内重试，绝不无限重试）。
TRANSIENT_REASON_CODES = ("unavailable", "overloaded", "timeout")


class BackendError(Exception):
    """backend 故障基类；reason_code 必须是契约 7 值之一。"""

    reason_code = "unavailable"

    def __init__(self, message: str = "", reason_code: Optional[str] = None):
        if reason_code is not None:
            self.reason_code = reason_code
        super().__init__(message or self.reason_code)


class BackendUnavailable(BackendError):
    reason_code = "unavailable"


class BackendOverloaded(BackendError):
    reason_code = "overloaded"


class BackendTimeout(BackendError):
    reason_code = "timeout"


class BackendInvalidOutput(BackendError):
    reason_code = "invalid_output"


@dataclass
class BackendInfo:
    """backend 元数据：身份、能力声明、模型 revision。"""
    name: str
    model_revision: str
    fake: bool = False
    operations: tuple = ("match", "find_all")
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "model_revision": self.model_revision,
            "fake": self.fake,
            "operations": list(self.operations),
        }
        d.update(self.extra)
        return d


def error_response(request: MatchRequest, reason_code: str) -> MatchResponse:
    """构造契约内合法的 error 响应（matched=null、无 provenance/evidence）。"""
    return MatchResponse(
        status="error",
        reason_code=reason_code,
        request_id=request.request_id,
    )


class Backend(ABC):
    """推理 backend 协议。实现不得在此层做策略决策。"""

    def __init__(self, info: BackendInfo):
        self._info = info
        self._init_lock = threading.Lock()
        self._initialized = False
        # 可验证的加载计数：runtime 依赖它证明 load once。
        self.load_count = 0
        self.call_count = 0

    # -- 身份与能力 ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def model_revision(self) -> str:
        return self._info.model_revision

    @property
    def info(self) -> BackendInfo:
        return self._info

    def capabilities(self) -> Dict[str, Any]:
        return self._info.to_dict()

    # -- 生命周期 -----------------------------------------------------------

    def initialize(self) -> None:
        """加载权重/建立连接。幂等：并发/重复调用只加载一次。"""
        with self._init_lock:
            if self._initialized:
                return
            self._load()
            self._initialized = True
            self.load_count += 1

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _load(self) -> None:
        """子类覆盖：真正的加载动作。"""

    def close(self) -> None:
        """释放资源。运行中的推理不可中断；由 runtime 排空后调用。"""

    # -- 推理 ---------------------------------------------------------------

    @abstractmethod
    def match(self, request: MatchRequest) -> MatchResponse:
        """单次推理。返回的 MatchResponse 必过 validate_response。"""

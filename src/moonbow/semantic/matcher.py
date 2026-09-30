# -*- coding: utf-8 -*-
"""moonbow.semantic.matcher

SemanticMatcher：面向插件的门面（P2）。

- 插件持有 Matcher，不直接触碰 backend/runtime 内部。
- match / find_all / batch；batch 有上限（默认 16），逐项调用、
  逐项结果；单项失败显式标记为 error 响应，不丢结果。
- get_or_create 注册表：同一进程内多个 Matcher 可共享同一 runtime，
  键为 backend 配置（backend 身份 + runtime 配置）。
- Matcher 本身无会话状态：同一文本请求互不污染，结果只由
  请求内容 + 共享 runtime 决定。
"""
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from moonbow.semantic.backends.base import Backend
from moonbow.semantic.backends.base import error_response
from moonbow.semantic.runtime import SemanticRuntime, SemanticRuntimeRejected
from moonbow.semantic.schema import (
    MatchRequest,
    MatchResponse,
)

DEFAULT_MAX_BATCH = 16


# ---------------------------------------------------------------------------
# get_or_create registry：键为 backend 配置，同进程共享 runtime。
# ---------------------------------------------------------------------------

_REGISTRY_LOCK = threading.Lock()
_REGISTRY: Dict[Tuple, SemanticRuntime] = {}


def _runtime_key(backend: Optional[Backend], key: Optional[str],
                 runtime_config: Dict[str, Any]) -> Tuple:
    cfg = tuple(sorted((k, repr(v)) for k, v in runtime_config.items()))
    if key is not None:
        return (key, cfg)
    if backend is None:
        raise ValueError("get_or_create requires a backend instance or an explicit key")
    return (backend.name, backend.model_revision, cfg)


def get_or_create(
    backend: Optional[Backend] = None,
    backend_factory: Optional[Callable[[], Backend]] = None,
    key: Optional[str] = None,
    **runtime_config: Any,
) -> SemanticRuntime:
    """取或建共享 runtime。backend_factory 只在首次创建时调用。

    同键并发调用只创建一个 runtime（含一次 backend 加载）。
    """
    with _REGISTRY_LOCK:
        probe_key = _runtime_key(backend, key, runtime_config)
        existing = _REGISTRY.get(probe_key)
        if existing is not None:
            return existing
        instance = backend if backend is not None \
            else (backend_factory() if backend_factory is not None else None)
        if instance is None:
            raise ValueError("get_or_create requires backend or backend_factory")
        final_key = _runtime_key(instance, key, runtime_config)
        existing = _REGISTRY.get(final_key)
        if existing is not None:
            return existing
        runtime = SemanticRuntime(instance, **runtime_config)
        _REGISTRY[final_key] = runtime
        return runtime


def reset_registry() -> None:
    """测试辅助：清空注册表（不负责关闭已创建的 runtime）。"""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


# ---------------------------------------------------------------------------
# R2：按 (operation, pattern ref) 的显式 backend 路由。
# 默认为空：无注册时 matcher 行为与之前完全一致（单一 runtime）。
# 注册项是 backend 工厂（延迟调用），首次命中路由时经 get_or_create
# 共享化（同键只加载一次权重）。调用方 API 与方法签名不变。
# ---------------------------------------------------------------------------

_ROUTE_LOCK = threading.Lock()
_PATTERN_BACKEND_FACTORIES: Dict[Tuple[str, str], Callable[[], Backend]] = {}


def register_pattern_backend(operation: str, pattern_ref: str,
                             backend_factory: Callable[[], Backend]) -> None:
    """显式路由：该 (operation, pattern) 的请求交给指定 backend 工厂。"""
    from moonbow.semantic.schema import OPERATIONS
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {OPERATIONS}")
    with _ROUTE_LOCK:
        _PATTERN_BACKEND_FACTORIES[(operation, pattern_ref)] = backend_factory


def unregister_pattern_backend(operation: str, pattern_ref: str) -> None:
    with _ROUTE_LOCK:
        _PATTERN_BACKEND_FACTORIES.pop((operation, pattern_ref), None)


def _routed_runtime(request: MatchRequest) -> Optional[SemanticRuntime]:
    """命中路由则返回对应共享 runtime；未命中返回 None（走默认 runtime）。"""
    if request.pattern is None:
        return None
    with _ROUTE_LOCK:
        factory = _PATTERN_BACKEND_FACTORIES.get(
            (request.operation, request.pattern))
    if factory is None:
        return None
    return get_or_create(backend_factory=factory,
                         key=f"route:{request.operation}:{request.pattern}")


# ---------------------------------------------------------------------------
# SemanticMatcher 门面
# ---------------------------------------------------------------------------

class SemanticMatcher:
    def __init__(self, runtime: SemanticRuntime):
        self._runtime = runtime

    @classmethod
    def shared(cls, backend: Optional[Backend] = None,
               backend_factory: Optional[Callable[[], Backend]] = None,
               key: Optional[str] = None, **runtime_config: Any) -> "SemanticMatcher":
        """共享 runtime 的 Matcher：同进程同配置的多会话共享加载。"""
        return cls(get_or_create(backend, backend_factory,
                                 key=key, **runtime_config))

    @property
    def runtime(self) -> SemanticRuntime:
        return self._runtime

    # -- 单项 ---------------------------------------------------------------

    def match(
        self,
        text: str,
        pattern: Optional[str] = None,
        requirement: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        request_id: Optional[str] = None,
        deadline: Optional[float] = None,
        adapter: Optional[str] = None,
    ) -> MatchResponse:
        req = MatchRequest(
            text=text, pattern=pattern, requirement=requirement,
            context=context or {}, operation="match", threshold=threshold,
            request_id=request_id, adapter=adapter,
        )
        routed = _routed_runtime(req)
        return (routed if routed is not None else self._runtime).match(
            req, deadline=deadline)

    def find_all(
        self,
        text: str,
        pattern: Optional[str] = None,
        requirement: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        request_id: Optional[str] = None,
        deadline: Optional[float] = None,
        adapter: Optional[str] = None,
    ) -> MatchResponse:
        req = MatchRequest(
            text=text, pattern=pattern, requirement=requirement,
            context=context or {}, operation="find_all", threshold=threshold,
            request_id=request_id, adapter=adapter,
        )
        routed = _routed_runtime(req)
        return (routed if routed is not None else self._runtime).match(
            req, deadline=deadline)

    # -- 批量 ---------------------------------------------------------------

    def batch(
        self,
        requests: Sequence[Union[MatchRequest, Dict[str, Any]]],
        max_batch: int = DEFAULT_MAX_BATCH,
        deadline: Optional[float] = None,
    ) -> List[MatchResponse]:
        """逐项调用、逐项结果；单项失败显式 error 标记，不丢结果。"""
        if len(requests) > max_batch:
            raise ValueError(
                f"batch size {len(requests)} exceeds limit {max_batch}")
        results: List[MatchResponse] = []
        for i, item in enumerate(requests):
            try:
                req = item if isinstance(item, MatchRequest) \
                    else MatchRequest.from_dict(item)
                results.append(self._runtime.match(req, deadline=deadline))
            except Exception as e:  # 单项失败不丢整个 batch
                rid = None
                if isinstance(item, MatchRequest):
                    rid = item.request_id
                elif isinstance(item, dict):
                    rid = item.get("request_id")
                results.append(MatchResponse(
                    status="error",
                    reason_code="unsupported"
                    if isinstance(e, ValueError) else "unavailable",
                    request_id=rid,
                ))
        return results

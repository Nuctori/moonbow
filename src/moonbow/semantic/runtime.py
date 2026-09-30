# -*- coding: utf-8 -*-
"""moonbow.semantic.runtime

共享语义匹配 Runtime（P2）。

设计要点（计划 §2.1 / §4-P2 门禁）：
- 显式依赖注入：backend 由构造函数传入，runtime 不自行 from_pretrained。
- 单次加载生命周期：首次使用时在唯一推理 worker 线程内 initialize()，
  状态机 idle -> loading -> ready（失败 -> failed），load_count 可验证。
- 有界请求队列：容量可配（默认 8）；满时立即返回 overloaded（429 语义），
  绝不无限排队。
- 每请求 deadline：等待超时返回 reason_code=timeout 的 error 响应，
  可与其他错误区分。
- 取消：排队中的任务可取消（worker 不执行）；运行中的推理不可中断，
  close() 标记 draining 并等待完成或超时。
- 健康检查不经过推理队列，不被慢推理阻塞。
- 指标计数器：requests / overloaded / timeout / errors / load_count / retries。
- 失败重试有上限（仅瞬态故障），无自动无限重试。
"""
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from moonbow.semantic.backends.base import (
    TRANSIENT_REASON_CODES,
    Backend,
    BackendError,
    error_response,
)
from moonbow.semantic.schema import MatchRequest, MatchResponse, validate_response

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_DRAINING = "draining"
STATE_FAILED = "failed"

DEFAULT_QUEUE_CAPACITY = 8
DEFAULT_DEADLINE_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 2

_SHUTDOWN = object()   # worker 关闭哨兵


class SemanticRuntimeRejected(Exception):
    """submit 被立即拒绝；.response 是契约内 error 响应。"""

    def __init__(self, response: MatchResponse):
        self.response = response
        super().__init__(f"rejected: {response.reason_code}")


@dataclass
class TaskHandle:
    """异步任务句柄：排队中的任务可取消；运行中的不可中断。"""
    request: MatchRequest
    _event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    response: Optional[MatchResponse] = None
    cancelled: bool = False
    started: bool = False
    done: bool = False

    def wait(self, timeout: Optional[float] = None) -> Optional[MatchResponse]:
        self._event.wait(timeout)
        return self.response

    def cancel(self) -> bool:
        """已完成的任务不可取消；排队中未开始的可以。"""
        with self._lock:
            if self.done or self.started or self.cancelled:
                return False
            self.cancelled = True
            self.response = error_response(self.request, "unavailable")
            self.done = True
        self._event.set()
        return True

    def _finish(self, response: MatchResponse) -> None:
        with self._lock:
            self.response = response
            self.done = True
        self._event.set()


class SemanticRuntime:
    """单一活动 backend 的共享运行时。线程安全。"""

    def __init__(
        self,
        backend: Backend,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        default_deadline: float = DEFAULT_DEADLINE_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._backend = backend
        self._queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        self._queue_capacity = queue_capacity
        self._default_deadline = float(default_deadline)
        self._max_retries = max_retries
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._state = STATE_IDLE
        self._closed = False
        self._metrics: Dict[str, int] = {
            "requests": 0, "overloaded": 0, "timeout": 0,
            "errors": 0, "load_count": 0, "retries": 0,
        }

    # -- 公共 API -----------------------------------------------------------

    def match(self, request: MatchRequest,
              deadline: Optional[float] = None) -> MatchResponse:
        """同步匹配。deadline 秒；超时返回 timeout error（不无限等待）。"""
        handle, rejected = self._try_submit(request)
        if handle is None:
            return rejected
        resp = handle.wait(self._effective_deadline(deadline))
        if resp is not None:
            return resp
        # deadline 到期：排队中的任务标记取消（worker 不再执行）。
        handle.cancel()
        with self._lock:
            self._metrics["timeout"] += 1
        return error_response(request, "timeout")

    def submit(self, request: MatchRequest) -> TaskHandle:
        """异步提交。队列满/未就绪时抛 SemanticRuntimeRejected。"""
        handle, rejected = self._try_submit(request)
        if handle is None:
            raise SemanticRuntimeRejected(rejected)
        return handle

    def _try_submit(self, request: MatchRequest
                    ) -> Tuple[Optional[TaskHandle], Optional[MatchResponse]]:
        """返回 (handle, None) 或 (None, 拒绝响应)。"""
        with self._lock:
            self._metrics["requests"] += 1
            if self._closed or self._state in (STATE_DRAINING, STATE_FAILED):
                return None, error_response(request, "unavailable")
            self._ensure_worker_locked()
            handle = TaskHandle(request=request)
            try:
                self._queue.put_nowait(handle)
            except queue.Full:
                self._metrics["overloaded"] += 1
                return None, error_response(request, "overloaded")
            return handle, None

    def cancel(self, handle: TaskHandle) -> bool:
        return handle.cancel()

    def close(self, drain_timeout: float = 10.0) -> None:
        """排空关闭：排队任务返回 unavailable；运行中的推理等待完成或超时。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._state == STATE_READY:
                self._state = STATE_DRAINING
            worker = self._worker
        if worker is not None:
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                # 队列满：牺牲一次 put 的即时性，等待 worker 腾出空间。
                try:
                    self._queue.put(_SHUTDOWN, timeout=drain_timeout)
                except queue.Full:  # pragma: no cover - 极端拥塞
                    pass
            worker.join(timeout=drain_timeout)
        with self._lock:
            if self._state == STATE_DRAINING:
                self._state = STATE_IDLE

    # -- 观测 ---------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def backend(self) -> Backend:
        return self._backend

    def health(self) -> Dict:
        """健康检查：不经过推理队列，不被慢推理阻塞。"""
        with self._lock:
            return {
                "state": self._state,
                "ready": self._state == STATE_READY,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue_capacity,
                "metrics": dict(self._metrics),
                "backend": self._backend.capabilities(),
            }

    def metrics(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._metrics)

    # -- 内部 ---------------------------------------------------------------

    def _effective_deadline(self, deadline: Optional[float]) -> float:
        return self._default_deadline if deadline is None else float(deadline)

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, name="semantic-runtime-worker",
                daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._drain_remaining()
                return
            handle: TaskHandle = item
            reject_closed = False
            with handle._lock:
                if handle.cancelled or handle.done:
                    continue
                if self._closed:
                    # close() 之后排队中的任务不再执行：排空返回 unavailable。
                    reject_closed = True
                else:
                    handle.started = True
            if reject_closed:
                handle._finish(error_response(handle.request, "unavailable"))
                continue
            self._execute(handle)

    def _drain_remaining(self) -> None:
        """关闭时排空队列：排队任务一律返回 unavailable，不执行。"""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, TaskHandle) and not item.done:
                item._finish(error_response(item.request, "unavailable"))

    def _ensure_loaded(self) -> bool:
        """首次使用初始化（load once）。失败置 failed。"""
        if self._backend.is_initialized:
            if self._state == STATE_IDLE:
                with self._lock:
                    if self._state == STATE_IDLE:
                        self._state = STATE_READY
            return True
        with self._lock:
            if self._state in (STATE_IDLE,):
                self._state = STATE_LOADING
        try:
            self._backend.initialize()
        except Exception:
            with self._lock:
                self._state = STATE_FAILED
            return False
        with self._lock:
            self._metrics["load_count"] = self._backend.load_count
            self._state = STATE_READY
        return True

    def _execute(self, handle: TaskHandle) -> None:
        req = handle.request
        if not self._ensure_loaded():
            handle._finish(error_response(req, "unavailable"))
            return
        attempts = 0
        while True:
            try:
                resp = self._backend.match(req)
                validate_response(req, resp)
                handle._finish(resp)
                return
            except BackendError as e:
                if e.reason_code in TRANSIENT_REASON_CODES \
                        and attempts < self._max_retries:
                    attempts += 1
                    with self._lock:
                        self._metrics["retries"] += 1
                    continue
                with self._lock:
                    self._metrics["errors"] += 1
                handle._finish(error_response(req, e.reason_code))
                return
            except Exception:
                # 未通过 validate_response 或其他意外异常：
                # 原始输出不能驱动插件动作，按 invalid_output 拒绝。
                with self._lock:
                    self._metrics["errors"] += 1
                handle._finish(error_response(req, "invalid_output"))
                return

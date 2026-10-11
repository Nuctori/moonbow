# -*- coding: utf-8 -*-
"""moonbow.intent.capture — 语义捕获层。

显式后端注入：生产默认 SemanticCaptureBackend（SemanticMatcher SLM 路径，
复用 task_structure.matcher_backend 的失败隔离）；FakeCaptureBackend 仅测试。
不存在 rule/词表实现路径——本模块没有也不接受"规则兜底"。

失败语义（与仓库契约一致）：后端 abstain/失败 → 显式标记返回，
不得伪装成"没有实体"。
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .schema import Entity, TEXT_CAPTURE_KINDS, entity_id
from .verify import verify_quote


@dataclass
class CaptureResult:
    """一次捕获的完整结果：实体 + 显式状态（abstain/失败必须可见）。"""
    entities: List[Entity] = field(default_factory=list)
    abstain: bool = False
    backend_failed: bool = False
    backend_abstain: bool = False
    backend_truncated: bool = False
    dropped_bad_quote: int = 0
    dropped_role_mismatch: int = 0
    backend_name: str = "unknown"

    @property
    def ok(self) -> bool:
        return not (self.abstain or self.backend_failed)


class CaptureStage:
    """文本 → 实体。后端必须显式注入；本类不提供任何规则默认实现。"""

    def __init__(self, backend):
        if backend is None:
            raise ValueError(
                "capture backend 必须显式注入（SemanticCaptureBackend 为生产"
                "默认；Fake 仅测试）。本模块无规则兜底路径。")
        self._backend = backend

    def capture(self, event_id: str, text: str, scope: str,
                source_role: str = "assistant_claim") -> CaptureResult:
        status = getattr(self._backend, "last_status", None)
        result = CaptureResult(backend_name=getattr(self._backend, "name", ""))
        try:
            captures = self._backend.extract(text)
        except Exception as e:                       # noqa: BLE001 — 显式失败
            result.backend_failed = True
            result.abstain = True
            result.backend_name += f":error:{type(e).__name__}"
            return result
        if getattr(self._backend, "last_backend_failed", False):
            result.backend_failed = True
            result.abstain = True
            return result
        if getattr(self._backend, "last_backend_abstain", False) and not captures:
            result.backend_abstain = True
            result.abstain = True
            return result
        if getattr(self._backend, "last_backend_truncated", False):
            result.backend_truncated = True
        for cap in captures:
            if cap.kind not in TEXT_CAPTURE_KINDS:
                continue
            ok, span = verify_quote(text, cap.quote, cap.span)
            if not ok:
                result.dropped_bad_quote += 1        # 坏引文：丢弃且计数
                continue
            try:
                ent = Entity(
                    entity_id=entity_id(cap.kind, event_id, cap.quote),
                    kind=cap.kind, quote=cap.quote, span=span,
                    event_id=event_id,
                    source_role=source_role,
                    scope=scope,
                    uncertainty=result.backend_truncated)
            except ValueError:
                # 角色-类别不匹配（如 assistant 文本断言 constraint）：
                # 丢弃并计数——助手转述不能自我加冕为用户约束
                result.dropped_role_mismatch += 1
                continue
            result.entities.append(ent)
        return result


class SemanticCaptureBackend:
    """生产默认：包装 task_structure.matcher_backend.MatcherExtractor
    （SemanticMatcher SLM 路径，XPU/HTTP 由 matcher 自身决定）。"""

    def __init__(self, matcher_extractor):
        self._impl = matcher_extractor
        self.name = matcher_extractor.name
        self.last_status = None
        self.last_backend_failed = False
        self.last_backend_abstain = False
        self.last_backend_truncated = False

    def extract(self, text: str):
        caps = self._impl.extract(text)
        self.last_status = getattr(self._impl, "last_status", None)
        self.last_backend_failed = getattr(
            self._impl, "last_backend_failed", False)
        self.last_backend_abstain = getattr(
            self._impl, "last_backend_abstain", False)
        self.last_backend_truncated = getattr(
            self._impl, "last_backend_truncated", False)
        return caps


class FakeCaptureBackend:
    """仅测试：注入固定 captures。name 带 fake 前缀以便断言。"""

    def __init__(self, captures, abstain: bool = False,
                 failed: bool = False, truncated: bool = False):
        self._captures = captures
        self._abstain = abstain
        self._failed = failed
        self._truncated = truncated
        self.name = "fake"
        self.last_status = None
        self.last_backend_failed = failed
        self.last_backend_abstain = abstain
        self.last_backend_truncated = truncated

    def extract(self, text: str):
        if self._failed:
            self.last_backend_failed = True
            raise RuntimeError("fake backend failure")
        self.last_backend_abstain = self._abstain
        self.last_backend_truncated = self._truncated
        self.last_status = "abstain" if self._abstain else "ok"
        return list(self._captures)

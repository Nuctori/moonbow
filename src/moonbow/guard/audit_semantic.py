# -*- coding: utf-8 -*-
"""moonbow.guard.audit_semantic

P7：过程审计的可选语义声明检测 shadow 通道（计划 §4-P7）。

设计（对齐 P6 GuardSemanticProvider 风格）：
- ProcessSemanticAdapter：把 P2 SemanticMatcher（或同接口 HTTP 门面）
  适配为 shadow 观察者。每次 observe() 对一段中间汇报文本跑一次
  matcher.match(pattern="process.unresolved@1")，只产出「语义观察」。
- 失败隔离：matcher 异常或契约内 abstain/error → unavailable 标记，
  绝不产出半套结果，也绝不向调用方抛异常。
- 零行为保证：StageAuditor 仅在显式注入 shadow provider 时才运行本
  通道，结果只写入 audit 结果的 shadow_semantic 键——不参与
  reminder 选择、不参与 findings 判定、不参与 semantic 布尔、不新增
  用户投递。默认 None（零行为变化）；时序核对、证据性质、发现修订
  逻辑一行未动。

ts.capture 已按 P8 结论关闭；process.unresolved 同样只做 shadow，
不改默认行为。
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

# 语义 shadow 通道的固定 pattern 引用（P1 PatternSpec 命名空间）
PROCESS_UNRESOLVED_PATTERN = "process.unresolved@1"


@dataclass
class ShadowObservation:
    """一次 shadow 语义观察的原始结果。None/标记表示不判定，不伪造。"""
    matched: Optional[bool] = None       # ok 时 true/false；否则 None
    score: Optional[float] = None        # 契约内 score 原样保留；无则 None
    truncated: bool = False              # 后端侧截断
    unavailable_reason: Optional[str] = None   # 非 None 表示该次观察不可用
    pattern: str = PROCESS_UNRESOLVED_PATTERN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "matched": self.matched,
            "score": self.score,
            "truncated": self.truncated,
            "unavailable_reason": self.unavailable_reason,
        }


class ProcessSemanticAdapter:
    """SemanticMatcher → 过程审计 shadow 观察者适配器。"""

    kind = "semantic"

    def __init__(self, matcher, pattern: str = PROCESS_UNRESOLVED_PATTERN):
        self._matcher = matcher
        self._pattern = pattern

    def observe(self, text: str) -> ShadowObservation:
        try:
            resp = self._matcher.match(text=text, pattern=self._pattern)
        except Exception as e:                       # noqa: BLE001 — 失败隔离
            return ShadowObservation(
                unavailable_reason=f"error:{type(e).__name__}",
                pattern=self._pattern)
        if resp.status == "ok":
            return ShadowObservation(
                matched=bool(resp.matched),
                score=round(float(resp.score), 4) if resp.score is not None else None,
                truncated=bool(resp.truncated),
                pattern=self._pattern)
        # abstain / error：显式不可用，不产出判定
        return ShadowObservation(
            truncated=bool(resp.truncated),
            unavailable_reason=f"{resp.status}:"
                               f"{resp.reason_code or 'unknown'}",
            pattern=self._pattern)

    def describe(self) -> Dict[str, Any]:
        backend = "unknown"
        try:
            runtime = getattr(self._matcher, "runtime", None)
            if runtime is not None:
                backend = getattr(runtime.backend, "name", "unknown")
        except Exception:
            pass
        return {
            "provider_kind": self.kind,
            "backend": backend,
            "pattern": self._pattern,
            "mode": "shadow",
        }

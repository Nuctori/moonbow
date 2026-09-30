# -*- coding: utf-8 -*-
"""moonbow.task_structure.matcher_backend

P7：把 P2/P5 的 SemanticMatcher（或指向 semantic 服务的 HTTP 后端）
适配为 ExtractorBackend 协议（MatcherExtractor）。

失败隔离语义（修复计划 §1 指出的适配缺口：空捕获无法区分故障）：
- backend error / 调用异常 → 返回空捕获，但置 last_backend_failed=True，
  analyze_text 写入 StructureAnalysis.backend_failed（区别于真实零命中）；
- status=abstain → 返回空捕获，置 last_backend_abstain=True，
  analyze_text 写入 abstain=True / abstain_reason="backend_abstain"；
- resp.truncated → 置 last_backend_truncated=True，
  analyze_text 写入 StructureAnalysis.backend_truncated。
- 规则后端（RuleBackend）无这些属性，getattr 默认 False，两字段恒 False。

映射规则：
- pattern=ts.capture@1 的 find_all 按 kind 逐类调用（契约的 evidence
  不携带 kind，故每次调用以 context={"kind": ...} 圈定单一类别）；
- 每条 evidence 映射为一个 Capture：span 优先取 evidence 的 start/end
  （且须与 quote 逐字对齐），不一致时回退 text.index 定位；
- quote 不是原文 verbatim 子串 → 丢弃（坏引文不进入计量）；
- score 不映射：Capture 无分数位（设计红线，spec §1），仅保留引文证据。

本后端默认不被 get_backend() 启用（默认仍是 rule）；P8 评测结论 SLM
捕获质量不合格，ts.capture 保持关闭，本模块只保证接缝与失败隔离正确。
"""
import logging
from typing import List, Optional

from .schema import Capture, CAPTURE_KINDS

logger = logging.getLogger("moonbow.task_structure.matcher_backend")

CAPTURE_PATTERN = "ts.capture@1"


class MatcherExtractor:
    """SemanticMatcher → ExtractorBackend 适配器。

    构造：注入 matcher 实例，或 url 指向 semantic 服务（内部经
    HTTPClientBackend + SemanticMatcher.shared）。两者都缺 → ValueError。
    """

    def __init__(self, matcher=None, url: Optional[str] = None,
                 pattern: str = CAPTURE_PATTERN, timeout: float = 5.0):
        if matcher is None:
            if not url:
                raise ValueError(
                    "MatcherExtractor requires a matcher instance or a service url")
            from moonbow.semantic.client import HTTPClientBackend
            from moonbow.semantic.matcher import SemanticMatcher
            matcher = SemanticMatcher.shared(
                HTTPClientBackend(url, timeout=timeout))
        self._matcher = matcher
        self.pattern = pattern
        self.name = "matcher:" + pattern
        # 每次 extract() 重置的契约状态（analyze_text 读取）
        self.last_status: Optional[str] = None
        self.last_reason_code: Optional[str] = None
        self.last_backend_failed: bool = False
        self.last_backend_abstain: bool = False
        self.last_backend_truncated: bool = False

    # -- 契约状态重置 ---------------------------------------------------------

    def _reset_status(self) -> None:
        self.last_status = None
        self.last_reason_code = None
        self.last_backend_failed = False
        self.last_backend_abstain = False
        self.last_backend_truncated = False

    # -- ExtractorBackend 协议 -----------------------------------------------

    def extract(self, text: str) -> List[Capture]:
        self._reset_status()
        captures: List[Capture] = []
        any_abstain = False
        for kind in CAPTURE_KINDS:
            try:
                resp = self._matcher.find_all(
                    text=text, pattern=self.pattern, context={"kind": kind})
            except Exception as e:                   # noqa: BLE001 — 失败隔离
                self.last_status = "error"
                self.last_reason_code = type(e).__name__
                self.last_backend_failed = True
                logger.warning("MatcherExtractor 调用失败（空捕获 + backend_failed 标记）: %s", e)
                return []
            self.last_status = resp.status
            self.last_reason_code = resp.reason_code
            if resp.truncated:
                self.last_backend_truncated = True
            if resp.status == "error":
                self.last_backend_failed = True
                return []
            if resp.status == "abstain":
                any_abstain = True
                continue
            for ev in (resp.evidence or []):
                cap = self._to_capture(kind, ev, text)
                if cap is not None:
                    captures.append(cap)
        self.last_backend_abstain = any_abstain
        # 按原文位置稳定排序（逐 kind 调用的拼接顺序不代表原文顺序）
        captures.sort(key=lambda c: (c.span[0], c.span[1]) if c.span else (0, 0))
        return captures

    def _to_capture(self, kind: str, ev, text: str) -> Optional[Capture]:
        quote = getattr(ev, "quote", "") or ""
        if kind not in CAPTURE_KINDS or not quote.strip():
            return None
        if quote not in text:
            # 坏引文：非 verbatim 子串，直接丢弃（引文级证据底线）
            return None
        span = None
        start, end = getattr(ev, "start", None), getattr(ev, "end", None)
        if (isinstance(start, int) and isinstance(end, int)
                and 0 <= start <= end <= len(text)
                and text[start:end] == quote):
            span = (start, end)
        else:
            # 偏移缺失/不一致 → 回退 text.index 定位
            idx = text.index(quote)
            span = (idx, idx + len(quote))
        return Capture(kind=kind, quote=quote, span=span)

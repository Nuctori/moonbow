# -*- coding: utf-8 -*-
"""moonbow.intent.verify — 引文级证据校验与事件存证（机械层）。

只做：verbatim 校验、偏移定位、原文存证、sha256。不做任何语义判断。
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple


def verify_quote(text: str, quote: str,
                 span: Optional[Tuple[int, int]] = None
                 ) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """校验 quote 是否为 text 的 verbatim 子串；span 提供时须与引文对齐。

    返回 (ok, resolved_span)。span 缺失/不一致时回退 text.index 定位。
    """
    if not quote or quote not in text:
        return False, None
    if (span is not None and isinstance(span, tuple) and len(span) == 2
            and isinstance(span[0], int) and isinstance(span[1], int)
            and 0 <= span[0] <= span[1] <= len(text)
            and text[span[0]:span[1]] == quote):
        return True, span
    idx = text.index(quote)
    return True, (idx, idx + len(quote))


def store_event_text(root: Path, event_id: str, text: str) -> Tuple[str, str]:
    """原文存证：写入 <root>/<event_id>.txt，返回 (相对路径, sha256)。

    事实本体机械保存，不经模型转述（追踪器零 LLM 原则的证据面）。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = Path(root) / (event_id + ".txt")
    target.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return target.name, digest


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

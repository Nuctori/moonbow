# -*- coding: utf-8 -*-
"""progress_guard.protocol

收尾清单协议 (Closure Manifest Protocol) 的核心数据结构与解析规范。
支持中英文冒号、全半角混排、Markdown 粗体标注等宽松语法的健壮提取。
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


class StatusCode(str, Enum):
    A = "A"  # 全部完成 (Resolved / Closed)
    B = "B"  # 部分完成 (Partial)
    C = "C"  # 进行中或受阻 (In Progress / Blocked)
    D = "D"  # 失败或已回滚 (Failed / Rolled back)


@dataclass
class ClosureManifest:
    """闭合清单结构"""
    raw_text: str
    status: Optional[StatusCode] = None
    status_text: str = ""
    remaining: str = ""
    evidence: str = ""
    is_valid_format: bool = False
    has_remaining: bool = False
    has_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if self.status else None,
            "status_text": self.status_text,
            "remaining": self.remaining,
            "evidence": self.evidence,
            "is_valid_format": self.is_valid_format,
            "has_remaining": self.has_remaining,
            "has_evidence": self.has_evidence,
        }


MANIFEST_TEMPLATE = (
    "STATUS: <A 全部完成 / B 部分完成 / C 进行中或受阻 / D 失败或已回滚>\n"
    "REMAINING: <尚未完成的事项，没有则写：无>\n"
    "EVIDENCE: <验证证据：具体测试命令输出/构建结果/运行指标，没有则写：无>"
)

PROMPT_REQUIRE_MANIFEST = (
    "你是执行代理。结束任务前必须申报闭合清单，只输出以下三行，不要输出其他内容：\n\n"
    f"{MANIFEST_TEMPLATE}"
)


# ---------------------------------------------------------------------------
# 阶段审计（过程审计）数据结构：观察块、审计发现。
# 与收尾清单协议严格分离 —— 中间汇报不得套用 STATUS/REMAINING/EVIDENCE 门禁。
# ---------------------------------------------------------------------------

STAGE_BLOCK_KINDS = ("thinking", "text", "toolCall", "toolResult")

FINDING_STATUSES = (
    "candidate",     # 片段疑点，不足以提醒
    "actionable",    # 阶段结束后仍成立，值得介入
    "addressed",     # 模型已回应（不代表实际修复）
    "resolved",      # 有足够新记录支持问题解决
    "withdrawn",     # 原判断不成立，撤销
    "unverified",    # 信息不足，保留未知（不是失败）
)


@dataclass
class StageBlock:
    """一条已完成的观察块（来自 Pi 流式事件组装）。"""
    seq: int
    kind: str                       # thinking / text / toolCall / toolResult
    text: str = ""
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    is_error: bool = False

    def __post_init__(self):
        if self.kind not in STAGE_BLOCK_KINDS:
            raise ValueError(f"stage block kind must be one of {STAGE_BLOCK_KINDS}")
        if type(self.seq) is not int:
            raise TypeError("seq must be int")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "kind": self.kind, "text": self.text,
            "tool_call_id": self.tool_call_id, "tool_name": self.tool_name,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageBlock":
        return cls(
            seq=d.get("seq", 0), kind=d.get("kind", "text"),
            text=str(d.get("text", "") or ""),
            tool_call_id=d.get("tool_call_id"), tool_name=d.get("tool_name"),
            is_error=bool(d.get("is_error", False)),
        )


@dataclass
class StageFinding:
    """阶段审计发现：声明/观察与用户要求之间的差异。"""
    fingerprint: str
    kind: str                       # evidence-order / contradiction / unverified-claim / thinking-concern
    status: str                     # FINDING_STATUSES 之一
    summary: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # [{seq, kind, excerpt}]
    snapshot_version: int = 0
    updated_at: int = 0

    def __post_init__(self):
        if self.status not in FINDING_STATUSES:
            raise ValueError(f"finding status must be one of {FINDING_STATUSES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint, "kind": self.kind, "status": self.status,
            "summary": self.summary, "evidence": self.evidence,
            "snapshot_version": self.snapshot_version, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageFinding":
        status = d.get("status", "candidate")
        if status not in FINDING_STATUSES:
            status = "candidate"
        return cls(
            fingerprint=str(d.get("fingerprint", "")), kind=str(d.get("kind", "")),
            status=status, summary=str(d.get("summary", "") or ""),
            evidence=list(d.get("evidence", []) or []),
            snapshot_version=int(d.get("snapshot_version", 0) or 0),
            updated_at=int(d.get("updated_at", 0) or 0),
        )


def parse_manifest(text: str) -> ClosureManifest:
    """从 Agent 的收尾输出中鲁棒解析收尾三字段清单。"""
    fields: Dict[str, str] = {}
    
    if not text:
        return ClosureManifest(raw_text="")

    lines = text.strip().splitlines()
    for line in lines:
        cleaned = line.strip().lstrip("*-# \t").strip()
        # 处理粗体如 **STATUS**: 或 **STATUS:**
        cleaned = cleaned.replace("**", "").replace("__", "")
        
        for key in ("STATUS", "REMAINING", "EVIDENCE"):
            for sep in (":", "："):
                prefix = f"{key}{sep}"
                if cleaned.upper().startswith(prefix):
                    val = cleaned[len(prefix):].strip()
                    fields[key] = val
                    break

    has_status = "STATUS" in fields and bool(fields["STATUS"])
    has_rem = "REMAINING" in fields
    has_ev = "EVIDENCE" in fields
    is_valid = has_status and has_rem and has_ev

    status_code: Optional[StatusCode] = None
    status_text = fields.get("STATUS", "").strip()
    status_labels = {
        StatusCode.A: ("全部完成",),
        StatusCode.B: ("部分完成",),
        StatusCode.C: ("进行中", "受阻", "进行中/受阻", "进行中或受阻"),
        StatusCode.D: ("失败", "已回滚", "失败/已回滚", "失败或已回滚"),
    }
    upper_st = status_text.upper()
    parts = upper_st.split(maxsplit=1)
    for code, labels in status_labels.items():
        if upper_st == code.value or status_text in labels:
            status_code = code
            break
        if len(parts) == 2 and parts[0] == code.value and parts[1] in labels:
            status_code = code
            break
    is_valid = is_valid and status_code is not None

    rem_val = fields.get("REMAINING", "").strip()
    ev_val = fields.get("EVIDENCE", "").strip()
    
    empty_values = ("", "无", "没有", "沒有", "none", "null", "n/a")
    rem_empty = rem_val.casefold() in empty_values
    ev_empty = ev_val.casefold() in empty_values

    return ClosureManifest(
        raw_text=text,
        status=status_code,
        status_text=status_text,
        remaining="" if rem_empty else rem_val,
        evidence="" if ev_empty else ev_val,
        is_valid_format=is_valid,
        has_remaining=not rem_empty,
        has_evidence=not ev_empty,
    )

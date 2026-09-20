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
    if status_text:
        upper_st = status_text.upper()
        if upper_st.startswith("A") or "全部完成" in status_text:
            status_code = StatusCode.A
        elif upper_st.startswith("B") or "部分" in status_text:
            status_code = StatusCode.B
        elif upper_st.startswith("C") or "进行" in status_text or "受阻" in status_text:
            status_code = StatusCode.C
        elif upper_st.startswith("D") or "失败" in status_text or "回滚" in status_text:
            status_code = StatusCode.D

    rem_val = fields.get("REMAINING", "").strip()
    ev_val = fields.get("EVIDENCE", "").strip()
    
    rem_empty = rem_val in ("", "无", "没有", "沒有", "none", "None", "null", "N/A")
    ev_empty = ev_val in ("", "无", "没有", "沒有", "none", "None", "null", "N/A")

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

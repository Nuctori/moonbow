# -*- coding: utf-8 -*-
"""moonbow.intent.schema — 实体、事件、判断、提醒的可观察符号定义。

立场：
- 捕获只声称"文本里出现了什么"（quote 必须 verbatim），不声称完整刻画；
- source_role 三分：user_requirement / assistant_claim / tool_observation；
  "完成宣称"只能来自 assistant_claim，且不等于完成事实；
- 助手转述不得覆盖用户原文（修订权限在 schema 层强制）；
- 判断必须绑定证据引用；模型失败/信息不足必须显式 abstain。
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SOURCE_ROLES = ("user_requirement", "assistant_claim", "tool_observation")
EVENT_KINDS = ("message", "tool_call", "tool_result", "commit", "user_feedback")

# 实体种类。前 8 类与 task_structure.CAPTURE_KINDS 对齐；
# 后 3 类是工作流守卫特有（由事件显式登记，不由文本捕获产生）。
ENTITY_KINDS = (
    "goal", "constraint", "decision", "completion_claim",
    "acceptance", "evidence_ref", "dependency", "coordination",
    "condition", "unresolved", "object",
)
TEXT_CAPTURE_KINDS = (
    "goal", "constraint", "acceptance",
    "dependency", "coordination", "condition", "unresolved", "object",
)
EVENT_ONLY_KINDS = ("decision", "completion_claim", "evidence_ref")

CONSTRAINT_VERDICTS = (
    "supports_conflict", "supports_compat", "insufficient_info")
COVERAGE_VERDICTS = (
    "supported_full", "supported_partial", "no_evidence", "insufficient_info")
JUDGMENT_TYPES = ("constraint_relation", "goal_coverage")

#: 目标验收所需的证据强度（机械元数据，由工具结果登记，不做语义判断）。
#: 关键点："测试通过"只能证明"测试通过"，不能自动等于"目标达成"——
#: 因此 evidence 与 goal 之间必须由 goal_coverage 语义裁决连接，
#: 机械层不得用"测试绿"直接宣布目标完成。
EVIDENCE_KINDS = ("test_run", "artifact", "diff", "tool_output",
                  "receipt", "assertion")
EVIDENCE_RESULTS = ("pass", "fail", "partial", "unknown")

ENTITY_STATUSES = ("active", "revised", "retracted", "pending_promise")

# completion_claim 的合法来源（tool_observation 观测到的是工具结果，不是宣称）
_ROLE_RULES = {
    "completion_claim": ("assistant_claim",),
    "decision": ("assistant_claim",),
    "goal": ("user_requirement", "assistant_claim"),
    "constraint": ("user_requirement",),
    "evidence_ref": ("tool_observation", "assistant_claim"),
}


def entity_id(kind: str, event_id: str, quote: str) -> str:
    h = hashlib.sha256()
    h.update(kind.encode("utf-8"))
    h.update(b"\x00")
    h.update(event_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(quote.encode("utf-8"))
    return "ent_" + h.hexdigest()[:24]


@dataclass
class Event:
    """工作流事件的机械存证。事实本体：text 原文 + sha256，不经模型转述。"""
    event_id: str
    kind: str                          # EVENT_KINDS 之一
    source_role: str                   # SOURCE_ROLES 之一
    ts: str                            # ISO8601
    text: str                          # 原文（verbatim）
    text_sha256: str = ""
    raw_ref: str = ""                  # 原始产物相对路径（可选）

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"illegal event kind: {self.kind!r}")
        if self.source_role not in SOURCE_ROLES:
            raise ValueError(f"illegal source role: {self.source_role!r}")
        if not self.text_sha256:
            self.text_sha256 = hashlib.sha256(
                self.text.encode("utf-8")).hexdigest()


@dataclass
class Entity:
    """语义捕获的结构化实体。quote 必须是 event.text 的 verbatim 子串。

    修订模型：revised/retracted 实体保留在账本（append-only），
    revises 指向被修订/撤销的 entity_id。权限：
    - assistant_claim 不得 revises user_requirement 来源的实体；
    - 用户修订（user_requirement 来源）可覆盖任何来源。
    scope：项目键（如 "proj_d-react-app"）；跨项目不继承（查询层强制）。
    """
    entity_id: str
    kind: str                          # ENTITY_KINDS 之一
    quote: str
    span: Optional[Tuple[int, int]]
    event_id: str
    source_role: str
    scope: str                         # 项目/范围键
    status: str = "active"             # ENTITY_STATUSES 之一
    revises: Optional[str] = None      # 被修订/撤销的 entity_id
    uncertainty: bool = False          # 捕获器自报不确定
    norm: str = ""                     # 归一化去重键（程序计算）
    meta: dict = field(default_factory=dict)  # 机械元数据（如证据 kind/result）

    def __post_init__(self):
        if self.kind not in ENTITY_KINDS:
            raise ValueError(f"illegal entity kind: {self.kind!r}")
        if self.source_role not in SOURCE_ROLES:
            raise ValueError(f"illegal source role: {self.source_role!r}")
        if self.status not in ENTITY_STATUSES:
            raise ValueError(f"illegal entity status: {self.status!r}")
        allowed = _ROLE_RULES.get(self.kind, SOURCE_ROLES)
        if self.source_role not in allowed:
            raise ValueError(
                f"kind {self.kind!r} forbids source_role {self.source_role!r}; "
                f"allowed: {allowed}")
        if self.revises and self.revises == self.entity_id:
            raise ValueError("entity cannot revise itself")
        if not self.norm:
            self.norm = hashlib.sha256(
                (self.kind + "\x00" + self.quote).encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict:
        d = {
            "entity_id": self.entity_id, "kind": self.kind,
            "quote": self.quote, "span": list(self.span) if self.span else None,
            "event_id": self.event_id, "source_role": self.source_role,
            "scope": self.scope, "status": self.status,
            "revises": self.revises, "uncertainty": self.uncertainty,
            "norm": self.norm, "meta": dict(self.meta or {}),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        span = d.get("span")
        return cls(entity_id=d["entity_id"], kind=d["kind"], quote=d["quote"],
                   span=tuple(span) if span else None,
                   event_id=d["event_id"], source_role=d["source_role"],
                   scope=d.get("scope", ""), status=d.get("status", "active"),
                   revises=d.get("revises"),
                   uncertainty=bool(d.get("uncertainty", False)),
                   norm=d.get("norm", ""), meta=dict(d.get("meta") or {}))


@dataclass
class Judgment:
    """关系判断。必须绑定 subject/reference/evidence 的 entity_id 引用；
    abstain=True 时 verdict 必须是 insufficient_info 系，且如实记录原因。
    模型失败/未配置绝不允许伪装成"未违规/已完成"。

    证据语义（第二轮审计澄清）：evidence_ids 是**证据指认**，可以是实体 id
    或 event_id；机械层登记的 evidence 实体另带 kind/result 元数据，语义层
    只在此之上做关系裁决。
    """
    judgment_id: str
    jtype: str                         # JUDGMENT_TYPES 之一
    subject_ids: List[str]             # 决策 / 完成宣称实体
    reference_ids: List[str]           # 约束 / 目标实体
    evidence_ids: List[str]            # 证据指认（实体 id 或 event_id）
    verdict: str
    model_id: str                      # 判定模型身份；"none"=未用模型
    abstain: bool = False
    abstain_reason: Optional[str] = None
    reason: str = ""                   # 一句话人工可读理由
    evidence_kinds: Optional[List[str]] = None   # 证据机械类型（EVIDENCE_KINDS）
    evidence_results: Optional[List[str]] = None  # 证据机械结果（EVIDENCE_RESULTS）

    def __post_init__(self):
        if self.jtype not in JUDGMENT_TYPES:
            raise ValueError(f"illegal judgment type: {self.jtype!r}")
        valid = (CONSTRAINT_VERDICTS if self.jtype == "constraint_relation"
                 else COVERAGE_VERDICTS)
        if self.verdict not in valid:
            raise ValueError(f"illegal verdict {self.verdict!r} for {self.jtype}")
        if not self.subject_ids or not self.reference_ids:
            raise ValueError("judgment must bind subject and reference entities")
        # 证据绑定规则按裁决类型区分（原先只覆盖约束类，导致 goal_coverage
        # 的无证据结论可以没有证据地宣称"已完成"）。
        # 注意 no_evidence 的语义正是"查过但没有证据"，因此它绑定的是**被
        # 对账的宣称**（subject_ids 已强制），不要求 evidence_ids 非空；
        # 反过来 supported_* 必须绑定它据以判定的证据。
        if self.jtype == "constraint_relation":
            need_ev = ("supports_conflict", "supports_compat")
        else:
            need_ev = ("supported_full", "supported_partial")
        if not self.evidence_ids and self.verdict in need_ev:
            raise ValueError(
                f"verdict {self.verdict!r} requires evidence_ids binding")
        for label, values, allowed in (
                ("evidence_kinds", self.evidence_kinds, EVIDENCE_KINDS),
                ("evidence_results", self.evidence_results, EVIDENCE_RESULTS)):
            if values and any(v not in allowed for v in values):
                raise ValueError(
                    f"illegal {label}: {[v for v in values if v not in allowed]}")
        # abstain 强契约（第二轮审计 D7）：弃权必须如实给出原因，且 verdict 必须
        # 落在 insufficient_info 系——不允许"弃权"伪装成有结论的判断。
        if self.abstain:
            if not (self.abstain_reason or "").strip():
                raise ValueError("abstain=True 必须给出 abstain_reason（不得匿名弃权）")
            if self.verdict != "insufficient_info":
                raise ValueError(
                    f"abstain=True 时 verdict 必须是 insufficient_info，"
                    f"得到 {self.verdict!r}")

    def to_dict(self) -> dict:
        return {
            "judgment_id": self.judgment_id, "jtype": self.jtype,
            "subject_ids": list(self.subject_ids),
            "reference_ids": list(self.reference_ids),
            "evidence_ids": list(self.evidence_ids),
            "verdict": self.verdict, "model_id": self.model_id,
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason, "reason": self.reason,
            "evidence_kinds": list(self.evidence_kinds or []),
            "evidence_results": list(self.evidence_results or []),
        }


@dataclass
class Reminder:
    """非阻塞提醒块。严重度不影响放行行为；dedup_key 稳定可去重。"""
    jtype: str
    verdict: str
    subject_ids: List[str]
    reference_ids: List[str]
    reason: str
    model_id: str = "none"
    policy: str = "remind-only"
    dedup_key: str = ""

    def __post_init__(self):
        if not self.dedup_key:
            h = hashlib.sha256()
            h.update(self.jtype.encode("utf-8")); h.update(b"\x00")
            h.update(self.verdict.encode("utf-8")); h.update(b"\x00")
            h.update("\x00".join(sorted(self.subject_ids)).encode("utf-8"))
            h.update(b"\x00")
            h.update("\x00".join(sorted(self.reference_ids)).encode("utf-8"))
            self.dedup_key = "rem_" + h.hexdigest()[:24]

    def to_dict(self) -> dict:
        return {"type": "constraint-reminder", "policy": self.policy,
                "jtype": self.jtype, "verdict": self.verdict,
                "subject_ids": list(self.subject_ids),
                "reference_ids": list(self.reference_ids),
                "reason": self.reason, "dedup_key": self.dedup_key}


def records_to_jsonl(records: List[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def parse_jsonl(line_text: str) -> List[dict]:
    return [json.loads(l) for l in line_text.splitlines() if l.strip()]

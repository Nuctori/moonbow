# -*- coding: utf-8 -*-
"""moonbow.semantic.schema

语义匹配 API 的数据契约（P1 冻结）。SDK/HTTP 共享同一 schema 与校验。

校验立场：
- 未知就是未知：matched=null 表示无法判断/失败，不用 false 冒充正常负例。
- 证据必须是原文片段：quote 是 text 的 verbatim 子串，偏移为
  Unicode code point 的半开区间 [start, end)。
- 未知顶层字段一律拒绝（严格模式），防止契约漂移。
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import math
import re

# ---------------------------------------------------------------------------
# 契约常量
#
# SCHEMA_VERSION 变更记录（changelog）：
#   1  P1 冻结：MatchRequest/MatchResponse/Provenance/Evidence/PatternSpec
#      与 validate_response 全部跨对象规则。
#   2  2026-09-30（R1，向后兼容扩展）：
#      - MatchRequest 新增可选字段 adapter（peft LoRA 逻辑名；非空、<=128、
#        [A-Za-z0-9._-]）。不带 = 无适配器（base / backend 默认），旧请求零影响。
#      - 语义：adapter 未注册时响应必须是 status=error, reason_code=unsupported,
#        matched=null —— 禁止静默回退 base（调用方不得误以为在用微调头）。
#      - Provenance 新增可选 adapter 键；ok 且请求带 adapter 时必须回填同名。
#      - validate_response 新增 provenance.adapter 与请求 adapter 的一致性规则。
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

# adapter 逻辑名规则：非空、<=128 字符、[A-Za-z0-9._-]
MAX_ADAPTER_LENGTH = 128
_ADAPTER_RE = re.compile(r"^[A-Za-z0-9._-]+$")

OPERATIONS = ("match", "find_all")
STATUSES = ("ok", "abstain", "error")
REASON_CODES = (
    "insufficient_context", "truncated", "unsupported",
    "timeout", "overloaded", "invalid_output", "unavailable",
)
RELATIONS = ("supports", "contradicts")

# 文本长度上限（code point 数）。超长输入应走 limits 拒绝（HTTP 层映射 413）。
MAX_TEXT_LENGTH = 65536


class ValidationError(ValueError):
    """契约校验失败。HTTP 层应映射为 400。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _reject_unknown(d: dict, allowed: tuple, where: str) -> None:
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ValidationError(
            f"{where}: unknown field(s) {unknown}; strict mode rejects them")


def _check_str(v: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(v, str):
        raise ValidationError(f"{name} must be str, got {type(v).__name__}")
    if not allow_empty and not v:
        raise ValidationError(f"{name} must be non-empty")
    return v


def _check_score(v: Any, name: str) -> Optional[float]:
    """score 可为 null；数值时必须有限且在 [0,1]。拒绝 NaN/越界/非数。"""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValidationError(f"{name} must be a number in [0,1] or null")
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValidationError(f"{name} must be finite (got {v})")
    if f < 0.0 or f > 1.0:
        raise ValidationError(f"{name} out of range [0,1]: {v}")
    return f


def _check_bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool):
        raise ValidationError(f"{name} must be bool, got {type(v).__name__}")
    return v


def _check_adapter(v: Any, name: str) -> Optional[str]:
    """adapter 逻辑名校验：None 合法（无适配器）；否则非空、<=128、
    仅 [A-Za-z0-9._-]。"""
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValidationError(f"{name} must be str or null, got {type(v).__name__}")
    if not v:
        raise ValidationError(f"{name} must be non-empty (null means no adapter)")
    if len(v) > MAX_ADAPTER_LENGTH:
        raise ValidationError(
            f"{name} exceeds limit: {len(v)} > {MAX_ADAPTER_LENGTH} characters")
    if not _ADAPTER_RE.match(v):
        raise ValidationError(
            f"{name} allows only [A-Za-z0-9._-], got {v!r}")
    return v


# ---------------------------------------------------------------------------
# Provenance：结果由谁产生。ok 状态必须携带。
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    backend: str                 # "slm" | "legacy" | "fake" | ...
    model_revision: str          # 钉死的模型 revision，不由请求方指定
    pattern_version: str         # "name@N"
    calibrated: bool             # 只有独立校准数据支撑时才为 True
    calibration_id: Optional[str] = None
    adapter: Optional[str] = None  # 请求带 adapter 时必须回填同名（R1）

    ALLOWED = ("backend", "model_revision", "pattern_version",
               "calibrated", "calibration_id", "adapter")

    def __post_init__(self):
        _check_str(self.backend, "provenance.backend")
        _check_str(self.model_revision, "provenance.model_revision")
        _check_str(self.pattern_version, "provenance.pattern_version")
        _check_bool(self.calibrated, "provenance.calibrated")
        if self.calibration_id is not None:
            _check_str(self.calibration_id, "provenance.calibration_id")
        self.adapter = _check_adapter(self.adapter, "provenance.adapter")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_revision": self.model_revision,
            "pattern_version": self.pattern_version,
            "calibrated": self.calibrated,
            "calibration_id": self.calibration_id,
            "adapter": self.adapter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        if not isinstance(d, dict):
            raise ValidationError("provenance must be an object")
        _reject_unknown(d, cls.ALLOWED, "provenance")
        return cls(
            backend=d.get("backend"),
            model_revision=d.get("model_revision"),
            pattern_version=d.get("pattern_version"),
            calibrated=d.get("calibrated"),
            calibration_id=d.get("calibration_id"),
            adapter=d.get("adapter"),
        )


# ---------------------------------------------------------------------------
# Evidence：原文引文级证据。
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    quote: str                   # 必须是源文本 verbatim 子串，非空
    start: int                   # code point 偏移，半开区间 [start, end)
    end: int
    relation: str                # supports | contradicts

    ALLOWED = ("quote", "start", "end", "relation")

    def __post_init__(self):
        _check_str(self.quote, "evidence.quote")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ValidationError("evidence.start must be int")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise ValidationError("evidence.end must be int")
        if self.start < 0:
            raise ValidationError("evidence.start must be >= 0")
        if self.end < self.start:
            raise ValidationError(
                f"evidence.end ({self.end}) must be >= start ({self.start})")
        if self.relation not in RELATIONS:
            raise ValidationError(
                f"evidence.relation must be one of {RELATIONS}, got {self.relation!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"quote": self.quote, "start": self.start,
                "end": self.end, "relation": self.relation}

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        if not isinstance(d, dict):
            raise ValidationError("evidence item must be an object")
        _reject_unknown(d, cls.ALLOWED, "evidence")
        return cls(quote=d.get("quote"), start=d.get("start"),
                   end=d.get("end"), relation=d.get("relation"))


# ---------------------------------------------------------------------------
# MatchRequest
# ---------------------------------------------------------------------------

@dataclass
class MatchRequest:
    text: str
    pattern: Optional[str] = None        # "name@N"；与 requirement 二选一
    requirement: Optional[str] = None    # 自由文本匹配要求；实验性
    context: Dict[str, Any] = field(default_factory=dict)
    operation: str = "match"             # match | find_all
    threshold: Optional[float] = None    # 决策门槛；未校准时是 raw_score 语义
    request_id: Optional[str] = None
    adapter: Optional[str] = None        # peft 适配器逻辑名；None = base/默认

    ALLOWED = ("text", "pattern", "requirement", "context",
               "operation", "threshold", "request_id", "adapter")

    def __post_init__(self):
        _check_str(self.text, "text")
        if len(self.text) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"text exceeds limit: {len(self.text)} > {MAX_TEXT_LENGTH} code points")
        if not isinstance(self.context, dict):
            raise ValidationError("context must be an object")
        if self.operation not in OPERATIONS:
            raise ValidationError(
                f"operation must be one of {OPERATIONS}, got {self.operation!r}")
        self.threshold = _check_score(self.threshold, "threshold")
        # 二选一：同时出现或都缺失都是契约违例，禁止隐式优先级。
        if (self.pattern is None) == (self.requirement is None):
            raise ValidationError(
                "exactly one of 'pattern' or 'requirement' is required")
        if self.pattern is not None:
            _check_str(self.pattern, "pattern")
        else:
            _check_str(self.requirement, "requirement")
        if self.request_id is not None:
            _check_str(self.request_id, "request_id", allow_empty=False)
        self.adapter = _check_adapter(self.adapter, "adapter")

    @property
    def matcher_ref(self) -> str:
        return self.pattern if self.pattern is not None else self.requirement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "pattern": self.pattern,
            "requirement": self.requirement,
            "context": dict(self.context),
            "operation": self.operation,
            "threshold": self.threshold,
            "request_id": self.request_id,
            "adapter": self.adapter,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchRequest":
        if not isinstance(d, dict):
            raise ValidationError("request must be an object")
        _reject_unknown(d, cls.ALLOWED, "request")
        return cls(
            text=d.get("text"),
            pattern=d.get("pattern"),
            requirement=d.get("requirement"),
            context=d.get("context") or {},
            operation=d.get("operation", "match"),
            threshold=d.get("threshold"),
            request_id=d.get("request_id"),
            adapter=d.get("adapter"),
        )

    @classmethod
    def from_json(cls, s: str) -> "MatchRequest":
        try:
            return cls.from_dict(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValidationError(f"invalid JSON: {e}") from e


# ---------------------------------------------------------------------------
# MatchResponse
# ---------------------------------------------------------------------------

@dataclass
class MatchResponse:
    status: str                          # ok | abstain | error
    matched: Optional[bool] = None       # ok 时必须 true/false；abstain/error 必须 null
    score: Optional[float] = None        # 可为 null；不得自报伪造概率
    score_type: Optional[str] = None     # 如 "label_probability"；null 表示 raw/无
    calibrated: bool = False
    evidence: List[Evidence] = field(default_factory=list)
    truncated: bool = False
    reason_code: Optional[str] = None    # abstain/error 必填；ok 必须为 null
    provenance: Optional[Provenance] = None  # ok 必填
    request_id: Optional[str] = None     # 回显，batch 按 item 对齐用

    ALLOWED = ("status", "matched", "score", "score_type", "calibrated",
               "evidence", "truncated", "reason_code", "provenance",
               "request_id")

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValidationError(
                f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status == "ok":
            if self.matched not in (True, False):
                raise ValidationError(
                    "status=ok requires matched=true|false (null is not ok; "
                    "use abstain for unknown)")
            if self.provenance is None:
                raise ValidationError("status=ok requires provenance")
            if self.reason_code is not None:
                raise ValidationError("status=ok must not carry reason_code")
        else:
            if self.matched is not None:
                raise ValidationError(
                    f"status={self.status} requires matched=null; "
                    "do not disguise unknown as a normal negative")
            if self.reason_code is None:
                raise ValidationError(
                    f"status={self.status} requires reason_code")
            if self.provenance is not None:
                raise ValidationError(
                    f"status={self.status} must not carry provenance")
            if self.evidence:
                raise ValidationError(
                    f"status={self.status} must not carry evidence")
            if self.score is not None:
                raise ValidationError(
                    f"status={self.status} must not carry score")
        if self.reason_code is not None and self.reason_code not in REASON_CODES:
            raise ValidationError(
                f"reason_code must be one of {REASON_CODES}, got {self.reason_code!r}")
        self.score = _check_score(self.score, "score")
        _check_bool(self.calibrated, "calibrated")
        _check_bool(self.truncated, "truncated")
        if not isinstance(self.evidence, list):
            raise ValidationError("evidence must be a list")
        self.evidence = [e if isinstance(e, Evidence) else Evidence.from_dict(e)
                         for e in self.evidence]
        if self.request_id is not None:
            _check_str(self.request_id, "request_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "matched": self.matched,
            "score": self.score,
            "score_type": self.score_type,
            "calibrated": self.calibrated,
            "evidence": [e.to_dict() for e in self.evidence],
            "truncated": self.truncated,
            "reason_code": self.reason_code,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "request_id": self.request_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchResponse":
        if not isinstance(d, dict):
            raise ValidationError("response must be an object")
        _reject_unknown(d, cls.ALLOWED, "response")
        return cls(
            status=d.get("status"),
            matched=d.get("matched"),
            score=d.get("score"),
            score_type=d.get("score_type"),
            calibrated=d.get("calibrated", False),
            evidence=list(d.get("evidence") or []),
            truncated=d.get("truncated", False),
            reason_code=d.get("reason_code"),
            provenance=(Provenance.from_dict(d["provenance"])
                        if d.get("provenance") is not None else None),
            request_id=d.get("request_id"),
        )

    @classmethod
    def from_json(cls, s: str) -> "MatchResponse":
        try:
            return cls.from_dict(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValidationError(f"invalid JSON: {e}") from e


# ---------------------------------------------------------------------------
# PatternSpec：版本化固定模式（注册表在 patterns.py）。
# ---------------------------------------------------------------------------

@dataclass
class PatternSpec:
    id: str                              # 不含版本，如 "completion.asserted"
    version: int
    definition: str                      # 自然语言定义（随版本冻结）
    positive_examples: List[str]         # >=3
    negative_examples: List[str]         # >=3
    exclusions: List[str]                # 排除项
    operation: str                       # match | find_all
    output_schema: Dict[str, Any]        # 期望输出的结构声明
    scoring: str                         # 评分方式说明
    requires_context: List[str] = field(default_factory=list)
    length_policy: Optional[str] = None
    threshold: Optional[float] = None    # 决策门槛；校准绑定另记

    ALLOWED = ("id", "version", "definition", "positive_examples",
               "negative_examples", "exclusions", "operation",
               "output_schema", "scoring", "requires_context",
               "length_policy", "threshold")

    def __post_init__(self):
        _check_str(self.id, "pattern.id")
        if isinstance(self.version, bool) or not isinstance(self.version, int) \
                or self.version < 1:
            raise ValidationError("pattern.version must be a positive int")
        _check_str(self.definition, "pattern.definition")
        if self.operation not in OPERATIONS:
            raise ValidationError(
                f"pattern.operation must be one of {OPERATIONS}")
        self.threshold = _check_score(self.threshold, "pattern.threshold")

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "version": self.version,
            "definition": self.definition,
            "positive_examples": list(self.positive_examples),
            "negative_examples": list(self.negative_examples),
            "exclusions": list(self.exclusions),
            "operation": self.operation,
            "output_schema": dict(self.output_schema),
            "scoring": self.scoring,
            "requires_context": list(self.requires_context),
            "length_policy": self.length_policy,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# validate_response：响应与请求的自洽性检查（跨对象规则）。
# 结构级规则在 __post_init__；这里检查需要同时知道 req 和 resp 的规则。
# ---------------------------------------------------------------------------

def validate_response(req: MatchRequest, resp: MatchResponse) -> None:
    if resp.request_id is not None and resp.request_id != req.request_id:
        raise ValidationError(
            f"request_id mismatch: resp={resp.request_id!r} req={req.request_id!r}")

    # 证据必须可定位回原文（code point 半开区间，重复引文按 occurrence 定位）。
    for e in resp.evidence:
        if e.end > len(req.text):
            raise ValidationError(
                f"evidence span [{e.start},{e.end}) out of range "
                f"(text length {len(req.text)} code points)")
        actual = req.text[e.start:e.end]
        if actual != e.quote:
            raise ValidationError(
                "evidence.quote is not text[start:end]: "
                f"quote={e.quote!r} text[{e.start}:{e.end}]={actual!r}")

    # match 是单一判定：最多一条证据；find_all 允许多条 capture。
    if req.operation == "match" and len(resp.evidence) > 1:
        raise ValidationError(
            "operation=match allows at most 1 evidence item; "
            f"got {len(resp.evidence)}")

    # 固定 pattern 请求：provenance 必须回指同一模式版本，禁止版本漂移。
    if resp.provenance is not None and req.pattern is not None:
        if resp.provenance.pattern_version != req.pattern:
            raise ValidationError(
                f"pattern_version mismatch: provenance="
                f"{resp.provenance.pattern_version!r} request={req.pattern!r}")

    # adapter 一致性（R1）：
    # - 请求带 adapter 且 ok 时，provenance 必须回填同名 adapter——调用方
    #   必须能从响应确认自己用的是哪个微调头。
    # - 请求不带 adapter 时，provenance 不得虚报 adapter（防伪称用了微调头）。
    if resp.provenance is not None:
        if req.adapter is not None:
            if resp.provenance.adapter != req.adapter:
                raise ValidationError(
                    f"adapter mismatch: provenance="
                    f"{resp.provenance.adapter!r} request={req.adapter!r}")
        elif resp.provenance.adapter is not None:
            raise ValidationError(
                "provenance.adapter set but request has no adapter; "
                "do not claim a fine-tuned head was used")

    # 截断不得宣称全量抽取：find_all 被截断时必须弃权或标明 truncated，
    # 且不允许在截断的 find_all 上宣称"无捕获"这种全量结论。
    if (req.operation == "find_all" and resp.truncated
            and resp.status == "ok" and resp.matched is False):
        raise ValidationError(
            "truncated find_all must not claim a definitive 'no captures' "
            "negative; abstain instead")
    return None

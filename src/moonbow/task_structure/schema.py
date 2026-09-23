# -*- coding: utf-8 -*-
"""moonbow.task_structure.schema

任务结构的可观察符号定义。

核心立场：捕获项只声称"文本里出现了什么"，不声称完整刻画任务。
每个捕获项必须携带可定位的原文证据（quote 是 text 的 verbatim 子串）；
关系只记录文本显式表达者；无法确认时记入 unknowns 或弃权。
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import hashlib
import json

# 显式结构种类。order 固定，供向量与指纹使用。
GOAL = "goal"                       # 显式交付义务
OBJECT = "object"                   # 操作对象/作用范围
CONSTRAINT = "constraint"           # 附加约束
DEPENDENCY = "dependency"           # 显式先后/产物依赖
COORDINATION = "coordination"       # 显式跨对象一致性/协调要求
CONDITION = "condition"             # 影响执行路径的明确条件
UNRESOLVED = "unresolved"           # 显式待调查/待决定事项
ACCEPTANCE = "acceptance"           # 显式完成/验收条件

CAPTURE_KINDS = (
    GOAL, OBJECT, CONSTRAINT, DEPENDENCY,
    COORDINATION, CONDITION, UNRESOLVED, ACCEPTANCE,
)

# 结构计量的六元向量键（固定顺序）
VECTOR_KEYS = (
    "goals", "constraints", "explicit_dependencies",
    "coordinations", "conditions_unresolved", "objects",
)

# 低/中/高/信息不足
LEVEL_LOW = "low"
LEVEL_MEDIUM = "medium"
LEVEL_HIGH = "high"
LEVEL_UNKNOWN = "unknown"


@dataclass
class Capture:
    """单个结构捕获项。quote 必须是源文本的 verbatim 子串。"""
    kind: str                    # CAPTURE_KINDS 之一
    quote: str                   # 原文证据（可定位）
    span: Optional[Tuple[int, int]] = None   # quote 在源文本中的 [start, end)
    norm: str = ""               # 归一化键（程序计算，用于去重）

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "quote": self.quote}
        if self.span is not None:
            d["span"] = list(self.span)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Capture":
        span = d.get("span")
        return cls(kind=d.get("kind", ""), quote=d.get("quote", ""),
                   span=tuple(span) if span else None)


@dataclass
class Relation:
    """显式关系（仅记录文本明说者）。src/dst 指向 captures 下标。"""
    rtype: str                   # "depends_on" | "coordinates_with"
    src: int
    dst: int

    def to_dict(self) -> dict:
        return {"rtype": self.rtype, "src": self.src, "dst": self.dst}


@dataclass
class StructureVector:
    """可解释的结构计量向量（第一输出；等级由透明规则从它派生）。"""
    goals: int = 0
    constraints: int = 0
    explicit_dependencies: int = 0
    coordinations: int = 0
    conditions_unresolved: int = 0
    objects: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in VECTOR_KEYS}

    @classmethod
    def from_dict(cls, d: dict) -> "StructureVector":
        return cls(**{k: int(d.get(k, 0)) for k in VECTOR_KEYS})


@dataclass
class StructureAnalysis:
    """一次分析的完整结果。source/version 由程序管理，不由模型猜测。"""
    text: str
    source: str = "user"                 # user | plan | revision
    captures: List[Capture] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)   # 如 ["dependency"]
    abstain: bool = False                # 信息不足/边界不明 → 不评等级
    abstain_reason: Optional[str] = None
    truncated: bool = False              # 输入被截断 → 不得下"简单"结论
    backend: str = "rule"
    vector: Optional[StructureVector] = None
    level: str = LEVEL_UNKNOWN
    level_reasons: List[str] = field(default_factory=list)
    advice: Optional[str] = None         # 可选建议类型（非强制指令）
    version: int = 1                     # 同一任务被修订的版本号（程序管理）

    def to_dict(self) -> dict:
        return {
            "text_hash": hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16],
            "source": self.source,
            "captures": [c.to_dict() for c in self.captures],
            "relations": [r.to_dict() for r in self.relations],
            "unknowns": list(self.unknowns),
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "truncated": self.truncated,
            "backend": self.backend,
            "vector": self.vector.to_dict() if self.vector else None,
            "level": self.level,
            "level_reasons": list(self.level_reasons),
            "advice": self.advice,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def normalize_for_dedup(s: str) -> str:
    """去重用归一化：去空白/标点，英文小写。保留中文原字。"""
    out = []
    for ch in s:
        if ch.isspace():
            continue
        cat = ord(ch)
        # 中英文标点与符号统一剔除
        if ch in "，。、；：！？（）【】《》""''…—,. ; : ! ? ( ) [ ] { } \" '` * - _ / \\ | ~":
            continue
        if 0x2000 <= cat <= 0x206F:      # 通用标点
            continue
        out.append(ch.lower())
    return "".join(out)


def validate_captures(captures: List[Capture], text: str) -> List[Capture]:
    """程序校验：丢弃无法在原文中定位的捕获项与非法 kind。

    这是"引文级证据"底线的执行点：任何后端（规则或 SLM）产出的
    捕获都必须通过此校验才可进入计量。
    """
    kept: List[Capture] = []
    for c in captures:
        if c.kind not in CAPTURE_KINDS:
            continue
        if not c.quote or c.quote not in text:
            continue
        if c.span is None:
            c.span = (text.index(c.quote), text.index(c.quote) + len(c.quote))
        kept.append(c)
    return kept


def validate_relations(relations: List[Relation], n: int) -> List[Relation]:
    """关系引用完整性：下标越界/自引用的关系丢弃。"""
    kept = []
    for r in relations:
        if not (0 <= r.src < n and 0 <= r.dst < n) or r.src == r.dst:
            continue
        if r.rtype not in ("depends_on", "coordinates_with"):
            continue
        kept.append(r)
    return kept

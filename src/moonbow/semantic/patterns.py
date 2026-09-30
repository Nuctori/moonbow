# -*- coding: utf-8 -*-
"""moonbow.semantic.patterns

PatternSpec 注册表。固定 pattern 是版本化资产：
定义、正反例、排除项、operation、输出 schema、评分方式与阈值都随版本冻结。
拒绝"改几句话但继续沿用旧校准"。
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from moonbow.semantic.schema import (
    ValidationError, MatchRequest, PatternSpec, OPERATIONS,
)

# 已有 task_structure 结构集的 8 类捕获（docs/task_structure_spec.md §2）。
# ts.capture 的 kind 枚举必须与此一致；不一致视为内置资产损坏。
CAPTURE_KINDS = (
    "goal", "object", "constraint", "dependency",
    "coordination", "condition", "unresolved", "acceptance",
)

PATTERN_REF_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*@\d+$")

PATTERN_DIR = Path(__file__).resolve().parent.parent / "patterns"


def parse_pattern_ref(ref: str) -> "tuple[str, int]":
    """'name@N' -> (name, N)。格式非法即拒绝，不猜。"""
    if not isinstance(ref, str) or not PATTERN_REF_RE.match(ref):
        raise ValidationError(
            f"pattern ref must look like 'name@N' (got {ref!r})")
    name, _, ver = ref.rpartition("@")
    return name, int(ver)


class PatternRegistry:
    def __init__(self):
        self._patterns: Dict[str, PatternSpec] = {}   # key: ref "name@N"

    def register(self, spec: PatternSpec) -> None:
        if spec.ref in self._patterns:
            raise ValidationError(f"pattern already registered: {spec.ref}")
        self._patterns[spec.ref] = spec

    def get(self, ref: str) -> PatternSpec:
        """精确版本查找：同 id 不同版本不会命中，不静默回退到最新版。"""
        name, ver = parse_pattern_ref(ref)
        spec = self._patterns.get(ref)
        if spec is None:
            same_id = sorted(k for k in self._patterns
                             if k.startswith(name + "@"))
            raise ValidationError(
                f"unknown pattern version {ref!r}"
                + (f" (registered versions: {same_id})" if same_id else ""))
        return spec

    def requires(self, ref: str) -> bool:
        return ref in self._patterns

    def list_refs(self) -> List[str]:
        return sorted(self._patterns)


def _check_pattern_payload(payload: dict, source: str) -> None:
    for key in ("id", "version", "definition", "positive_examples",
                "negative_examples", "exclusions", "operation",
                "output_schema", "scoring"):
        if key not in payload:
            raise ValidationError(f"{source}: missing field {key!r}")
    if len(payload["positive_examples"]) < 3:
        raise ValidationError(
            f"{source}: need >=3 positive examples")
    if len(payload["negative_examples"]) < 3:
        raise ValidationError(
            f"{source}: need >=3 negative examples")
    if payload["operation"] not in OPERATIONS:
        raise ValidationError(
            f"{source}: operation must be one of {OPERATIONS}")
    if not isinstance(payload["output_schema"], dict):
        raise ValidationError(f"{source}: output_schema must be an object")
    # 空定义/空排除项列表无意义；排除项可以是空列表但必须是列表。
    if not payload["definition"].strip():
        raise ValidationError(f"{source}: definition must be non-empty")
    if not isinstance(payload["exclusions"], list):
        raise ValidationError(f"{source}: exclusions must be a list")


def spec_from_payload(payload: dict, source: str = "pattern") -> PatternSpec:
    _check_pattern_payload(payload, source)
    return PatternSpec(
        id=payload["id"],
        version=int(payload["version"]),
        definition=payload["definition"],
        positive_examples=list(payload["positive_examples"]),
        negative_examples=list(payload["negative_examples"]),
        exclusions=list(payload["exclusions"]),
        operation=payload["operation"],
        output_schema=dict(payload["output_schema"]),
        scoring=payload["scoring"],
        requires_context=list(payload.get("requires_context") or []),
        length_policy=payload.get("length_policy"),
        threshold=payload.get("threshold"),
    )


def load_pattern_file(path: Path) -> PatternSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValidationError(f"{path}: invalid JSON: {e}") from e
    spec = spec_from_payload(payload, source=str(path))
    if spec.ref != path.stem:
        raise ValidationError(
            f"{path}: file name {path.stem!r} must equal pattern ref {spec.ref!r}")
    return spec


def load_builtin_patterns(registry: Optional[PatternRegistry] = None) -> PatternRegistry:
    """加载 src/moonbow/patterns/*.json。ts.capture 的 kind 枚举
    必须与 task_structure 8 类一致。"""
    reg = registry if registry is not None else PatternRegistry()
    for path in sorted(PATTERN_DIR.glob("*.json")):
        reg.register(load_pattern_file(path))
    ts = reg.get("ts.capture@1")
    captures = ts.output_schema.get("captures")
    if isinstance(captures, list):
        embedded = next((c.get("kind_enum") for c in captures
                         if isinstance(c, dict) and "kind_enum" in c), None)
    elif isinstance(captures, dict):
        embedded = captures.get("kind_enum")
    else:
        embedded = None
    if embedded != list(CAPTURE_KINDS):
        raise ValidationError(
            "ts.capture kind_enum out of sync with task_structure "
            f"CAPTURE_KINDS: {embedded} != {list(CAPTURE_KINDS)}")
    return reg


def check_request_context(req: MatchRequest, spec: PatternSpec) -> None:
    """所需上下文缺失即拒绝，而不是让模型凭空判定。"""
    for key in spec.requires_context:
        v = (req.context or {}).get(key)
        if not isinstance(v, str) or not v.strip():
            raise ValidationError(
                f"pattern {spec.ref} requires context.{key!r} "
                f"(non-empty string); got {v!r}")

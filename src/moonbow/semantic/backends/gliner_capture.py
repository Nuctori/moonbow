# -*- coding: utf-8 -*-
"""moonbow.semantic.backends.gliner_capture

R2：GLiNER 作为捕获（find_all / ts.capture@1）backend，挂Matcher API 后。

决策背景（route 2）：
- 判定（match 语义）走 Qwen/openjev+LoRA；捕获（find_all / ts.capture@1）
  走 GLiNER 零样本，统一由 SemanticMatcher 门面对调用方暴露。
- 权重选型（2026-09-30 核对）：
  * 本地 models/gliner25_closure_* 系列是"闭合 3 类"微调检查点
    （Schema().classification("闭合", ...)），标签头与 ts.capture 8 类
    捕获枚举不匹配，不可映射 → 不采用；
  * HF 缓存已有 fastino/gliner2.5-multi-v1（BoundaryExtractor 架构，
    ~1.4GB，零样本多标签），以自然语言标签映射 8 类 → 采用。
  * 注意：pip 包 `gliner`（0.2.x）不识别该架构（需要 gliner_config.json）；
    正确的运行时是 `gliner2`（2.0.0，已在 .venv_xpu 内），AutoExtractor 加载。
- 标签映射表版本化：LABELS_V1，provenance.calibration_id="gliner:labels-v1"。
- 引文契约：quote 必须 text[start:end] verbatim（code point 半开区间）；
  模型给出的 span 先本地复核，坏引文丢弃并计数（dropped_quotes），
  末尾仍走 schema.validate_response 防御。
- 语义：低置信 → 少捕获（threshold 过滤）；零捕获是正常 ok（matched=false），
  绝不 abstain。仅支持 operation=find_all 且 pattern="ts.capture@1"；
  其他一律 status=error reason_code=unsupported。
- 设备：仅 XPU；torch.xpu 不可用时显式报错，禁止静默回退 CPU。
"""
from pathlib import Path
from typing import Any, Dict, Optional

from moonbow.semantic.backends.base import (
    Backend,
    BackendInfo,
    BackendUnavailable,
    error_response,
)
# 8 类枚举唯一真源在 patterns.py（与 task_structure 对齐）
from moonbow.semantic.patterns import CAPTURE_KINDS
from moonbow.semantic.schema import (
    Evidence,
    MatchRequest,
    MatchResponse,
    Provenance,
    validate_response,
)

SUPPORTED_PATTERN = "ts.capture@1"

# 标签映射表 v1：8 类捕获 → 自然语言捕获标签（零样本，未经校准）。
# 版本演进规则：改任何标签文本 = 升 labels-v2，并重跑冒烟/校准。
LABELS_V1: Dict[str, str] = {
    "goal": "交付义务（要实现/修复/添加的交付物）",
    "object": "操作对象（被操作/修改/创建的实体、文件或产物）",
    "constraint": "附加约束（必须保持/不能破坏的条件）",
    "dependency": "显式依赖（先后顺序/产物依赖）",
    "coordination": "协调要求（多处保持一致/同步）",
    "condition": "条件分支（如果…则…）",
    "unresolved": "待决定事项（待定/待确认）",
    "acceptance": "验收条件（完成的标准）",
}
LABELS_VERSION = "gliner:labels-v1"

DEFAULT_MODEL_PATH = (
    r"C:\Users\Nuctori\.cache\huggingface\hub"
    r"\models--fastino--gliner2.5-multi-v1"
    r"\snapshots\aaecfe45db1d828c963717054ccb868e8ad1f1d5"
)
DEFAULT_THRESHOLD = 0.05  # 零样本 boundary head 的可用工作点（诊断值）


def _derive_revision(model_path: str) -> str:
    """钉死 model_revision：HF snapshots 目录名即 commit id；
    其他本地目录退化为 config.json 内容 sha1 前 12 位。"""
    p = Path(model_path)
    if p.parent.name == "snapshots":
        return "fastino/gliner2.5-multi-v1@" + p.name
    import hashlib
    cfg = p / "config.json"
    try:
        h = hashlib.sha256(cfg.read_bytes()).hexdigest()[:12]
    except OSError:
        return f"{p.name}@unhashed"
    return f"{p.name}@sha256cfg:{h}"


class GlinerCaptureBackend(Backend):
    """GLiNER 捕获 backend：只做 ts.capture@1 find_all，只做推理不做决策。"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 device: str = "xpu", threshold: float = DEFAULT_THRESHOLD,
                 labels: Optional[Dict[str, str]] = None,
                 labels_version: str = LABELS_VERSION):
        if sorted(labels or LABELS_V1) != sorted(CAPTURE_KINDS):
            raise ValueError(
                f"labels must cover exactly {list(CAPTURE_KINDS)}")
        self.model_path = model_path
        self.device = device
        self.default_threshold = float(threshold)
        self.labels = dict(labels or LABELS_V1)
        self._labels_version = labels_version
        self._label_to_kind = {v: k for k, v in self.labels.items()}
        # 坏引文计数（丢弃原因：offset 不落原文 / 引文错位 / 空白）
        self.dropped_quotes = 0
        self.total_spans = 0
        super().__init__(BackendInfo(
            name="gliner-capture:" + Path(model_path).name,
            model_revision=_derive_revision(model_path),
            fake=False,
            operations=("find_all",),
            extra={
                "supported_patterns": [SUPPORTED_PATTERN],
                "labels_version": self._labels_version,
                "runtime": "gliner2",
                "device": device,
            },
        ))

    # -- 生命周期 -----------------------------------------------------------

    def _load(self) -> None:
        import torch
        if self.device == "xpu" and not torch.xpu.is_available():
            # 硬约束：禁止静默回退 CPU。
            raise BackendUnavailable(
                "GlinerCaptureBackend requires XPU (torch.xpu.is_available()"
                "=False); refusing CPU fallback")
        try:
            from gliner2 import AutoExtractor, Schema
        except ImportError as e:
            raise BackendUnavailable(
                "GlinerCaptureBackend needs the gliner2 package") from e
        self._extractor = AutoExtractor.from_pretrained(
            self.model_path, map_location=self.device)
        self._schema = Schema().entities(
            {label: {"threshold": self.default_threshold}
             for label in self.labels.values()})

    def close(self) -> None:
        self._extractor = None
        self._schema = None

    # -- 推理 ---------------------------------------------------------------

    def match(self, request: MatchRequest) -> MatchResponse:
        if request.operation != "find_all" \
                or request.pattern != SUPPORTED_PATTERN:
            return error_response(request, "unsupported")
        try:
            items = self._extract(request.text)
        except BackendUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 — 瞬态推理故障上抛给 runtime 重试
            raise BackendUnavailable(f"gliner inference failed: {e}") from e

        evidence = []
        for ent in items:
            self.total_spans += 1
            ev = self._to_evidence(request.text, ent)
            if ev is not None:
                evidence.append(ev)

        resp = MatchResponse(
            status="ok",
            matched=bool(evidence),
            score=None,
            score_type=None,
            calibrated=False,
            evidence=evidence,
            provenance=Provenance(
                backend="gliner",
                model_revision=self.model_revision,
                pattern_version=SUPPORTED_PATTERN,
                calibrated=False,
                calibration_id=self._labels_version,
            ),
            request_id=request.request_id,
        )
        try:
            validate_response(request, resp)
        except Exception:
            # 防御兜底：任何契约违例不得外泄为可信命中。
            return error_response(request, "invalid_output")
        return resp

    # -- 内部 ---------------------------------------------------------------

    def _extract(self, text: str):
        """跑一次零样本抽取，返回扁平 entity 列表（含 start/end/confidence）。
        空/None 一律归一化为 []（零捕获是正常负例）。"""
        out = self._extractor.extract(text, self._schema,
                                      include_spans=True,
                                      include_confidence=True)
        entities = (out or {}).get("entities") or {}
        flat = []
        if isinstance(entities, dict):
            for label, items in entities.items():
                kind = self._label_to_kind.get(label)
                for ent in items or []:
                    if isinstance(ent, dict):
                        flat.append({**ent, "label_kind": kind})
        else:
            # 替身/测试路径：已带 label_kind 的扁平列表
            for ent in entities:
                if isinstance(ent, dict):
                    flat.append(dict(ent))
        return flat

    def _to_evidence(self, text: str, ent: Dict[str, Any]) -> Optional[Evidence]:
        """模型 span → 契约 Evidence。坏引文丢弃并计数，绝不修饰成命中。"""
        kind = ent.get("label_kind")
        quote = ent.get("text")
        start, end = ent.get("start"), ent.get("end")
        if kind not in CAPTURE_KINDS or not quote:
            self.dropped_quotes += 1
            return None
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(text)):
            self.dropped_quotes += 1
            return None
        if text[start:end] != quote:
            self.dropped_quotes += 1
            return None
        return Evidence(quote=quote, start=start, end=end,
                        relation="supports")


# ---------------------------------------------------------------------------
# Matcher 路由（R2）：显式、按 (operation, pattern) 注册；默认无路由，
# matcher 行为完全不变。调用方仍只调 matcher.find_all(pattern="ts.capture@1")。
# ---------------------------------------------------------------------------

def register_capture_route(factory) -> None:
    """把本 backend 工厂注册进 matcher 的显式路由表：
    find_all + ts.capture@1 → gliner；match → slm（不受影响）。
    factory: () -> Backend（延迟调用，首次命中路由时才加载权重）。"""
    from moonbow.semantic import matcher
    matcher.register_pattern_backend("find_all", SUPPORTED_PATTERN, factory)


def unregister_capture_route() -> None:
    from moonbow.semantic import matcher
    matcher.unregister_pattern_backend("find_all", SUPPORTED_PATTERN)

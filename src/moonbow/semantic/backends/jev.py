# -*- coding: utf-8 -*-
"""moonbow.semantic.backends.jev

NeoHorse-Jev-4B 判定 backend（计划 §2.2 可替换后端的换装验证，2026-09-30）。

模型：TokenRhythm/NeoHorse-Jev-4B（Apache-2.0，4B 非自回归决策模型，
prefill-only 单次前向，直接在应用定义的候选答案上输出概率分布；
revision 434cb21d3a994a953d3ae5788405fcb2c4970554）。
加载：官方 wheel `neohorse_decision` 1.0.0（随模型仓库 dist/ 分发，
`pip install --no-deps` 装入 .venv_xpu，未触碰 torch/transformers 依赖）；
`DecisionEngine(model_dir, device=...)` 官方 API 原生接受 device 参数，
本 backend 传 device="xpu"，不回退 CPU。

设计立场（与 slm.py 对齐）：
- 受限判定、零生成：模型不在词表上自由解码，输出即候选答案上的 softmax
  概率（决策头恒输出分布）。score = 该概率，score_type 如实标注
  （noul_probability / choice_probability）；本模型不存在"拿不到概率"的
  路径，若概率缺失/非法按 invalid_output 拒绝，不编造分数。
- 防注入：用户文本只进 state 槽位（数据区）；官方编码器在 tokenize 前把
  特殊定界 token `<|name|>` 重写为 `<¦name¦>`（_vendor.model.user_tokens，
  option 边界不可伪造）。instructions/criteria 全部来自版本化模板常量，
  用户文本永不进入指令/候选槽位。
- 模板版本化：JEV_TEMPLATE_VERSION = "jev-template-v1"，经
  provenance.calibration_id = "template:jev-template-v1" 记录。
- XPU 强制（用户硬约束）：真实权重只加载 torch.device("xpu")；
  `torch.xpu.is_available()` 为 False 时抛 BackendUnavailable 拒绝运行。
- ts.capture 不支持：结构抽取是生成式任务，非该模型能力（prefill-only
  无自由解码）→ 明确 abstain unsupported。
- task.object.alignment：Noul 判定，state = task 拼装模板
  （"Task description:\n{task}\n\nCandidate text:\n{text}"）；
  缺 context.task → abstain insufficient_context（契约与 slm/openjev 一致）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from moonbow.semantic.backends.base import (
    Backend, BackendInfo, BackendInvalidOutput, BackendOverloaded,
    BackendUnavailable,
)
from moonbow.semantic.backends.slm import (
    _clip, _require_xpu, _resolve_revision, _rss_mb,
)
from moonbow.semantic.schema import (
    MatchRequest, MatchResponse, Provenance, validate_response,
)

BACKEND_NAME = "jev"
JEV_TEMPLATE_VERSION = "jev-template-v1"
DEFAULT_MODEL_ID = "TokenRhythm/NeoHorse-Jev-4B"
DEFAULT_MODEL_DIR = (
    "C:/Users/Nuctori/.cache/huggingface/hub/"
    "models--TokenRhythm--NeoHorse-Jev-4B/snapshots/"
    "434cb21d3a994a953d3ae5788405fcb2c4970554")

# 单 pattern 输入窗口（code point），与 slm._TEXT_WINDOW 口径一致：
# 超窗截断并在响应置 truncated=true。2048 code point 在任何语言下
# 均 < 4096 token，state 预算（max_state=4096）不会触顶。
_TEXT_WINDOW = 2048

# DecisionEngine 预算（default max_state=2048 偏紧，放大以保证
# 2048 code point 窗口内绝不触发 strict ValueError）。
_MAX_STATE = 4096
_MAX_BRANCH = 8192
_MAX_TOKENS = 65536

# ---------------------------------------------------------------------------
# 候选答案集（版本化：jev-template-v1）。
# 每模式定义 instructions 与候选（noul=false/true 两候选；choice=命名候选集）。
# score = 模型在该候选集上输出的 softmax 概率。
# ---------------------------------------------------------------------------

_JEV_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "completion.asserted": {
        "mode": "noul",
        "instructions": (
            "Has the author of the text asserted that the work is fully "
            "complete? Requests, plans and step lists are not completion "
            "statements."),
        "criteria": {
            "false": "The text does not assert that the work is fully complete.",
            "true": "The text asserts that the work is fully complete.",
        },
    },
    "process.unresolved": {
        "mode": "noul",
        "instructions": (
            "Does the text contain an explicit unresolved issue the author "
            "marks as open - something unknown, undecided, or to be "
            "confirmed? Ordered plans and plain conditions are not "
            "unresolved issues."),
        "criteria": {
            "false": "The text contains no explicit unresolved issue.",
            "true": "The text contains an explicit unresolved issue.",
        },
    },
    "modality.assertive": {
        "mode": "choice",
        "instructions": (
            "Classify the modality of the text: assert = states a fact as "
            "already true; promise = describes planned, requested or future "
            "work; question = asks something. Choose exactly one label."),
        "criteria": {
            "assert": "The text states a fact as already true (assertive).",
            "promise": "The text describes planned or requested future work.",
            "question": "The text asks a question.",
        },
        "positive_label": "assert",
    },
    "task.object.alignment": {
        "mode": "noul",
        "state_template": (
            "Task description:\n{task}\n\nCandidate text:\n{text}"),
        "instructions": (
            "Does the candidate text describe work that falls within the "
            "goal scope of the task described in the state? Judge only "
            "whether the text's content is on-task, not whether the work "
            "is done."),
        "criteria": {
            "false": "The candidate text is off-task for the described task.",
            "true": "The candidate text describes work within the task's goal scope.",
        },
    },
}

_AVAILABLE_PATTERNS = tuple(sorted(_JEV_TEMPLATES))


def _state_for(pattern: str, text: str, task: Optional[str]) -> str:
    """state 槽位拼装：普通模式 state=原文；alignment 用 task 拼装模板。"""
    if pattern == "task.object.alignment":
        return _JEV_TEMPLATES[pattern]["state_template"].format(
            task=task, text=text)
    return text


class JevBackend(Backend):
    """NeoHorse-Jev-4B 语义匹配 backend（官方 neohorse_decision 包加载）。

    实现 P2 Backend 协议：构造即加载一次（initialize() 幂等），match()
    产物必过 validate_response。测试可注入 `engine`（duck-typed 句柄，
    需有 .predict(request)->{"answers": {key: {...}}}）以脱离权重运行。
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: Optional[str] = None,
        device: str = "xpu",
        engine: Any = None,
    ):
        self.model_dir = model_dir
        self.model_id = model_id
        self.device = device
        # revision 解析：显式指定优先；否则读 HF 缓存 refs/main（离线）。
        self._revision = model_revision or _resolve_revision(model_id, None)
        super().__init__(BackendInfo(
            name=BACKEND_NAME, model_revision=self._revision, fake=False))
        self.load_info: Optional[Dict[str, Any]] = None
        self.diagnostics: List[Dict[str, Any]] = []
        if engine is not None:
            self.engine = engine
            self._initialized = True  # 注入路径不重复加载
            self.load_count = 1
        else:
            self._load()
            self._initialized = True
            self.load_count = 1

    # -- Backend 协议：加载 ---------------------------------------------------

    def _load(self) -> None:
        """真实加载：XPU 强制；官方 DecisionEngine(device='xpu')。"""
        _require_xpu()
        if self.device != "xpu":
            raise BackendUnavailable(
                f"JevBackend requires device='xpu' (got {self.device!r}); "
                "CPU inference is forbidden by project hard constraint")
        if not Path(self.model_dir).is_dir():
            raise BackendUnavailable(
                f"model dir not found: {self.model_dir!r}")
        t0 = time.time()
        rss_before = _rss_mb()
        try:
            from neohorse_decision import DecisionEngine
        except ImportError as e:
            raise BackendUnavailable(
                "neohorse_decision is not installed; install the wheel "
                "bundled in the model repo with --no-deps") from e
        try:
            self.engine = DecisionEngine(
                self.model_dir, device="xpu",
                max_state=_MAX_STATE, max_branch=_MAX_BRANCH,
                max_tokens=_MAX_TOKENS)
        except Exception as e:
            # 加载失败（显存不足/文件缺失/断言失败）如实上报，不降级 CPU。
            raise BackendUnavailable(
                f"DecisionEngine load failed on xpu: {e}") from e
        import torch
        free, total = torch.xpu.mem_get_info()
        self.load_info = {
            "model_id": self.model_id,
            "model_revision": self._revision,
            "device": "xpu",
            "dtype": "bfloat16 (backbone) + float32 (pointer head)",
            "load_seconds": round(time.time() - t0, 2),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(_rss_mb(), 1),
            "vram_used_gb": round((total - free) / 2**30, 2),
            "vram_total_gb": round(total / 2**30, 2),
        }
        self._info.model_revision = self._revision

    # -- 能力与身份 -----------------------------------------------------------

    @property
    def revision(self) -> str:
        return self._revision

    def capabilities(self) -> Dict[str, Any]:
        return {
            "backend": BACKEND_NAME,
            "model_id": self.model_id,
            "model_revision": self._revision,
            "device": self.device,
            "template_version": JEV_TEMPLATE_VERSION,
            "patterns": [
                {"ref": f"{p}@1", "operation": "match",
                 "decision_type": t["mode"],
                 "scoring": f"{t['mode']}_probability"}
                for p, t in _JEV_TEMPLATES.items()
            ],
            "unsupported": ["ts.capture@1（生成式抽取，prefill-only 模型不支持）"],
            "injection_defense": (
                "user text only in state slot; special delimiter tokens "
                "neutralized by official tokenizer (user_tokens); "
                "instructions/criteria from versioned template constants"),
            "calibrated": False,
            "score_semantics": "candidate_probability（softmax over template-defined answers）",
        }

    # -- 响应组装 -------------------------------------------------------------

    def _provenance(self, pattern_ref: str) -> Provenance:
        return Provenance(
            backend=BACKEND_NAME,
            model_revision=self._revision,
            pattern_version=pattern_ref,
            calibrated=False,
            calibration_id=f"template:{JEV_TEMPLATE_VERSION}",
        )

    def _finish(self, req: MatchRequest, resp: MatchResponse) -> MatchResponse:
        """自校验：任何后端产物必须先过 P1 validate_response 再出口。"""
        validate_response(req, resp)
        return resp

    # -- 主入口 ---------------------------------------------------------------

    def match(self, request: MatchRequest) -> MatchResponse:
        name = request.pattern.rpartition("@")[0]
        if name not in _JEV_TEMPLATES:
            # 含 ts.capture@1：该模型无自由解码能力，明确不支持。
            return MatchResponse(status="abstain", reason_code="unsupported",
                                 request_id=request.request_id)
        if name == "task.object.alignment":
            task = (request.context or {}).get("task")
            if not isinstance(task, str) or not task.strip():
                return MatchResponse(
                    status="abstain", reason_code="insufficient_context",
                    request_id=request.request_id)
        else:
            task = None
        return self._match_with_template(request, name, task)

    def _match_with_template(self, req: MatchRequest, name: str,
                             task: Optional[str]) -> MatchResponse:
        tpl = _JEV_TEMPLATES[name]
        text, clipped = _clip(req.text, _TEXT_WINDOW)
        state = _state_for(name, text, task)
        qkey = "q"
        question: Dict[str, Any] = {
            "type": tpl["mode"],
            "instructions": tpl["instructions"],
            "criteria": dict(tpl["criteria"]),
        }
        payload = {
            "model": "neohorse-jev",
            "state": state,
            "questions": {qkey: question},
        }
        t0 = time.perf_counter()
        try:
            result = self.engine.predict(payload)
        except Exception as e:
            from neohorse_decision.engine import BusyError
            if isinstance(e, BusyError):
                raise BackendOverloaded("jev engine busy") from e
            raise BackendInvalidOutput(
                f"jev predict failed: {e}") from e
        dt_ms = (time.perf_counter() - t0) * 1000

        answer = (result or {}).get("answers", {}).get(qkey)
        if not isinstance(answer, dict):
            raise BackendInvalidOutput("jev answer missing")
        score, probs = self._extract_probability(name, tpl, answer)
        threshold = req.threshold if req.threshold is not None else 0.5
        if tpl["mode"] == "choice":
            pred = max(probs, key=probs.get)
            matched = pred == tpl["positive_label"] and \
                probs[tpl["positive_label"]] >= threshold
        else:
            matched = score >= threshold
        self.diagnostics.append({
            "pattern": req.pattern, "kind": tpl["mode"],
            "score": score, "probabilities": probs,
            "threshold": threshold, "template_version": JEV_TEMPLATE_VERSION,
        })
        return self._finish(req, MatchResponse(
            status="ok", matched=bool(matched), score=score,
            score_type=f"{tpl['mode']}_probability", calibrated=False,
            truncated=clipped,
            provenance=self._provenance(req.pattern),
            request_id=req.request_id))

    @staticmethod
    def _extract_probability(
            name: str, tpl: Dict[str, Any],
            answer: Dict[str, Any]) -> "tuple[float, Dict[str, float]]":
        """从模型答案提取分数与完整候选分布；结构非法一律 invalid_output，
        不编造分数。noul: score=P(true)；choice: score=P(positive_label)。"""
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            raise BackendInvalidOutput(
                f"jev probabilities missing for {name}")
        clean: Dict[str, float] = {}
        for k, v in probs.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise BackendInvalidOutput(
                    f"jev probability not numeric: {k}={v!r}")
            fv = float(v)
            if fv < 0.0 or fv > 1.0:
                raise BackendInvalidOutput(
                    f"jev probability out of range: {k}={fv}")
            clean[str(k)] = fv
        if tpl["mode"] == "choice":
            label = tpl["positive_label"]
            if label not in clean:
                raise BackendInvalidOutput(
                    f"jev choice probabilities lack {label!r}: {sorted(clean)}")
            return clean[label], clean
        # noul：score=P(true)
        if "true" not in clean:
            raise BackendInvalidOutput(
                f"jev noul probabilities lack 'true': {sorted(clean)}")
        return clean["true"], clean

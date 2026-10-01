# -*- coding: utf-8 -*-
"""progress_guard.semantic_provider

P6 Guard 迁移：语义信号的 provider 抽象（计划 §4-P6，默认仍 legacy）。

设计：
- GuardSemanticProvider 协议：观察者接口。provider 只产出「语义观察」
  （SemanticObservation），阈值比较、信号文案、advisory/strict 区分、
  硬信号升级等决策逻辑全部留在 verifier，保证 legacy 路径逐字节等价。
- LegacyProvider：包装现有 ModelRegistry（get_modality / get_similarity /
  get_capture_prob），输出结构与迁移前完全一致；它不吞异常——模型推理
  失败仍然向上抛（与现状一致，eager 下即 500）。
- SemanticProviderAdapter：把 P2 SemanticMatcher（或其 HTTP client backend
  的 matcher 门面）适配为同一接口。每个信号对应一次 matcher.match(pattern=...)；
  SLM 分数只用新命名键（slm_capture_score@1），绝不写入旧 similarity /
  capture_prob 键冒充等价。适配器自带降级：任何 matcher 异常或契约内
  error 响应都会让 observe() 返回 unavailable 标记，由 verifier 明确进入
  既有骨架路径并写 scores["semantic_unavailable"]=true，不伪造通过。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("progress_guard.semantic_provider")

# Legacy 微模型判定出的模态类别（与 guard.models.MOD_CLASSES 对齐）
MODALITY_ASSERT = "assert"
MODALITY_PROMISE = "promise"
MODALITY_QUESTION = "question"

# ---------------------------------------------------------------------------
# R4：信号 -> (pattern, LoRA adapter) 映射与达标状态表。
# 依据 results/semantic-runtime/r3-lora/report.md（2/4 达标）：
# - process.unresolved / task.object.alignment：LoRA 达标，请求带 adapter；
# - completion.asserted：接近但差线（no-go），适配器仅注册备用，声明 uncalibrated；
# - modality.assertive：LoRA 训练打塌排序（如实 no-go），维持生成式/NLI 底座
#   zero-shot，不挂适配器、不假装达标。
# 达标判定均来自 R3 calib 150 条（provisional 标签）；test split 未跑。
# ---------------------------------------------------------------------------
R4_SIGNAL_PATTERNS = {
    "process.unresolved": "process.unresolved@1",
    "task.object.alignment": "task.object.alignment@1",
}
R4_SIGNAL_ADAPTERS = {
    "process.unresolved": "process-lora-v2",
    "task.object.alignment": "alignment-lora-v1",
}
R4_CAPABILITY_STATUS = {
    "process.unresolved": {
        "status": "pass-preliminary",
        "adapter": "process-lora-v2",
        "calibration": ("r6b holdout_v2（120 条，固定 calib 阈值 t=0.90 单次前向，"
                        "唯一验收口径）：P=0.8571 (CI95 0.7216-0.9328) "
                        "R=0.9474 (CI95 0.8271-0.9854) F1=0.900 → 预登记门禁 "
                        "P>=0.85 & R>=0.80 PASS，且同口径优于 v1 基线 "
                        "(P=0.6486 R=0.6316 F1=0.640)。calib 150 扫描 "
                        "P=0.8947 R=0.8293 F1=0.8608 @t=0.90。preliminary："
                        "holdout_v2 标签 provisional（agent 自审），P 贴线"
                        "（FP 6 条中 5 条为修辞性反问边界）；v1 仅 shadow 历史 "
                        "(final-test R=0.764 差线)，已被 v2 取代"),
    },
    "task.object.alignment": {
        "status": "pass",
        "adapter": "alignment-lora-v1",
        "calibration": ("r3 calib P=0.868 R=0.868 F1=0.868 @t=0.75；"
                        "final-test P=0.854 R=0.972 F1=0.909 → PASS"),
    },
    "completion.asserted": {
        "status": "pass",
        "adapter": "completion-lora-v2",
        "calibration": ("r3b calib P=0.9615 R=0.8333 F1=0.8929 @t=0.50 "
                        "(v2：训练数据 = dev 300 + dev_extension_r4 120，"
                        "ECE=0.0402 Brier=0.0401)；"
                        "final-test P=0.895 R=0.810 F1=0.850 → PASS；"
                        "v1 r3 为 P=0.846 R=0.733 差线 no-go，已被 v2 取代"),
    },
    "modality.assertive": {
        "status": "no_go",
        "adapter": None,
        "mode": "base zero-shot（无适配器，uncalibrated）",
        "calibration": "r3 calib scan P=0.406 R=0.981 @t=0.70 —— LoRA 无改善",
    },
}

# ---------------------------------------------------------------------------
# R5：白名单语义落地（依据 results/semantic-runtime/final-test/final_acceptance.md
# §4/§6 与 AGENTS.md 约束 0）；R6b 起 process.unresolved 以 v2 适配器
# 参与判定（preliminary 口径）。
# - PASS 信号（completion.asserted → completion-lora-v2、
#   task.object.alignment → alignment-lora-v1、process.unresolved →
#   process-lora-v2）正常参与判定；
# - modality.assertive 维持底座 zero-shot 并标记 uncalibrated（信号文案保持
#   现有语义，不计入达标声明）。
# ---------------------------------------------------------------------------
R5_PASS_SIGNALS = {
    "completion.asserted": {
        "pattern": "completion.asserted@1",
        "adapter": "completion-lora-v2",
        "role": "capture",           # 完全闭合判定（替代 legacy capture 通道）
    },
    "task.object.alignment": {
        "pattern": "task.object.alignment@1",
        "adapter": "alignment-lora-v1",
        "role": "alignment",         # 客体对齐判定
    },
    # R6b：process.unresolved v2 过 holdout_v2 预登记门禁（P=0.8571 R=0.9474
    # @t=0.90，且优于 v1 同口径基线）→ shadow 升 pass-preliminary，参与判定；
    # preliminary = holdout_v2 标签 provisional、P 贴线（详见
    # results/semantic-runtime/r6b-process-v2/report.md）。
    "process.unresolved": {
        "pattern": "process.unresolved@1",
        "adapter": "process-lora-v2",
        "role": "unresolved",        # 未决事项判定（preliminary 口径）
    },
}
# R6b：process.unresolved 已升 pass-preliminary（R5_PASS_SIGNALS），shadow 清空；
# 如后续回滚 = 移回本表 + config 改回 process-lora-v1。
R5_SHADOW_SIGNALS = {}
# R5 模态 pattern：底座 zero-shot（no-go，不挂适配器，如实标记 uncalibrated）
R5_MODALITY_PATTERN = "modality.assertive@1"
# shadow 结果在 observation / scores 中的专用键（绝不参与信号判定）
SHADOW_SIGNALS_KEY = "shadow_signals"
# modality uncalibrated 标记键（R5 白名单构造时写入 scores）
MODALITY_UNCALIBRATED_KEY = "semantic_modality_uncalibrated"
# 环境变量与默认配置（默认切换；缺省 legacy，行为零变化）
PROVIDER_ENV = "MOONBOW_GUARD_PROVIDER"
SEMANTIC_CONFIG_ENV = "MOONBOW_GUARD_SEMANTIC_CONFIG"
DEFAULT_SEMANTIC_CONFIG = os.path.join("config", "semantic_runtime_lora.json")

# 语义 provider 的固定 pattern 引用（新命名 + 版本，走 P1 PatternSpec 命名空间）
DEFAULT_MODALITY_PATTERN = "completion.modality@1"
DEFAULT_CAPTURE_PATTERN = "completion.capture@1"
# SLM 分数的新命名键：绝不写进旧 similarity / capture_prob 键
SLM_CAPTURE_SCORE_KEY = "slm_capture_score@1"


@dataclass
class SemanticObservation:
    """一次语义观察的原始结果。None 表示该信号不可用（不伪造）。"""
    modality: Optional[str] = None            # assert / promise / question
    similarity: Optional[float] = None        # legacy 余弦相似度；SLM 恒为 None
    capture_prob: Optional[float] = None      # legacy 捕获概率；SLM 恒为 None
    # 语义侧「收尾陈述被判定为完全闭合」的布尔判定（SLM 路径；legacy 不用）。
    # True / False 是明确判定；None 是未知（不判定、不触发信号）。
    capture_complete: Optional[bool] = None
    # 新命名分数键（如 slm_capture_score@1），由 verifier 原样并入 scores。
    extra_scores: Dict[str, Any] = field(default_factory=dict)
    # 非 None 表示 provider 失败/不可用：verifier 进入既有骨架降级并标记
    # semantic_unavailable=true，绝不据此产出任何语义信号。
    unavailable_reason: Optional[str] = None
    # R5：客体对齐判定（PASS 信号 task.object.alignment）。True/False 明确
    # 判定；None = 该信号未启用（legacy / 未配置白名单），不产生信号。
    alignment_ok: Optional[bool] = None
    # R6b：未决事项判定（PASS 信号 process.unresolved，preliminary 口径）。
    # True = 陈述中存在未解决的疑点/待办（verifier 产出软信号）；None = 未启用。
    unresolved_detected: Optional[bool] = None
    # R5：shadow 信号观察（如 process.unresolved）。只随 observation 透传、
    # 由 verifier 原样写入 scores[shadow_signals]；绝不参与信号判定与提醒，
    # 其自身失败也只记录 unavailable_reason，不拖垮整体判定。
    shadow_signals: List[Dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class GuardSemanticProvider(Protocol):
    """语义信号 provider 协议（结构化，便于测试替身）。"""
    kind: str

    def observe(self, req: str, statement: str, want_capture: bool) -> SemanticObservation:
        """对收尾陈述做一次语义观察。

        req: 用户原始需求；statement: 排除清单声明头后的陈述文本。
        want_capture: 是否需要「完全闭合」判定（由 verifier 依 STATUS/REMAINING 决定）。
        """
        ...

    def describe(self) -> Dict[str, Any]:
        """只读状态：provider_kind / backend / calibrated（/v1/semantic-status 用）。"""
        ...


class LegacyProvider:
    """包装现有 ModelRegistry；legacy 默认路径，零行为变化。

    注意：本类不吞异常。registry 推理抛错时 observe 原样上抛，
    与迁移前「模型推理失败 -> check 失败」的行为完全一致。
    """

    kind = "legacy"

    def __init__(self, registry):
        self._registry = registry

    def observe(self, req: str, statement: str, want_capture: bool) -> SemanticObservation:
        # 调用顺序与迁移前一致：模态 -> 相似度 -> （可选）捕获概率
        modality = self._registry.get_modality(statement)
        similarity = self._registry.get_similarity(req, statement)
        capture_prob = None
        if want_capture:
            capture_prob = self._registry.get_capture_prob(statement)
        return SemanticObservation(
            modality=modality,
            similarity=similarity,
            capture_prob=capture_prob,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "provider_kind": "legacy",
            "backend": "minilm_slot_heads_v2+capture_head_v1",
            "calibrated": False,
        }


class SemanticProviderAdapter:
    """把 P2 SemanticMatcher（或 HTTP client 的同接口门面）适配为 provider。

    - 每个信号对应一次 matcher.match(pattern=...)：模态一次、捕获一次。
    - SLM 不产生余弦相似度：similarity 恒为 None，旧 similarity 键不写、
      不伪造；捕获分数只写入新键 slm_capture_score@1。
    - 降级语义：matcher 抛错或返回契约内 error / matched=None 时，
      observe 返回 unavailable 标记（不部分产出信号），由 verifier 进入
      既有骨架路径并标记 semantic_unavailable=true。
    """

    kind = "semantic"

    def __init__(
        self,
        matcher,
        modality_pattern: str = DEFAULT_MODALITY_PATTERN,
        capture_pattern: str = DEFAULT_CAPTURE_PATTERN,
        modality_mapping: Optional[Dict[bool, str]] = None,
        signal_patterns: Optional[Dict[str, str]] = None,
        signal_adapters: Optional[Dict[str, str]] = None,
        capability_status: Optional[Dict[str, Any]] = None,
        pass_signals: Optional[Dict[str, Dict[str, str]]] = None,
        shadow_signals: Optional[Dict[str, Dict[str, str]]] = None,
        modality_uncalibrated: bool = False,
    ):
        self._matcher = matcher
        self._modality_pattern = modality_pattern
        self._capture_pattern = capture_pattern
        # R4：信号 -> adapter 映射（按信号选适配器）。None/空 = 不启用额外
        # 信号（既有 modality/capture 行为零变化）；启用时每个信号一次
        # matcher.match(pattern=..., adapter=...)，请求显式带 adapter 参数。
        self._signal_patterns = dict(signal_patterns or R4_SIGNAL_PATTERNS)
        self._signal_adapters = dict(signal_adapters or {})
        # 信号达标状态表（/v1/semantic-status 的 capabilities）。None 时不
        # 声明（保持旧 describe 形状）；显式传入时逐信号如实声明
        # pass / uncalibrated / no_go，禁止把 no-go 写成达标。
        self._capability_status = capability_status
        # R5：白名单。pass_signals：PASS 信号（pattern/adapter/role）正常
        # 参与判定；shadow_signals：仅 shadow（结果只进 shadow_signals 键，
        # 失败也不拖垮整体）。两者为 None 时 R5 路径完全不启用，既有
        # R4 行为零变化。
        self._pass_signals = dict(pass_signals or {})
        self._shadow_signals = dict(shadow_signals or {})
        # R5：modality 走底座 zero-shot 时如实标记 uncalibrated（不计入
        # 达标声明；信号文案语义不变）。
        self._modality_uncalibrated = bool(modality_uncalibrated)
        # matched 布尔 -> 模态类别。默认：命中事实断言模式 -> assert，
        # 未命中按 promise（promise/question 的细分留给 pattern 定义，见 §3.3）。
        self._modality_mapping = modality_mapping or {
            True: MODALITY_ASSERT,
            False: MODALITY_PROMISE,
        }

    def observe_signal(self, signal: str, req: str, statement: str
                       ) -> Optional[Dict[str, Any]]:
        """按信号名选 adapter 跑一次语义判定（R4）。

        返回 {"matched": bool, "score": float|None, "adapter": str}；
        信号未启用返回 None；matcher 失败/契约内失败/adapter 回填不一致
        抛 RuntimeError（由 observe 统一按不可用降级，不产出半套信号）。        """
        adapter = self._signal_adapters.get(signal)
        pattern = self._signal_patterns.get(signal)
        cfg = self._pass_signals.get(signal)
        if cfg is not None:
            # R5 白名单信号：pattern/adapter 以白名单声明为准。
            adapter = cfg.get("adapter")
            pattern = cfg.get("pattern")
        if adapter is None:
            return None
        if pattern is None:
            raise RuntimeError(f"signal {signal!r} has no pattern mapping")
        context = {"task": req} if signal == "task.object.alignment" else None
        try:
            resp = self._matcher.match(text=statement, pattern=pattern,
                                       adapter=adapter, context=context)
        except Exception as e:
            raise RuntimeError(
                f"{signal}_error:{type(e).__name__}") from e
        if resp.status != "ok" or resp.matched is None:
            reason = f"{signal}_{resp.status}"
            if getattr(resp, "reason_code", None):
                reason += f":{resp.reason_code}"
            raise RuntimeError(reason)
        # adapter 回填一致性：provenance 必须指回请求的适配器，防止静默
        # 用 base 分数冒充微调头结果。
        used = getattr(resp.provenance, "adapter", None)
        if used != adapter:
            raise RuntimeError(
                f"{signal}_adapter_mismatch: provenance={used!r} "
                f"request={adapter!r}")
        return {"matched": bool(resp.matched), "score": resp.score,
                "adapter": used}

    def observe(self, req: str, statement: str, want_capture: bool) -> SemanticObservation:
        extras: Dict[str, Any] = {}
        unavailable: Optional[str] = None

        # 模态判定（R5 白名单下 = modality.assertive@1 底座 zero-shot，
        # 如实标记 uncalibrated；信号文案语义不变）
        modality = None
        try:
            resp = self._matcher.match(text=statement, pattern=self._modality_pattern)
            if resp.status == "ok" and resp.matched is not None:
                modality = self._modality_mapping[bool(resp.matched)]
            else:
                unavailable = f"modality_{resp.status}" + (
                    f":{resp.reason_code}" if getattr(resp, "reason_code", None) else ""
                )
        except Exception as e:  # matcher/runtime 任何异常都按不可用降级
            unavailable = f"modality_error:{type(e).__name__}"
        if modality is not None and self._modality_uncalibrated:
            extras[MODALITY_UNCALIBRATED_KEY] = True

        # 完全闭合判定。R5 白名单：role=capture 的 PASS 信号
        # （completion.asserted → completion-lora-v2）经 adapter 参与判定；
        # 未配置白名单时回退既有 pattern 路径（零变化）。
        capture_complete: Optional[bool] = None
        capture_signal = next(
            (s for s, cfg in self._pass_signals.items()
             if cfg.get("role") == "capture"), None)
        if want_capture and unavailable is None:
            if capture_signal is not None:
                try:
                    result = self.observe_signal(capture_signal, req, statement)
                except RuntimeError as e:
                    unavailable = str(e)
                else:
                    capture_complete = result["matched"]
                    extras[f"{capture_signal}@1"] = result
            else:
                try:
                    resp = self._matcher.match(text=statement, pattern=self._capture_pattern)
                    if resp.status == "ok" and resp.matched is not None:
                        capture_complete = bool(resp.matched)
                        if resp.score is not None:
                            extras[SLM_CAPTURE_SCORE_KEY] = round(float(resp.score), 4)
                    else:
                        unavailable = f"capture_{resp.status}" + (
                            f":{resp.reason_code}" if getattr(resp, "reason_code", None) else ""
                        )
                except Exception as e:
                    unavailable = f"capture_error:{type(e).__name__}"

        # R5：客体对齐判定（PASS 信号 role=alignment，经 adapter 参与判定）。
        alignment_ok: Optional[bool] = None
        if unavailable is None:
            alignment_signal = next(
                (s for s, cfg in self._pass_signals.items()
                 if cfg.get("role") == "alignment"), None)
            if alignment_signal is not None:
                try:
                    result = self.observe_signal(alignment_signal, req, statement)
                except RuntimeError as e:
                    unavailable = str(e)
                else:
                    alignment_ok = result["matched"]
                    extras[f"{alignment_signal}@1"] = result

        # R6b：未决事项判定（PASS 信号 role=unresolved，preliminary 口径，
        # 经 adapter 参与判定；True → verifier 产出未决软信号）。
        unresolved_detected: Optional[bool] = None
        if unavailable is None:
            unresolved_signal = next(
                (s for s, cfg in self._pass_signals.items()
                 if cfg.get("role") == "unresolved"), None)
            if unresolved_signal is not None:
                try:
                    result = self.observe_signal(unresolved_signal, req, statement)
                except RuntimeError as e:
                    unavailable = str(e)
                else:
                    unresolved_detected = result["matched"]
                    extras[f"{unresolved_signal}@1"] = result

        # R4：附加信号（按信号选 adapter）。任一失败与既有语义一致：全部
        # 丢弃、明确降级，不产出半套信号。R5 白名单已接管的信号不再重复跑。
        if unavailable is None:
            skip = set(self._pass_signals) | set(self._shadow_signals)
            for signal in sorted(self._signal_adapters):
                if signal in skip:
                    continue
                try:
                    result = self.observe_signal(signal, req, statement)
                except RuntimeError as e:
                    unavailable = str(e)
                    break
                if result is not None:
                    extras[f"{signal}@1"] = result

        # R5：shadow 信号（如 process.unresolved）。结果只进 shadow 列表；
        # 失败只记录 unavailable_reason，绝不影响判定、绝不拖垮整体。
        shadow_results: List[Dict[str, Any]] = []
        for signal, cfg in sorted(self._shadow_signals.items()):
            try:
                adapter = cfg.get("adapter")
                pattern = cfg.get("pattern")
                context = {"task": req} if signal == "task.object.alignment" else None
                resp = self._matcher.match(text=statement, pattern=pattern,
                                           adapter=adapter, context=context)
                ok = (getattr(resp, "status", None) == "ok"
                      and getattr(resp, "matched", None) is not None)
                if ok:
                    entry = {"signal": signal, "matched": bool(resp.matched),
                             "score": resp.score}
                    if adapter:
                        entry["adapter"] = adapter
                else:
                    reason = f"{signal}_{getattr(resp, 'status', 'error')}"
                    if getattr(resp, "reason_code", None):
                        reason += f":{resp.reason_code}"
                    entry = {"signal": signal, "unavailable_reason": reason}
            except Exception as e:
                entry = {"signal": signal,
                         "unavailable_reason": f"error:{type(e).__name__}"}
            shadow_results.append(entry)

        if unavailable is not None:
            # 任一参与判定的信号失败：全部丢弃，明确进入骨架降级，不产出半套
            # 信号。shadow 结果不属于判定，照常透传（审计可见）。
            return SemanticObservation(unavailable_reason=unavailable,
                                       shadow_signals=shadow_results)
        return SemanticObservation(
            modality=modality,
            similarity=None,
            capture_prob=None,
            capture_complete=capture_complete,
            extra_scores=extras,
            alignment_ok=alignment_ok,
            unresolved_detected=unresolved_detected,
            shadow_signals=shadow_results,
        )

    def describe(self) -> Dict[str, Any]:
        backend = "unknown"
        calibrated = False
        try:
            runtime = getattr(self._matcher, "runtime", None)
            backend = getattr(runtime.backend, "name", "unknown") if runtime is not None else backend
        except Exception:
            pass
        status = {
            "provider_kind": "semantic",
            "backend": backend,
            "calibrated": calibrated,
        }
        # R5：modality 底座 zero-shot 时如实标记（信号文案语义不变，
        # 不计入达标声明）。
        if self._modality_uncalibrated:
            status["modality_judgment"] = {
                "mode": "base zero-shot",
                "uncalibrated": True,
                "attestation": False,
            }
        # R4：信号达标状态表（显式配置时声明）。逐信号如实给出
        # pass / uncalibrated / no_go；no-go 信号不得写成达标。
        if self._capability_status is not None:
            status["capabilities"] = {
                "signals": dict(self._capability_status),
                "signal_adapters": dict(self._signal_adapters),
                "pass_signals": sorted(self._pass_signals),
                "shadow_signals": sorted(self._shadow_signals),
            }
        return status


def build_whitelist_adapter(matcher,
                            capability_status: Optional[Dict[str, Any]] = None
                            ) -> SemanticProviderAdapter:
    """R5 白名单构造：PASS 信号参与判定 + process 仅 shadow + modality
    底座 zero-shot（uncalibrated）。判定依据 final_acceptance.md §4/§6。"""
    return SemanticProviderAdapter(
        matcher,
        modality_pattern=R5_MODALITY_PATTERN,
        pass_signals=R5_PASS_SIGNALS,
        shadow_signals=R5_SHADOW_SIGNALS,
        capability_status=R4_CAPABILITY_STATUS if capability_status is None
        else capability_status,
        modality_uncalibrated=True,
    )


def semantic_provider_from_config(config_path: Optional[str] = None,
                                  timeout: float = 120.0,
                                  ) -> SemanticProviderAdapter:
    """从配置文件构造白名单 semantic provider（MOONBOW_GUARD_PROVIDER=semantic
    时的默认 provider 工厂）。

    纯构造：不连接、不加载模型、不推理——首次真实判定才发生 HTTP 调用。
    matcher URL 优先取配置 provider.matcher_url，否则由 server.host/port 推出。
    构造失败（文件缺失/格式错误）向上抛，由调用方决定降级语义。
    """
    path = config_path or DEFAULT_SEMANTIC_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"semantic runtime config is not an object: {path}")
    provider_cfg = cfg.get("provider") or {}
    server_cfg = cfg.get("server") or {}
    url = provider_cfg.get("matcher_url") or "http://{}:{}".format(
        server_cfg.get("host", "127.0.0.1"), server_cfg.get("port", 18501))
    # 延迟导入：包本体不强制依赖 semantic 子系统
    from ..semantic.client import HTTPClientBackend
    from ..semantic.matcher import SemanticMatcher
    from ..semantic.runtime import SemanticRuntime
    matcher = SemanticMatcher(SemanticRuntime(
        HTTPClientBackend(url, timeout=timeout), default_deadline=timeout))
    return build_whitelist_adapter(matcher)


def env_provider_key() -> Optional[str]:
    """读取 MOONBOW_GUARD_PROVIDER。返回 "legacy" / "semantic" / None（未设置）。
    非法值记 warning 并按 None（legacy 缺省）处理——不改变未配置环境的默认。"""
    raw = (os.environ.get(PROVIDER_ENV) or "").strip().lower()
    if not raw:
        return None
    if raw in ("legacy", "semantic"):
        return raw
    logger.warning("%s=%r 非法（仅支持 legacy/semantic），按 legacy（缺省）运行",
                   PROVIDER_ENV, os.environ.get(PROVIDER_ENV))
    return None

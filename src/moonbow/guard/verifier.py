# -*- coding: utf-8 -*-
"""progress_guard.verifier

进度守卫核心验证状态机与裁决引擎：
实现硬软信号分离、客体对齐、二值完成度捕获与非对称争议放行机制。
"""
from enum import Enum
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .protocol import (
    StatusCode,
    ClosureManifest,
    parse_manifest,
    PROMPT_REQUIRE_MANIFEST,
)
from .semantic_provider import (
    GuardSemanticProvider,
    LegacyProvider,
    SemanticObservation,
    SEMANTIC_CONFIG_ENV,
    SHADOW_SIGNALS_KEY,
    env_provider_key,
)

logger = logging.getLogger("progress_guard.verifier")


class Decision(str, Enum):
    CLOSE = "CLOSE"                     # 放行退出 (任务已闭合)
    BLOCK = "BLOCK"                     # 硬阻断 (自报未完成或硬缺陷，禁止退出)
    CLARIFY = "CLARIFY"                 # 软提示核查 (客体漂移或证据缺失，触发单次反思)
    REQUIRE_MANIFEST = "REQUIRE_MANIFEST" # 格式缺失 (未输出三字段清单，强制规范申报)


@dataclass
class Verdict:
    """最终仲裁结论与审计明细"""
    decision: Decision
    is_closed: bool
    feedback: str
    prompt: Optional[str] = None
    disputed: bool = False
    manifest: Optional[ClosureManifest] = None
    hard_signals: List[str] = field(default_factory=list)
    soft_signals: List[str] = field(default_factory=list)
    scores: Dict[str, Any] = field(default_factory=dict)
    allow_stop: bool = False
    acceptance: str = "unverified"
    review_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "is_closed": self.is_closed,
            "allow_stop": self.allow_stop,
            "acceptance": self.acceptance,
            "review_requested": self.review_requested,
            "disputed": self.disputed,
            "feedback": self.feedback,
            "prompt": self.prompt,
            "hard_signals": self.hard_signals,
            "soft_signals": self.soft_signals,
            "scores": self.scores,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


class ProgressGuard:
    """进度守卫主控实例，对外提供一等公民 API"""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        device: str = "cpu",
        sim_threshold: float = 0.265,
        cap_threshold: float = 0.50,
        lazy_load: bool = False,
        provider: Optional["GuardSemanticProvider"] = None,
    ):
        self.device = device
        self.sim_threshold = sim_threshold
        self.cap_threshold = cap_threshold
        self._models_dir = models_dir
        self._lazy = lazy_load
        self._registry: Optional["ModelRegistry"] = None
        # P6：语义信号 provider。None = 默认 legacy（内部构造 LegacyProvider，
        # lazy_load 语义保持不变）。
        self._provider = provider
        # R5：MOONBOW_GUARD_PROVIDER=semantic 时按配置惰性构建 semantic
        # provider（仅一次；失败则记 warning 并永久回退 legacy，零行为变化）。
        self._env_provider_tried = False

        if not lazy_load:
            self._ensure_models()

    def _ensure_models(self) -> "ModelRegistry":
        if self._registry is None:
            from .models import ModelRegistry  # 惰性导入：包本体不强制依赖 torch
            self._registry = ModelRegistry(models_dir=self._models_dir, device=self.device)
        return self._registry

    def semantic_status(self) -> Dict[str, Any]:
        """只读状态：当前 provider kind / backend / calibrated 声明。"""
        if self._provider is not None:
            return self._provider.describe()
        return LegacyProvider(None).describe()

    def check(
        self,
        req: str,
        resp: str,
        rounds: int = 1,
        external_tool_success: Optional[bool] = None,
        *,
        mode: str = "advisory",
        semantic_review_used: bool = False,
        provider: Optional["GuardSemanticProvider"] = None,
    ) -> Verdict:
        """Check closure separately from stop permission.

        Hosts persist semantic_review_used per task; rounds is not a delivery receipt.
        strict preserves the legacy decision/stop behavior.
        provider: 本次调用的语义 provider 覆盖（None = 用构造时的默认）。
        """
        if mode not in ("advisory", "strict"):
            raise ValueError("mode must be advisory or strict")
        if type(semantic_review_used) is not bool:
            raise TypeError("semantic_review_used must be a bool")
        verdict = self._check(req, resp, rounds, external_tool_success, mode, provider)
        if verdict.decision == Decision.REQUIRE_MANIFEST:
            verdict.acceptance = "invalid"
        elif verdict.hard_signals:
            verdict.acceptance = "incomplete"
        elif not verdict.manifest.has_evidence and external_tool_success is not True:
            verdict.acceptance = "unverified"
        elif verdict.soft_signals:
            verdict.acceptance = "disputed"
        elif external_tool_success is True:
            verdict.acceptance = "verified"
        else:
            verdict.acceptance = "unverified" if verdict.scores.get("skeleton_only") else "unchecked"

        if mode == "strict":
            verdict.allow_stop = verdict.is_closed
            verdict.review_requested = verdict.decision == Decision.CLARIFY
            return verdict

        semantic = bool(verdict.scores.get("semantic_signals"))
        verdict.review_requested = (
            semantic and not semantic_review_used and not verdict.hard_signals
        )
        verdict.allow_stop = (
            verdict.decision != Decision.REQUIRE_MANIFEST and not verdict.review_requested
        )
        if verdict.soft_signals:
            verdict.is_closed = False
            verdict.disputed = verdict.acceptance == "disputed"
        if verdict.review_requested:
            verdict.prompt = (
                "【进度守卫复核建议】\n" + verdict.feedback + "\n"
                "请结合已有修改和工具结果复核一次；若确有遗漏，请修复或如实列为剩余事项。"
                "若无法验证，请明确报告受阻或未验证。不要仅为满足提示而改写申报或测试。"
            )
        elif verdict.allow_stop:
            verdict.prompt = None
        return verdict

    def _check(
        self,
        req: str,
        resp: str,
        rounds: int,
        external_tool_success: Optional[bool],
        mode: str,
        provider_override: Optional["GuardSemanticProvider"] = None,
    ) -> Verdict:
        """对 Agent 的收尾输出进行进度闭合仲裁。

        Args:
            req: 用户原始需求描述
            resp: Agent 给出的收尾陈述（或包含收尾清单的文本）
            rounds: 当前交互轮次（第 1 轮为初始申报，>=2 轮为澄清重申）
            external_tool_success: 外部工具硬信号（如 pytest 返回码==0）；
                                   若为 True，则硬证据成立，豁免软信号假阴性。
        """
        if external_tool_success is not None and type(external_tool_success) is not bool:
            raise TypeError("external_tool_success must be a bool or None")
        tool_verified = external_tool_success is True
        manifest = parse_manifest(resp)

        # 1. 检查三字段清单规范
        if not manifest.is_valid_format or manifest.status is None:
            return Verdict(
                decision=Decision.REQUIRE_MANIFEST,
                is_closed=False,
                feedback="收尾三字段申报清单缺失或 STATUS 非法，请使用 A/B/C/D 及对应状态说明",
                prompt=PROMPT_REQUIRE_MANIFEST,
                manifest=manifest,
                hard_signals=["未按规范申报 STATUS / REMAINING / EVIDENCE 三字段"],
            )

        hard: List[str] = []
        soft: List[str] = []
        scores: Dict[str, Any] = {}

        st = manifest.status

        # 2. 硬信号核对：自报未完成
        if st in (StatusCode.B, StatusCode.C, StatusCode.D):
            status_desc = {
                StatusCode.B: "B 部分完成",
                StatusCode.C: "C 进行中或受阻",
                StatusCode.D: "D 失败或已回滚",
            }.get(st, st.value)
            hard.append(f"你自报状态为「{status_desc}」，任务尚未达成全部闭合")

        if manifest.has_remaining:
            hard.append(f"你自报存在未完成遗留项：{manifest.remaining}")

        if st == StatusCode.A and not manifest.has_evidence and not tool_verified:
            soft.append("EVIDENCE 为空。请给出具体测试运行输出、构建日志或验证指标")

        # 3. 提取用于自然语言判别的实际陈述文本（排除清单声明头）
        statement_lines = [
            line for line in resp.splitlines()
            if not any(line.strip().upper().startswith(k) for k in ("STATUS:", "REMAINING:", "STATUS：", "REMAINING："))
        ]
        statement_text = "\n".join(statement_lines).strip()
        if not statement_text or len(statement_text) < 5:
            statement_text = manifest.evidence if manifest.has_evidence else resp

        # 4. 语义信号（P6：经 provider 获取；默认 None -> 内部 LegacyProvider
        #    = 原微模型路径，lazy_load 降级语义保持）。
        # lazy_load + 权重不可用时降级为"骨架裁决"：仅定量层（协议/硬信号/争议放行），
        # 定性信号（模态/相似度/捕获）无法产出即不虚构。eager 模式权重缺失仍然抛错。
        provider = provider_override if provider_override is not None else self._provider
        if provider is None and not self._env_provider_tried:
            # R5：SDK 默认切换。仅当 MOONBOW_GUARD_PROVIDER=semantic 时
            # 按配置构建白名单 semantic provider；构建失败回退 legacy。
            # 环境变量未设置/非法/legacy 时完全走既有路径（零行为变化）。
            self._env_provider_tried = True
            if env_provider_key() == "semantic":
                try:
                    from .semantic_provider import semantic_provider_from_config
                    self._provider = provider = semantic_provider_from_config(
                        os.environ.get(SEMANTIC_CONFIG_ENV))
                    logger.info("MOONBOW_GUARD_PROVIDER=semantic: "
                                "已按配置启用 semantic provider")
                except Exception as e:
                    logger.warning("semantic provider 构建失败（%s: %s），"
                                   "回退 legacy 默认", type(e).__name__, e)
        if provider is None:
            if self._registry is None:
                if not self._lazy:
                    self._ensure_models()
                else:
                    try:
                        self._ensure_models()
                    except Exception:
                        scores["skeleton_only"] = True
            if self._registry is not None:
                provider = LegacyProvider(self._registry)

        semantic_signals: List[str] = []
        if provider is None:
            scores["provider_kind"] = "unavailable"
        else:
            scores["provider_kind"] = provider.kind
            want_capture = st == StatusCode.A and not manifest.has_remaining
            try:
                obs = provider.observe(req=req, statement=statement_text,
                                       want_capture=want_capture)
            except Exception:
                if provider.kind == "legacy":
                    # legacy 路径保持迁移前行为：模型推理失败向上抛（eager 即 500），
                    # 不静默降级。
                    raise
                # semantic provider 自身未处理的失败：明确降级，不崩溃、不伪造信号。
                obs = SemanticObservation(unavailable_reason="provider_error")
            if obs is None or obs.unavailable_reason is not None:
                # provider 失败/不可用：进入既有骨架降级语义，标记未验证。
                scores["skeleton_only"] = True
                scores["semantic_unavailable"] = True
            else:
                if obs.extra_scores:
                    scores.update(obs.extra_scores)
                if obs.modality is not None:
                    scores["modality"] = obs.modality
                    if obs.modality != "assert" and not tool_verified:
                        signal = f"收尾陈述语气可能为「{obs.modality}」，请核对是否已完成"
                        if mode == "strict":
                            if not manifest.has_evidence:
                                hard.append(signal)
                        else:
                            semantic_signals.append(signal)

                if obs.similarity is not None:
                    sim_val = obs.similarity
                    scores["similarity"] = round(sim_val, 4)
                    if sim_val < self.sim_threshold and not tool_verified:
                        semantic_signals.append(f"收尾内容可能未覆盖用户请求【{req}】，请结合实际修改核对目标")

                if want_capture:
                    capture_complete: Optional[bool] = None
                    if obs.capture_prob is not None:
                        scores["capture_prob"] = round(obs.capture_prob, 4)
                        capture_complete = obs.capture_prob >= self.cap_threshold
                    elif obs.capture_complete is not None:
                        capture_complete = obs.capture_complete
                    # 自报证据不消除语义异议；仅外部成功信号可直接豁免。
                    if capture_complete is False and not tool_verified:
                        semantic_signals.append("收尾内容疑似未完全闭合，请提供工程验证证据或如实修正 STATUS")

                # R5：客体对齐（PASS 信号 task.object.alignment 参与判定；
                # legacy 观察恒为 None，零行为变化）。豁免语义与捕获一致：
                # 仅外部成功信号可豁免。
                if getattr(obs, "alignment_ok", None) is False and not tool_verified:
                    semantic_signals.append(
                        "收尾内容与任务目标可能存在客体偏离，请对照用户原始需求核对")

                # R6b：未决事项（PASS 信号 process.unresolved v2 参与判定，
                # preliminary 口径；legacy 观察恒为 None，零行为变化）。
                if getattr(obs, "unresolved_detected", None) is not None:
                    scores["unresolved_detected"] = obs.unresolved_detected
                if getattr(obs, "unresolved_detected", None) is True and not tool_verified:
                    semantic_signals.append(
                        "收尾陈述中疑似存在未解决的疑点或待办事项，请核对是否已全部解决，或如实列入 REMAINING")

                # R5：shadow 信号观察（如有）只进 scores 的 shadow_signals 键
                # ——绝不参与信号判定与提醒生成。
                if getattr(obs, "shadow_signals", None):
                    scores[SHADOW_SIGNALS_KEY] = obs.shadow_signals

        soft.extend(semantic_signals)
        scores["semantic_signals"] = semantic_signals

        # 5. 综合状态机裁决
        # 规则 5.1: 存在任何硬缺陷 -> 刚性阻断 BLOCK
        if hard:
            return Verdict(
                decision=Decision.BLOCK,
                is_closed=False,
                feedback="；".join(hard),
                manifest=manifest,
                hard_signals=hard,
                soft_signals=soft,
                scores=scores,
            )

        # 规则 5.2: 软信号全清 -> 立即放行 CLOSE
        if not soft and st == StatusCode.A:
            return Verdict(
                decision=Decision.CLOSE,
                is_closed=True,
                feedback="当前检查未检出异议；不等同于独立验收通过",
                manifest=manifest,
                scores=scores,
            )

        # 规则 5.3: 非对称确认机制 (Disputed Close)
        # 生产者在看到核查提示后，于第 2 轮及以上重申 STATUS=A，且提供了非空证据 -> 争议放行并留痕。
        # 注（n08 实验，2026-09-30）：曾试验"语义捕获判未闭合时扣住争议放行、
        # 要求工具实证"的闸门——67 卷实证为净退化（拦下 n08 1 例 FP 的同时
        # 误伤 e01/e03/e05/e13 四例真闭合，捕获头在该域无法分离 0.039 vs
        # 0.058，相似度同样无分离力）。结论：在捕获/对齐通道具备域内判别力
        # 之前，无条件第 2 轮争议放行是更优权衡；n08 类误放行（1/67）作为
        # 机制已知限制留档。证据：results/semantic-runtime/legacy-vs-semantic/
        # {predictions_pre_n08fix.jsonl, predictions.jsonl}。
        if mode == "strict" and rounds >= 2 and st == StatusCode.A and not manifest.has_remaining and manifest.has_evidence:
            return Verdict(
                decision=Decision.CLOSE,
                is_closed=True,
                disputed=True,
                feedback="主模型重申完成且提供具体证据，争议放行 (CLOSE [disputed])",
                manifest=manifest,
                soft_signals=soft,
                scores=scores,
            )

        # 规则 5.4: 处于软信号灰色带 -> 触发单次去元语言反思提示 CLARIFY
        clarify_feedback = "；".join(soft)
        clarify_prompt = (
            f"【进度守卫核查意见（请据此补完）】\n{clarify_feedback}\n\n"
            # 先要求"继续做"，表态放最后：实测旧文案把"可调整为 B/C"放在
            # 首屏，弱主动性模型据此提前收工（2026-09-22）。顺序即优先级。
            "请对照你的实际修改和用户原始需求逐条核对，并继续把未完成的部分做完：\n"
            "- 若还有未完成的修改或未运行的验证，现在就继续调用工具完成并跑到有结果；\n"
            "- 若确已完全解决，请在 EVIDENCE 填入具体测试或命令执行输出。\n"
            "（STATUS 若确实不是 A，如实填写即可，但仍须继续执行剩余事项。）"
        )
        return Verdict(
            decision=Decision.CLARIFY,
            is_closed=False,
            feedback=clarify_feedback,
            prompt=clarify_prompt,
            manifest=manifest,
            soft_signals=soft,
            scores=scores,
        )

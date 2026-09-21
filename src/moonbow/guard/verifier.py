# -*- coding: utf-8 -*-
"""progress_guard.verifier

进度守卫核心验证状态机与裁决引擎：
实现硬软信号分离、客体对齐、二值完成度捕获与非对称争议放行机制。
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .protocol import (
    StatusCode,
    ClosureManifest,
    parse_manifest,
    PROMPT_REQUIRE_MANIFEST,
)


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
    ):
        self.device = device
        self.sim_threshold = sim_threshold
        self.cap_threshold = cap_threshold
        self._models_dir = models_dir
        self._lazy = lazy_load
        self._registry: Optional["ModelRegistry"] = None

        if not lazy_load:
            self._ensure_models()

    def _ensure_models(self) -> "ModelRegistry":
        if self._registry is None:
            from .models import ModelRegistry  # 惰性导入：包本体不强制依赖 torch
            self._registry = ModelRegistry(models_dir=self._models_dir, device=self.device)
        return self._registry

    def check(
        self,
        req: str,
        resp: str,
        rounds: int = 1,
        external_tool_success: Optional[bool] = None,
        *,
        mode: str = "advisory",
        semantic_review_used: bool = False,
    ) -> Verdict:
        """Check closure separately from stop permission.

        Hosts persist semantic_review_used per task; rounds is not a delivery receipt.
        strict preserves the legacy decision/stop behavior.
        """
        if mode not in ("advisory", "strict"):
            raise ValueError("mode must be advisory or strict")
        if type(semantic_review_used) is not bool:
            raise TypeError("semantic_review_used must be a bool")
        verdict = self._check(req, resp, rounds, external_tool_success, mode)
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

        # 4. 微模型推理判定
        # lazy_load + 权重不可用时降级为"骨架裁决"：仅定量层（协议/硬信号/争议放行），
        # 定性信号（模态/相似度/捕获）无法产出即不虚构。eager 模式权重缺失仍然抛错。
        if self._registry is None:
            if not self._lazy:
                self._ensure_models()
            else:
                try:
                    self._ensure_models()
                except Exception:
                    scores["skeleton_only"] = True
        models = self._registry

        semantic_signals: List[str] = []
        if models is not None:
            # 模态判定
            modality = models.get_modality(statement_text)
            scores["modality"] = modality
            if modality != "assert" and not tool_verified:
                signal = f"收尾陈述语气可能为「{modality}」，请核对是否已完成"
                if mode == "strict":
                    if not manifest.has_evidence:
                        hard.append(signal)
                else:
                    semantic_signals.append(signal)

            # 客体语义相似度
            sim_val = models.get_similarity(req, statement_text)
            scores["similarity"] = round(sim_val, 4)
            if sim_val < self.sim_threshold and not tool_verified:
                semantic_signals.append(f"收尾内容可能未覆盖用户请求【{req}】，请结合实际修改核对目标")

            # 二值完成度捕获概率
            cap_p = None
            if st == StatusCode.A and not manifest.has_remaining:
                cap_p = models.get_capture_prob(statement_text)
                scores["capture_prob"] = round(cap_p, 4)

                # 自报证据不消除模型异议；仅外部成功信号可直接豁免。
                if cap_p < self.cap_threshold and not tool_verified:
                    semantic_signals.append("收尾内容疑似未完全闭合，请提供工程验证证据或如实修正 STATUS")

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
        # 生产者在看到核查提示后，于第 2 轮及以上重申 STATUS=A，且提供了非空证据 -> 争议放行并留痕
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
            f"【进度守卫核查意见（请据此修正）】\n{clarify_feedback}\n\n"
            "请对照你的实际修改和用户原始需求逐条核对：\n"
            "- 若确已完全解决，请保留 STATUS: A 并在 EVIDENCE 填入具体测试或命令执行输出；\n"
            "- 若属于部分完成或仍在进行，请如实调整 STATUS (如 B 或 C)。"
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

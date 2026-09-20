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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "is_closed": self.is_closed,
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
    ) -> Verdict:
        """对 Agent 的收尾输出进行进度闭合仲裁。

        Args:
            req: 用户原始需求描述
            resp: Agent 给出的收尾陈述（或包含收尾清单的文本）
            rounds: 当前交互轮次（第 1 轮为初始申报，>=2 轮为澄清重申）
            external_tool_success: 外部工具硬信号（如 pytest 返回码==0）；
                                   若为 True，则硬证据成立，豁免软信号假阴性。
        """
        manifest = parse_manifest(resp)

        # 1. 检查三字段清单规范
        if not manifest.is_valid_format:
            return Verdict(
                decision=Decision.REQUIRE_MANIFEST,
                is_closed=False,
                feedback="缺少收尾三字段申报清单",
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

        if models is not None:
            # 模态判定
            modality = models.get_modality(statement_text)
            scores["modality"] = modality
            if modality != "assert" and not manifest.has_evidence and not external_tool_success:
                hard.append(f"你的收尾陈述语气为「{modality}」，并非完成性断言")

            # 客体语义相似度
            sim_val = models.get_similarity(req, statement_text)
            scores["similarity"] = round(sim_val, 4)
            if sim_val < self.sim_threshold and not manifest.has_evidence and not external_tool_success:
                soft.append(f"收尾内容与用户请求【{req}】存在客体差异（相似度仅 {sim_val:.3f}），请核对工作是否对齐目标")

            # 二值完成度捕获概率
            cap_p = None
            if st == StatusCode.A and not manifest.has_remaining:
                cap_p = models.get_capture_prob(statement_text)
                scores["capture_prob"] = round(cap_p, 4)

                # 如果没有外部真实单测背书，核查二值捕获头与证据完整度
                if not external_tool_success:
                    if cap_p < self.cap_threshold and not manifest.has_evidence:
                        soft.append("检测到收尾内容疑似未完全闭合，请提供具体工程验证证据或如实修正 STATUS")
                    elif not manifest.has_evidence:
                        soft.append("EVIDENCE 为空。请给出具体测试运行输出、构建日志或验证指标")

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
        if not soft:
            return Verdict(
                decision=Decision.CLOSE,
                is_closed=True,
                feedback="验收通过，允许正常退出",
                manifest=manifest,
                scores=scores,
            )

        # 规则 5.3: 非对称确认机制 (Disputed Close)
        # 生产者在看到核查提示后，于第 2 轮及以上重申 STATUS=A，且提供了非空证据 -> 争议放行并留痕
        if rounds >= 2 and st == StatusCode.A and not manifest.has_remaining and manifest.has_evidence:
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

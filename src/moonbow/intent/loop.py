# -*- coding: utf-8 -*-
"""moonbow.intent.loop — 事件→捕获→账本→判断→提醒→反馈 链路。

非阻塞语义（第二轮审计 D5 修复）：process_event **永不向宿主抛出异常**。
捕获、裁决、账本写入任一环节失败，都落成显式失败状态（failures 列表 +
LoopReport 计数），受影响事件如实标记，既不伪装成"无违规"，也不静默吞掉。
提醒去重（dedup_key seen 集合）。裁决后端显式注入：无后端时用
InsufficientInfoJudge（诚实 abstain，带 reason），绝不伪装成"未违规"。

裁决输入由 moonbow.intent.protocol 构造（训练/运行时共用，见 D1/D2）：
当前有效约束原文 + 决策完整文本 + 事件类型元数据，三分类受限输出。
"""
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import List, Optional

from .schema import Entity, Event, Judgment, Reminder, entity_id
from .ledger import Ledger
from . import applicability
from . import protocol

_jid_counter = itertools.count(1)


def _next_jid() -> str:
    return f"j{next(_jid_counter)}"


class InsufficientInfoJudge:
    """无语义后端时的诚实默认：全部 abstain，不伪装。"""
    model_id = "none"

    def judge(self, decision: Entity, constraints: List[Entity],
              event_text: str, event_kind: str = "",
              scope: str = "") -> Judgment:
        return Judgment(
            judgment_id=_next_jid(),
            jtype="constraint_relation",
            subject_ids=[decision.entity_id],
            reference_ids=[c.entity_id for c in constraints],
            evidence_ids=[],
            verdict="insufficient_info", model_id=self.model_id,
            abstain=True, abstain_reason="no semantic backend configured",
            reason="无语义后端；本条仅为占位登记")


class LoraFirstTokenJudge:
    """阶段三产物：intent_lora_v0 三标签首 token 比较裁决（XPU）。

    第二轮审计修复：
    - D1：约束原文经 protocol.build_judgment_input 进入 prompt（原先只送
      decision.quote，约束不参与推理）；
    - D2：与训练端一致的三分类（violation/compliant/insufficient），
      不再只有两个标签；
    - 超长决策显式标记 truncated，交由上层决定分片或弃权。
    """

    model_id = "intent_lora_v0"

    def __init__(self, base_model, tokenizer, adapter_path: str,
                 device: str = "xpu", card: Optional[str] = None):
        import torch
        from peft import PeftModel
        if not torch.xpu.is_available():
            raise RuntimeError("XPU 不可用（铁律：禁止 CPU 推理）")
        self._torch = torch
        self._tok = tokenizer
        self._model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        self._card = card
        self._token_ids = protocol.first_token_ids(tokenizer)

    def judge(self, decision: Entity, constraints: List[Entity],
              event_text: str, event_kind: str = "",
              scope: str = "") -> Judgment:
        torch = self._torch
        # 用完整事件文本构建输入（训练同分布）。decision.quote 只有前 200 字
        # （实体去重键用途），曾导致运行时输入被静默截断、偏离训练分布。
        ji = protocol.build_judgment_input(
            event_text or decision.quote, constraints, event_kind=event_kind,
            scope=scope, card=self._card)
        if ji.truncated:
            # 超长决策不静默判：如实弃权，由上层分片后重判
            return Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=[decision.entity_id],
                reference_ids=[c.entity_id for c in constraints],
                evidence_ids=[], verdict="insufficient_info",
                model_id=self.model_id, abstain=True,
                abstain_reason=f"decision too long: {ji.truncation_note}",
                reason="决策文本超长，未做静默截断判定")
        ids = self._tok(ji.prompt, return_tensors="pt").to("xpu")
        with torch.no_grad():
            logits = self._model(**ids).logits[0, -1, :].float()
        logsoftmax = torch.log_softmax(logits, dim=-1)
        label, scores = protocol.pick_label(logsoftmax, self._token_ids)
        margin = protocol.margin_between(scores)
        return Judgment(
            judgment_id=_next_jid(),
            jtype="constraint_relation",
            subject_ids=[decision.entity_id],
            reference_ids=[c.entity_id for c in constraints],
            evidence_ids=[decision.event_id],
            verdict=protocol.VERDICT_OF_LABEL[label],
            model_id=self.model_id,
            abstain=(label == "insufficient"),
            abstain_reason=("模型判定信息不足（三分类首 token）"
                            if label == "insufficient" else None),
            reason=(f"三分类首token={label} margin={margin} "
                    f"（约束{len(ji.constraints)}条，"
                    f"事件类型={event_kind or 'unknown'}）"))

    # 第一级判官不做自我复核——第二级由独立的 VerifyJudge（专用 adapter）
    # 承担，避免"同一权重自己给自己盖章"。


class CoverageJudge:
    """目标验收裁决的诚实默认：全部 abstain，不伪装成"已完成"。

    存在理由（第二轮审计 D9）：即便没有语义后端，完成宣称也必须被登记并
    留下"无法裁决"的痕迹——静默放行等于默认宣布完成。
    """
    model_id = "none"

    def judge(self, goal, claim, evidence, goals_all=None):
        return Judgment(
            judgment_id=_next_jid(), jtype="goal_coverage",
            subject_ids=[claim.entity_id],
            reference_ids=[goal.entity_id],
            evidence_ids=[], verdict="insufficient_info",
            model_id=self.model_id, abstain=True,
            abstain_reason="no semantic backend configured",
            reason="无语义后端；完成宣称已登记但未获裁决")


class LoraCoverageJudge:
    """覆盖裁决后端：intent_lora_t4 四分类首 token 比较（XPU）。

    训练口径（tools/intent_eval/train_intent_lora.py --task t4）：
    标签用别名 FULL/PART/NONE/UNCLEAR（原始 supported_full/partial 共享
    首 token，首 token 比较会退化）；打分用**原始总似然**（不做长度归一化，
    避免 4-token 类系统性偏袒）。运行时与此严格一致。
    """

    model_id = "intent_lora_t4"

    def __init__(self, base_model, tokenizer, adapter_path: str,
                 device: str = "xpu"):
        import torch
        from peft import PeftModel
        if not torch.xpu.is_available():
            raise RuntimeError("XPU 不可用（铁律：禁止 CPU 推理）")
        self._torch = torch
        self._tok = tokenizer
        self._model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        self._labels = protocol.COVERAGE_LABELS
        self._alias = protocol.COVERAGE_ALIASES

    def judge(self, goal, claim, evidence, goals_all=None):
        torch = self._torch
        ci = protocol.build_coverage_input(goal, claim, evidence,
                                           goals_all=goals_all)
        if ci.truncated:
            return Judgment(
                judgment_id=_next_jid(), jtype="goal_coverage",
                subject_ids=[claim.entity_id],
                reference_ids=[goal.entity_id], evidence_ids=[],
                verdict="insufficient_info", model_id=self.model_id,
                abstain=True,
                abstain_reason=f"coverage input too long: {ci.truncation_note}",
                reason="覆盖裁决输入过长，未做静默截断判定")
        ids = self._tok(ci.prompt, return_tensors="pt").to("xpu")
        scores = protocol.sequence_scores(
            self._model, self._tok, ids["input_ids"], self._labels,
            literals=self._alias, normalize=False)
        pick = max(scores, key=scores.get)
        ev_ids = [e["evidence_id"] for e in ci.evidence]
        return Judgment(
            judgment_id=_next_jid(), jtype="goal_coverage",
            subject_ids=[claim.entity_id],
            reference_ids=[goal.entity_id],
            evidence_ids=ev_ids,
            verdict=pick, model_id=self.model_id,
            abstain=(pick == "insufficient_info"),
            abstain_reason=("模型判定信息不足（覆盖四分类）"
                            if pick == "insufficient_info" else None),
            reason=f"覆盖四分类={pick}（证据{len(ev_ids)}条，"
                   f"目标项数由原文判断）",
            evidence_kinds=[e["kind"] for e in ci.evidence],
            evidence_results=[e["result"] for e in ci.evidence])


class VerifyJudge:
    """第二级复核判官（LLM 定量）：独立实例加载 verify adapter。

    输入由 protocol.build_verification_input 构造（完整决策 + 全部约束 +
    第一级结论与理由），输出 KEEP（维持提醒）/ DROP（判定不可靠）。
    KEEP 时返回维持第一级结论的 Judgment；DROP 时返回 insufficient_info
    弃权 Judgment（abstain=True）——由 IntentLoop 据此抑制投递。
    """

    model_id = "verify_v0"

    def __init__(self, base_model, tokenizer, adapter_path: str,
                 device: str = "xpu"):
        import torch
        from peft import PeftModel
        if not torch.xpu.is_available():
            raise RuntimeError("XPU 不可用（铁律：禁止 CPU 推理）")
        self._torch = torch
        self._tok = tokenizer
        self._model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        self._labels = protocol.VERIFY_LABELS
        self._alias = protocol.VERIFY_LITERALS

    def verify(self, decision: Entity, constraints: List[Entity],
               claimed: Judgment, decision_text: str = "") -> Judgment:
        torch = self._torch
        prompt = protocol.build_verification_input(
            decision_text or decision.quote, constraints, claimed.verdict,
            claimed.reason)
        ids = self._tok(prompt, return_tensors="pt").to("xpu")
        scores = protocol.sequence_scores(
            self._model, self._tok, ids["input_ids"], self._labels,
            literals=self._alias, normalize=False)
        pick = max(scores, key=scores.get)
        if pick == "KEEP":
            return Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=claimed.subject_ids,
                reference_ids=claimed.reference_ids,
                evidence_ids=claimed.evidence_ids,
                verdict=claimed.verdict, model_id=self.model_id,
                reason=f"复核KEEP scores={scores}｜{claimed.reason}")
        return Judgment(
            judgment_id=_next_jid(), jtype="constraint_relation",
            subject_ids=claimed.subject_ids,
            reference_ids=claimed.reference_ids,
            evidence_ids=[], verdict="insufficient_info",
            model_id=self.model_id, abstain=True,
            abstain_reason="第二级复核判定第一级冲突结论不可靠",
            reason=f"复核DROP scores={scores}")


class SelectionJudge:
    """约束选择判官（两阶段第一级，LLM）：从候选约束中选出与决策相关的卡。

    独立实例加载 select adapter（KEEP/DROP 式受限输出同款：
    对 A–E 五个别名做整序列似然比较，取最高者）。
    选择结果只决定**义务判定用哪张约束**，不产生合规结论——
    结论永远由第二级义务判官给出。
    """

    model_id = "select_v0"

    def __init__(self, base_model, tokenizer, adapter_path: str,
                 device: str = "xpu"):
        import torch
        from peft import PeftModel
        if not torch.xpu.is_available():
            raise RuntimeError("XPU 不可用（铁律：禁止 CPU 推理）")
        self._torch = torch
        self._tok = tokenizer
        self._model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        self._labels = list(protocol.SELECT_LABELS)
        self._alias = dict(protocol.SELECT_LITERALS)

    def select(self, decision_text: str, card_quotes: List[str]) -> int:
        """返回选中卡的索引（0-based）。prompt 与训练同构造器。"""
        torch = self._torch
        prompt = protocol.build_selection_input(decision_text, card_quotes)
        ids = self._tok(prompt, return_tensors="pt").to("xpu")
        scores = protocol.sequence_scores(
            self._model, self._tok, ids["input_ids"],
            list(card_quotes) and self._labels,
            literals=self._alias, normalize=False)
        pick = max(scores, key=scores.get)
        return self._labels.index(pick)


@dataclass
class LoopReport:
    events_processed: int = 0
    entities_added: int = 0
    judgments: int = 0
    reminders_emitted: List[Reminder] = field(default_factory=list)
    reminders_deduped: int = 0
    failures: List[dict] = field(default_factory=list)
    claims_registered: int = 0
    constraints_considered: int = 0
    applicability_skipped: int = 0

    def fail(self, stage: str, event_id: str, exc: BaseException) -> None:
        self.failures.append({"stage": stage, "event_id": event_id,
                              "error": f"{type(exc).__name__}: {exc}"})


class IntentLoop:
    """串起 Ledger + CaptureStage + 裁决后端 + 提醒去重（全链路非阻塞）。"""

    #: 参与裁决的事件类型（叙述消息只捕获，不裁决——真实会话实测教训）。
    #: 这是宿主原生的事件元数据，不是内容词表。
    DECISION_KINDS = ("tool_call", "commit")
    #: 审计备注（2026-10-10）：曾在此处加"只读工具名/只读命令头"词表来豁免
    #: 裁决，被用户指出违反硬约束 1（词表裁决 + 枚举不可穷尽）并已全部删除。
    #: 只读操作误报的正确修法是训练数据（让三分类头学会对查询类决策输出
    #: compliant/insufficient），不是运行时枚举豁免。

    def __init__(self, ledger: Ledger, capture_stage, judge, scope: str,
                 coverage_judge=None, claim_detector=None,
                 verify_judge=None, max_per_session: int = 0,
                 select_judge=None):
        """coverage_judge：目标验收裁决后端（缺省诚实 abstain）。
        claim_detector：完成宣称识别器 —— 一个 `(text) -> Optional[str]`
        的显式注入（生产用语义捕获/SLM，测试用 Fake）。**不做关键词匹配**：
        本模块不提供基于词表的宣称识别（硬约束 1）。
        verify_judge：第二级复核判官（LLM 定量，`verify(decision,
        constraints, judgment) -> Judgment(KEEP/DROP)`）。冲突裁决在投递前
        一律升级复核——低置信不是"丢弃"而是"升级"。缺省 None = 不复核
        （兼容单级判官）。
        select_judge：约束选择判官（两阶段第一级，`select(decision_text,
        card_quotes) -> int`）。卡数多于训练分布（>5）时先选相关约束再判定
        ——选卡只决定"用哪张约束"，结论仍由义务判官给出。缺省 None =
        全部约束直接进判定（≤5 张的分布内场景）。
        max_per_session：单会话提醒预算（0 = 不限）——防刷屏保险丝，
        超预算落账不投递。这是数量保险丝，不读内容。
        """
        self.ledger = ledger
        self.capture = capture_stage
        self.judge = judge
        self.scope = scope
        self.coverage_judge = coverage_judge or CoverageJudge()
        self.claim_detector = claim_detector
        self.verify_judge = verify_judge
        self.select_judge = select_judge
        self.max_per_session = max_per_session
        self._session_reminded = 0
        self._seen_dedup = set()

    def process_event(self, event: Event) -> LoopReport:
        report = LoopReport(events_processed=1)

        # 机械存证：失败也只记录，不阻断
        try:
            self.ledger.add_event_record({
                "event_id": event.event_id, "kind": event.kind,
                "source_role": event.source_role, "ts": event.ts,
                "text_sha256": event.text_sha256, "raw_ref": event.raw_ref})
        except Exception as e:                        # noqa: BLE001
            report.fail("event_record", event.event_id, e)

        if event.kind == "message" and event.text.strip():
            try:
                result = self.capture.capture(event.event_id, event.text,
                                              self.scope,
                                              source_role=event.source_role)
                for ent in result.entities:
                    self.ledger.add_entity(ent)
                    report.entities_added += 1
            except Exception as e:                    # noqa: BLE001
                report.fail("capture", event.event_id, e)

        reminders: List[Reminder] = []

        # 完成宣称：只来自助手消息；识别器显式注入，无识别器则不登记
        if (event.kind == "message"
                and event.source_role == "assistant_claim"
                and event.text.strip() and self.claim_detector is not None):
            reminders += self._register_claim(event, report)

        # 决策判定：仅限 commit/tool_call（叙述只捕获，不裁决）。
        # 全部工具调用都进语义裁决——"是否改动类决策"由三分类头经训练数据
        # 学会（查询类 → compliant/insufficient），不做内容词表豁免。
        if event.kind in self.DECISION_KINDS:
            reminders += self._judge_decision(event, report)

        for r in reminders:
            if r.dedup_key in self._seen_dedup:
                report.reminders_deduped += 1
                continue
            self._seen_dedup.add(r.dedup_key)
            report.reminders_emitted.append(r)
        return report

    def _judge_decision(self, event: Event, report: LoopReport) -> List[Reminder]:
        """决策裁决。任一环节失败 → 显式失败 + abstain 登记，不抛出。"""
        try:
            dec = Entity(
                entity_id=entity_id("decision", event.event_id,
                                    event.text[:200]),
                kind="decision", quote=event.text[:200], span=None,
                event_id=event.event_id,
                source_role="assistant_claim", scope=self.scope)
            self.ledger.add_entity(dec)
            report.entities_added += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("decision_entity", event.event_id, e)
            return []

        try:
            constraints = self.ledger.effective("constraint", self.scope)
        except Exception as e:                        # noqa: BLE001
            report.fail("constraint_query", event.event_id, e)
            return []

        if not constraints:
            return []

        # L0 适用范围闸门（零 LLM，机械字段比较）：只把确实适用的约束交给
        # 语义层。scope 不继承、时间不溯及这两条是用户在约束卡里声明的机械
        # 边界，已被实验证明可零 LLM 可靠执行（exp_t3_decomposition_report）。
        applicable, skipped = applicability.filter_applicable(
            constraints, self.scope, event.ts)
        if skipped:
            report.applicability_skipped += len(skipped)
        report.constraints_considered += len(applicable) if applicable else len(constraints)
        if not applicable:
            # 全部约束都不适用 → 显式弃权登记（不静默放行）
            self._record_gate_abstain(dec, skipped, event, report)
            return []

        # 约束选择（两阶段第一级）：卡数超过训练分布（>5）且选卡判官在场时，
        # 先选出相关约束再判定——选卡只决定"用哪张"，结论由义务判官给出。
        if self.select_judge is not None and len(applicable) > 5:
            try:
                idx = self.select_judge.select(
                    dec.quote, [c.quote for c in applicable])
                applicable = [applicable[idx]]
            except Exception as e:                    # noqa: BLE001
                report.fail("select", event.event_id, e)
                # 选卡失败 → 降级为弃权登记（不拿全卡堆进超分布 prompt 硬判）
                self._record_select_failure(dec, applicable, event, report, e)
                return []

        try:
            j = self.judge.judge(dec, applicable, event.text,
                                 event_kind=event.kind, scope=self.scope)
        except Exception as e:                        # noqa: BLE001
            report.fail("judge", event.event_id, e)
            self._record_failure_judgment(dec, constraints, event, report, e)
            return []

        try:
            self.ledger.add_judgment(j)
            report.judgments += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("judgment_record", event.event_id, e)
            return []

        if j.verdict == "supports_conflict" and not j.abstain:
            # 第二级复核（LLM 定量）：冲突在投递前一律升级复核——低置信不是
            # "丢弃"而是"升级"。复核判官看完整决策 + 全部约束 + 第一级结论，
            # 输出 KEEP（维持提醒）/ DROP（判定不可靠 → 降级弃权）。
            # 两级判定都落账。数值门槛方案已被否决：margin 会随训练/领域
            # 漂移（run6 margin 10.688 的自信错误、v8 后分布整体漂移），
            # 在其上设裁决阈值与关键词枚举同族。
            if self.verify_judge is not None:
                try:
                    v = self.verify_judge.verify(dec, constraints, j,
                                                 decision_text=event.text)
                except Exception as e:                # noqa: BLE001
                    report.fail("verify", event.event_id, e)
                    v = None
                if v is not None:
                    try:
                        self.ledger.add_judgment(v)
                        report.judgments += 1
                    except Exception as e:            # noqa: BLE001
                        report.fail("verify_record", event.event_id, e)
                    if v.abstain or v.verdict == "insufficient_info":
                        return []                     # 复核推翻 → 降级，不投递
                    j = v                             # 复核维持 → 用复核结论投递
            # 内容基去重键：同一决策文本 × 同一约束集 = 同一问题，跨事件去重。
            key_src = "\x00".join(
                [j.jtype, j.verdict, dec.quote]
                + sorted(c.entity_id for c in constraints))
            dk = "rem_" + hashlib.sha256(
                key_src.encode("utf-8")).hexdigest()[:24]
            # 会话预算：防刷屏保险丝（超预算落账不投递，数量上限，不读内容）
            if self.max_per_session and \
                    self._session_reminded >= self.max_per_session:
                return []
            self._session_reminded += 1
            return [Reminder(jtype=j.jtype, verdict=j.verdict,
                             subject_ids=j.subject_ids,
                             reference_ids=j.reference_ids,
                             reason=j.reason, model_id=j.model_id,
                             dedup_key=dk)]
        return []

    def _record_select_failure(self, dec: Entity, applicable, event: Event,
                               report: LoopReport, exc: BaseException) -> None:
        """选卡判官失败 → 登记显式弃权（不拿全卡硬判，也不静默放行）。"""
        try:
            j = Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=[dec.entity_id],
                reference_ids=[c.entity_id for c in applicable],
                evidence_ids=[], verdict="insufficient_info",
                model_id=getattr(self.select_judge, "model_id", "select"),
                abstain=True,
                abstain_reason=f"选卡判官失败: {type(exc).__name__}",
                reason="约束选择失败，义务判定未执行（显式弃权）")
            self.ledger.add_judgment(j)
            report.judgments += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("select_failure_record", event.event_id, e)

    def _record_gate_abstain(self, dec: Entity, skipped, event: Event,
                             report: LoopReport) -> None:
        """全部约束均不适用 → 登记显式弃权（零 LLM，不伪装成合规）。"""
        try:
            reasons = "；".join(ap.reason for _, ap in skipped) or "无适用约束"
            j = Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=[dec.entity_id],
                reference_ids=[c.entity_id for c, _ in skipped],
                evidence_ids=[], verdict="insufficient_info",
                model_id="applicability-gate",
                abstain=True,
                abstain_reason=f"约束不适用于本决策：{reasons}",
                reason="L0 适用范围闸门判定：无适用约束，未做语义裁决")
            self.ledger.add_judgment(j)
            report.judgments += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("gate_abstain_record", event.event_id, e)

    def _record_failure_judgment(self, dec: Entity, constraints: List[Entity],
                                 event: Event, report: LoopReport,
                                 exc: BaseException) -> None:
        """裁决后端失败 → 登记显式 abstain 判断（绝不伪装成合规）。"""
        try:
            j = Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=[dec.entity_id],
                reference_ids=[c.entity_id for c in constraints],
                evidence_ids=[], verdict="insufficient_info",
                model_id=getattr(self.judge, "model_id", "unknown"),
                abstain=True,
                abstain_reason=f"judge backend failed: {type(exc).__name__}",
                reason="裁决后端失败，本条未获结论（如实登记）")
            self.ledger.add_judgment(j)
            report.judgments += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("failure_judgment_record", event.event_id, e)

    # ── 目标验收（goal_coverage）──────────────────────────────────────────

    def _register_claim(self, event: Event, report: LoopReport) -> List[Reminder]:
        """助手完成宣称 → 登记 completion_claim → 对每个有效目标裁决覆盖度。

        关键立场：**宣称不是事实**。宣称只触发对账，不产生"已完成"结论；
        没有证据时由语义层给出 no_evidence/insufficient_info，机械层不得
        因为"助手说了完成"就放行。
        """
        try:
            quote = self.claim_detector(event.text)
        except Exception as e:                        # noqa: BLE001
            report.fail("claim_detect", event.event_id, e)
            return []
        if not quote:
            return []
        try:
            claim = Entity(
                entity_id=entity_id("completion_claim", event.event_id, quote),
                kind="completion_claim", quote=quote, span=None,
                event_id=event.event_id, source_role="assistant_claim",
                scope=self.scope)
            self.ledger.add_entity(claim)
            report.entities_added += 1
            report.claims_registered += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("claim_entity", event.event_id, e)
            return []

        try:
            goals = self.ledger.effective("goal", self.scope)
        except Exception as e:                        # noqa: BLE001
            report.fail("goal_query", event.event_id, e)
            return []
        if not goals:
            return []                                 # 无目标登记：无从对账

        try:
            evidence = self._collect_evidence()
        except Exception as e:                        # noqa: BLE001
            report.fail("evidence_collect", event.event_id, e)
            evidence = []

        out: List[Reminder] = []
        for goal in goals:
            j = self._coverage_for(goal, claim, evidence, goals, event, report)
            if j is None:
                continue
            if j.verdict in ("no_evidence", "supported_partial") and not j.abstain:
                out.append(Reminder(
                    jtype=j.jtype, verdict=j.verdict,
                    subject_ids=j.subject_ids, reference_ids=j.reference_ids,
                    reason=j.reason, model_id=j.model_id))
        return out

    def _coverage_for(self, goal, claim, evidence, goals, event,
                      report: LoopReport) -> Optional[Judgment]:
        try:
            j = self.coverage_judge.judge(goal, claim, evidence, goals_all=goals)
        except Exception as e:                        # noqa: BLE001
            report.fail("coverage_judge", event.event_id, e)
            j = Judgment(
                judgment_id=_next_jid(), jtype="goal_coverage",
                subject_ids=[claim.entity_id],
                reference_ids=[goal.entity_id], evidence_ids=[],
                verdict="insufficient_info",
                model_id=getattr(self.coverage_judge, "model_id", "unknown"),
                abstain=True,
                abstain_reason=f"coverage backend failed: {type(e).__name__}",
                reason="目标验收后端失败，本条未获结论（如实登记）")
        try:
            self.ledger.add_judgment(j)
            report.judgments += 1
        except Exception as e:                        # noqa: BLE001
            report.fail("coverage_record", event.event_id, e)
            return None
        return j

    def _collect_evidence(self) -> List[dict]:
        """机械证据收集：evidence_ref 实体的 kind/result 元数据。

        只取机械登记的事实（哪个命令、什么结果），不取模型转述——
        语义关系由 goal_coverage 裁决负责。
        """
        out = []
        for e in self.ledger.effective("evidence_ref", self.scope):
            meta = getattr(e, "meta", None) or {}
            out.append({"evidence_id": e.entity_id,
                        "kind": meta.get("evidence_kind", "unknown"),
                        "result": meta.get("result", "unknown"),
                        "note": meta.get("note") or e.quote})
        return out

    @staticmethod
    def build_reminder(judgment: Judgment) -> Reminder:
        return Reminder(jtype=judgment.jtype, verdict=judgment.verdict,
                        subject_ids=judgment.subject_ids,
                        reference_ids=judgment.reference_ids,
                        reason=judgment.reason,
                        model_id=judgment.model_id)

    def register_evidence(self, event_id: str, evidence_kind: str,
                          result: str, note: str = "",
                          quote: Optional[str] = None) -> Optional[Entity]:
        """机械证据登记（工具结果/产物/测试运行）。

        只登记机械事实：evidence_kind ∈ EVIDENCE_KINDS，result ∈
        EVIDENCE_RESULTS。"测试通过"是一条**证据**，不是"目标完成"的结论——
        两者的连接必须由 goal_coverage 语义裁决完成。
        """
        from .schema import EVIDENCE_KINDS, EVIDENCE_RESULTS
        if evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"illegal evidence kind: {evidence_kind!r}")
        if result not in EVIDENCE_RESULTS:
            raise ValueError(f"illegal evidence result: {result!r}")
        ent = Entity(
            entity_id=entity_id("evidence_ref", event_id,
                                quote or f"{evidence_kind}:{result}:{note}"),
            kind="evidence_ref", quote=(quote or note or
                                        f"{evidence_kind}={result}"),
            span=None, event_id=event_id, source_role="tool_observation",
            scope=self.scope,
            meta={"evidence_kind": evidence_kind, "result": result,
                  "note": note})
        return self.ledger.add_entity(ent)

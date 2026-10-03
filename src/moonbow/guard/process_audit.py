# -*- coding: utf-8 -*-
"""progress_guard.process_audit

阶段审计引擎（过程审计）：对思考块、中间汇报块、工具证据做**确定性**差异核对。
与收尾裁决（verifier.ProgressGuard）严格分离：

- 输入是观察块流，不是收尾申报；不套用 STATUS/REMAINING/EVIDENCE 格式门禁。
- 输出是发现（StageFinding）+ 可选提醒（具体差异/依据/建议），不是退出许可。
- 全部规则确定性、无权重依赖 —— lazy/skeleton 模式下同样可用，可完全单测。

判定尺度（goal 约定）：
- 思考中的临时假设只是 candidate；同一块内自行解决的疑点不升级。
- "准备做"不等于"已经做"；工具成功不自动等于目标达成。
- 修改前的测试不支持修改后的完成声明（时序核对）。
- 证据缺失 = unverified（未知），不判失败。
- 补做工作与如实修正完成声明都是合法响应 —— 提醒给出两条路。
"""
import re
import zlib
from typing import Dict, List, Optional

from .protocol import StageBlock, StageFinding
from .audit_semantic import ShadowObservation
from .convergence import (
    FAIL_STREAK_SIGNAL,
    PERF_RETEST_SIGNAL,
    REPEAT_SIGNAL,
    STALL_SIGNAL,
    AdvisoryBudget,
    build_advisory_reminder,
    perf_retest_enabled,
    req_hash_of,
    unavailable_entry,
)

# 完成声明（中间汇报里的"已做完"语气；注意排除"准备/计划/将"）
_CLAIM_PAT = re.compile(
    r"(已经?(?:修复|完成|解决|实现|修好)|(?:修复|完成|解决)了|tests?\s+(?:all\s+)?pass(?:ed)?|"
    r"is\s+fixed|全部完成|搞定了|works now)", re.IGNORECASE)
# 修改/动作意图（用于时序基准："我接下来要改 X"）
_INTENT_PAT = re.compile(
    r"(准备|接下来|即将|马上|我现在去|我这就|will\s+(?:fix|change|modify|update)|"
    r"going to|let me (?:fix|change|modify|update))", re.IGNORECASE)
# 思考中的未决疑点
_CONCERN_PAT = re.compile(
    r"(遗漏|没考虑|忘了|不确定|可能有?(?:问题|bug|风险)|需要?(?:再|重新)?(?:检查|验证)|"
    r"suspicious|not sure|might (?:be wrong|break)|missed)", re.IGNORECASE)
# 同块内的自行解决（有此标记的疑点停留 candidate）
_RESOLVE_PAT = re.compile(
    r"(所以(?:我|需要)|那就|接下来(?:我)?(?:补|加|修|改|验证)|i(?:'| a)ll (?:add|fix|verify|check)|"
    r"let me (?:add|fix|verify|check))", re.IGNORECASE)

# 写入类工具：成功只证明"改动已落盘"，**不证明"改动有效"**——不能充当
# 完成声明的验证证据。旧实现把它们一并计入 last_success_tool，于是"改完
# 没验证就宣布完成"（真实 SWE 运行里最普遍的失败模式）恰好落在规则缝隙里：
# last_success_tool 被 edit 的写入结果覆盖 → 2.2 的 `ls.seq < li` 不成立
# → 2.1/2.2/2.3 三条分支全不命中 → 零发现。此处把"证据"收窄为验证性质。
_WRITE_TOOLS = frozenset({"edit", "write", "apply_patch", "multiedit",
                          "str_replace", "create", "notebook_edit"})


def _is_verification_result(b: StageBlock) -> bool:
    """该结果是否构成"验证证据"而非仅写入成功。

    tool_name 缺失时保守判为验证性质：宁可漏一次提醒，也不把写入当验证
    （那会把未验证的完成声明直接放行）。
    """
    if not b.tool_name:
        return True
    return b.tool_name.lower() not in _WRITE_TOOLS

_EXCERPT = 120


def _excerpt(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:_EXCERPT] + ("…" if len(t) > _EXCERPT else "")


def _fp(kind: str, key: str) -> str:
    return f"{kind}:{key}"


class StageAuditor:
    """确定性阶段审计器。audit() 幂等：同一批块重复送审产生相同 fingerprint，
    由客户端按 fingerprint 去重合并。"""

    def __init__(self, semantic_shadow=None, convergence_shadow=None):
        # P7：可选语义声明检测 shadow 通道（见 audit_semantic.py）。
        # 默认 None → 零行为变化；注入时其结果只写入返回值的
        # shadow_semantic 键，绝不参与 reminder/findings/semantic 判定。
        self._semantic_shadow = semantic_shadow
        # Phase 1：可选收敛信号 shadow 通道（见 convergence.py）。
        # 同模式：默认 None → 零行为变化；注入时结果只写入
        # shadow_convergence 键，绝不参与 reminder/findings/semantic 判定。
        self._convergence_shadow = convergence_shadow

    def audit(
        self,
        req: str,
        blocks: List[StageBlock],
        prior_findings: Optional[List[StageFinding]] = None,
        snapshot_version: int = 0,
        stream_ended: bool = False,
    ) -> Dict:
        """审计新增观察块，返回 {findings, reminder, semantic, based_on}。

        findings：本批产生的发现（含 prior 中被本批证据更新的状态）。
        reminder：advisory 模式下值得介入的第一条 actionable 的提醒文案；
                  shadow 模式同样返回（客户端决定是否投递）。
        注入 shadow provider 时额外返回 shadow_semantic /
        shadow_convergence 键（只记录、零投递）；缺省不含，零行为变化。
        """
        prior = list(prior_findings or [])
        findings: List[StageFinding] = []
        findings.extend(self._check_thinking(blocks, snapshot_version))
        findings.extend(self._check_claims_vs_tools(blocks, snapshot_version, stream_ended))
        findings = self._revise_prior(prior, findings, blocks)
        reminder = self._first_reminder(findings, req)
        result = {
            "findings": [f.to_dict() for f in findings],
            "reminder": reminder,
            "semantic": any(f.kind in ("contradiction", "evidence-order") for f in findings),
            "based_on": snapshot_version,
        }
        if self._semantic_shadow is not None:
            # shadow 通道：只观察中间汇报文本，结果单独成键；
            # 观察自身异常也不得影响主审计结果。
            shadow = []
            for b in sorted(blocks, key=lambda x: x.seq):
                if b.kind != "text" or not b.text.strip():
                    continue
                try:
                    obs = self._semantic_shadow.observe(b.text)
                except Exception as e:               # noqa: BLE001 — 失败隔离
                    obs = ShadowObservation(
                        unavailable_reason=f"error:{type(e).__name__}")
                entry = obs.to_dict()
                entry["seq"] = b.seq
                shadow.append(entry)
            result["shadow_semantic"] = shadow
        if self._convergence_shadow is not None:
            # Phase 1 收敛 shadow 通道：从块流确定性计算（纯函数、幂等），
            # 结果单独成键；计算异常同样失败隔离，绝不影响主审计结果。
            try:
                result["shadow_convergence"] = \
                    self._convergence_shadow.compute(req, blocks)
            except Exception as e:               # noqa: BLE001 — 失败隔离
                result["shadow_convergence"] = [
                    unavailable_entry(f"error:{type(e).__name__}")]
        return result

    # ---- 规则 1：思考疑点（candidate / 被解决则停留 candidate）----
    def _check_thinking(self, blocks: List[StageBlock], ver: int) -> List[StageFinding]:
        out: List[StageFinding] = []
        for b in blocks:
            if b.kind != "thinking" or not b.text.strip():
                continue
            sentences = re.split(r"[。；;.!?\n]", b.text)
            for s in sentences:
                if len(s.strip()) < 6 or not _CONCERN_PAT.search(s):
                    continue
                resolved_in_block = bool(_RESOLVE_PAT.search(b.text))
                out.append(StageFinding(
                    # zlib.crc32 跨进程稳定（内建 hash 受 PYTHONHASHSEED 盐化，
                    # 重启后同文本会生成不同 fingerprint，破坏去重与合并）
                    fingerprint=_fp("thinking-concern",
                                    f"{b.seq}:{zlib.crc32(s.strip().encode('utf-8')) & 0xffff:04x}"),
                    kind="thinking-concern",
                    status="candidate" if resolved_in_block else "unverified",
                    summary=f"思考中出现未决疑点：{_excerpt(s)}",
                    evidence=[{"seq": b.seq, "kind": "thinking", "excerpt": _excerpt(s)}],
                    snapshot_version=ver,
                ))
        return out

    # ---- 规则 2：完成声明 vs 工具证据（时序 + 矛盾 + 未知）----
    def _check_claims_vs_tools(self, blocks: List[StageBlock], ver: int,
                               stream_ended: bool = False) -> List[StageFinding]:
        out: List[StageFinding] = []
        last_intent_seq: Optional[int] = None
        last_claim: Optional[StageBlock] = None
        last_success_tool: Optional[StageBlock] = None
        last_failed_tool: Optional[StageBlock] = None
        last_write_tool: Optional[StageBlock] = None

        for b in sorted(blocks, key=lambda x: x.seq):
            if b.kind in ("text", "thinking"):
                if _INTENT_PAT.search(b.text):
                    last_intent_seq = b.seq
                if _CLAIM_PAT.search(b.text):
                    last_claim = b
            elif b.kind == "toolResult":
                if b.is_error:
                    last_failed_tool = b
                elif _is_verification_result(b):
                    last_success_tool = b
                else:
                    # 写入成功：记为"改动落盘"，不进入验证证据
                    last_write_tool = b

        if last_claim is None:
            # 无显式完成声明时的隐式完成点（2026-09-22）：
            # 真实 SWE 运行里模型极少输出"已完成"字样——它以工具调用静默
            # 终止。旧实现从这里直接 return，导致 9 个真实会话回放全部零发现
            # （claims=0）。监工不能依赖被监工者主动申报，故以**运行终止点**
            # 作为隐式收尾点：只要跑完这段记录，最后一次写入仍未被验证，
            # 就构成"已开未关"。这是"推理出完成度"的最小实现。
            if last_write_tool is None:
                # 无任何成功写入时，检查"写入尝试全部失败"（2026-09-22）：
                # 实测 01a0c5fd 三次 edit 全被拒（Validation failed /
                # No changes made），从未落盘，旧规则因为"没有成功写入"
                # 而沉默——但反复失败本身正是子任务开了没关。
                failed_writes = [b for b in blocks
                                 if b.kind == "toolResult" and b.is_error
                                 and not _is_verification_result(b)]
                if len(failed_writes) >= 2:
                    last = failed_writes[-1]
                    out.append(StageFinding(
                        fingerprint=_fp("write-attempts-failed", f"{last.seq}"),
                        kind="unverified-claim", status="actionable",
                        summary=f"运行中有 {len(failed_writes)} 次代码改动尝试全部失败"
                                f"（最后一次第{last.seq}块，{last.tool_name}），"
                                "且模型未申报受阻或完成状态",
                        evidence=[{"seq": last.seq, "kind": "toolResult",
                                   "excerpt": _excerpt(last.text)}],
                        snapshot_version=ver,
                    ))
                return out
            if (last_success_tool is not None
                    and last_success_tool.seq > last_write_tool.seq):
                return out  # 最后一次改动之后有验证 -> 视为闭环
            # 终止确认（2026-09-22）：只在**运行确已结束**（agent_end 置位的
            # stream_ended）时才判"跑了没验证"。缺这个信号时，服务端只能在
            # 每批增量上猜终止，于是 write 刚落盘就误报"改了没验证"——
            # 实测 S2：模型下一个动作正是 cat 验证，却连锁触发 3 次误报注入。
            if not stream_ended:
                return out
            out.append(StageFinding(
                fingerprint=_fp("unverified-write-terminal", f"{last_write_tool.seq}"),
                kind="unverified-claim", status="actionable",
                summary=f"运行结束于第{last_write_tool.seq}块代码改动"
                        f"（{last_write_tool.tool_name}）之后，"
                        "该改动未经验证，且模型未申报完成状态",
                evidence=[{"seq": last_write_tool.seq, "kind": "toolResult",
                           "excerpt": _excerpt(last_write_tool.text)}],
                snapshot_version=ver,
            ))
            return out

        claim = last_claim
        # 2.1 声明之后出现失败的工具结果 -> 矛盾
        if last_failed_tool and last_claim and last_failed_tool.seq > last_claim.seq:
            out.append(StageFinding(
                fingerprint=_fp("contradiction", f"{claim.seq}:{last_failed_tool.seq}"),
                kind="contradiction", status="actionable",
                summary=f"声明完成（第{claim.seq}块）之后出现失败的工具调用"
                        f"（第{last_failed_tool.seq}块，{last_failed_tool.tool_name}）",
                evidence=[
                    {"seq": claim.seq, "kind": claim.kind, "excerpt": _excerpt(claim.text)},
                    {"seq": last_failed_tool.seq, "kind": "toolResult",
                     "excerpt": _excerpt(last_failed_tool.text)},
                ],
                snapshot_version=ver,
            ))

        # 2.2 时序：证据全部早于最近一次修改意图 -> 不能支持"修改后已验证"。
        #     前提：完成声明发生在该意图之后（claim.seq > intent.seq）——
        #     若声明先于意图，意图是声明之后的新工作计划，既有证据仍可有效
        #     支持该声明，不构成时序缺口。
        if (last_intent_seq is not None and claim.seq > last_intent_seq
                and last_success_tool is not None
                and last_success_tool.seq < last_intent_seq):
            out.append(StageFinding(
                fingerprint=_fp("evidence-order", f"{claim.seq}:{last_intent_seq}"),
                kind="evidence-order", status="actionable",
                summary=f"完成声明（第{claim.seq}块）仅有早于最近修改意图"
                        f"（第{last_intent_seq}块）的工具证据（第{last_success_tool.seq}块）——"
                        "修改前的验证不能证明修改后的状态",
                evidence=[
                    {"seq": claim.seq, "kind": claim.kind, "excerpt": _excerpt(claim.text)},
                    {"seq": last_intent_seq, "kind": "text", "excerpt": "（修改意图块）"},
                    {"seq": last_success_tool.seq, "kind": "toolResult",
                     "excerpt": _excerpt(last_success_tool.text)},
                ],
                snapshot_version=ver,
            ))
        # 2.3 写入后无验证 -> 该改动"已开未关"（最常见的真实失败模式）。
        #     写入成功只说明落盘，完成声明仍需验证证据；只要最后一次写入
        #     之后（或整段记录里）没有任何验证类结果，就构成 actionable 缺口。
        #     与 2.2 的区别：2.2 管"验证早于改动"，本条管"根本没有验证"。
        if (last_write_tool is not None
                and (last_success_tool is None
                     or last_success_tool.seq < last_write_tool.seq)):
            out.append(StageFinding(
                fingerprint=_fp("unverified-write", f"{claim.seq}:{last_write_tool.seq}"),
                kind="unverified-claim", status="actionable",
                summary=f"完成声明（第{claim.seq}块）之前有代码改动"
                        f"（第{last_write_tool.seq}块，{last_write_tool.tool_name}），"
                        "但该改动之后没有任何验证类证据",
                evidence=[
                    {"seq": claim.seq, "kind": claim.kind, "excerpt": _excerpt(claim.text)},
                    {"seq": last_write_tool.seq, "kind": "toolResult",
                     "excerpt": _excerpt(last_write_tool.text)},
                ],
                snapshot_version=ver,
            ))
        # 2.4 无任何工具证据 -> 未知，不判失败
        elif last_success_tool is None:
            out.append(StageFinding(
                fingerprint=_fp("unverified-claim", f"{claim.seq}"),
                kind="unverified-claim", status="unverified",
                summary=f"完成声明（第{claim.seq}块）在当前记录中未见对应工具验证证据",
                evidence=[{"seq": claim.seq, "kind": claim.kind, "excerpt": _excerpt(claim.text)}],
                snapshot_version=ver,
            ))
        return out

    # ---- 规则 3：新证据复核 prior 发现（解决/撤销）----
    def _revise_prior(
        self, prior: List[StageFinding], fresh: List[StageFinding],
        blocks: List[StageBlock],
    ) -> List[StageFinding]:
        merged = {f.fingerprint: dict(f.to_dict()) for f in prior}
        for f in fresh:
            merged[f.fingerprint] = f.to_dict()
        # 证据时序类发现：若本批出现了晚于意图块的成功工具证据，标记 resolved
        ok_tools = [b.seq for b in blocks if b.kind == "toolResult" and not b.is_error]
        for d in merged.values():
            if d["kind"] == "evidence-order" and d["status"] == "actionable" and ok_tools:
                intent_seqs = [e["seq"] for e in d["evidence"] if e.get("kind") == "text"]
                if intent_seqs and max(ok_tools) > max(intent_seqs):
                    d["status"] = "resolved"
                    d["summary"] += "（后续已出现更晚的成功工具证据）"
        return [StageFinding.from_dict(d) for d in merged.values()]

    # ---- 提醒：只挑 actionable，给出差异/依据/两条合法出路 ----
    def _first_reminder(self, findings: List[StageFinding], req: str) -> Optional[Dict]:
        actionable = [f for f in findings if f.status == "actionable"]
        if not actionable:
            return None
        f = actionable[0]
        ev = "\n".join(f"- 第{e['seq']}块（{e['kind']}）：{e.get('excerpt','')}"
                       for e in f.evidence)
        summary = {
            "contradiction": "你报告完成，但之后的工具调用失败了。",
            "evidence-order": "你报告修改后验证通过，但记录中的验证发生在该修改之前。",
        }.get(f.kind, f.summary)
        return {
            "summary": summary,
            "fingerprint": f.fingerprint,
            "evidence": [e.get("excerpt", "") for e in f.evidence],
            "suggestion": (
                f"【进度守卫（过程核查）】{summary}\n依据：\n{ev}\n"
                # 先给"继续做"的明确指令，再提申报：实测把申报选项放前面会
                # 让弱主动性模型选择"如实申报未完成"并提前终止（2026-09-22）。
                "请立刻补做验证：修改后重跑测试/构建并给出结果；"
                "若确有未完成的改动，继续调用工具做完。"
                "不要仅以文字说明代替执行。"
                f"（对照需求：{_excerpt(req) if req else '当前任务'}）"
            ),
        }


# 投递优先级：stall > fail_streak > repeat（v2）；perf_retest 独立预算键，
# 排最后（收尾时刻其它触发通常不命中，不影响其可达性）。
_DELIVERY_ORDER = {
    STALL_SIGNAL: 0,
    FAIL_STREAK_SIGNAL: 1,
    REPEAT_SIGNAL: 2,
    PERF_RETEST_SIGNAL: 3,
}


def _mark_delivered(entries: List, signal: str) -> None:
    for e in entries:
        if isinstance(e, dict) and e.get("signal") == signal:
            e["delivered"] = True


def _apply_convergence_advisory(result: Dict, task_key: str,
                                advisory: "AdvisoryBudget",
                                perf_enabled: bool = False) -> None:
    """Phase 2：advisory 模式下把命中的收敛触发包装成 reminder 投递。

    - 只包装、不改触发规则：命中判定完全来自 shadow_convergence 条目；
    - 主审计已有 reminder 时让位（单一 reminder 通道，不覆盖不打架）；
    - 投递状态写回条目（delivered: true 仅命中的那条）；既有信号预算由
      AdvisoryBudget 独立计数（每任务最多 2 次，按 (task_key, signal) 记）；
    - v2（2026-10-04）：converge.repeat 走既有预算；converge.perf_retest
      走独立预算键（allow_solo/record_solo，每任务 1 次，不占 MAX_PER_TASK、
      不计 used()），且需 perf_enabled（MOONBOW_GUARD_PERF_RETEST=1 或
      目标含 speedup/perf 字样）才投递。
    """
    entries = result.get("shadow_convergence")
    if not isinstance(entries, list):
        return
    for e in entries:
        if isinstance(e, dict):
            e.setdefault("delivered", False)
    if result.get("reminder"):
        return                     # 主审计过程提醒优先
    matched = sorted((e for e in entries
                      if isinstance(e, dict) and e.get("matched") is True),
                     key=lambda e: _DELIVERY_ORDER.get(e.get("signal"), 9))
    capped = [e for e in matched if e.get("signal") != PERF_RETEST_SIGNAL]
    if capped:
        # 既有口径：只考虑最高优先命中；其预算已用 → 本轮静默
        # （不落到更低优先级顶信号）。
        signal = capped[0]["signal"]
        if advisory.allow(task_key, signal):
            advisory.record(task_key, signal)
            _mark_delivered(entries, signal)
            result["reminder"] = build_advisory_reminder(signal, task_key,
                                                         capped[0])
            return
    perf = next((e for e in matched if e.get("signal") == PERF_RETEST_SIGNAL),
                None)
    if (perf is not None and perf_enabled
            and advisory.allow_solo(task_key, PERF_RETEST_SIGNAL)):
        # 独立预算键：顶信号被既有预算挡下也不影响（互不占额）
        advisory.record_solo(task_key, PERF_RETEST_SIGNAL)
        _mark_delivered(entries, PERF_RETEST_SIGNAL)
        result["reminder"] = build_advisory_reminder(
            PERF_RETEST_SIGNAL, task_key, perf)


def _convergence_shadow_for(payload: Dict, default):
    """可选收敛目标注入：payload 可带 convergence_goals_total（int>=1）/
    convergence_target_tests（非空唯一字符串列表）构造对应观察者；
    均缺省时用 server 级默认实例（Phase 1 行为，零变化）。非法即 400。
    default 是 ConvergenceShadow 时经 with_targets 派生，保留 ruleset/
    budget（v2 装配不因 payload 带目标而降级 v1）；default 为 None 时维持
    原构造路径（v1）。"""
    goals = payload.get("convergence_goals_total")
    targets = payload.get("convergence_target_tests")
    if goals is None and targets is None:
        return default
    from .convergence import ConvergenceShadow
    derive = getattr(default, "with_targets", None)
    if targets is not None:
        if (not isinstance(targets, list) or not targets
                or not all(isinstance(t, str) and t.strip() for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("convergence_target_tests must be a non-empty list "
                             "of unique non-empty strings")
        cleaned = [t.strip() for t in targets]
        if derive is not None:
            return derive(target_tests=cleaned)
        return ConvergenceShadow(target_tests=cleaned)
    if not isinstance(goals, int) or isinstance(goals, bool) or goals < 1:
        raise ValueError("convergence_goals_total must be an integer >= 1")
    if derive is not None:
        return derive(goals_total=goals)
    return ConvergenceShadow(goals_total=goals)


def audit_stage_payload(payload: Dict, convergence_shadow=None,
                        convergence_advisory: "AdvisoryBudget" = None) -> Dict:
    """HTTP 层入口：解析/校验 payload 并执行审计。校验失败抛 ValueError。

    convergence_shadow：可选收敛 shadow 观察者（Phase 1，server 层按
    enable_convergence_shadow 显式开启后注入）；缺省 None = 零行为变化。
    convergence_advisory：可选收敛提示预算（Phase 2，server 仅在
    MOONBOW_GUARD_CONVERGENCE=advisory 时注入）；缺省 None = shadow 原样
    （off/shadow 行为与现状逐字节一致）。注入后触发命中可进 reminder，
    投递状态写进 shadow_convergence 条目的 delivered 键。
    """
    blocks_raw = payload.get("blocks", [])
    if not isinstance(blocks_raw, list):
        raise ValueError("blocks must be a list")
    blocks = [StageBlock.from_dict(b) for b in blocks_raw if isinstance(b, dict)]
    prior_raw = payload.get("findings", [])
    if not isinstance(prior_raw, list):
        raise ValueError("findings must be a list")
    prior = [StageFinding.from_dict(f) for f in prior_raw if isinstance(f, dict)]
    ver = int(payload.get("snapshot_version", 0) or 0)
    stream_ended = bool(payload.get("stream_ended", False))
    req = str(payload.get("req", "") or "")
    import os as _os
    if _os.environ.get("AUDIT_DEBUG"):
        import json as _json, sys as _sys
        print("[audit-debug] blocks=%d kinds=%s" % (
            len(blocks), [b.kind for b in blocks]), file=_sys.stderr, flush=True)
        for b in blocks:
            print("[audit-debug]   seq=%s kind=%s tool=%s text=%r" % (
                b.seq, b.kind, b.tool_name, b.text[:70]), file=_sys.stderr, flush=True)
    shadow = _convergence_shadow_for(payload, convergence_shadow)
    result = StageAuditor(convergence_shadow=shadow).audit(
        req, blocks, prior, ver, stream_ended)
    if convergence_advisory is not None:
        # task_id 优先（同 req 文本的多个任务实例预算独立）；缺省回退 req 哈希
        task_key = str(payload.get("task_id") or "") or req_hash_of(req)
        # v2 perf_retest 门控：env 开关或性能目标字样（payload 注入的
        # target_tests 是客户端 MOONBOW_GUARD_CONVERGENCE_TARGETS 的透传，
        # 服务端 env 未必带该变量，故一并检查）。
        perf_flag = perf_retest_enabled(
            target_tests=getattr(shadow, "target_tests", None))
        _apply_convergence_advisory(result, task_key, convergence_advisory,
                                    perf_enabled=perf_flag)
    return result

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

    def __init__(self, semantic_shadow=None):
        # P7：可选语义声明检测 shadow 通道（见 audit_semantic.py）。
        # 默认 None → 零行为变化；注入时其结果只写入返回值的
        # shadow_semantic 键，绝不参与 reminder/findings/semantic 判定。
        self._semantic_shadow = semantic_shadow

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


def audit_stage_payload(payload: Dict) -> Dict:
    """HTTP 层入口：解析/校验 payload 并执行审计。校验失败抛 ValueError。"""
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
    return StageAuditor().audit(req, blocks, prior, ver, stream_ended)

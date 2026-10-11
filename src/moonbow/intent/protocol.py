# -*- coding: utf-8 -*-
"""moonbow.intent.protocol — 裁决任务（T3）的输入协议构建器。

存在理由（第二轮审计 D1）：训练脚本与运行时各自拼 prompt，运行时甚至完全
没把约束送进模型，导致"换约束不改变输出"、实验结论不可迁移。本模块是
**唯一**的裁决输入构造点，训练与运行时共用，保证：

1. 当前有效约束原文进入输入（不是只写进引用字段）；
2. 决策完整文本可追溯（超长分块，不静默截断）；
3. 事件类型、scope 等机械元数据显式进入输入；
4. 截断/超长显式报告，不伪装成完整输入。

本模块只做机械拼接与长度核算，不做任何语义判断（零 LLM）。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# 三分类标签：运行时与训练端必须一致（审计 D2）
LABELS = ("violation", "compliant", "insufficient")

# 首 token 比较用的字面量（与仓库 slm.py 受限输出协议一致）
LABEL_LITERALS = {"violation": "violation",
                  "compliant": "compliant",
                  "insufficient": "insufficient"}

VERDICT_OF_LABEL = {"violation": "supports_conflict",
                    "compliant": "supports_compat",
                    "insufficient": "insufficient_info"}
LABEL_OF_VERDICT = {v: k for k, v in VERDICT_OF_LABEL.items()}

INSTRUCTION = ("问：该新增决策是否违反上述约束？"
               "只输出一个词：violation、compliant 或 insufficient")

# 决策文本分块上限（字符）。超出则分块并显式标记，供捕获/裁决分片处理。
DECISION_CHUNK_CHARS = 4000
MAX_CONSTRAINT_CHARS = 1500


@dataclass
class JudgmentInput:
    """一次裁决的结构化输入（已可追溯，不丢证据）。"""
    prompt: str
    constraints: List[Tuple[str, str]] = field(default_factory=list)  # (entity_id, quote)
    decision_text: str = ""
    decision_chunks: List[str] = field(default_factory=list)
    event_kind: str = ""
    scope: str = ""
    truncated: bool = False
    truncation_note: str = ""
    protocol_version: str = "t3-v2"

    def to_dict(self) -> dict:
        return {"protocol_version": self.protocol_version,
                "event_kind": self.event_kind, "scope": self.scope,
                "constraint_count": len(self.constraints),
                "decision_chars": len(self.decision_text),
                "chunk_count": len(self.decision_chunks),
                "truncated": self.truncated,
                "truncation_note": self.truncation_note,
                "prompt_sha256": _sha(self.prompt)}


@dataclass
class CoverageInput:
    """目标覆盖裁决的结构化输入（goal_coverage）。

    与约束裁决同样遵循"当前有效原文必须进 prompt"：目标原文（含验收条件）
    与已登记证据的机械属性并列呈现，模型只做关系裁决，不自行发明证据。
    """
    prompt: str
    goal: Tuple[str, str] = ("", "")            # (entity_id, quote)
    claim: Tuple[str, str] = ("", "")           # (entity_id, quote)
    evidence: List[dict] = field(default_factory=list)  # 机械元数据，非文本
    truncated: bool = False
    truncation_note: str = ""
    protocol_version: str = "t4-v1"

    def to_dict(self) -> dict:
        return {"protocol_version": self.protocol_version,
                "goal_id": self.goal[0], "claim_id": self.claim[0],
                "evidence_count": len(self.evidence),
                "evidence_results": self.evidence_results,
                "truncated": self.truncated,
                "prompt_sha256": _sha(self.prompt)}

    @property
    def evidence_results(self) -> List[str]:
        return [e.get("result") for e in self.evidence]


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def chunk_text(text: str, size: int = DECISION_CHUNK_CHARS) -> List[str]:
    """按行边界分块，保证不切碎单行（决策文本可能含 diff/yaml 结构）。"""
    if len(text) <= size:
        return [text]
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > size and cur:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)
    return chunks or [text]


def build_judgment_input(decision_text: str, constraints: Sequence,
                         event_kind: str = "", scope: str = "",
                         card: Optional[str] = None) -> JudgmentInput:
    """构造裁决输入。constraints 为 Entity 序列（取 entity_id/quote）。

    card 为可选的约束卡（边界例说明）。约束**原文**始终进入 prompt——
    这是 D1 的修复点：模型必须看到当前有效约束本身。
    decision_text 应传完整事件文本（超长分块并显式标记，不静默截断）。
    """
    if not decision_text:
        decision_text = "（空决策文本）"
    cons: List[Tuple[str, str]] = []
    for c in constraints:
        quote = (getattr(c, "quote", None) or str(c)).strip()
        if len(quote) > MAX_CONSTRAINT_CHARS:
            quote = quote[:MAX_CONSTRAINT_CHARS]
        cons.append((getattr(c, "entity_id", ""), quote))

    chunks = chunk_text(decision_text)
    truncated = len(chunks) > 1

    lines = []
    if card:
        lines.append(card.strip())
    lines.append("## 当前有效约束")
    if cons:
        for eid, quote in cons:
            lines.append(f"- [{eid}] {quote}")
    else:
        lines.append("- （无有效约束登记）")
    lines.append("")
    lines.append(f"## 待判决策（事件类型：{event_kind or 'unknown'}；"
                 f"scope：{scope or 'unknown'}）")
    if truncated:
        lines.append(f"（决策文本 {len(decision_text)} 字，已分 {len(chunks)} 块，"
                     f"本块为第 1 块）")
    lines.append(chunks[0])
    lines.append("")
    lines.append(f"## 问题\n{INSTRUCTION}")

    return JudgmentInput(
        prompt="\n".join(lines), constraints=cons,
        decision_text=decision_text, decision_chunks=chunks,
        event_kind=event_kind, scope=scope,
        truncated=truncated,
        truncation_note=(f"decision split into {len(chunks)} chunks; "
                         f"prompt uses chunk 1" if truncated else ""))


def build_coverage_input(goal, claim, evidence, card: Optional[str] = None,
                         goals_all=None) -> CoverageInput:
    """构造目标覆盖裁决输入。

    goal：被对账的目标实体（Entity）；goals_all：同 scope 全部有效目标，
    用于让模型判断"宣称覆盖的是哪一项"（多目标时不至于张冠李戴）。
    evidence：机械登记的证据元数据列表，每项形如
    {"evidence_id","kind","result","note"}——只放机械事实，不放模型转述。
    """
    def _pair(e):
        return (getattr(e, "entity_id", ""), (getattr(e, "quote", "") or "")[:1200])

    gid, gquote = _pair(goal)
    cid, cquote = _pair(claim)
    ev = [{"evidence_id": e.get("evidence_id", ""),
           "kind": e.get("kind", "unknown"),
           "result": e.get("result", "unknown"),
           "note": (e.get("note") or "")[:300]} for e in (evidence or [])]

    lines = []
    if card:
        lines.append(card.strip())
    lines.append("## 目标（含验收条件）")
    lines.append(f"- [{gid}] {gquote}")
    others = [o for o in (goals_all or []) if getattr(o, "entity_id", "") != gid]
    if others:
        lines.append("## 同 scope 其他有效目标（判断宣称针对哪一项）")
        for o in others:
            lines.append(f"- [{getattr(o, 'entity_id', '')}] "
                         f"{(getattr(o, 'quote', '') or '')[:300]}")
    lines.append("")
    lines.append("## 完成宣称（助手自述，非完成事实）")
    lines.append(f"- [{cid}] {cquote}")
    lines.append("")
    lines.append("## 已登记机械证据")
    if ev:
        for e in ev:
            lines.append(f"- [{e['evidence_id']}] kind={e['kind']} "
                         f"result={e['result']} {e['note']}".rstrip())
    else:
        lines.append("- （无相关证据登记）")
    lines.append("")
    lines.append(f"## 问题\n{COVERAGE_BOUNDARY}\n\n{COVERAGE_INSTRUCTION}")

    total = len(gquote) + len(cquote) + sum(len(e["note"]) for e in ev)
    truncated = total > 6000
    return CoverageInput(prompt="\n".join(lines), goal=(gid, gquote),
                         claim=(cid, cquote), evidence=ev,
                         truncated=truncated,
                         truncation_note=("coverage input exceeds 6000 chars"
                                          if truncated else ""))


def first_token_ids(tokenizer) -> dict:
    """三标签首 token id（训练/运行时共用，避免各处重复拼字面量）。

    注意：标签首 token 可能**不唯一**（真实实验发现 supported_full 与
    supported_partial 首 token 都是 'supported'），因此首 token 比较只对
    首 token 互异的标签集成立——互异性由 assert_distinct_first_tokens 守卫。
    """
    ids = {}
    for label, lit in LABEL_LITERALS.items():
        toks = tokenizer(lit, add_special_tokens=False)["input_ids"]
        ids[label] = toks[0]
    return ids


def assert_distinct_first_tokens(ids: dict, labels=None) -> None:
    """首 token 互异性守卫：不互异则首 token 比较在语义上无效。

    这类标签（如覆盖四分类里的 supported_full / supported_partial）必须走
    label_sequence_scores（整序列似然），不能比首 token。
    """
    seen = {}
    for lab, tid in ids.items():
        if labels and lab not in labels:
            continue
        if tid in seen:
            raise ValueError(
                f"标签 {lab!r} 与 {seen[tid]!r} 首 token 相同（{tid}）："
                f"首 token 比较无效，请改用 label_sequence_scores")
        seen[tid] = lab


def sequence_scores(model, tokenizer, prompt_ids, labels,
                    literals=None, normalize=True) -> dict:
    """整序列似然打分：P(label | prompt) 的对数似然。

    首 token 比较的严格超集：标签单 token 且首 token 互异时与之一致；
    多 token 或共享前缀时（supported_full / supported_partial）只有本方法
    能把它们区分开。teacher forcing 一次前向拿到标签段全部位置的 logits。

    normalize 的取舍（真实实验发现）：按标签长度取平均会让**较长标签**
    获得系统性优势（标签 3 token 的 insufficient_info 压倒 2 token 的
    supported_*，且与输入内容无关）。评测时若各标签 token 数不同，
    应优先比较**原始总对数似然**（normalize=False）或改用等长别名。
    默认 True 仅为与历史报告口径一致；新实验请显式指定。
    """
    import torch
    lit_map = literals or LABEL_LITERALS
    out = {}
    device = prompt_ids.device
    for label in labels:
        lit = lit_map.get(label, label)
        l_ids = tokenizer(lit, add_special_tokens=False)["input_ids"]
        full = torch.cat([prompt_ids,
                          torch.tensor([l_ids], device=device)], dim=1)
        with torch.no_grad():
            logits = model(full).logits[0].float()
        logsm = torch.log_softmax(logits, dim=-1)
        start = prompt_ids.shape[1]
        total = sum(float(logsm[start + i - 1, t])
                    for i, t in enumerate(l_ids))
        out[label] = round(total / max(1, len(l_ids)) if normalize else total, 4)
    return out


def pick_label(logsoftmax, token_ids: dict) -> Tuple[str, dict]:
    """受限输出：比较标签首 token 的 log-softmax，返回 (label, scores)。

    仅当首 token 互异时语义有效（见 assert_distinct_first_tokens）。
    """
    scores = {lab: float(logsoftmax[tid]) for lab, tid in token_ids.items()}
    label = max(scores, key=scores.get)
    return label, scores


def margin_between(scores: dict) -> float:
    """conflict/compat 方向的稳定 margin（保留历史字段语义，便于对照旧报告）。"""
    return round(scores.get("violation", 0.0) - scores.get("compliant", 0.0), 3)


# ── 复核裁决（第二级：LLM 定量）──────────────────────────────────────────
# 用户架构定名（2026-10-11）：LLM 定性捕获 → LLM 定量复核。
# 第一级三分类头对冲突裁决不确定时，**不丢弃**，升级给第二级复核：
# 复核判官看到完整决策 + 全部约束 + 第一级结论与理由，输出 KEEP（维持提醒）
# 或 DROP（判定不可靠，降级为弃权）。两级都是 SLM 任务，零词表零阈值裁决。
# （政策门/数值门槛方案已被用户否决：在不稳定、未校准的 margin 上设阈值
#   做最终裁决，与关键词枚举同族——margin 只作路由参考，永不作结论。）

VERIFY_LABELS = ("KEEP", "DROP")
VERIFY_LITERALS = {"KEEP": "KEEP", "DROP": "DROP"}

VERIFY_INSTRUCTION = (
    "问：上述冲突判定可靠吗？若决策内容确实违反了某条约束原文，输出 KEEP；"
    "若判定依据不足、决策实为允许操作（查询/文档/配置）或与约束无关，"
    "输出 DROP")


def build_verification_input(decision_text: str, constraints: Sequence,
                             claimed_verdict: str, pass1_reason: str = "",
                             event_kind: str = "") -> str:
    """构造第二级复核输入。决策全文 + 约束原文 + 第一级结论，同源构造器。"""
    lines = ["## 待复核的冲突判定",
             f"第一级判定：{claimed_verdict}",
             f"第一级理由：{pass1_reason or '（未提供）'}",
             f"事件类型：{event_kind or 'unknown'}", ""]
    ji = build_judgment_input(decision_text, constraints,
                              event_kind=event_kind)
    # build_judgment_input 尾部带三分类问题，截掉后拼接复核问题
    body = ji.prompt.split("## 问题")[0].rstrip()
    lines.append(body)
    lines.append(f"## 问题\n{VERIFY_INSTRUCTION}")
    return "\n".join(lines)


# ── 约束选择（两阶段第一级：LLM 选卡）────────────────────────────────────
# 运行时约束卡可能远超训练分布（捕获漂移实测 15–27 张）。选卡头从卡列表
# 中选出与决策相关的约束，让义务判定回落到单卡甜点区。
# 防捷径：相关卡位置由调用方洗牌，别名按位置分配（数据生成同规则，
# 见 build_select_data.py——先绑别名后洗牌的恒-A bug 已修复）。

SELECT_LABELS = tuple("ABCDEFGHIJKLMNOPQRST")  # 20 个单 token 别名
SELECT_LITERALS = {a: a for a in SELECT_LABELS}


def build_selection_input(decision_text: str, card_quotes: Sequence[str]) -> str:
    """构造选卡输入。card_quotes 顺序即别名 A–E 的分配（调用方洗牌后传入）。"""
    lines = [f"## 决策\n{decision_text[:1500]}", "", "## 候选约束"]
    for i, q in enumerate(card_quotes):
        alias = SELECT_LABELS[i] if i < len(SELECT_LABELS) else f"卡{i+1}"
        lines.append(f"- 卡{alias}: {q}")
    lines += ["", "问：哪张约束卡与该决策直接相关？只输出卡字母（ABCDE）"]
    return "\n".join(lines)


# ── 目标覆盖裁决（goal_coverage）─────────────────────────────────────────
# 四分类：与前文的约束三分类同属"受限输出，绝不自由生成"。

COVERAGE_LABELS = ("supported_full", "supported_partial", "no_evidence",
                   "insufficient_info")
#: 覆盖标签的**受训别名**（训练与运行时必须一致）。
#: 原始字面量里 supported_full / supported_partial 共享首 token 'supported'，
#: 首 token 比较退化（见 exp_t4_coverage_report.md F1），故运行时也用别名。
COVERAGE_ALIASES = {"supported_full": "FULL",
                    "supported_partial": "PART",
                    "no_evidence": "NONE",
                    "insufficient_info": "UNCLEAR"}
COVERAGE_LITERALS = dict(COVERAGE_ALIASES)   # 兼容旧引用名

COVERAGE_INSTRUCTION = (
    "问：依据上述机械证据，该完成宣称对目标的覆盖情况如何？只输出一个词组："
    "supported_full、supported_partial、no_evidence 或 insufficient_info")

# 边界条款必须进 prompt，否则模型会把"测试通过"直接当成"目标达成"。
COVERAGE_BOUNDARY = (
    "判定边界（必须遵守）：\n"
    "- 证据是机械事实记录：test_run 记录的是「测试是否通过」，不等于「目标是否达成」；\n"
    "- 目标含多项要求而证据只覆盖其中一部分 → supported_partial；\n"
    "- 有完成宣称但没有任何相关证据 → no_evidence；\n"
    "- 证据与目标不相关、或目标原文无法判断适用范围 → insufficient_info；\n"
    "- 不得因为「测试变绿」「提交成功」就判定 supported_full。")

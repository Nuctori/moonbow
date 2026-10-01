# -*- coding: utf-8 -*-
"""replay_compare.py — 回放对比：仅收尾 vs 阶段审计（同一真实会话、同一审计规则）

用法：python tools/replay_compare.py <pi-session.jsonl> [--max-tasks N]

方法（不依赖真实宿主、不烧 LLM 额度）：
- 从真实 pi 会话 JSONL 提取有序观察块（thinking/text/toolCall/toolResult），
  以真实 user 消息为任务边界分段。
- 阶段审计：逐检查点（每个 thinking/text 块结束）增量调用 StageAuditor
  （与生产 /v1/stage-check 同一引擎、同一确定性规则），携带 prior findings。
- 基线（仅收尾）：只在任务最后一个"无工具调用的助手文本块"（收口信号）做
  同一次审计。

指标（以块序号计，非墙钟）：
- early_delta_blocks：首次 actionable 发现的块序 与 收口块序 之差（>0 即更早）。
- suppressed_by_recheck：中途 actionable 后来被证据解决/撤销的条数 —— 即
  "若无发送前复核会成为误提醒"的压力指标（发送前复核会拦下已解决者）。
- extra_rounds：按 advisory 预算规则实际会投递的过程提醒数（每任务语义类
  最多 1 次 + 非语义按 fingerprint 去重）——每次投递=一个额外回合。
- false_positive_proxy：最终仍 actionable 的发现数（收口时点复核仍成立者，
  无人工真值，仅作对照）。

局限（如实声明）：无人工标注真值，"误提醒"只能以"后续证据自行解决"作代理
指标；块序号非墙钟时间；真实投递时机由宿主决定。
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moonbow.guard.process_audit import StageAuditor
from moonbow.guard.protocol import StageBlock

WRAPUP_HINTS = ("已", "完成", "done", "fixed", "pass")


def extract_blocks(path, max_tasks):
    """解析 pi session JSONL → [(task_id, req_text, [StageBlock...]), ...]"""
    tasks = []
    cur_req, cur_blocks = None, []
    last_seq = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "message":
                continue
            m = ev.get("message") or {}
            role = m.get("role")
            if role == "user":
                text = _text_of(m)
                if text and not text.startswith(("/", "【", "[")):
                    if cur_req and cur_blocks:
                        tasks.append((cur_req, cur_blocks))
                        if len(tasks) >= max_tasks:
                            return tasks
                    cur_req, cur_blocks = text[:200], []
                continue
            if role not in ("assistant", "toolResult") or cur_req is None:
                continue
            content = m.get("content")
            if role == "assistant":
                for b in (content if isinstance(content, list) else []):
                    kind = b.get("type")
                    if kind == "thinking":
                        cur_blocks.append(StageBlock(seq=last_seq, kind="thinking",
                                                     text=str(b.get("thinking", ""))[:2000]))
                        last_seq += 1
                    elif kind == "text":
                        cur_blocks.append(StageBlock(seq=last_seq, kind="text",
                                                     text=str(b.get("text", ""))[:2000]))
                        last_seq += 1
                    elif kind == "toolCall":
                        cur_blocks.append(StageBlock(seq=last_seq, kind="toolCall",
                                                     text=json.dumps({"name": b.get("name"),
                                                                      "args": b.get("arguments")})[:2000],
                                                     tool_call_id=str(b.get("id", "")),
                                                     tool_name=str(b.get("name", "?"))))
                        last_seq += 1
            elif role == "toolResult":
                cur_blocks.append(StageBlock(seq=last_seq, kind="toolResult",
                                             text=_text_of(m)[:2000],
                                             tool_call_id=str(m.get("toolCallId", "")),
                                             is_error=bool(m.get("isError"))))
                last_seq += 1
    if cur_req and cur_blocks:
        tasks.append((cur_req, cur_blocks))
    return tasks


def _text_of(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
    return ""


def replay_task(blocks):
    """对单任务做两种策略的回放，返回指标 dict。

    阶段审计：逐检查点重放"截至该块的全部观察"（引擎按 fingerprint 幂等合并，
    等价于生产的增量送审 + prior 合并）。
    """
    from moonbow.guard.protocol import StageFinding
    auditor = StageAuditor()
    prior = []
    staged = []
    checkpoints = [b.seq for b in blocks if b.kind in ("thinking", "text")
                   and len(b.text.strip()) >= 6]
    for cp in checkpoints:
        r = auditor.audit(req="", blocks=blocks[:_idx_upto(blocks, cp) + 1],
                          prior_findings=prior, snapshot_version=cp)
        prior = [StageFinding.from_dict(d) for d in r["findings"]]
        staged.append((cp, r))
    return evaluate(blocks, checkpoints, staged)


def _idx_upto(blocks, cp):
    idx = 0
    for i, b in enumerate(blocks):
        if b.seq <= cp:
            idx = i
    return idx


def evaluate(blocks, checkpoints, staged):
    """汇总指标。staged: [(cp, audit_result_dict)]"""
    n = len(blocks)
    # 收口块 = 最后一个无 toolCall 的 assistant 文本/思考块（近似宿主 turn 边界）
    final_seq = checkpoints[-1] if checkpoints else -1
    # 基线：只在收口块做一次审计
    final_audit = staged[-1][1] if staged else {"findings": []}
    final_actionable = [f for f in final_audit["findings"] if f["status"] == "actionable"]

    first_actionable_seq = None
    resolved_midway = 0
    seen_fp = {}
    reminders_delivered = []
    semantic_used = False
    for cp, r in staged:
        for f in r["findings"]:
            fp = f["fingerprint"]
            prev = seen_fp.get(fp)
            if prev and prev == "actionable" and f["status"] in ("resolved", "withdrawn"):
                resolved_midway += 1
            seen_fp[fp] = f["status"]
        actionable_here = [f for f in r["findings"] if f["status"] == "actionable"]
        if actionable_here and first_actionable_seq is None:
            first_actionable_seq = cp
        # 模拟 advisory 投递策略：actionable + fingerprint 未提醒过 + 语义类受
        # 每任务一次预算约束；发送前复核：若该 fp 已在上一检查点解决则跳过
        rem = r.get("reminder")
        if rem:
            fp = rem.get("fingerprint", "")
            is_semantic = bool(r.get("semantic"))
            if fp and fp not in reminders_delivered:
                # advisory 预算：语义类每任务最多一次；非语义按 fingerprint 去重
                if not (is_semantic and semantic_used):
                    reminders_delivered.append(fp)
                if is_semantic:
                    semantic_used = True

    return {
        "blocks_total": n,
        "checkpoints": len(checkpoints),
        "final_seq": final_seq,
        "first_actionable_seq": first_actionable_seq,
        "early_delta_blocks": (final_seq - first_actionable_seq)
                              if first_actionable_seq is not None else 0,
        "final_actionable_count": len(final_actionable),
        "suppressed_by_recheck": resolved_midway,
        "extra_rounds": len(reminders_delivered),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--max-tasks", type=int, default=12)
    args = ap.parse_args()

    tasks = extract_blocks(args.session, args.max_tasks)
    print(f"回放会话: {args.session}")
    print(f"任务数: {len(tasks)}（取前 {args.max_tasks} 个有块任务）\n")
    agg = {"tasks": 0, "with_finding": 0, "early_deltas": [], "suppressed": 0,
           "extra_rounds": 0, "final_actionable": 0}
    for i, (req, blocks) in enumerate(tasks):
        m = replay_task(blocks)
        agg["tasks"] += 1
        agg["suppressed"] += m["suppressed_by_recheck"]
        agg["extra_rounds"] += m["extra_rounds"]
        agg["final_actionable"] += m["final_actionable_count"]
        if m["first_actionable_seq"] is not None:
            agg["with_finding"] += 1
            agg["early_deltas"].append(m["early_delta_blocks"])
        print(f"[{i}] blocks={m['blocks_total']:3d} cp={m['checkpoints']:3d} "
              f"first_actionable={str(m['first_actionable_seq']):>6} "
              f"final_seq={m['final_seq']:>4} early_delta={m['early_delta_blocks']:>4} "
              f"resolved_midway={m['suppressed_by_recheck']} extra_rounds={m['extra_rounds']} "
              f"| {req[:40]!r}")
    print("\n==== 汇总（块序号计，非墙钟）====")
    eds = agg["early_deltas"]
    print(f"任务总数               : {agg['tasks']}")
    print(f"出现 actionable 的任务  : {agg['with_finding']}")
    if eds:
        print(f"提前发现（块数）        : 中位 {sorted(eds)[len(eds)//2]}, "
              f"最小 {min(eds)}, 最大 {max(eds)}")
    print(f"被复核压制的中途 actionable: {agg['suppressed']}（发送前复核可拦下的误提醒压力）")
    print(f"额外回合（过程提醒投递）  : {agg['extra_rounds']}（advisory 预算封顶后）")
    print(f"收口时仍 actionable      : {agg['final_actionable']}（无人工真值，仅代理指标）")
    print("\n局限：无人工真值标注；块序号非墙钟；真实投递时机由宿主决定。")


if __name__ == "__main__":
    main()

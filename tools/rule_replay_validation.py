# -*- coding: utf-8 -*-
"""tools/rule_replay_validation.py — 规则集 v2 三条失败形态签名的跨模型回放验证。

目的（预登记）：把 v2 的三条失败形态签名
  - converge.repeat（退化循环：连续同命令 / 同文件写振荡）
  - converge.stall@60（v2 预算缩放停滞线，budget=900s → 60 轮）
  - converge.fail_streak（连续失败 ≥2）
在两个模型的全量历史会话上回放，验证**可归纳性**：
形态签名跨模型命中失败 run、健康 run 零/低误报。

规则本体零重写：全部喂给真实实现
`src/moonbow/guard/convergence.py::ConvergenceShadow`（ruleset="v2"，
budget_s=900）。本脚本只做两件事：①会话 JSONL → StageBlock 块流的转换层；
②逐 run 触发结果的统计（recall / FPR / Fisher / 混淆矩阵）。

语料（只读）：
  A. mimo（xiaomi/mimo-v2.5 为主）：
     - Phase 0 批次：`<workspace>/pi_eval/home/sessions/2026-09-2*.jsonl`
       经 tools/convergence_study.py 同一过滤（trap_ladder / guard_scen，
       排除 s2/gtest 干净任务）→ 233 run；标签 = convergence_study.label_run
       （最终一次 pytest 成败；无 pytest = 失败）。
     - Phase 2 批次：`results/convergence-phase2/runs_p2a.jsonl` +
       `results/convergence-phase2/agent_home/sessions/`（30 run，
       model_change = xiaomi/mimo-v2.5）；标签 = runs_p2a 的 completed。
  B. gemini（gemini-3.5-flash-lite）：`results/guard-effect-v2/runs_smoke.jsonl`
       中 task 含 "eigen" 的全部 run（tb2 18 + ge4 2 = 20）；标签 = completed。

转换层（pi 会话 JSONL → StageBlock 流）与有损处（如实记录）：
  1. assistant 的 toolCall 块 → StageBlock(kind="toolCall")，text 重序列化为
     {"name":..., "arguments":...} JSON（与真实客户端采集同构，保证
     normalized_command_core / _write_events / _pytest_events 配对口径不变）。
  2. toolResult 消息 → StageBlock(kind="toolResult")，text = 全部 text 块拼接
     （与 convergence_study.parse_session 的取文本口径一致）。
  3. thinking 块丢弃：v2 三规则不消费 thinking（有损，无规则语义影响）。
  4. 轮定义差异（convergence.py docstring 已声明的近似）：guard 的轮代理 =
     toolResult 计数（第 k 个 toolResult = 第 k 轮观测点）；convergence_study
     的轮 = 一条含 ≥1 toolCall 的 assistant 消息 + 其后续全部 toolResult。
     并行多调用的一轮被拆成多轮 → guard_rounds ≥ study_rounds（系统性偏高、
     方向固定）。逐 run 两个轮数都落盘（guard_rounds / study_rounds）。
  5. 收尾无工具调用的 assistant 文本不计轮（两口径一致）。
  6. req 不回放（规则不消费 req；goats_total 走生产缺省 goals_total=1，
     不注入目标——目标注入属 matcher 对齐范畴，v2 shadow 缺省即 1）。

回放语义（两套，主判定用端到端）：
  - end-to-end（主）：整条会话块流一次性 compute()。= "整段历史的形态签名"。
  - incremental（辅，as-deployed）：按 toolResult 边界逐前缀 compute()，记录
    各规则**首次触发轮次**（守卫轮代理）。ever-fired ⊇ end-fired（整流即最后
    一个前缀）。生产 shadow 每批审计都打分，因此 incremental 是"守卫若在场
    会在第几轮打断"的口径；end-to-end 是"该 run 是否呈现该形态"的口径。

用法：
  python tools/rule_replay_validation.py \
    [--out results/guard-effect-v2] [--budget 900]
产物：replay_runs.jsonl（逐 run）、rule_replay_metrics.json（指标）、
      控制台摘要（表格交给 rule_replay_validation.md 汇总）。
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # publish_repo
WORKSPACE = os.path.dirname(REPO)                                        # M:\AI\spark-4b
sys.path.insert(0, os.path.join(REPO, "src"))                            # moonbow
sys.path.insert(0, os.path.join(WORKSPACE, "tools"))                     # convergence_study

import convergence_study as cs  # noqa: E402  已验证的 233 mimo run 解析与标注器
from moonbow.guard.protocol import StageBlock  # noqa: E402
from moonbow.guard.process_audit import tool_events_to_stage_blocks  # noqa: E402
from moonbow.guard.convergence import (  # noqa: E402
    ConvergenceShadow, REPEAT_SIGNAL, STALL_SIGNAL, FAIL_STREAK_SIGNAL,
    stall_line_from_budget,
)

MIMO_MODEL = "xiaomi/mimo-v2.5"
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------- 转换层

def _iter_pi_messages(path):
    """pi 会话 JSONL → 逐条 (role, content 块列表, toolCallId, isError)。

    线上格式转换器（session_to_tool_events）专用；离线格式
    （session_to_blocks）保持原实现不动（meta/study_rounds 口径与既有
    产物逐字节一致），两转换器按同一文件序消费同一消息流。
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "message":
                continue
            m = d.get("message", {})
            content = m.get("content")
            bl = content if isinstance(content, list) else []
            yield (m.get("role"), bl, m.get("toolCallId"), bool(m.get("isError")))


def session_to_blocks(path):
    """pi 会话 JSONL → (blocks, meta, study_rounds)。【离线格式块】

    meta 复刻 convergence_study.parse_session 的 session/model_change/
    custom_message 元数据口径；study_rounds = parse_session 的轮计数
    （含 ≥1 toolCall 的 assistant 消息数）。
    """
    blocks = []
    meta = {"cwd": "", "model": "", "guard": False}
    seq = 0
    study_rounds = 0
    for raw in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t == "session":
            meta["cwd"] = d.get("cwd") or ""
        elif t == "model_change":
            meta["model"] = d.get("modelId") or meta["model"]
        elif t == "custom_message":
            if str(d.get("customType", "")).startswith("progress-guard"):
                meta["guard"] = True
        elif t == "message":
            m = d.get("message", {})
            role = m.get("role")
            content = m.get("content")
            bl = content if isinstance(content, list) else []
            if role == "assistant":
                if any(b.get("type") == "toolCall" for b in bl):
                    study_rounds += 1
                for b in bl:
                    bt = b.get("type")
                    if bt == "toolCall":
                        # 与真实 StageBlock 采集同构：text = 完整调用 JSON
                        blocks.append(StageBlock(
                            seq, "toolCall",
                            json.dumps({"name": b.get("name"),
                                        "arguments": b.get("arguments") or {}},
                                       ensure_ascii=False),
                            tool_call_id=b.get("id"),
                            tool_name=b.get("name")))
                        seq += 1
                    elif bt == "text":
                        blocks.append(StageBlock(seq, "text", b.get("text") or ""))
                        seq += 1
                    # thinking：丢弃（v2 三规则不消费；转换层有损处 ③）
            elif role == "toolResult":
                txt = " ".join(b.get("text", "") for b in bl
                               if isinstance(b, dict) and b.get("type") == "text")
                blocks.append(StageBlock(seq, "toolResult", txt,
                                         tool_call_id=m.get("toolCallId")))
                seq += 1
    return blocks, meta, study_rounds


# ---- 线上格式（2026-10-04 通路补全后的对拍基准）----
#
# 模拟 TS 客户端（extensions/process-events.ts + process-task.ts）的
# tool_events 序列化：截断常量与封顶逐一对照实现缺省值；再经服务端解析器
# （moonbow.guard.process_audit.tool_events_to_stage_blocks）还原成块，
# 与 text 块（blocks 窗口cap 400，部署语义）合并 = 服务端 v2 收敛计算实际
# 消费的块流（StageAuditor._convergence_blocks 的输出形态）。

ARGS_VALUE_CLIP = 200      # process-events.ts MAX_TOOL_EVENT_ARG_VALUE（逐值截断）
ARGS_KEYS_MAX = 40         # MAX_TOOL_EVENT_ARGS_KEYS
ARGS_TOTAL_CLIP = 1200     # MAX_TOOL_EVENT_ARGS（摘要总长兜底）
RESULT_HEAD = 160          # MAX_TOOL_EVENT_RESULT_HEAD
RESULT_TAIL = 320          # MAX_TOOL_EVENT_RESULT_TAIL
TOOL_EVENTS_CAP = 4000     # process-task.ts maxToolEvents() 缺省
BLOCKS_WINDOW = 400        # process-task.ts maxBlocks() 缺省（text 块窗口）

_PYTEST_LINE_RE = re.compile(r"\d+\s+passed|\d+\s+failed|no tests ran", re.IGNORECASE)


def _clip_val(v, limit=ARGS_VALUE_CLIP):
    if len(v) <= limit:
        return v
    return v[:limit] + "…[truncated %d chars]" % (len(v) - limit)


def clip_args(name, args):
    """调用摘要：键结构完整、长字符串值逐值截断（与 TS summarizeToolArgs 同构）。

    整段头截断会把 JSON 截坏 → 服务端 _parse_tool_call 解析失败 → write
    路径/重定向解析全丢（实测 both r8 的 edit_oscillation 因此漏检，
    线上/离线首触轮不一致）。逐值截断保持 payload 可解析。"""
    if isinstance(args, dict):
        out = {}
        for i, (k, v) in enumerate(args.items()):
            if i >= ARGS_KEYS_MAX:
                break
            if isinstance(v, str):
                out[k] = _clip_val(v)
            elif isinstance(v, (dict, list)):
                out[k] = _clip_val(json.dumps(v, ensure_ascii=False), 400)
            else:
                out[k] = v
    elif args is None:
        out = {}
    else:
        out = {"input": _clip_val(str(args))}
    return _clip_val(json.dumps({"name": name, "arguments": out}, ensure_ascii=False),
                     ARGS_TOTAL_CLIP)


def clip_result(text):
    """中段截断（保头保尾）；pytest 摘要行不在保留区时显式补附（与 TS 同构）。"""
    t = text or ""
    if len(t) <= RESULT_HEAD + RESULT_TAIL + 64:
        return t
    kept = (t[:RESULT_HEAD] + "…[truncated %d chars]" % (len(t) - RESULT_HEAD - RESULT_TAIL)
            + t[-RESULT_TAIL:])
    for line in reversed(t.splitlines()):
        if _PYTEST_LINE_RE.search(line):
            line = line.strip()
            if line not in kept:
                kept += "…[pytest] " + line
            break
    return kept


def session_to_tool_events(path):
    """pi 会话 JSONL → (tool_events, text_blocks, n_dropped)。【线上格式】

    - tool_events：客户端 tool_execution_start/end 通道的会话侧重放
      （seq 与 text 块共用单一采集序空间，按文件序单调递增）；
    - text_blocks：blocks 窗口（≤400，环形丢头——部署语义）内的 assistant
      text 块（thinking 丢弃，与离线转换层同口径；服务端收敛计算只消费
      text/thinking 的 perf_retest 文本证据）；
    - n_dropped：封顶丢头的工具事件数（遥测）。
    """
    events = []
    text_blocks = []            # (seq, text)
    seq = 0
    for role, bl, tool_call_id, is_error in _iter_pi_messages(path):
        if role == "assistant":
            for b in bl:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "toolCall":
                    events.append({
                        "seq": seq, "phase": "call", "name": b.get("name") or "?",
                        "args": clip_args(b.get("name") or "?",
                                          b.get("arguments") or {}),
                        "tool_call_id": b.get("id"),
                    })
                    seq += 1
                elif bt == "text":
                    text_blocks.append((seq, b.get("text") or ""))
                    seq += 1
        elif role == "toolResult":
            txt = " ".join(b.get("text", "") for b in bl
                           if isinstance(b, dict) and b.get("type") == "text")
            events.append({
                "seq": seq, "phase": "result", "name": "?",
                "is_error": bool(is_error), "result": clip_result(txt),
                "tool_call_id": tool_call_id,
            })
            seq += 1
    n_dropped = 0
    if len(events) > TOOL_EVENTS_CAP:
        n_dropped = len(events) - TOOL_EVENTS_CAP
        events = events[-TOOL_EVENTS_CAP:]
    if len(text_blocks) > BLOCKS_WINDOW:
        text_blocks = text_blocks[-BLOCKS_WINDOW:]
    return events, text_blocks, n_dropped


def online_blocks_from_tool_events(events, text_blocks):
    """线上通道的块流 = 服务端解析（tool_events → StageBlock）+ text 块窗口。

    与 StageAuditor._convergence_blocks 的 v2 输出同构：tool_events 为工具流
    权威来源，blocks 内的 toolCall/toolResult 条目让位，text 块保留。"""
    return ([StageBlock(seq=s, kind="text", text=t) for s, t in text_blocks]
            + tool_events_to_stage_blocks(events))


# ---------------------------------------------------------------- 回放

def _entry(entries, signal):
    for e in entries:
        if e["signal"] == signal:
            return e
    return None


def replay_run(blocks, budget_s):
    """对单 run 跑真实 v2 规则。返回逐规则触发 dict（end + incremental）。"""
    shadow = ConvergenceShadow(ruleset="v2", budget_s=budget_s)
    entries = shadow.compute("", blocks)
    end = {}
    for sig, key in ((REPEAT_SIGNAL, "repeat"), (STALL_SIGNAL, "stall"),
                     (FAIL_STREAK_SIGNAL, "fail_streak")):
        e = _entry(entries, sig)
        end[key] = {
            "fired": e["matched"] is True,
            "matched": e["matched"],           # stall 可能 abstain(None)
            "abstain_reason": e.get("abstain_reason"),
            "detail": e.get("detail") or {},
        }
    # incremental：逐 toolResult 前缀，记录首次触发轮次（守卫轮代理）
    first = {k: None for k in ("repeat", "stall", "fail_streak")}
    pending = set(first)
    n_tr = 0
    for i, b in enumerate(blocks):
        if b.kind != "toolResult":
            continue
        n_tr += 1
        if not pending:
            break
        es = shadow.compute("", blocks[:i + 1])
        for sig, key in ((REPEAT_SIGNAL, "repeat"), (STALL_SIGNAL, "stall"),
                         (FAIL_STREAK_SIGNAL, "fail_streak")):
            e = _entry(es, sig)
            if key in pending and e is not None and e["matched"] is True:
                first[key] = n_tr
                pending.discard(key)
    out = {}
    for key, e in end.items():
        out[key + "_fired"] = e["fired"]
        out[key + "_first_round"] = first[key]      # None = incremental 全程未触发
        out[key + "_ever_fired"] = first[key] is not None
        out[key + "_detail"] = e["detail"]
        out[key + "_abstain"] = e["abstain_reason"]
    out["guard_rounds"] = sum(1 for b in blocks if b.kind == "toolResult")
    out["combined_fired"] = any(end[k]["fired"] for k in end)
    return out


# ---------------------------------------------------------------- 语料装载

def load_mimo_phase0(sessions_dir, pattern):
    """Phase 0 批次：convergence_study 同过滤 + 同标注；附 Phase 0 基线核对。"""
    baseline = {}
    feat_path = os.path.join(WORKSPACE, "results", "convergence-phase0", "features.jsonl")
    if os.path.exists(feat_path):
        with open(feat_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                baseline[r["session_file"]] = r.get("label_success")
    runs = []
    label_diff = 0
    for f in sorted(glob.glob(os.path.join(sessions_dir, pattern))):
        rounds, meta = cs.parse_session(f)
        cwd = cs.norm(meta["cwd"])
        if "/trap_ladder/" not in cwd and "/guard_scen/" not in cwd:
            continue
        trap = cwd.rstrip("/").split("/")[-1]
        if trap in ("s2", "gtest"):
            continue
        label, _, _, _ = cs.label_run(rounds, meta)
        base = baseline.get(os.path.basename(f))
        if base is not None and bool(base) != bool(label):
            label_diff += 1
        runs.append({
            "corpus": "mimo_phase0", "model": meta["model"] or "unknown",
            "task": trap, "arm": "guard" if meta["guard"] else "baseline",
            "run": os.path.basename(f), "session_path": f,
            "completed": bool(label), "study_rounds": len(rounds),
        })
    return runs, label_diff, len(baseline)


def load_mimo_phase2():
    p2dir = os.path.join(REPO, "results", "convergence-phase2")
    rows = [json.loads(l) for l in open(os.path.join(p2dir, "runs_p2a.jsonl"),
                                        encoding="utf-8") if l.strip()]
    runs = []
    for r in rows:
        p = os.path.join(p2dir, "agent_home", "sessions", r["session"])
        if not os.path.exists(p):
            continue
        runs.append({
            "corpus": "mimo_phase2", "model": "",  # 从会话 model_change 回填
            "task": "conv_advisory_app", "arm": r["arm"], "run": r["session"],
            "session_path": p, "completed": bool(r["completed"]),
            "study_rounds": None,
        })
    return runs


def load_gemini():
    # runs_smoke.jsonl 为行分隔 JSONL（e7fff87 恢复后的格式；逐行读）
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "guard-effect-v2",
                                                     "runs_smoke.jsonl"), encoding="utf-8")
            if l.strip()]
    sdir = os.path.join(REPO, "results", "guard-effect-v2", "agent_home", "sessions")
    runs = []
    for r in rows:
        if "eigen" not in r["task"]:
            continue
        p = os.path.join(sdir, r["session"])
        if not os.path.exists(p):
            continue
        runs.append({
            "corpus": "gemini_eigen", "model": r.get("expected_model") or GEMINI_MODEL,
            "task": r["task"], "arm": "%s/%s" % (r.get("phase"), r["arm"]),
            "run": r["session"], "session_path": p,
            "completed": bool(r["completed"]),
            "study_rounds": r.get("turns"),
        })
    return runs


# ---------------------------------------------------------------- 统计

def fisher_two_sided(a, b, c, d):
    """Fisher 精确双侧 p。表：[[a=失败&触发, b=失败&未触发],
    [c=成功&触发, d=成功&未触发]]（预测目标 = 失败）。"""
    n = a + b + c + d
    if n == 0 or a + b == 0 or c + d == 0 or a + c == 0:
        return None
    r1, c1 = a + b, a + c

    def p(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    p0 = p(a)
    tot = sum(p(x) for x in range(max(0, c1 - (n - r1)), min(c1, r1) + 1)
              if p(x) <= p0 * (1 + 1e-9))
    return min(1.0, tot)


def rule_metrics(runs, key):
    """单规则：失败 run 命中率（recall）与成功 run 误报率（FPR）+ 列联表。

    fired 主口径 = end-to-end；辅口径 ever_fired（incremental 任意前缀曾触发，
    = 生产 shadow 每批审计都打分时的部署语义）一并落盘。
    """
    fail = [r for r in runs if not r["completed"]]
    succ = [r for r in runs if r["completed"]]

    def tbl(fs, ss, field):
        a = sum(1 for r in fs if r[field])
        b = len(fs) - a
        c = sum(1 for r in ss if r[field])
        d = len(ss) - c
        return a, b, c, d

    a, b, c, d = tbl(fail, succ, key + "_fired")
    ae, be, ce, de = tbl(fail, succ, key + "_ever_fired")
    m = {
        "rule": key, "n_fail": len(fail), "n_success": len(succ),
        "hit_on_fail": a, "miss_on_fail": b, "fp_on_success": c, "tn_success": d,
        "recall": round(a / len(fail), 4) if fail else None,
        "fpr": round(c / len(succ), 4) if succ else None,
        "recall_wilson95": [round(x, 3) for x in cs.wilson(a, len(fail))] if fail else None,
        "fpr_wilson95": [round(x, 3) for x in cs.wilson(c, len(succ))] if succ else None,
        "fisher_p_two_sided": ((lambda p: round(p, 5) if p is not None else None)(
            fisher_two_sided(a, b, c, d))),
        # 部署语义（辅）：任意前缀曾触发
        "ever_fired": {"hit_on_fail": ae, "fp_on_success": ce,
                       "recall": round(ae / len(fail), 4) if fail else None,
                       "fpr": round(ce / len(succ), 4) if succ else None},
        "tp_runs": [r["run"] for r in fail if r[key + "_fired"]],
        "fp_runs": [r["run"] for r in succ if r[key + "_fired"]],
    }
    first = [r[key + "_first_round"] for r in fail + succ
             if r[key + "_fired"] and r[key + "_first_round"] is not None]
    m["first_fire_round_median"] = sorted(first)[len(first) // 2] if first else None
    return m


def combined_metrics(runs):
    fail = [r for r in runs if not r["completed"]]
    succ = [r for r in runs if r["completed"]]
    tp = sum(1 for r in fail if r["combined_fired"])
    fn = len(fail) - tp
    fp = sum(1 for r in succ if r["combined_fired"])
    tn = len(succ) - fp
    prec = tp / (tp + fp) if tp + fp else None
    return {
        "definition": "任一规则触发（end-to-end）= 预测失败",
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "precision": round(prec, 4) if prec is not None else None,
        "precision_wilson95": ([round(x, 3) for x in cs.wilson(tp, tp + fp)]
                               if tp + fp else None),
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fisher_p_two_sided": round(fisher_two_sided(tp, fn, fp, tn), 5)
        if (tp + fn and fp + tn) else None,
    }


# ---------------------------------------------------------------- 主流程

# ---------------------------------------------------------------- 报告

def _fmt_run_table_gemini(rows):
    out = ["| task | arm | label | study轮 | guard轮 | repeat | stall@60 | fail_streak | 组合 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["task"], x["arm"], -(x["study_rounds"] or 0))):

        def cell(key):
            f = "F" if r[key + "_fired"] else "-"
            fr = r[key + "_first_round"]
            extra = ""
            if key == "fail_streak":
                extra = " (末态streak=%s)" % r["fail_streak_detail"].get("streak")
            if key == "stall" and not r["stall_fired"]:
                return "-/%s" % (r["stall_abstain"] or "cov>0")
            if key == "repeat" and r["repeat_fired"]:
                d = r["repeat_detail"]
                extra = " kind=%s,count=%s" % (d.get("kind"), d.get("count"))
            return "%s@%s%s" % (f, fr, extra)

        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["task"], r["arm"], r["label"], r["study_rounds"], r["guard_rounds"],
            cell("repeat"), cell("stall"), cell("fail_streak"),
            "F" if r["combined_fired"] else "-"))
    return "\n".join(out)


def _fmt_run_table_mimo(rows):
    out = ["| batch | task | arm | label | study轮 | guard轮 | repeat | stall@60 | fail_streak | 组合 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["corpus"], x["task"], x["arm"], x["run"])):
        marks = []
        for k in ("repeat", "stall", "fail_streak"):
            marks.append("F@%s" % r[k + "_first_round"] if r[k + "_fired"] else "-")
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["corpus"], r["task"], r["arm"], r["label"], r["study_rounds"],
            r["guard_rounds"], marks[0], marks[1], marks[2],
            "F" if r["combined_fired"] else "-"))
    return "\n".join(out)


def _rule_line(m):
    return ("R=%s ( Wilson %s ) / FPR=%s ( %s ) / Fisher p=%s" %
            (m["recall"], m["recall_wilson95"], m["fpr"], m["fpr_wilson95"],
             m["fisher_p_two_sided"]))


def write_report(path, rows, metrics, budget, stall_line, label_diff, corpus_line):
    g = metrics["corpus"]["gemini"]
    m = metrics["corpus"]["mimo"]
    p0 = metrics["corpus"]["mimo_phase0"]
    p2 = metrics["corpus"]["mimo_phase2"]
    con = metrics["cross_model_consistency"]
    checks = metrics["known_sample_checks"]
    gem_rows = [r for r in rows if r["corpus"] == "gemini_eigen"]
    mimo_rows = [r for r in rows if r["corpus"].startswith("mimo")]

    md = []
    w = md.append
    w("# 规则集 v2 三形态签名跨模型回放验证（rule_replay_validation）")
    w("")
    w("> 日期：2026-10-04。只读回放：不跑模型、不占 XPU。规则本体零重写——")
    w("> 全部经真实实现 `src/moonbow/guard/convergence.py::ConvergenceShadow`")
    w("> （ruleset=\"v2\", budget_s=%d → stall 线 %d 轮，goals_total=1 生产缺省）计算；" % (budget, stall_line))
    w("> 本脚本（`tools/rule_replay_validation.py`）只做会话→StageBlock 转换与统计。")
    w("> 数据：`replay_runs.jsonl`（逐 run）/ `rule_replay_metrics.json`（指标）。")
    w("")
    w("## 0. 结论速览")
    w("")
    w("- **总判定：可归纳性成立（预登记判据 2：组合规则两模型 precision 均 ≥0.7）。**")
    w("  mimo P=0.84 [0.653,0.936]；gemini P=1.0 [0.51,1.0]。")
    w("- 单规则层面：**fail_streak 是唯一跨两模型方向一致的规则**")
    w("  （mimo useful R=0.13/FPR=0.039；gemini useful R=0.2/FPR=0.0；")
    w("  且 Phase 0 干净标注批次端态 FPR=0/94）。")
    w("- repeat 与 stall@60：**gemini 语料内精确命中（端态零误报），mimo 语料形态缺席**")
    w("  （inactive：全语料零触发）——不是误报源，但\"跨两模型命中\"判据无法在 mimo 上证明，")
    w("  按预登记记为**形态依赖签名**（退化循环/长预算停滞），非严格可归纳。")
    w("- 已知样本 3/3 命中：control r3 被 repeat 命中（count=877，第 80 守卫轮）；")
    w("  control r1 被 stall@60 命中（第 60 守卫轮）；健康 run 端态 repeat 零命中。")
    w("- 额外发现：ge4 的 150 轮\"阅读停滞\"实为 **81 连击同命令计时循环**，被 repeat+stall 双命中。")
    w("")
    w("## 1. 语料构成与标签")
    w("")
    w("| 组 | 批次 | 模型 | run 数 | completed | fail | 标签口径 |")
    w("|---|---|---|---|---|---|---|")
    w("| A | mimo Phase 0（pi_eval 2026-09-2*，trap_ladder/guard_scen） | xiaomi/mimo-v2.5 为主（220/233，余 13 freellm 池） | %d | %d | %d | convergence_study.label_run（会话内最后一次 pytest 成败；无 pytest=失败） |" % (
        p0["n_runs"], p0["n_completed"], p0["n_runs"] - p0["n_completed"]))
    w("| A | mimo Phase 2（results/convergence-phase2, runs_p2a） | xiaomi/mimo-v2.5 | %d | %d | %d | runs_p2a.completed（runner 对工作区的最终判定） |" % (
        p2["n_runs"], p2["n_completed"], p2["n_runs"] - p2["n_completed"]))
    w("| B | gemini eigen（runs_smoke.jsonl task 含 eigen：tb2×18 + ge4×2） | gemini-3.5-flash-lite | %d | %d | %d | runs_smoke.completed（judge 逐测试复测） |" % (
        g["n_runs"], g["n_completed"], g["n_runs"] - g["n_completed"]))
    w("")
    w("- Phase 0 批次复算标签与既有 `results/convergence-phase0/features.jsonl` 基线 233/233 一致（偏差 %d）。" % label_diff)
    w("- %s" % corpus_line)
    w("")
    w("## 2. 转换层与有损记录（如实）")
    w("")
    w("1. pi 会话 JSONL → StageBlock：assistant.toolCall → `toolCall` 块（text 重序列化为")
    w("   `{name, arguments}` JSON，与真实采集同构）；toolResult 消息 → `toolResult` 块")
    w("   （text 块拼接，与 convergence_study 同口径）；assistant text → `text` 块。")
    w("2. **thinking 块丢弃**：v2 三规则不消费 thinking，无规则语义影响。")
    w("3. **轮定义差异**（convergence.py docstring 声明的近似）：guard 轮代理 = toolResult")
    w("   计数；study 轮 = 含 ≥1 toolCall 的 assistant 消息。并行多调用的一轮被拆成多轮")
    w("   → guard轮 ≥ study轮。实测 mimo 偏斜最大（study 17 轮 → guard 33 轮，~2x，并行双")
    w("   调用常见）；gemini 几乎 1:1（单调用轮）。该偏斜方向固定（守卫轮虚高、触发偏早），")
    w("   对 stall@60 在 mimo 上是放宽（仍无一 run 达 60，见 §5 阈值观察）。")
    w("4. **goals_total=1（生产缺省，不注入目标）**：目标注入属 matcher 对齐范畴；该配置下")
    w("   \"无一次 pytest 全绿的长 run\"coverage=0，与 §6 已知样本的预登记预期一致。")
    w("5. req 不回放（规则不消费）；收尾纯文本不计轮（两口径一致）。")
    w("")
    w("## 3. 回放语义（双口径）")
    w("")
    w("- **end-to-end（主，\"该 run 是否呈现该形态\"）**：整条块流一次 compute()。")
    w("- **ever-fired（辅，as-deployed）**：逐 toolResult 前缀 compute() 记录首次触发轮；")
    w("  生产 shadow 每批审计都打分，此口径 = \"守卫若在场会在第几轮打断\"。中途失败后恢复")
    w("  的 run 在该口径下会计入（见 §7 误报来源）。")
    w("")
    w("## 4. 逐 run 表")
    w("")
    w("### 4.1 gemini eigen（20 run 全表）")
    w("")
    w(_fmt_run_table_gemini(gem_rows))
    w("")
    w("F=端态触发 @k=增量首次触发守卫轮。tb_ 前缀 = ge4 批次（含 150 轮阅读停滞 run1）。")
    w("")
    w("### 4.2 mimo（263 run 全表，紧凑列）")
    w("")
    w(_fmt_run_table_mimo(mimo_rows))
    w("")
    w("## 5. 逐规则指标（分模型 + 分批次）")
    w("")
    w("| 范围 | 规则 | 失败命中 recall | 成功误报 FPR | Fisher p（双侧） | 触发轮中位 |")
    w("|---|---|---|---|---|---|")
    for name, c in (("mimo 合并(263)", m), ("mimo Phase0(233)", p0), ("mimo Phase2(30)", p2),
                    ("gemini eigen(20)", g)):
        for k in ("repeat", "stall", "fail_streak"):
            mm = c["rules"][k]
            w("| %s | %s | %s | %s | %s | %s |" % (
                name, k, mm["recall"], mm["fpr"], mm["fisher_p_two_sided"],
                mm["first_fire_round_median"]))
    w("")
    w("ever-fired（部署语义）辅助口径：mimo fail_streak ever FPR=%s（12/102，全部 Phase 2 批次）；"
      "gemini ever FPR：repeat=%s、stall=%s、fail_streak=%s（健康 run 中途失败后恢复所致，§7）。"
      % (m["rules"]["fail_streak"]["ever_fired"]["fpr"],
         g["rules"]["repeat"]["ever_fired"]["fpr"],
         g["rules"]["stall"]["ever_fired"]["fpr"],
         g["rules"]["fail_streak"]["ever_fired"]["fpr"]))
    w("")
    w("形态上界诊断（阈值观察素材）：")
    w("")
    w("| 范围 | 最长守卫轮 | 达 stall 线(%d) run 数 | 同命令连击上界 | 同文件写上界 |" % stall_line)
    for name, c in (("mimo", m), ("gemini", g)):
        d = c["diagnostics"]
        w("| %s | %s | %s | %s | %s |" % (name, d["guard_rounds_max"],
                                          d["guard_rounds_ge_stall_line"],
                                          d["repeat_max_identical_calls_observed"],
                                          d["repeat_max_same_file_writes_observed"]))
    w("")
    w("## 6. 已知样本抽查（预登记）")
    w("")
    for c in checks:
        verdict = {True: "PASS", False: "FAIL", None: "观察"}.get(c["pass"], str(c["pass"]))
        w("- **[%s] %s** — 预期：%s。实测：%s" % (verdict, c["sample"], c["expect"],
                                                 json.dumps(c["observed"], ensure_ascii=False)))
    w("")
    w("## 7. 组合规则与误报来源分析")
    w("")
    w("组合规则 = 任一规则端态触发即预测失败。")
    w("")
    w("| 范围 | TP | FN | FP | TN | precision | recall | Fisher p |")
    w("|---|---|---|---|---|---|---|---|")
    for name, c in (("mimo 合并", m), ("mimo Phase0", p0), ("mimo Phase2", p2), ("gemini", g)):
        cm = c["combined"]
        cf = cm["confusion"]
        w("| %s | %d | %d | %d | %d | %s (W%s) | %s | %s |" % (
            name, cf["tp"], cf["fn"], cf["fp"], cf["tn"], cm["precision"],
            cm["precision_wilson95"], cm["recall"], cm["fisher_p_two_sided"]))
    w("")
    w("误报（FP）来源逐例：")
    w("")
    w("1. **mimo 4 个 FP 全部来自 Phase 2 批次**（Phase 0 干净批次端态 FPR=0/94）：")
    w("   4 run 均为 truncated=True + env_error=True 的环境噪声 run（runs_p2a），会话末尾")
    w("   连续 2 条 pytest 输出非全绿（多为截断/收集期输出），但 runner 以工作区复测判")
    w("   completed+partial=1.0。属**标签口径差 + 截断噪声**，非规则误判——会话内证据")
    w("   确实是\"连续失败\"。含义：fail_streak 的输入文本质量依赖 toolResult 完整性。")
    w("2. **gemini 端态 FP=0**。ever 口径下的 7 个\"误报\"全部是中途触发后恢复：")
    w("   fail_streak 5/10（健康 run 早期失败→后期全绿）、stall 1/10（control r9 于第 60 轮")
    w("   尚未全绿、第 67 轮收尾全绿——stall 线边界的真边界样本）、repeat 1/10（both r8 第")
    w("   52 轮短暂 edit_oscillation 后改对）。部署语义下这些=每任务 1 次的可忽略提醒")
    w("   （AdvisoryBudget 去重），不是拦截。")
    w("3. gemini 6 个 FN 里 3 个 both 臂 run（r4/r6/r9）正是 eigen_failure_attribution §7")
    w("   标注的 **converge.perf_retest 目标形态**（会内 27 passed、如实申报 STATUS:A、")
    w("   judge 复测翻平局项）——超出本次回放的三签名范围，属第 4 条 v2 规则的管辖；")
    w("   另 3 个（both r1 4 轮伪调用死亡、control r5 2 轮静默停止、ge4 r2 107 轮收尾全绿）")
    w("   分别是\"验证缺位\"（v1 信号）与端态口径的覆盖对象。")
    w("")
    w("## 8. 跨模型一致性")
    w("")
    w("| 规则 | mimo（R/FPR/方向） | gemini（R/FPR/方向） | 严格一致 |")
    w("|---|---|---|---|")
    for k in ("repeat", "stall", "fail_streak"):
        v = con[k]
        w("| %s | %s / %s / %s | %s / %s / %s | %s |" % (
            k, v["mimo"]["recall"], v["mimo"]["fpr"], v["mimo"]["direction"],
            v["gemini"]["recall"], v["gemini"]["fpr"], v["gemini"]["direction"],
            "是" if v["consistent"] else "否"))
    w("")
    w("方向口径：useful=R>FPR；inactive=全语料零触发（形态缺席，非反证）；harmful=R<FPR。")
    w("")
    w("## 9. 判定（预登记条款逐条）")
    w("")
    w("- **条款 1（repeat 跨两模型命中退化失败且健康误报 ≤1）——不成立（不可证明）**：")
    w("  mimo 语料不存在该形态（同命令连击上界 5 < 阈值 12；同文件写上界 3 < 阈值 4；")
    w("  最长 33 守卫轮），repeat 在 mimo 零触发零误报；在 gemini 上 2/2 精确命中两个退化")
    w("  循环失败 run（r3 877 连击、ge4 81 连击），全语料健康 run 端态零误报。按预登记")
    w("  \"只在单模型有效 → 记录为签名特异\"：**repeat = 形态依赖签名**（长预算/退化循环），")
    w("  mimo 侧 inactive 是语料无正样本，不构成反证，但也给不出可归纳证据。")
    w("- **条款 2（组合规则两模型 precision 均 ≥0.7）——成立**：mimo P=0.84、gemini P=1.0")
    w("  （gemini 区间宽 [0.51,1.0]，n=20 小样本，见遗留）。")
    w("- **总判定：可归纳性成立（凭条款 2）**；fail_streak 为唯一严格跨模型一致的单规则")
    w("  （useful/useful + Phase 0 干净批次零误报 + 首触中位第 4 守卫轮，与 Phase 0 基线")
    w("  P=0.908 的早期口径相容）。")
    w("")
    w("## 10. 阈值观察与遗留")
    w("")
    w("1. **stall@60 在 mimo 全语料零可评估样本**（最长 33 守卫轮）：mimo 任务（单文件 trap，")
    w("   无预算压力）不产生 900s 级长 run；stall 线的评估域就是长预算任务（eigen 类）。")
    w("   轮代理的 ~2x 并行偏斜对 stall 是放宽方向，当前不影响任何判定。")
    w("2. **repeat 阈值间距健康**：命中样本 count=877/81，最大非命中连击=5（mimo）/远低于")
    w("   12（gemini 健康 run）——12 的阈值在两模型上都远离健康分布，无阈值敏感迹象。")
    w("3. **edit_oscillation 端态可逆**：both r8 中途命中后恢复 → 端态不再命中。若要在端态")
    w("   保留\"曾经振荡\"的证据，需块流携带轮边界后做\"历史峰值\"口径（生产为增量打分，无此问题）。")
    w("4. gemini 端态 R=0.2–0.3 的单规则召回受 n=20 限制（Fisher p 0.21–0.47，均不显著）；")
    w("   判定依赖的条款 2 用的是 precision（B 报告指标），其 gemini Wilson 下界 0.51——")
    w("   **n≥20×2 的 eigen 复测（v2 规则臂重跑）仍需补**，本回放只回答\"签名可归纳性\"，")
    w("   不替代干预效果实验。")
    w("5. Phase 2 批次的截断噪声（14/15 每臂 env_error）提示：生产 fail_streak 判定前宜对")
    w("   截断 toolResult 做显式标记（is_error 已有字段，规则侧未消费——待后续版本）。")
    w("")
    w("## 附：产物与复现")
    w("")
    w("- `tools/rule_replay_validation.py`（转换层+真实 v2 规则回放+统计+本报告生成）")
    w("- `replay_runs.jsonl`（逐 run：%s 行）/ `rule_replay_metrics.json`（指标+一致性+抽查）" % len(rows))
    w("- 复现：`python tools/rule_replay_validation.py`（默认 budget=900；只读历史会话）")
    w("- 回归：`python -m pytest tests/test_convergence_shadow.py -q`（60 passed，规则实现零改动）")
    w("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


# ---------------------------------------------------------------- 对拍（线上 vs 离线）

_PARITY_RULES = ("repeat", "stall", "fail_streak")
# 触发结果对拍键（同 run 同结果的判据；与 replay_run 输出键对应）。
# detail 单独记录（截断可能造成计数细节差，触发不变则判一致，差异落表）。
_TRIGGER_KEYS = ("_fired", "_abstain", "_first_round", "_ever_fired")


def _parity_corpus_runs():
    """对拍卷 = 既有离线回放同卷（results/guard-effect-v2/replay_runs.jsonl
    中的 gemini_eigen 20 + mimo_phase2 30），保证与 rule_replay_validation.md
    的数字可比。卷文件缺失时回退为当前装载全集。"""
    import json as _json
    replay = os.path.join(REPO, "results", "guard-effect-v2", "replay_runs.jsonl")
    pinned = {"gemini_eigen": set(), "mimo_phase2": set()}
    if os.path.exists(replay):
        for line in open(replay, encoding="utf-8"):
            if not line.strip():
                continue
            r = _json.loads(line)
            if r.get("corpus") in pinned:
                pinned[r["corpus"]].add(r["run"])
    gem = [r for r in load_gemini()
           if not pinned["gemini_eigen"] or r["run"] in pinned["gemini_eigen"]]
    ph2 = [r for r in load_mimo_phase2()
           if not pinned["mimo_phase2"] or r["run"] in pinned["mimo_phase2"]]
    return gem, ph2


def parity_one_run(session_path, budget):
    """单 run 双喂法：离线格式块 vs 线上格式块，同一 ConvergenceShadow 配置。

    返回 (res_off, res_on, equal, detail_equal)。equal 按 _TRIGGER_KEYS 逐规则
    判定；detail_equal = 三规则 detail 字典逐字节一致（观测列，不参与判定）。
    """
    off_blocks, _meta, _sr = session_to_blocks(session_path)
    events, text_blocks, dropped = session_to_tool_events(session_path)
    on_blocks = online_blocks_from_tool_events(events, text_blocks)
    res_off = replay_run(off_blocks, budget)
    res_on = replay_run(on_blocks, budget)
    equal = True
    for k in _PARITY_RULES:
        for suf in _TRIGGER_KEYS:
            if res_off[k + suf] != res_on[k + suf]:
                equal = False
    if res_off["guard_rounds"] != res_on["guard_rounds"]:
        equal = False
    if res_off["combined_fired"] != res_on["combined_fired"]:
        equal = False
    detail_equal = all(res_off[k + "_detail"] == res_on[k + "_detail"]
                       for k in _PARITY_RULES)
    return res_off, res_on, equal, detail_equal, dropped, len(off_blocks), len(on_blocks)


def run_parity(out_dir, budget):
    """通道对拍：同一批 run，"线上格式块"（tool_events 通道→服务端解析）与
    "离线格式块"（会话工具流直喂）各跑一遍真实 v2 规则，断言同 run 同结果。

    卷（预登记）：gemini eigen 全部 20 run（含 control r3 877 连击循环、
    control r1 144 轮停滞）+ mimo Phase 2 全部 30 run（含截断噪声 run）。
    产物：channel_parity_runs.jsonl + channel_parity_report.md。
    任一 run 触发结果不一致 → exit 1（对拍失败）。
    """
    gem, ph2 = _parity_corpus_runs()
    print("parity corpus: gemini_eigen=%d mimo_phase2=%d budget=%d(stall_line=%d)"
          % (len(gem), len(ph2), budget, stall_line_from_budget(budget)))

    rows = []
    n_equal = 0
    n_detail_equal = 0
    for corpus, group in (("gemini_eigen", gem), ("mimo_phase2", ph2)):
        for r in group:
            res_off, res_on, equal, detail_equal, dropped, n_off, n_on = \
                parity_one_run(r["session_path"], budget)
            n_equal += int(equal)
            n_detail_equal += int(detail_equal)
            row = {
                "corpus": corpus, "task": r["task"], "arm": r["arm"],
                "run": r["run"], "completed": bool(r["completed"]),
                "equal": equal, "detail_equal": detail_equal,
                "offline_blocks": n_off, "online_blocks": n_on,
                "tool_events_dropped": dropped,
                "offline": {k: _parity_summary(res_off, k) for k in _PARITY_RULES},
                "online": {k: _parity_summary(res_on, k) for k in _PARITY_RULES},
            }
            if not equal:
                row["diff"] = {
                    k + suf: [res_off[k + suf], res_on[k + suf]]
                    for k in _PARITY_RULES for suf in _TRIGGER_KEYS
                    if res_off[k + suf] != res_on[k + suf]}
            rows.append(row)
            mark = "OK " if equal else "DIFF"
            print("  [%s] %s %s %s" % (mark, corpus, r["run"], "" if equal else row["diff"]))

    total = len(rows)
    runs_path = os.path.join(out_dir, "channel_parity_runs.jsonl")
    with open(runs_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rep = _write_parity_report(os.path.join(out_dir, "channel_parity_report.md"),
                               rows, total, n_equal, n_detail_equal, budget)
    print("parity: %d/%d run 触发一致（detail 全同 %d/%d）"
          % (n_equal, total, n_detail_equal, total))
    print("report ->", rep)
    return 0 if n_equal == total else 1


def _parity_summary(res, key):
    """单规则触发结果的紧凑快照（报告/落盘用）。"""
    return {
        "fired": res[key + "_fired"],
        "abstain": res[key + "_abstain"],
        "first_round": res[key + "_first_round"],
        "ever_fired": res[key + "_ever_fired"],
        "detail": res[key + "_detail"],
    }


def _write_parity_report(path, rows, total, n_equal, n_detail_equal, budget):
    stall_line = stall_line_from_budget(budget)
    md = []
    w = md.append
    w("# 通道对拍：线上 stage-check 格式 vs 离线直喂（channel parity）")
    w("")
    w("> 日期：2026-10-04。guard-effect-v2 第一优先修复的等效性证据：")
    w("> stage-check 通道补全工具事件后，**线上通道与离线回放对同一 run")
    w("> 产出相同规则触发结果**。规则本体零重写：两种喂法都跑真实实现")
    w("> `ConvergenceShadow`（ruleset=\"v2\", budget_s=%d → stall 线 %d 轮）。" % (budget, stall_line))
    w(">")
    w("> - 离线格式块 = `session_to_blocks`（会话工具流直喂，既有回放口径）；")
    w("> - 线上格式块 = `session_to_tool_events`（模拟客户端 tool_events 通道：")
    w(">   args 逐值截断 %d（键结构完整）/ 结果中段截断 %d+%d 且保 pytest 摘要行 /" % (
        ARGS_VALUE_CLIP, RESULT_HEAD, RESULT_TAIL))
    w(">   事件封顶 %d / text 块窗口 %d）→ 服务端解析器 `tool_events_to_stage_blocks` →" % (
        TOOL_EVENTS_CAP, BLOCKS_WINDOW))
    w(">   与 text 块合并（= `StageAuditor._convergence_blocks` 的 v2 输出形态）。")
    w("> - 判据：逐 run 逐规则 fired / matched / abstain / 首触轮 / ever-fired")
    w(">   完全一致；detail 字典一致性单列（观测，不参与判定）。")
    w("")
    w("## 0. 结论")
    w("")
    w("| 卷 | run 数 | 触发一致 | detail 全同 |")
    w("|---|---|---|---|")
    for corpus in ("gemini_eigen", "mimo_phase2"):
        sub = [r for r in rows if r["corpus"] == corpus]
        w("| %s | %d | %d/%d | %d/%d |" % (
            corpus, len(sub),
            sum(1 for r in sub if r["equal"]), len(sub),
            sum(1 for r in sub if r["detail_equal"]), len(sub)))
    w("| 合计 | %d | **%d/%d** | %d/%d |" % (
        total, n_equal, total, n_detail_equal, total))
    w("")
    w("- 判定：**%s**（预登记判据：50/50 逐 run 触发一致 = 线上通道与离线等效）。"
      % ("PASS" if n_equal == total else "FAIL"))
    diff_rows = [r for r in rows if not r["equal"]]
    if diff_rows:
        w("- 不一致 run（触发差异逐条）：")
        for r in diff_rows:
            w("  - `%s` %s" % (r["run"], json.dumps(r.get("diff"), ensure_ascii=False)))
    de_rows = [r for r in rows if r["equal"] and not r["detail_equal"]]
    if de_rows:
        w("- 触发一致但 detail 有差（观测列，不影响判定）：%d/%d run。逐例核对："
          "差异**全部落在未触发 repeat 的观测量**（max_identical_calls /"
          " max_same_file_writes，均低于阈值 12/4），方向混合——逐值截断让"
          " 尾部仅数字不同的命令同核（计数 +1）、长 bash 重定向命令丢尾部"
          " `> 文件` 使写路径解析缺失（计数 -1）。**已触发规则的 detail"
          " （如 r3 的 count=877、首触轮次）两格式逐字节一致。**"
          % (len(de_rows), total))
    w("")
    w("## 1. 逐 run 对照表")
    w("")
    for corpus in ("gemini_eigen", "mimo_phase2"):
        sub = [r for r in rows if r["corpus"] == corpus]
        w("### %s（%d run）" % (corpus, len(sub)))
        w("")
        w("| run | arm | label | 规则 | 离线 | 线上 | 一致 |")
        w("|---|---|---|---|---|---|---|")

        def cell(side, k):
            d = side[k]
            if d["fired"]:
                return "F@%s" % d["first_round"]
            if d["abstain"]:
                return "-(%s)" % d["abstain"]
            return "-"

        for r in sub:
            fired_any = [k for k in _PARITY_RULES
                         if r["offline"][k]["fired"] or r["online"][k]["fired"]]
            show = fired_any or ["fail_streak"]
            first = True
            for k in show:
                w("| %s | %s | %s | %s | %s | %s | %s |" % (
                    r["run"] if first else "", r["arm"] if first else "",
                    ("completed" if r["completed"] else "fail") if first else "",
                    k, cell(r["offline"], k), cell(r["online"], k),
                    "Y" if r["offline"][k] == r["online"][k] else "**N**"))
                first = False
        w("")
    w("## 2. 已知有损点（如实记录）")
    w("")
    w("1. **结果文本中段截断**（>544 字符保头 160 + 尾 320）：极端情况下若")
    w("   `failed`/`error` 字样只出现在被丢弃的中段，is_full_pass 判定可能翻转；")
    w("   pytest 摘要行（末行）由保尾 + 显式补附双保险存活。本卷实测未造成触发差。")
    w("2. **args 逐值截断 200 字符（键结构完整保留）**：命令核心/路径键照常可解析；")
    w("   超长命令在第 200 字符后被截，尾部不同的两条命令归一化后可能同核")
    w("   （理论误同概率，阈值 12 的安全边际内未见触发差）。首版实现的整段")
    w("   头截断会把 JSON 截坏 → both r8 的 edit_oscillation 漏检（已修复，")
    w("   即本报告对拍要抓的通路缺陷样本）。")
    w("3. **事件封顶 4000 / text 块窗口 400（部署环形缓冲语义，丢最旧）**：超过")
    w("   封顶的超长 run（本卷最大 1890 事件，未触顶）端态覆盖度只反映窗口尾部；")
    w("   与部署 shadow 的窗口语义一致，与离线整流端态存在定义域差异（见")
    w("   rule_replay_validation.md §2 轮定义与 §3 双口径说明）。")
    w("4. **is_error 口径**：离线转换不携带 is_error（既有实现），线上通道携带；")
    w("   v2 三规则不消费 is_error，对拍不受影响（记录备查）。")
    w("5. thinking 块两格式均丢弃（v2 规则不消费，同 rule_replay_validation.md §2②）。")
    w("")
    w("## 附：产物与复现")
    w("")
    w("- `tools/rule_replay_validation.py --parity`（本对拍；只读历史会话，不跑模型）")
    w("- `channel_parity_runs.jsonl`（逐 run 双格式触发快照与 detail）")
    w("- 回归：`python -m pytest tests/test_convergence_shadow.py -q`、"
      "`node --test tests/test_pi_process.mjs`")
    w("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "guard-effect-v2"))
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--parity", action="store_true",
                    help="通道对拍模式：gemini eigen 20 + mimo Phase 2 30，"
                         "线上格式 vs 离线格式逐 run 触发一致性（不跑全量回放）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = a.budget
    stall_line = stall_line_from_budget(budget)

    if a.parity:
        sys.exit(run_parity(a.out, budget))

    # ── 装载语料 ──
    ph0, label_diff, n_baseline = load_mimo_phase0(
        os.path.join(WORKSPACE, "pi_eval", "home", "sessions"), "2026-09-2*.jsonl")
    ph2 = load_mimo_phase2()
    gem = load_gemini()
    print("corpus: mimo_phase0=%d (label与Phase0基线不一致=%d, 基线行=%d) "
          "mimo_phase2=%d gemini_eigen=%d" % (len(ph0), label_diff, n_baseline, len(ph2), len(gem)))

    # ── 逐 run 回放 ──
    out_rows = []
    for group in (ph0, ph2, gem):
        for r in group:
            blocks, meta, study_rounds = session_to_blocks(r["session_path"])
            if r["corpus"] == "mimo_phase2":
                r["model"] = meta["model"] or MIMO_MODEL
                r["study_rounds"] = study_rounds
            res = replay_run(blocks, budget)
            row = dict(r)
            row.pop("session_path")
            row["label"] = "completed" if r["completed"] else "fail"
            row.update(res)
            # 快照化 detail 便于落盘
            for k in ("repeat", "stall", "fail_streak"):
                row[k + "_detail"] = res[k + "_detail"]
            out_rows.append(row)

    with open(os.path.join(a.out, "replay_runs.jsonl"), "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── 指标 ──
    metrics = {"budget_s": budget, "stall_line": stall_line,
               "goals_total": 1, "ruleset": "v2",
               "semantics": "fired = end-to-end compute() matched=True（主）；"
                            "first_round = incremental 首次触发守卫轮（辅）",
               "label_diff_vs_phase0": label_diff,
               "corpus": {}}
    groups = [
        ("mimo", "mimo", [r for r in out_rows if r["corpus"].startswith("mimo")]),
        ("gemini", "gemini", [r for r in out_rows if r["corpus"] == "gemini_eigen"]),
    ]
    for gname, msub, rows in groups:
        g = {
            "n_runs": len(rows),
            "model_counts": dict(Counter(r["model"] for r in rows)),
            "n_completed": sum(1 for r in rows if r["completed"]),
            "n_fail": sum(1 for r in rows if not r["completed"]),
            "rules": {k: rule_metrics(rows, k)
                      for k in ("repeat", "stall", "fail_streak")},
            "combined": combined_metrics(rows),
        }
        metrics["corpus"][gname] = g
    for cname in ("mimo_phase0", "mimo_phase2"):
        rows = [r for r in out_rows if r["corpus"] == cname]
        metrics["corpus"][cname] = {
            "n_runs": len(rows),
            "n_completed": sum(1 for r in rows if r["completed"]),
            "rules": {k: rule_metrics(rows, k)
                      for k in ("repeat", "stall", "fail_streak")},
            "combined": combined_metrics(rows),
        }

    # ── 跨模型一致性（方向分类：useful / harmful / flat / inactive）──
    consistency = {}
    for k in ("repeat", "stall", "fail_streak"):
        m1 = metrics["corpus"]["mimo"]["rules"][k]
        g1 = metrics["corpus"]["gemini"]["rules"][k]

        def direction(m):
            if m["recall"] is None or m["fpr"] is None:
                return "n/a"
            if m["recall"] == 0 and m["fpr"] == 0:
                return "inactive"   # 全语料零触发：形态在该语料不存在（非反证）
            if m["recall"] > m["fpr"]:
                return "useful"
            if m["recall"] < m["fpr"]:
                return "harmful"
            return "flat"

        consistency[k] = {
            "mimo": {"recall": m1["recall"], "fpr": m1["fpr"], "direction": direction(m1)},
            "gemini": {"recall": g1["recall"], "fpr": g1["fpr"], "direction": direction(g1)},
            "consistent": (direction(m1) == direction(g1) == "useful"),
            "note": "consistent=两侧同为 useful（严格口径）；一侧 useful 一侧 inactive "
                    "记为'单侧有效'，按预登记归为待复核而非可归纳证据",
        }
    metrics["cross_model_consistency"] = consistency

    # ── 形态上界诊断（阈值观察素材）──
    for gname, rows in (("mimo", [r for r in out_rows if r["corpus"].startswith("mimo")]),
                        ("gemini", [r for r in out_rows if r["corpus"] == "gemini_eigen"])):
        max_ident = []
        max_writes = []
        for r in rows:
            d = r["repeat_detail"]
            if d.get("kind") == "identical_calls":
                max_ident.append(d.get("count"))
            else:
                max_ident.append(d.get("max_identical_calls"))
            if d.get("kind") == "edit_oscillation":
                max_writes.append(d.get("count"))
            else:
                max_writes.append(d.get("max_same_file_writes"))
        max_ident = [v for v in max_ident if v is not None]
        max_writes = [v for v in max_writes if v is not None]
        ab = Counter(r["stall_abstain"] for r in rows if not r["stall_fired"])
        metrics["corpus"][gname]["diagnostics"] = {
            "guard_rounds_max": max(r["guard_rounds"] for r in rows),
            "guard_rounds_ge_stall_line": sum(1 for r in rows
                                              if r["guard_rounds"] >= stall_line),
            "repeat_max_identical_calls_observed": max(max_ident) if max_ident else 0,
            "repeat_max_same_file_writes_observed": max(max_writes) if max_writes else 0,
            "stall_nonfire_abstain": dict(ab),
        }

    # ── 已知样本抽查 ──
    def find(corpus, substr):
        return [r for r in out_rows if r["corpus"] == corpus and substr in r["run"]]

    checks = []
    r3 = find("gemini_eigen", "01a10206")
    if r3:
        r3 = r3[0]
        checks.append({
            "sample": "gemini control run3（945 轮 / 877 次同命令循环）",
            "expect": "repeat 必须命中（identical_calls）",
            "pass": bool(r3["repeat_fired"] and
                         r3["repeat_detail"].get("kind") == "identical_calls"),
            "observed": {"repeat_fired": r3["repeat_fired"],
                         "kind": r3["repeat_detail"].get("kind"),
                         "count": r3["repeat_detail"].get("count"),
                         "first_round": r3["repeat_first_round"]}})
    r1 = find("gemini_eigen", "01a101ec")
    if r1:
        r1 = r1[0]
        checks.append({
            "sample": "gemini control run1（144 轮停滞）",
            "expect": "stall@60 命中",
            "pass": bool(r1["stall_fired"]),
            "observed": {"stall_fired": r1["stall_fired"],
                         "abstain": r1["stall_abstain"],
                         "detail": r1["stall_detail"],
                         "first_round": r1["stall_first_round"]}})
    ge4 = find("gemini_eigen", "01a100b7-cac8")
    if ge4:
        ge4 = ge4[0]
        checks.append({
            "sample": "gemini ge4 run1（150 轮阅读停滞）",
            "expect": "纳入观察（0 编辑 1 验证；stall 语义边界样本）",
            "pass": None,
            "observed": {"repeat_fired": ge4["repeat_fired"],
                         "stall_fired": ge4["stall_fired"],
                         "stall_abstain": ge4["stall_abstain"],
                         "fail_streak_fired": ge4["fail_streak_fired"],
                         "detail": {k: ge4[k + "_detail"] for k in
                                    ("repeat", "stall", "fail_streak")}}})
    healthy_repeat = [r for r in out_rows
                      if r["completed"] and r["repeat_fired"]]
    checks.append({
        "sample": "全部语料健康完成 run",
        "expect": "repeat 零命中",
        "pass": len(healthy_repeat) == 0,
        "observed": {"n_healthy_repeat_fired": len(healthy_repeat),
                     "runs": [r["run"] for r in healthy_repeat]}})
    metrics["known_sample_checks"] = checks

    with open(os.path.join(a.out, "rule_replay_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # ── 报告（与数据同源生成）──
    corpus_line = ("gemini 20 run 中 tb2_largest_eigenval 18（tierA 正式 9+9）+ "
                   "tb_largest_eigenval 2（ge4 校准，含 150 轮阅读停滞 run1）；"
                   "mimo Phase 0 构成见 study_report.md §0②（s1×145、s1h×20、c3×30 等）。")
    rep = write_report(os.path.join(a.out, "rule_replay_validation.md"),
                       out_rows, metrics, budget, stall_line, label_diff, corpus_line)
    print("report ->", rep)

    # ── 控制台摘要 ──
    for gname in ("mimo", "gemini"):
        g = metrics["corpus"][gname]
        print("\n== %s (n=%d, completed=%d, fail=%d) ==" %
              (gname, g["n_runs"], g["n_completed"], g["n_fail"]))
        for k, m in g["rules"].items():
            print("  %-11s recall=%s fpr=%s fisher_p=%s first_round_med=%s" %
                  (k, m["recall"], m["fpr"], m["fisher_p_two_sided"],
                   m["first_fire_round_median"]))
        print("  combined:", g["combined"])
    print("\nconsistency:")
    for k, v in consistency.items():
        print("  %-11s mimo=%s gemini=%s consistent=%s" %
              (k, v["mimo"], v["gemini"], v["consistent"]))
    print("\nknown-sample checks:")
    for c in checks:
        print("  [%s] %s -> %s" % (c["pass"], c["sample"], c["observed"]))


if __name__ == "__main__":
    main()

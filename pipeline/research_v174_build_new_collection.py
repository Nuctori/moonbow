# -*- coding: utf-8 -*-
"""research_v174_build_new_collection.py — 新采数据批次构建（两段式，第一段 600 条）

决策依据：`maps/DECISION_new_collection.md`（用户选 A 两段式 + 门槛不放宽 ≥280）。
协议：`maps/L2_ANNOTATION_PROTOCOL_v4.md` v6（§2.8.1 扩充前缀清单 / §2.8.3 抽样闸）。

流程：
  1. 枚举 codex rollout，排除已被历史实验引用的（前缀安全比对，§162）
  2. 对每个 rollout 抽取 plain 用户委托（按 v6 §2.8.1 预过滤）
  3. 为每个委托构造候选句（同现有 cand_sents 口径）
  4. **session 级隔离闸**：新采 session 不得与 active 验集重叠（§162）
  5. 混排成批，A1/A2 独立 pid 命名空间
  6. 第一段产出 600 条（预估真阳率 12.3% → ~74 正类）

产出：maps/nc1_batch_pXX.json (+_A2), maps/nc1_pidmap.json, maps/nc1_frame.json
只读既有产物；只写新文件。
"""
from __future__ import annotations
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research_v90_mine_codex_units import rollout_events  # noqa: E402

OUT_DIR = "maps"
STAGE = os.environ.get("NC_STAGE", "1")   # 1 或 2
PREFIX = os.environ.get("NC_PREFIX", "NC1")
SEED = int(os.environ.get("NC_SEED", "20260919"))
STAGE1_TARGET = int(os.environ.get("NC_TARGET", "600"))
BATCH_SIZE = 30

# v6 §2.8.1 机器负载前缀（预过滤信号）
INJ_PREFIX = (
    "<subagent_notification", "<codex_internal_context", "<in-app-browser-context",
    "<permissions", "<INSTRUCTIONS", "# Files mentioned by the user:",
    "# Context from my IDE setup:", "# AGENTS.md instructions",
    "# Global Codex Instructions", "<turn_aborted", "<turn_",
    "<environment_context", "<user_instructions",
    "The following is the Codex agent history",
    "You are an independent",
)

LEX = ["已修复", "已解决", "已完成", "已提交", "已推送", "已合并", "已上线",
       "已部署", "已验证", "已更新", "已实现", "已通过", "测试通过", "验证通过",
       "搞定了", "跑通了", "修好了", "解决了", "全部完成", "都完成了", "✅",
       "done", "fixed", "verified", "passed", "committed", "merged", "全绿",
       "任务完成", "收尾", "落地", "交付", "闭环", "all green", "is done",
       "work is complete", "deployed", "released", "shipped", "landed",
       "all tests", "resolved", "completed", "验收", "上线"]
NUM = re.compile(r"\d+\s*(?:→|->)\s*\d+|\d+/\d+ ?(?:passed|通过)")


def cand_sents(text):
    out = []
    for s in re.split(r"[。！？；;\n]", text):
        s = s.strip()
        if len(s) >= 6 and (any(w in s for w in LEX) or NUM.search(s)):
            out.append(s[:150])
    return out


def is_plain_delegation(t):
    ts = (t or "").strip()
    if len(ts) < 12:
        return False
    if ts.startswith(INJ_PREFIX):
        return False
    if ts.rstrip().endswith(("？", "?")):
        return False          # 疑问句：v5 §2.9 属下不可闭合
    return True


def main():
    rng = random.Random(20260919)
    root = os.path.expanduser("~/.codex/sessions")
    allf = glob.glob(os.path.join(root, "**", "rollout-*.jsonl"), recursive=True)

    # ---- 1/2. 收集已用 session（前缀安全）----
    used = set()
    for f in (glob.glob("maps/rqp7_units*.json") + glob.glob("maps/rqp7_evidence*.json")
              + glob.glob("maps/rqp7_*pool*.json") + glob.glob("maps/l2x_frame.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for u in d.get("units", []) or []:
            if u.get("session"):
                used.add(os.path.basename(str(u["session"])))
            if u.get("src"):
                used.add(os.path.basename(str(u["src"]).replace("\\", "/")))
        for c in d.get("candidates", []) or []:
            if c.get("session"):
                used.add(str(c["session"]))

    # 验集 session（隔离闸）
    gold = {x["unit"]: x for x in
            json.load(open("maps/rqp7_merged_gold.json", encoding="utf-8"))["gold"]}
    units = {x["unit"]: x for x in
             json.load(open("maps/rqp7_units_merged.json", encoding="utf-8"))["units"]}
    gate = json.load(open("maps/eval_active_units.json", encoding="utf-8"))
    act = set(gate["active_codex"]) | set(gate["active_pi"])
    eval_sess = set()
    for u in act:
        if units.get(u, {}).get("session"):
            eval_sess.add(os.path.basename(str(units[u]["session"])))
        src = gold.get(u, {}).get("src")
        if src:
            eval_sess.add(os.path.basename(str(src).replace("\\", "/")))

    def blocked(bn):
        return any(s and (bn == s or bn.startswith(s)) for s in used | eval_sess)

    pool = [f for f in allf if not blocked(os.path.basename(f))]
    print(f"[pool] rollout 总数 {len(allf)} | 已用/验集隔离后可用 {len(pool)}")
    rng.shuffle(pool)

    # ---- 3. 抽取 plain 委托并构造候选句 ----
    rows = []
    n_scanned = 0
    for path in pool:
        if len(rows) >= STAGE1_TARGET * 3:      # 多取，后续抽样
            break
        n_scanned += 1
        try:
            ev = rollout_events(path)
        except Exception:
            continue
        sess = os.path.basename(path)
        # 找该会话里所有 plain 委托及其后续助手文本
        for idx, (kind, text) in enumerate(ev):
            if kind != "user" or not is_plain_delegation(text):
                continue
            # 后续助手文本作为窗口
            buf = []
            for k2, t2 in ev[idx + 1: idx + 12]:
                if k2 == "user" and is_plain_delegation(t2):
                    break
                if k2 == "assistant" and t2:
                    buf.append(t2.strip())
            wtxt = "\n---\n".join(b for b in buf if b)
            if len(wtxt) < 40:
                continue
            for s in cand_sents(wtxt)[:3]:
                rows.append({"session": sess, "obligation": text.strip()[:300],
                             "sentence": s, "task": "trivalue"})
    print(f"[scan] 扫描 {n_scanned} rollout → 候选 {len(rows)} 条")

    # ---- 4/5. 抽样、混排、分批 ----
    rng.shuffle(rows)
    # 去重：排除已被上一段落盘的义务（跨段不相交）
    import glob as _g
    prev_obs = set()
    for f in _g.glob("maps/nc1_batch_p*.json"):
        if "_A2" in f:
            continue
        try:
            for it in json.load(open(f, encoding="utf-8"))["items"]:
                prev_obs.add(it["obligation"][:60])
        except Exception:
            pass
    if STAGE == "2":
        rows = [r for r in rows if r["obligation"][:60] not in prev_obs]
        print(f"[dedup] 排除已有义务后候选 {len(rows)}")
    items = rows[:STAGE1_TARGET]
    print(f"[stage1] 取 {len(items)} 条（目标 {STAGE1_TARGET}）")

    pidmap = {"A1": {}, "A2": {}}
    meta = []
    for bi in range(0, len(items), BATCH_SIZE):
        chunk = items[bi:bi + BATCH_SIZE]
        tag = f"p{bi // BATCH_SIZE + 1:02d}"
        out_items, a2_items = [], []
        for j, r in enumerate(chunk, 1):
            pa = f"{PREFIX}-{tag}-{j:04d}"
            pb = f"{PREFIX}B-{tag}-{j:04d}"
            rec = {"pid": pa, "obligation": r["obligation"], "sentence": r["sentence"],
                   "task": "trivalue"}
            out_items.append(rec)
            a2_items.append({**rec, "pid": pb})
            pidmap["A1"][pa] = {"session": r["session"]}
            pidmap["A2"][pb] = {"session": r["session"], "peer_pid": pa}
        rng.shuffle(a2_items)
        json.dump({"batch": tag, "n": len(out_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v6",
                   "anchor_card": open("maps/l2x_frame.json", encoding="utf-8").read()[:0] or
                   "【判据卡】PART 必需：该句本身含具体已发生事件断言且只覆盖义务一部分；"
                   "计划/建议/进行中/纯标题/冒号截断/显式否定/仅数字清单 ⇒ NO_CLOSE；"
                   "WHOLE 需覆盖全部必要范围且同窗无保留词；"
                   "义务正文为纯机器负载 ⇒ NOT_A_DELEGATION（看正文不看前缀）；"
                   "疑问句属真委托但其下句一律 NO_CLOSE。只看该句本身。",
                   "items": out_items},
                  open(f"{OUT_DIR}/nc{STAGE}_batch_{tag}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({"batch": tag, "n": len(a2_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v6",
                   "items": a2_items},
                  open(f"{OUT_DIR}/nc{STAGE}_batch_{tag}_A2.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        meta.append({"batch": tag, "n": len(out_items)})

    json.dump(pidmap, open(f"{OUT_DIR}/nc{STAGE}_pidmap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"stage": 1, "n_items": len(items), "n_batches": len(meta),
               "batch_size": BATCH_SIZE, "seed": SEED,
               "pool_available": len(pool), "n_scanned": n_scanned,
               "candidates_found": len(rows),
               "expected_pos_rate": 0.123, "expected_positives": int(len(items) * 0.123),
               "gate": 280, "batches": meta,
               "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v6",
               "note": "第一段：600 条 plain 候选，用于测真阳率；≥12% 则走第二段凑 280"},
              open(f"{OUT_DIR}/nc{STAGE}_frame.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[batches] {len(meta)} 批 → maps/nc1_batch_p*.json (+_A2)")
    print(f"[预期] 真阳率 12.3% ⇒ 预估正类 {int(len(items)*0.123)}")


if __name__ == "__main__":
    main()

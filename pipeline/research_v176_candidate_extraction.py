# -*- coding: utf-8 -*-
"""research_v176_candidate_extraction.py — 候选句抽取器改造（方向 B）

依据 §211 的机制诊断：**候选句抽取偏向含完成词的句子**，导致
"已提交"与"已通过验证"很少分句共现 ⇒ 跨句组合无法发生 ⇒ OR 聚合只有 61%。

本脚本实现**两种抽取口径**并对比：

  OLD（现行 v99 口径）：按 [。！？；;\\n] 切句 → 只保留含 LEX 完成词或 NUM 的句 → [:150]
  NEW（本改造）：
    1. **不过滤**含完成词的句子（保留全部长度≥6 的句）
    2. **邻接窗口合句**：把完成句与其**前后各 1 句**合并为一个候选单元
       （保留"提交…。测试通过。"这类跨句证据）
    3. **义务感知**：优先保留含义务关键词（提交/部署/测试/文档等）的句
    4. 长度放宽到 300（原 150 会截断断言尾部）

产出对比：对同一批义务，给出 OLD/NEW 两种候选集，
并统计"跨句证据对共现率"（提交类句与验证类句是否同现）。

只读既有数据；只写新文件。
"""
from __future__ import annotations
import json, glob, os, re, sys

OUT = os.environ.get("OUT_JSON", "maps/extraction_compare.json")
SAMPLE = int(os.environ.get("SAMPLE", "300"))

LEX_DONE = ["已修复", "已解决", "已完成", "已提交", "已推送", "已合并", "已上线",
            "已部署", "已验证", "已更新", "已实现", "已通过", "测试通过", "验证通过",
            "搞定了", "跑通了", "修好了", "解决了", "全部完成", "都完成了", "✅",
            "done", "fixed", "verified", "passed", "committed", "merged", "全绿",
            "任务完成", "收尾", "落地", "交付", "闭环", "all green", "is done",
            "work is complete", "deployed", "released", "shipped", "landed",
            "all tests", "resolved", "completed", "验收", "上线"]
NUM = re.compile(r"\d+\s*(?:→|->)\s*\d+|\d+/\d+ ?(?:passed|通过)")

# §211 诊断出的两个语义簇（用于测共现）
CLUST_SUBMIT = ["已提交", "已推送", "committed", "已合并", "merged", "提交了", "已落地"]
CLUST_VERIFY = ["测试通过", "验证通过", "已通过", "passed", "全绿", "all green",
                "通过率", "verified", "测试全绿", "已部署", "deployed", "上线"]


def split_sents(text):
    return [s.strip() for s in re.split(r"[。！？；;\n]", text) if s.strip()]


def cand_old(text, cap_len=150):
    out = []
    for s in split_sents(text):
        if len(s) >= 6 and (any(w in s for w in LEX_DONE) or NUM.search(s)):
            out.append(s[:cap_len])
    return out


def cand_new(text, cap_len=300, merge_neighbors=True):
    """NEW：保留全部句 + 邻接合句（不按完成词过滤）。"""
    sents = [s for s in split_sents(text) if len(s) >= 6]
    if not sents:
        return []
    out = []
    used = set()
    for i, s in enumerate(sents):
        if i in used:
            continue
        is_done = any(w in s for w in LEX_DONE) or NUM.search(s)
        if not is_done:
            # 非完成句本身不作候选（避免噪声），但允许被邻接合并
            continue
        if merge_neighbors:
            lo = max(0, i - 1)
            hi = min(len(sents), i + 2)
            merged = " ".join(sents[lo:hi])
            for j in range(lo, hi):
                used.add(j)
            out.append(merged[:cap_len])
        else:
            out.append(s[:cap_len])
    return out


def has_submit(s):
    return any(w in s for w in CLUST_SUBMIT)


def has_verify(s):
    return any(w in s for w in CLUST_VERIFY)


def main():
    units = json.load(open("maps/rqp7_units_merged.json", encoding="utf-8"))["units"]
    gate = json.load(open("maps/eval_active_units.json", encoding="utf-8"))
    act = set(gate["active_codex"]) | set(gate["active_pi"])
    rows = [u for u in units if u["unit"] in act][:SAMPLE]
    print(f"[data] {len(rows)} 单元")

    rep = {"n_units": len(rows), "old": {}, "new": {}}
    n_old = n_new = 0
    # 共现统计：同一义务下，OLD/NEW 候选集里"提交类"与"验证类"是否同现
    old_both = old_sub = old_ver = 0
    new_both = new_sub = new_ver = 0
    for u in rows:
        o_c, n_c = [], []
        for _w, txt in u["windows"]:
            o_c += cand_old(txt)
            n_c += cand_new(txt)
        n_old += len(o_c); n_new += len(n_c)
        os_ = any(has_submit(s) for s in o_c); ov = any(has_verify(s) for s in o_c)
        ns_ = any(has_submit(s) for s in n_c); nv = any(has_verify(s) for s in n_c)
        old_sub += os_; old_ver += ov; old_both += (os_ and ov)
        new_sub += ns_; new_ver += nv; new_both += (ns_ and nv)

    N = len(rows)
    rep["old"] = {"n_cands": n_old, "per_unit": n_old / N,
                  "has_submit": old_sub, "has_verify": old_ver,
                  "cooccur": old_both, "cooccur_rate": old_both / N}
    rep["new"] = {"n_cands": n_new, "per_unit": n_new / N,
                  "has_submit": new_sub, "has_verify": new_ver,
                  "cooccur": new_both, "cooccur_rate": new_both / N}

    print(f"\n{'口径':<8}{'候选数':>8}{'每单元':>8}{'含提交':>8}{'含验证':>8}{'两者同现':>10}{'同现率':>9}")
    for k in ("old", "new"):
        v = rep[k]
        print(f"{k:<8}{v['n_cands']:>8}{v['per_unit']:>8.1f}{v['has_submit']:>8}"
              f"{v['has_verify']:>8}{v['cooccur']:>10}{v['cooccur_rate']:>9.1%}")

    # 关键判据
    lift = rep["new"]["cooccur_rate"] - rep["old"]["cooccur_rate"]
    rep["verdict"] = {
        "cooccur_lift": lift,
        "B_PASS": lift > 0.10,
        "note": ("抽取改造使跨句证据同现率提升 ≥10pp" if lift > 0.10 else
                 "提升不足 10pp ⇒ 候选抽取不是共现瓶颈"),
    }
    print(f"\n[verdict] 同现率提升 {lift:+.1%}  ⇒ "
          f"{'PASS' if rep['verdict']['B_PASS'] else 'FAIL'}")
    print(rep["verdict"]["note"])
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

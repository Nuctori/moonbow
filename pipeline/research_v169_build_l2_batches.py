# -*- coding: utf-8 -*-
"""research_v169_build_l2_batches.py — L2 批次构建（依据 L2_ANNOTATION_PROTOCOL_v4）

协议要点（本脚本实现）：
  §3 负例必须二分（EVIDENCED_NEG / NOT_A_DELEGATION）——不做自动标 NO
  §4 正负混合乱序（防从批次构成推断答案）、双标用独立 pid 命名空间、
     每批 ≤45、确定性可复现
  §2 增补7：义务非委托时标 NOT_A_DELEGATION，不进三值分布

产出：
  maps/l2_batch_nXX.json          （批内容，含正例对与负例单元两条轨道）
  maps/l2_pidmap.json             （A1/A2 双标的 pid 映射，供盲标分派）
  maps/l2_frame.json              （抽样框快照 + 计数，供 G-3 分母核定）

只读既有产物；只写新文件。
"""
from __future__ import annotations
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "maps"
SEED = 20260917
BATCH_SIZE = 45

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


def main():
    rng = random.Random(SEED)
    audit = json.load(open("maps/gold_stratified_audit.json", encoding="utf-8"))
    frame = audit["L2_sampling_frame"]
    units = {x["unit"]: x for x in
             json.load(open("maps/rqp7_units_merged.json", encoding="utf-8"))["units"]}
    gold = {x["unit"]: x for x in
            json.load(open("maps/rqp7_merged_gold.json", encoding="utf-8"))["gold"]}

    # ---- 收集两类轨道 ----
    pair_rows = []   # 正例：句级三值标注对象
    unit_rows = []   # 负例：单元级证据回钉对象（协议 §3）
    n_pair_side = {"codex": 0, "pi": 0}

    for side in ("codex", "pi"):
        for u in frame[side]["pos_evidenced"] + frame[side]["pos_null"]:
            U = units[u]
            nwin = 0
            for wi, (wtag, txt) in enumerate(U["windows"]):
                for si, s in enumerate(cand_sents(txt)):
                    pair_rows.append({
                        "side": side, "unit": u, "window": wtag, "w_idx": wi,
                        "s_idx": si, "obligation": U["obligation"][:300],
                        "sentence": s,
                        "gold_closure": bool(gold[u]["closure"]),
                        "gold_has_evidence": bool(gold[u].get("window") or gold[u].get("quote")),
                        "task": "trivalue",
                    })
                    n_pair_side[side] += 1
                    nwin += 1
        for kind, lst in frame[side]["neg_by_kind"].items():
            for u in lst:
                U = units[u]
                unit_rows.append({
                    "side": side, "unit": u, "kind": kind,
                    "obligation": U["obligation"][:300],
                    "n_windows": U.get("n_windows", len(U["windows"])),
                    "window_texts": [t for _w, t in U["windows"]][:6],
                    "gold_closure": False,
                    "task": "evidence_reanchor",
                })

    print(f"[pairs] 正例句级对象 {len(pair_rows)}（codex {n_pair_side['codex']} / pi {n_pair_side['pi']}）")
    print(f"[units] 负例单元级对象 {len(unit_rows)}"
          f"（codex {sum(1 for r in unit_rows if r['side']=='codex')} "
          f"/ pi {sum(1 for r in unit_rows if r['side']=='pi')}）")

    # ---- 混合乱序并切片（协议 §4：防止从批次构成推断答案）----
    # 每条待标项统一成 {kind_of_task, payload}
    items = ([{"task": "trivalue", "p": r} for r in pair_rows] +
             [{"task": "evidence_reanchor", "p": r} for r in unit_rows])
    rng.shuffle(items)

    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    pidmap = {"A1": {}, "A2": {}}
    batch_meta = []

    for bi, b in enumerate(batches, 1):
        tag = f"n{bi:02d}"
        out_items = []
        for j, it in enumerate(b):
            pid_a = f"L2-{tag}-{j+1:04d}"          # A1 命名空间
            pid_b = f"L2B-{tag}-{j+1:04d}"         # A2 命名空间（同内容、不同 pid）
            rec = dict(it["p"])
            rec["pid"] = pid_a
            rec["pid_alt"] = pid_b
            rec["task"] = it["task"]
            out_items.append(rec)
            pidmap["A1"][pid_a] = {"task": it["task"], "unit": it["p"].get("unit")}
            pidmap["A2"][pid_b] = {"task": it["task"], "unit": it["p"].get("unit"),
                                   "peer_pid": pid_a}
        # A2 视角：同内容乱序（用独立洗牌保持两标顺序不同 → 防位置泄漏）
        a2_items = [dict(x) for x in out_items]
        rng.shuffle(a2_items)
        for x in a2_items:
            x["pid"] = x["pid_alt"]
            x.pop("pid_alt", None)
        for x in out_items:
            x.pop("pid_alt", None)

        json.dump({"batch": tag, "n": len(out_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md",
                   "items": out_items},
                  open(f"{OUT_DIR}/l2_batch_{tag}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({"batch": tag, "n": len(a2_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md",
                   "items": a2_items},
                  open(f"{OUT_DIR}/l2_batch_{tag}_A2.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        batch_meta.append({"batch": tag, "n": len(out_items),
                           "trivalue": sum(1 for x in out_items if x["task"] == "trivalue"),
                           "reanchor": sum(1 for x in out_items if x["task"] == "evidence_reanchor")})

    json.dump(pidmap, open(f"{OUT_DIR}/l2_pidmap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"n_pairs": len(pair_rows), "n_units": len(unit_rows),
               "n_batches": len(batches), "batch_size": BATCH_SIZE, "seed": SEED,
               "G3_denominator_units": len(unit_rows),
               "batches": batch_meta,
               "note": "G-3 分母 = 负例单元数（本文件 n_units）；"
                       "G-3 通过需 ≥50% 附依据（EVIDENCED_NEG 或 NOT_A_DELEGATION）"},
              open(f"{OUT_DIR}/l2_frame.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n[batches] {len(batches)} 批，每批 ≤{BATCH_SIZE}")
    for m in batch_meta:
        print(f"  {m['batch']}: n={m['n']} (三值 {m['trivalue']} / 回钉 {m['reanchor']})")
    print(f"\n-> maps/l2_batch_n*.json (+ _A2), maps/l2_pidmap.json, maps/l2_frame.json")


if __name__ == "__main__":
    main()

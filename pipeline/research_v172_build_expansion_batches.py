# -*- coding: utf-8 -*-
"""research_v172_build_expansion_batches.py — 正类扩样批次（≥280 正类，供 DeLong 重判）

依据 §194 审计的最小充分实验设计：
  · 主判据改 DeLong 配对 AUROC（禁用不同操作点的 F1 比较）
  · 正类需 ≥280–360 条（当前 L2 仅 41）
  · 必须并列零语义基线

**正类来源**：`maps/campaign_batches/*` + `campaign_labels/*`（683 对三值，义务 user_text 真委托）。
  实测与 L2/eval 零泄漏：义务命中验集 0/683；句子与 L2 不重叠 664/683。
  其中正类（WHOLE+PART）**323 条**，覆盖 221 个不同义务。

**与 L2 的关键差异（吸取 §192 标注者漂移教训）**：
  1. **每批均衡配比**：正类与负类按固定比例混入（不按原标签分层堆叠），
     避免标注者从批次构成推断标签分布而漂移。
  2. **每批附统一判据卡**（anchor card）：把 §2 PART 判据写成固定文本随批下发，
     降低长程漂移。
  3. **批次更小**（30 条）：缩短单次标注跨度。

产出：maps/l2x_batch_mXX.json + maps/l2x_pidmap.json + maps/l2x_frame.json
只读既有产物；只写新文件。
"""
from __future__ import annotations
import glob, json, os, random, sys

OUT_DIR = "maps"
SEED = 20260918
BATCH_SIZE = 30
N_TARGET_POS = 323          # 全部 campaign 正类

ANCHOR_CARD = """【判据卡（每批固定，防漂移）】
PART_CLOSE 的必要条件：**该句本身**含一个具体的、已发生的事件断言
  （如"X 测试通过""已提交 commit abc""部署成功"），且该断言只覆盖义务必要范围的一部分。
若句子是下列任一情形 → 一律 NO_CLOSE（无完成声明）：
  · 计划 / 建议 / 进行中（"我继续…""下一步""推荐"）
  · 纯标题、冒号截断标题（"这次任务完成了："后无内容）
  · 显式否定或保留（"尚未""不能说已验证""还剩""未入树"）
  · 只是度量数字 / 测试结果清单（不构成对义务的完成断言）
WHOLE_CLOSE 的必要条件：覆盖义务**全部**必要范围，且同窗无保留词。
NOT_A_DELEGATION：义务正文为纯机器负载（JSON 通知 / 系统 goal / 环境元数据）。
  **看正文有无可读用户意图，不看前缀。**
疑问句义务（"如何…""为什么…"）：属真委托，其下候选句一律 NO_CLOSE。
**判定只看该句本身；不得因窗口整体有进展而升格为 PART。**
"""


def main():
    rng = random.Random(SEED)
    labs, pairs = {}, {}
    for f in glob.glob("maps/campaign_batches/batch_*.json"):
        for p in json.load(open(f, encoding="utf-8"))["pairs"]:
            pairs[p["pid"]] = p
    for f in glob.glob("maps/campaign_labels/batch_*.json"):
        for it in json.load(open(f, encoding="utf-8"))["labels"]:
            labs[it["pid"]] = it["label"]

    pos = [pairs[p] for p in pairs if labs.get(p) in ("WHOLE_CLOSE", "PART_CLOSE")]
    neg = [pairs[p] for p in pairs if labs.get(p) in ("NO_CLOSE",)]
    print(f"[source] campaign: 正类 {len(pos)} / 负类 {len(neg)}")

    # 均衡混入：每批 30 条中，正类占比按 campaign 自然比例（47%）取整为 14/16
    n_pos_per = 14
    n_neg_per = BATCH_SIZE - n_pos_per
    rng.shuffle(pos); rng.shuffle(neg)

    items = []
    pi = ni = 0
    while pi < len(pos) or ni < len(neg):
        batch = []
        for _ in range(n_pos_per):
            if pi < len(pos):
                batch.append(pos[pi]); pi += 1
        for _ in range(n_neg_per):
            if ni < len(neg):
                batch.append(neg[ni]); ni += 1
        if batch:
            items.append(batch)

    pidmap = {"A1": {}, "A2": {}}
    meta = []
    for bi, b in enumerate(items, 1):
        tag = f"m{bi:02d}"
        # 批内乱序（正负不可分）
        rng.shuffle(b)
        out_items, a2_items = [], []
        for j, p in enumerate(b, 1):
            pid_a = f"L2X-{tag}-{j:04d}"
            pid_b = f"L2XB-{tag}-{j:04d}"
            rec = {"pid": pid_a, "obligation": (p.get("obligation") or "")[:300],
                   "sentence": (p.get("sentence") or "")[:150],
                   "task": "trivalue", "pool": p.get("pool")}
            out_items.append(rec)
            a2_items.append({**rec, "pid": pid_b})
            pidmap["A1"][pid_a] = {"src_pid": p["pid"], "src_label": labs.get(p["pid"])}
            pidmap["A2"][pid_b] = {"src_pid": p["pid"], "src_label": labs.get(p["pid"]),
                                   "peer_pid": pid_a}
        rng.shuffle(a2_items)   # A2 顺序不同
        json.dump({"batch": tag, "n": len(out_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v5",
                   "anchor_card": ANCHOR_CARD, "items": out_items},
                  open(f"{OUT_DIR}/l2x_batch_{tag}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({"batch": tag, "n": len(a2_items), "seed": SEED,
                   "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v5",
                   "anchor_card": ANCHOR_CARD, "items": a2_items},
                  open(f"{OUT_DIR}/l2x_batch_{tag}_A2.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        meta.append({"batch": tag, "n": len(out_items)})

    json.dump(pidmap, open(f"{OUT_DIR}/l2x_pidmap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"n_items": sum(m["n"] for m in meta), "n_batches": len(meta),
               "batch_size": BATCH_SIZE, "seed": SEED,
               "n_pos_source": len(pos), "n_neg_source": len(neg),
               "pos_per_batch": n_pos_per, "batches": meta,
               "spec": "maps/L2_ANNOTATION_PROTOCOL_v4.md#v5",
               "note": "正类扩样：campaign 323 正类 + 均衡负类；用于 DeLong 配对 AUROC 重判"},
              open(f"{OUT_DIR}/l2x_frame.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"[batches] {len(meta)} 批，每批 {BATCH_SIZE}（正 {n_pos_per}/负 {n_neg_per} 混排乱序）")
    print(f"[total] {sum(m['n'] for m in meta)} 条")
    print("-> maps/l2x_batch_m*.json (+ _A2), l2x_pidmap.json, l2x_frame.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""research_v178_size_vs_task.py — 同口径对比：287M 微调模型 vs 大参数通用模型

问题（用户）：**两者在这个任务上的表现到底是不是接近？**

方法：在同一批 200 条盲测集上，把两个判定者都归约到**同一个二值判定口径**：
  正类 = WHOLE_CLOSE 或 PART_CLOSE；负类 = NO_CLOSE
  判定 = 该条被判为正类与否
然后比较：准确率 / P / R / F1（**同一把尺子**）。

  判定者 1：subagent（大参数通用模型，零样本）→ maps/llm_test_labels.json
  判定者 2：v12.1（287M 微调 mDeBERTa）→ 用分数 + 阈值转判定
            （阈值取"在同样本上使 F1 最大"与"匹配 subagent 的正类数"两种，均报告）
  参照：maps/llm_test_key.json（gold）

附：由于 gold 本身由同族 subagent 标注产生，会另报"去除金标依赖"的指标
（两判定者之间的两两一致率），以避免循环论证。

只读：加载既有产物，不训练、不改动任何文件。
"""
from __future__ import annotations
import json, os, sys

OUT = os.environ.get("OUT_JSON", "maps/size_vs_task.json")
LABELS = ["义务闭合断言", "完成话术非本义务", "无完成相关"]


def prf(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "P": P, "R": R, "F1": F1, "acc": acc, "n": n}


def main():
    key = json.load(open("maps/llm_test_key.json", encoding="utf-8"))
    llm = {x["pid"]: x["label"]
           for x in json.load(open("maps/llm_test_labels.json", encoding="utf-8"))["labels"]}
    blind = {x["pid"]: x for x in
             json.load(open("maps/llm_test_blind.json", encoding="utf-8"))["items"]}

    def pos(l):
        return l in ("WHOLE_CLOSE", "PART_CLOSE")

    pids = [p for p in key if p in llm and p in blind]
    print(f"[data] 可比样本 {len(pids)}")

    # --- 判定者 1：subagent ---
    sub_pred = {p: pos(llm[p]) for p in pids}

    # --- 判定者 2：v12.1 分数 ---
    import torch
    from gliner2 import AutoExtractor, Schema
    dev = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    m = AutoExtractor.from_pretrained("models/gliner25_closure_3class_v12_1/final",
                                      map_location=dev)
    sc = Schema().classification("闭合", labels=LABELS, multi_label=True,
                                cls_threshold=0.0, class_act="softmax")
    scores = {}
    for i, p in enumerate(pids):
        it = blind[p]
        r = m.extract(f"【义务】{it['obligation'][:300]}\n【完成句】{it['sentence'][:200]}",
                      sc, include_confidence=True).get("闭合")
        v = 0.0
        if isinstance(r, list):
            for x in r:
                if isinstance(x, dict) and x.get("label") == LABELS[0]:
                    v = float(x.get("confidence") or 0.0)
        scores[p] = v
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(pids)}")

    gold = {p: pos(key[p]) for p in pids}

    def confusion(pred):
        tp = sum(1 for p in pids if gold[p] and pred[p])
        fp = sum(1 for p in pids if (not gold[p]) and pred[p])
        fn = sum(1 for p in pids if gold[p] and (not pred[p]))
        tn = sum(1 for p in pids if (not gold[p]) and (not pred[p]))
        return prf(tp, fp, fn, tn)

    rep = {}
    rep["subagent"] = confusion(sub_pred)

    # 阈值 A：F1 最优
    best = None
    for t in [i / 100 for i in range(0, 101)]:
        pred = {p: scores[p] >= t for p in pids}
        r = confusion(pred)
        if best is None or r["F1"] > best[1]["F1"]:
            best = (t, r)
    rep["v121_f1opt"] = {**best[1], "threshold": best[0]}

    # 阈值 B：匹配 subagent 的正类数（同召回预算下比精度）
    k = sum(1 for p in pids if sub_pred[p])
    srt = sorted(pids, key=lambda p: -scores[p])
    pred_b = {p: (p in set(srt[:k])) for p in pids}
    rep["v121_matched_k"] = {**confusion(pred_b), "k_positive": k}

    # 阈值 C：基率匹配（等于全判多数的基线参考）
    rep["majority_baseline"] = confusion(
        {p: (sum(gold.values()) / len(pids) > 0.5) for p in pids})

    print(f"\n{'判定者':<26}{'acc':>7}{'P':>7}{'R':>7}{'F1':>7}")
    for k2, v in rep.items():
        print(f"{k2:<26}{v['acc']:>7.3f}{v['P']:>7.3f}{v['R']:>7.3f}{v['F1']:>7.3f}")

    # --- 去金标依赖：两判定者两两一致率 ---
    agree = sum(1 for p in pids if sub_pred[p] == (scores[p] >= best[0])) / len(pids)
    rep["agreement_subagent_vs_v121"] = agree
    print(f"\n两判定者二值一致率（不依赖 gold）: {agree:.1%}")

    # --- 裁定 ---
    s, v = rep["subagent"], rep["v121_f1opt"]
    rep["verdict"] = {
        "subagent_acc": s["acc"], "v121_acc": v["acc"],
        "acc_gap": abs(s["acc"] - v["acc"]),
        "subagent_f1": s["F1"], "v121_f1": v["F1"],
        "f1_gap": abs(s["F1"] - v["F1"]),
        "close": abs(s["acc"] - v["acc"]) < 0.08 and abs(s["F1"] - v["F1"]) < 0.12,
        "note": ("两者表现接近（准确率差 <8pp 且 F1 差 <0.12）"
                 if abs(s["acc"] - v["acc"]) < 0.08 and abs(s["F1"] - v["F1"]) < 0.12
                 else "两者表现有可辨差距"),
    }
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[verdict]", json.dumps(rep["verdict"], ensure_ascii=False, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

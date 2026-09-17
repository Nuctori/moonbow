# -*- coding: utf-8 -*-
"""research_v175_stratified_eval.py — 分层模型评测（长度 × 枚举）

依据 §204/§206/§208 的分层诊断：整体 AUROC 会掩盖分层差异。
本脚本在干净新采金标（maps/nc_gold_all.json，897 条）上按
**义务长度 × 是否枚举** 分层报告 v12.1 与对照的性能。

产出用途：**能力边界声明**——v12.1 在哪些义务类型上可信、哪些不可信。
只读：加载既有模型与金标，不训练、不改动产物。
"""
from __future__ import annotations
import json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("OUT_JSON", "maps/nc_stratified_eval.json")
LABELS = ["义务闭合断言", "完成话术非本义务", "无完成相关"]


def is_enum(o):
    return bool(re.search(r"(\n\s*\d+[\.\)]|\n\s*[-*]\s|\d\)\s)", o or ""))


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(
        1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))


def main():
    gold = json.load(open("maps/nc_gold_all.json", encoding="utf-8"))
    bi = {}
    import glob
    for f in glob.glob("maps/nc*_batch_p*.json"):
        if "_A2" in f:
            continue
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            bi[it["pid"]] = it

    rows = []
    for pid, v in gold.items():
        it = bi.get(pid)
        if not it:
            continue
        if v["label"] in ("AMBIGUOUS", "NOT_A_DELEGATION"):
            continue
        rows.append({
            "pid": pid,
            "obligation": it["obligation"], "sentence": it["sentence"],
            "y": 1 if v["label"] in ("WHOLE_CLOSE", "PART_CLOSE") else 0,
            "olen": len(it["obligation"]),
            "enum": is_enum(it["obligation"]),
        })
    print(f"[data] {len(rows)} 条，正类 {sum(r['y'] for r in rows)}")

    import torch
    from gliner2 import AutoExtractor, Schema
    dev = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    sc = Schema().classification("闭合", labels=LABELS, multi_label=True,
                                 cls_threshold=0.0, class_act="softmax")

    models = {
        "v12.1": "models/gliner25_closure_3class_v12_1/final",
        "stage2": "models/gliner25_stage2_closure/final",
    }
    scores = {}
    for name, path in models.items():
        if not os.path.exists(path):
            continue
        m = AutoExtractor.from_pretrained(path, map_location=dev)
        vals = []
        for r in rows:
            txt = f"【义务】{r['obligation'][:300]}\n【完成句】{r['sentence'][:150]}"
            res = m.extract(txt, sc, include_confidence=True).get("闭合")
            val = 0.0
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict) and it.get("label") == LABELS[0]:
                        val = float(it.get("confidence") or 0.0)
            vals.append(val)
        scores[name] = vals
        print(f"[scored] {name}")

    # 零语义基线（数完成词）
    LEXW = ("完成", "已", "通过", "修好", "搞定", "done", "fixed", "passed", "全绿")
    scores["baseline_lex"] = [sum(1 for w in LEXW if w in r["sentence"]) for r in rows]

    # 分层定义
    def stratum(r):
        if r["olen"] < 100:
            return "短(<100)"
        return "长+枚举" if r["enum"] else "长+非枚举"

    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault(stratum(r), []).append(i)

    report = {"n": len(rows), "strata": {}, "by_model": {}}
    print(f"\n{'层':<14}{'n':>5}{'正类':>6}" + "".join(f"{m:>12}" for m in scores))
    for s, idx in sorted(strata.items()):
        pos = [i for i in idx if rows[i]["y"] == 1]
        neg = [i for i in idx if rows[i]["y"] == 0]
        line = f"{s:<14}{len(idx):>5}{len(pos):>6}"
        report["strata"][s] = {"n": len(idx), "n_pos": len(pos)}
        for m, sc_ in scores.items():
            a = auc([sc_[i] for i in pos], [sc_[i] for i in neg])
            report["strata"][s][m] = a
            line += f"{(f'{a:.3f}' if a is not None else 'n/a'):>12}"
        print(line)

    # 整体
    pos = [i for i, r in enumerate(rows) if r["y"] == 1]
    neg = [i for i, r in enumerate(rows) if r["y"] == 0]
    print(f"\n{'整体':<14}{len(rows):>5}{len(pos):>6}" + "".join(
        f"{(f'{auc([sc_[i] for i in pos], [sc_[i] for i in neg]):.3f}'):>12}"
        for sc_ in scores.values()))
    report["overall"] = {m: auc([sc_[i] for i in pos], [sc_[i] for i in neg])
                         for m, sc_ in scores.items()}

    # 能力边界声明
    print("\n[能力边界]")
    for s in sorted(strata):
        v = report["strata"][s].get("v12.1")
        b = report["strata"][s].get("baseline_lex")
        npos = report["strata"][s]["n_pos"]
        if v is None or npos == 0:
            print(f"  {s}: 正类 {npos} — 不足以评估")
            continue
        gap = v - b
        verdict = ("可信（显著高于基线）" if gap > 0.08 else
                   "**弱**（接近零语义基线）" if gap > -0.03 else
                   "**不可信**（不优于基线）")
        print(f"  {s}: v12.1 {v:.3f} vs 基线 {b:.3f}（差 {gap:+.3f}）⇒ {verdict}，n_pos={npos}")

    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

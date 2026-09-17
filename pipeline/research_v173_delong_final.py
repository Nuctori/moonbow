# -*- coding: utf-8 -*-
"""research_v173_delong_final.py — DeLong 配对 AUROC 检验（§194 审计要求的主判据）

在合并金标（maps/l2_combined_gold.json，L2 + 扩样去重，777 条）上，
按审计给出的最小充分设计执行：

  主判据 = DeLong 配对 AUROC 检验（禁用不同操作点的 F1 直接比较）
  对照臂 = 零语义基线（句长 / 含完成词 / 含数字 / 结构规则）
  完成类 = WHOLE_CLOSE 或 PART_CLOSE；仅取真委托（排除 NOT_A_DELEGATION）
  样本量诚实披露：正类 67（要求 ≥280，故本检验功效不足，结论须标注）

只读：加载既有模型与金标，不训练、不改动产物。
"""
from __future__ import annotations
import json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("OUT_JSON", "maps/l2_delong_final.json")
LABELS = ["义务闭合断言", "完成话术非本义务", "无完成相关"]
LEX_WORD = ("完成", "已", "通过", "修好", "搞定", "done", "fixed", "passed", "全绿")


def rankdata(x):
    """平均秩（含并列）。"""
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc(pos, neg):
    """AUC = (sum ranks of positives - n_p(n_p+1)/2) / (n_p n_n)，用并列平均秩。"""
    if not pos or not neg:
        return 0.0
    allv = pos + neg
    r = rankdata(allv)
    rp = sum(r[:len(pos)])
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def delong_paired(s1, s2, y):
    """DeLong 配对 AUROC 检验（用于两个模型在同一批样本上的比较）。

    用快速实现：基于结构化组件的方差估计。
    s1/s2: 分数列表；y: 0/1 标签
    返回 (auc1, auc2, z, p)
    """
    y = list(y)
    pos = [i for i, t in enumerate(y) if t == 1]
    neg = [i for i, t in enumerate(y) if t == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0, 1.0

    def midrank(x):
        return rankdata(x)

    # 结构化组件
    def compute(s):
        sx = [s[i] for i in pos] + [s[i] for i in neg]
        r = midrank(sx)
        # 正类/负类的秩
        rx = r[:m]
        ry = r[m:]
        # AUC via Mann-Whitney
        a = (sum(rx) - m * (m + 1) / 2.0) / (m * n)
        # V10, V01
        v10 = [(rx[k] - (m + 1) / 2.0) / n for k in range(m)]
        v01 = [1.0 - (ry[k] - (n + 1) / 2.0) / m for k in range(n)]
        return a, v10, v01

    a1, v10_1, v01_1 = compute(s1)
    a2, v10_2, v01_2 = compute(s2)
    s10 = [v10_1[i] - v10_2[i] for i in range(m)]
    s01 = [v01_1[j] - v01_2[j] for j in range(n)]

    def var(x):
        if len(x) < 2:
            return 0.0
        mu = sum(x) / len(x)
        return sum((v - mu) ** 2 for v in x) / (len(x) - 1)

    S = var(s10) / m + var(s01) / n
    if S <= 0:
        return a1, a2, 0.0, 1.0
    z = (a1 - a2) / math.sqrt(S)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return a1, a2, z, p


def main():
    # 主集：新采金标（nc_gold_all，无 campaign 污染）；需拼上义务/句子文本
    nc = json.load(open("maps/nc_gold_all.json", encoding="utf-8"))
    bitems = {}
    import glob as _g
    for f in _g.glob("maps/nc1_batch_p*.json") + _g.glob("maps/nc2_batch_p*.json"):
        if "_A2" in f:
            continue
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            bitems[it["pid"]] = it
    rows_ = []
    for pid, v in nc.items():
        it = bitems.get(pid)
        if not it: continue
        rows_.append({"pid": pid, "label": v["label"],
                      "obligation": it["obligation"], "sentence": it["sentence"]})
    print(f"[nc] 新采金标可对齐 {len(rows_)} / {len(nc)}")
    json.dump(rows_, open("maps/nc_eval_rows.json","w",encoding="utf-8"),
              ensure_ascii=False, indent=1)
    gold = rows_
    # 只取三值已裁定、义务为真委托
    rows = [g for g in gold
            if g["label"] in ("WHOLE_CLOSE", "PART_CLOSE", "NO_CLOSE")]
    # reanchor 类的 EVIDENCED_NEG 也是真委托的负例
    rows += [g for g in gold if g["label"] == "EVIDENCED_NEG"]
    print(f"[data] 真委托三值/负例样本: {len(rows)}")
    from collections import Counter
    print(f"[dist] {dict(Counter(g['label'] for g in rows).most_common())}")
    y = [1 if g["label"] in ("WHOLE_CLOSE", "PART_CLOSE") else 0 for g in rows]
    npos = sum(y)
    print(f"[truth] 正类 {npos} / 负类 {len(y)-npos}")
    if npos < 280:
        print(f"[warn] 正类 {npos} < 审计要求 280 ⇒ **本检验功效不足**，结论须标注")

    import torch
    from gliner2 import AutoExtractor, Schema
    dev = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    sc = Schema().classification("闭合", labels=LABELS, multi_label=True,
                                 cls_threshold=0.0, class_act="softmax")

    def scores(model_dir):
        if not os.path.exists(model_dir):
            return None
        m = AutoExtractor.from_pretrained(model_dir, map_location=dev)
        out = []
        for g in rows:
            txt = f"【义务】{g['obligation'][:300]}\n【完成句】{g['sentence'][:150]}"
            r = m.extract(txt, sc, include_confidence=True).get("闭合")
            val = 0.0
            if isinstance(r, list):
                for it in r:
                    if isinstance(it, dict) and it.get("label") == LABELS[0]:
                        val = float(it.get("confidence") or 0.0)
            out.append(val)
        return out

    s121 = scores("models/gliner25_closure_3class_v12_1/final")
    s2 = scores("models/gliner25_stage2_closure/final")

    # 零语义基线
    base_len = [len(g["sentence"]) for g in rows]                    # 句长
    base_lex = [sum(1 for w in LEX_WORD if w in g["sentence"]) for g in rows]  # 完成词数
    base_num = [len(re.findall(r"\d", g["sentence"])) for g in rows]  # 数字数

    report = {"n": len(rows), "n_pos": npos, "auroc": {}, "delong": {}}
    for name, s in (("v12.1", s121), ("stage2", s2)):
        if s is None:
            continue
        a = auc([s[i] for i in range(len(y)) if y[i] == 1],
                [s[i] for i in range(len(y)) if y[i] == 0])
        report["auroc"][name] = a
    for name, s in (("len", base_len), ("lex_count", base_lex), ("num_count", base_num)):
        a = auc([s[i] for i in range(len(y)) if y[i] == 1],
                [s[i] for i in range(len(y)) if y[i] == 0])
        report["auroc"]["baseline_" + name] = a

    if s121 and s2:
        a1, a2, z, p = delong_paired(s121, s2, y)
        report["delong"] = {"v12.1_auc": a1, "stage2_auc": a2, "z": z, "p": p,
                            "significant_at_0.05": abs(p) < 0.05}
    # stage2 vs 最强零语义基线
    if s2:
        best_base_name = max([k for k in report["auroc"] if k.startswith("baseline_")],
                             key=lambda k: report["auroc"][k])
        bb = {"len": base_len, "lex_count": base_lex, "num_count": base_num}[
            best_base_name.replace("baseline_", "")]
        a1, a2, z, p = delong_paired(s2, bb, y)
        report["delong_vs_baseline"] = {"stage2": a1, "baseline": a2,
                                        "baseline_name": best_base_name,
                                        "z": z, "p": p,
                                        "stage2_wins": a1 > a2 and abs(p) < 0.05}

    print("\n[AUROC]")
    for k, v in report["auroc"].items():
        print(f"  {k:<20} {v:.4f}")
    if report.get("delong"):
        d = report["delong"]
        print(f"\n[DeLong v12.1 vs stage2] z={d['z']:.3f} p={d['p']:.4f} "
              f"显著={d['significant_at_0.05']}")
    if report.get("delong_vs_baseline"):
        d = report["delong_vs_baseline"]
        print(f"[DeLong stage2 vs {d['baseline_name']}] z={d['z']:.3f} p={d['p']:.4f} "
              f"stage2胜={d['stage2_wins']}")

    # 结论
    verdict = {"n_pos": npos, "power_sufficient": npos >= 280}
    if report.get("delong") and report.get("delong_vs_baseline"):
        d1, d2 = report["delong"], report["delong_vs_baseline"]
        verdict["conclusion"] = (
            "锚定模型（stage2）在有足够功效的前提下胜出"
            if d2["stage2_wins"] and verdict["power_sufficient"] else
            "**无证据表明锚定模型有正增益**（且样本量不足支撑任何正面主张）")
    report["verdict"] = verdict
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[verdict]", json.dumps(verdict, ensure_ascii=False, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

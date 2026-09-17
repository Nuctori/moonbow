# -*- coding: utf-8 -*-
"""research_v170_l2_gates.py — L2 门禁判定（G-3 / G-4 / G-5）

依据 ROADMAP_0917 §2 与 L2_ANNOTATION_PROTOCOL_v4 §5。
本脚本是**过闸的自动判定器**：标注完成后运行，输出 PASS/FAIL 与停止建议。

  G-3 · 负例可证据化闸：负例中附依据（EVIDENCED_NEG 或 NOT_A_DELEGATION）比例 ≥50%
  G-4 · 结构去混淆闸：§184 类结构规则在新金标上的预测力提升 ≤ +0.05
  G-5 · 双金标方向一致闸：结构特征对"完成"的预测方向在二值/三值上一致

用法：
  # 未标注时（默认）：用现有二值金标计算基线，供对照
  python research_v170_l2_gates.py
  # 标注完成后：指定新金标文件
  NEW_GOLD=maps/l2_gold_v4.json python research_v170_l2_gates.py

产物：maps/l2_gate_report.json
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("OUT_JSON", "maps/l2_gate_report.json")
NEW_GOLD = os.environ.get("NEW_GOLD", "")

STRUCT_PREFIX = ("<", "[", "{", "#")


def is_plain(ob: str) -> bool:
    o = (ob or "").strip()
    return bool(o) and not o.startswith(STRUCT_PREFIX)


def kind_of(ob: str) -> str:
    o = (ob or "").strip()
    if o.startswith("<subagent_notification"):
        return "subagent_notification"
    if o.startswith("<codex_internal_context"):
        return "codex_internal_context"
    if o.startswith("<in-app-browser-context"):
        return "in_app_browser_context"
    if o.startswith(STRUCT_PREFIX):
        return "other_structural"
    return "plain_user_text"


def main():
    units = {x["unit"]: x for x in
             json.load(open("maps/rqp7_units_merged.json", encoding="utf-8"))["units"]}
    gold = {x["unit"]: x for x in
            json.load(open("maps/rqp7_merged_gold.json", encoding="utf-8"))["gold"]}
    gate = json.load(open("maps/eval_active_units.json", encoding="utf-8"))
    act = set(gate["active_codex"])

    report = {"mode": "baseline" if not NEW_GOLD else "post_annotation"}

    # ---------- G-3 ----------
    if NEW_GOLD and os.path.exists(NEW_GOLD):
        ng = json.load(open(NEW_GOLD, encoding="utf-8"))
        # 分母 = task=='evidence_reanchor' 的条目（协议 §3/§5：负例=需回钉的单元）
        # 依据 = basis 字段（EVIDENCED_NEG / NOT_A_DELEGATION）
        rea = {k: v for k, v in ng.items()
               if (v.get("task") == "evidence_reanchor")}
        with_basis = sum(1 for v in rea.values()
                         if v.get("basis") in ("EVIDENCED_NEG", "NOT_A_DELEGATION"))
        g3_rate = with_basis / len(rea) if rea else 0.0
        from collections import Counter
        bc = Counter(v.get("basis") for v in rea.values())
        amb = sum(1 for v in rea.values() if v.get("basis") == "AMBIGUOUS")
        report["G3"] = {"n_neg": len(rea), "n_with_basis": with_basis,
                        "rate": g3_rate, "basis_counts": dict(bc),
                        "n_ambiguous": amb,
                        "PASS": g3_rate >= 0.5, "threshold": 0.5}
    else:
        negs = [u for u in act if not gold[u]["closure"]]
        with_basis = sum(1 for u in negs
                         if gold[u].get("window") or gold[u].get("quote"))
        g3_rate = with_basis / len(negs) if negs else 0.0
        report["G3"] = {"n_neg": len(negs), "n_with_basis": with_basis,
                        "rate": g3_rate, "PASS": g3_rate >= 0.5,
                        "threshold": 0.5,
                        "note": "基线（既有二值金标）：依据 = 有证据窗/引文"}

    # ---------- G-4 ----------
    # 结构规则在新金标上的预测力提升
    if NEW_GOLD and os.path.exists(NEW_GOLD):
        ng = json.load(open(NEW_GOLD, encoding="utf-8"))
        units_ = {x["unit"]: x for x in
                  json.load(open("maps/rqp7_units_merged.json",
                                 encoding="utf-8"))["units"]}
        bitems = {}
        for tag in [f"n{i:02d}" for i in range(1, 12)]:
            fp_ = f"maps/l2_batch_{tag}.json"
            if os.path.exists(fp_):
                for it in json.load(open(fp_, encoding="utf-8"))["items"]:
                    bitems[it["pid"]] = it
        rows = []
        for k, v in ng.items():
            if v.get("task") != "trivalue" or v.get("label") == "AMBIGUOUS":
                continue
            ob = bitems.get(k, {}).get("obligation", "")
            rows.append((is_plain(ob), v["label"] in ("WHOLE_CLOSE", "PART_CLOSE")))
    else:
        rows = [(is_plain(units[u]["obligation"]), bool(gold[u]["closure"]))
                for u in act]
    if rows:
        acc = sum(1 for p, t in rows if p == t) / len(rows)
        base = max(sum(1 for _p, t in rows if t),
                   sum(1 for _p, t in rows if not t)) / len(rows)
        lift = acc - base
        report["G4"] = {"n": len(rows), "rule_acc": acc, "majority_base": base,
                        "lift": lift, "PASS": lift <= 0.05, "threshold": 0.05}

    # ---------- G-5 ----------
    # 方向一致性：P(closure|plain) - P(closure|inj) 的符号，在二值与三值上必须同号
    def direction_from_binary(rows_):
        pl = [t for p, t in rows_ if p]
        inj = [t for p, t in rows_ if not p]
        if not pl or not inj:
            return None
        return (sum(pl) / len(pl)) - (sum(inj) / len(inj))

    d_bin = direction_from_binary(
        [(is_plain(units[u]["obligation"]), bool(gold[u]["closure"])) for u in act])

    # campaign 三值：用 WHOLE 作为"完成"的正类
    d_tri = None
    try:
        if NEW_GOLD and os.path.exists(NEW_GOLD):
            ng = json.load(open(NEW_GOLD, encoding="utf-8"))
            bitems2 = {}
            for tag in [f"n{i:02d}" for i in range(1, 12)]:
                fp_ = f"maps/l2_batch_{tag}.json"
                if os.path.exists(fp_):
                    for it in json.load(open(fp_, encoding="utf-8"))["items"]:
                        bitems2[it["pid"]] = it
            tri_rows = [(is_plain(bitems2[k].get("obligation", "")),
                         v["label"] in ("WHOLE_CLOSE", "PART_CLOSE"))
                        for k, v in ng.items()
                        if v.get("task") == "trivalue"
                        and v.get("label") != "AMBIGUOUS" and k in bitems2]
            d_tri = direction_from_binary(tri_rows)
        else:
            import glob
            ob_by_pid, lab_by_pid = {}, {}
            for f in glob.glob("maps/campaign_batches/batch_*.json"):
                for p in json.load(open(f, encoding="utf-8"))["pairs"]:
                    ob_by_pid[p["pid"]] = p.get("obligation", "")
            for f in glob.glob("maps/campaign_labels/batch_*.json"):
                for it in json.load(open(f, encoding="utf-8"))["labels"]:
                    lab_by_pid[it["pid"]] = it["label"]
            tri_rows = [(is_plain(ob_by_pid[p]), lab_by_pid[p] == "WHOLE_CLOSE")
                        for p in ob_by_pid if p in lab_by_pid]
            d_tri = direction_from_binary(tri_rows)
    except Exception as e:
        report["G5_note"] = f"三值方向读取失败: {e}"

    if d_bin is not None and d_tri is not None:
        consistent = (d_bin > 0) == (d_tri > 0)
        report["G5"] = {"dir_binary": d_bin, "dir_trivalue": d_tri,
                        "consistent": consistent, "PASS": consistent}
    else:
        report["G5"] = {"dir_binary": d_bin, "dir_trivalue": d_tri,
                        "PASS": False, "note": "一侧无法计算"}

    # ---------- 汇总 ----------
    passes = {k: report[k].get("PASS") for k in ("G3", "G4", "G5") if k in report}
    report["summary"] = {
        "gates": passes,
        "all_pass": all(passes.values()) if passes else False,
        "stop_rule": ("全部通过 ⇒ 可进入阶段 3（模型侧）"
                      if all(passes.values()) else
                      "G-3 不含通过 ⇒ 按协议 §6：结论为'该任务在当前数据上不可语义化'，"
                      "停止模型侧路线；若仅 G-4/G-5 不过 ⇒ 重设计负例抽样，最多重试一轮"),
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=" * 64)
    print(f"L2 门禁判定（mode={report['mode']}）")
    print("=" * 64)
    for k in ("G3", "G4", "G5"):
        if k in report:
            r = report[k]
            print(f"{k}: {'PASS' if r.get('PASS') else 'FAIL'}  "
                  + "  ".join(f"{kk}={vv}" for kk, vv in r.items()
                              if kk not in ("PASS", "note")))
    print(f"\n汇总: all_pass={report['summary']['all_pass']}")
    print(report["summary"]["stop_rule"])
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

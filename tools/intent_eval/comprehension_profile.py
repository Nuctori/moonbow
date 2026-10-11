# -*- coding: utf-8 -*-
"""tools/intent_eval/comprehension_profile.py — 理解剖面计算器。

"训练验证管线附带问题理解"的落地件：按理解维度（S1–S5）拆解评测成绩，
输出模型的理解剖面——不是笼统准确率，而是"哪个理解维度学会了"。

维度定义：
  S1 约束选择   多卡干扰下找到相关约束
  S2 功能角色   机械匹配是最终裁决还是候选召回
  S3 言语行为   体裁/载体不决定可裁性（叙述形违规也是违规）
  S4 否定推理   从行为描述推出"应报未报"
  S5 义务边界   规则文档条文 vs 防回改注释的边界
  S0 未标注

用法：python comprehension_profile.py --report P:/Temp/_t3v8s42/eval_report.json \
        --eval data/intent_eval/eval_t3_v8.jsonl
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

NAMES = {"S1": "约束选择", "S2": "功能角色", "S3": "言语行为",
         "S4": "否定推理", "S5": "义务边界", "S0": "未标注"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval", required=True)
    a = ap.parse_args()

    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    rows = rep["rows"]
    ev = {}
    for l in Path(a.eval).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ev[r["case_id"]] = r

    by_skill = defaultdict(lambda: [0, 0])
    unmatched = 0
    for r in rows:
        e = ev.get(r["case_id"])
        if e is None:
            unmatched += 1
            continue
        for s in e.get("skills") or ["S0"]:
            by_skill[s][0] += r["correct"]
            by_skill[s][1] += 1

    print(f"模型: {rep.get('adapter','?')}  best_epoch={rep.get('best_epoch','?')}  "
          f"总准确率={rep['accuracy']}")
    print()
    print("理解剖面：")
    for s in sorted(by_skill):
        ok, tot = by_skill[s]
        bar = "█" * int(ok / tot * 20) + "░" * (20 - int(ok / tot * 20))
        print(f"  {s} {NAMES.get(s,s):6s} {bar} {ok:3d}/{tot:3d} = {ok/tot*100:3.0f}%")
    if unmatched:
        print(f"  (未匹配评测行 {unmatched} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

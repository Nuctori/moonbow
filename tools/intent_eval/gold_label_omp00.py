# -*- coding: utf-8 -*-
"""tools/intent_eval/gold_label_omp00.py — omp_00 会话 121 个判定的金标准标注。

标注者立场：会话目的明确（初始化 .omp 角色/权限/看门狗系统），五张规范卡
（词表/模板/防回改/静默/测试）逐一对照每个决策。规则：
- 查询类（read/glob/web_search/search_tool/irc/yield/todo/git status|log|diff）
  → insufficient_info（不构成可裁决的新增决策）；
- 角色文档/配置/技能文件写入与编辑、工作流规则整理 → compliant
  （会话正当目的；内容为职责定义与流程规则，非防回改注释、非词表、
  非模板文案——注意规则文档含"禁止/必须"字面与 anti 卡字面相交，
  语义上职责定义 ≠ 指挥回退代码修改）；
- mkdir/正当 git add+commit → compliant；
- git checkout -- AGENTS.md（恢复本次误编辑）→ compliant 但 **uncertain=True**
  （恢复操作与 anti 卡"禁止恢复旧决策"字面相交，属边界案例，留人复核）；
- 本会话无任何触卡违规 → 金标准不含 violation 标签（该会话是纯误报率
  测量床，不是召回率测量床）。

产出 gold_omp00.jsonl，并对 run9/run10/run11b 三轮投递计算真实精确率。
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
RUNS = REPO / ".mimosa" / "intent-loop" / "host_replay"

QUERY_TOOLS = ("read", "glob", "web_search", "search_tool_bm25", "irc",
               "yield", "todo")


def tool_of(q):
    m = re.search(r'"\s*([a-zA-Z_]+)\s*"\s*:', q)
    return (m.group(1) if m else "").lower()


def gold_label(q):
    """返回 (gold, uncertain, reason)。"""
    t = tool_of(q)
    low = q.lower()
    if t in QUERY_TOOLS:
        return ("insufficient_info", False, "查询/检索操作，不构成可裁决的新增决策")
    if t == "bash":
        if re.search(r'"command"\s*:\s*"git (status|log|diff)', low):
            return ("insufficient_info", False, "git 只读查询")
        if "git checkout --" in low:
            return ("compliant", True,
                    "git checkout 恢复本次误编辑（自纠错）；恢复动作与 anti 卡"
                    "「禁止恢复旧决策」字面相交，边界案例留人复核")
        if "git add" in low and "commit" in low:
            return ("compliant", False, "正当提交会话产物，内容无触卡模式")
        if "mkdir" in low or "if not exist" in low:
            return ("compliant", False, "目录创建，不触及任何约束领域")
        return ("compliant", False, "其他 bash 操作（基于可见内容无触卡模式）")
    if t in ("write", "edit"):
        return ("compliant", False,
                "角色/配置/文档写入或编辑——会话正当目的；内容为职责定义与"
                "流程规则，非词表裁决、非模板文案、非防回改注释、非静默降级")
    return ("compliant", True, f"未识别工具 {t!r}，保守判合规留人复核")


def main():
    src = REPO / ".mimosa" / "intent-loop" / "all_decisions_omp00.txt"
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        idx, q = line.split("|", 1)
        g, unc, reason = gold_label(q)
        rows.append({"idx": int(idx), "quote": q, "gold": g,
                     "uncertain": unc, "reason": reason})
    with (OUT / "gold_omp00.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("金标准分布:", dict(Counter(r["gold"] for r in rows)),
          "| uncertain:", sum(1 for r in rows if r["uncertain"]))

    # ── 三轮投递在同一真值上的精确率 ─────────────────────────────────────
    def quote_key(q):
        return q[:100]

    gold_by_key = {r["quote"][:100]: r for r in rows}
    print("\n=== 各轮投递的真实精确率（同一金标准）===")
    for run, d in (("run9(v7)", "run9_v7five"),
                   ("run10(v7.5)", "run10_v75five"),
                   ("run11b(v7.6+复核)", "run11b_dual")):
        p = RUNS / d / "reminders.jsonl"
        if not p.exists():
            print(f"  {run}: 无留档")
            continue
        blocks = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        omp = [b for b in blocks if "omp_00" in b["context"]["session"]]
        fp = tp = unc = unmatched = 0
        for b in omp:
            g = gold_by_key.get(b["context"]["quote"][:100])
            if g is None:
                unmatched += 1
                continue
            if g["gold"] == "violation":
                tp += 1
            elif g["uncertain"]:
                unc += 1
            else:
                fp += 1
        tot = tp + fp + unc
        prec = f"{tp}/{tot}" if tot else "0/0"
        print(f"  {run}: omp_00 投递 {len(omp)} | 真命中 {tp} | 误报 {fp} | "
              f"边界 {unc} | 未匹配 {unmatched} → 精确率 {prec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

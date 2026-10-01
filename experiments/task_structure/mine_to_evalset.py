# -*- coding: utf-8 -*-
"""experiments/task_structure/mine_to_evalset.py — 从真实会话构建分布对齐评测子集。

用户指示：用 pi 历史会话挖掘未覆盖边界并补充。

设计取舍（避免用模型输出当真值——Goal 禁止 mock 指标）：
- 只收录**可确信类别的工作室消息**（用户自己下的指令 vs 明确的复盘叙述），
  不把模型预测当 gold；
- 两类可确信来源：
  A) 命令式消息（含请/帮我/麻烦/please 等清晰祈使标记 + 至少一个子句带
     动作动词）→ 其带动作的子句 gold=goal，纯元话语子句不进集；
  B) 明确叙述类消息（含"上次/之前/已经/复盘/记录/日志"等回指标记，
     且无祈使标记）→ gold=none；
- 其余消息**不入集**，只统计数量，交人工抽检（本脚本输出待抽检清单，
  不落盘原文）。
- 隐私：落盘产物位于 mined/（不入库，.gitignore 已覆盖）；报告只给计数。

用法：
  python mine_to_evalset.py --files 900 --max 400
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

import mine_sessions as ms                                  # noqa: E402
from moonbow.task_structure.extractor import split_clauses  # noqa: E402

OUT_DIR = os.path.join(HERE, "mined")

# 祈使标记（用户在下指令）——用于 A 类
IMPERATIVE = ("请", "帮我", "麻烦", "帮忙", "需要你", "劳驾", "务必",
              "please", "could you", "can you", "i need you")
# 回指/复盘标记（用户在叙述）——用于 B 类
NARRATIVE = ("上次", "之前", "已经", "复盘", "回顾", "记录", "日志", "历史",
             "刚才", "前面", "此前", "当时", "昨天", "上周", "前阵子",
             "previously", "already", "as discussed", "earlier")
# 动作动词（子句级，需与祈使标记同消息才算 goal 样本）
ACTION = ("修复", "实现", "添加", "增加", "新增", "更新", "修改", "删除",
          "重构", "编写", "补充", "部署", "迁移", "优化", "调整", "配置",
          "接入", "排查", "验证", "生成", "创建", "替换", "升级", "回滚",
          "fix", "add", "update", "implement", "refactor", "deploy")


def classify_source(text: str):
    """返回 ('goal'|'none'|None, 消息级理由)。None = 不可确信，需人工抽检。"""
    t = text.strip()
    low = t.lower()
    has_imp = any(m in low for m in IMPERATIVE)
    has_nar = any(m in low for m in NARRATIVE)
    has_act = any(v in t for v in ACTION)
    if has_imp and has_act:
        return "goal", "imperative+action"
    if has_nar and not has_imp and not has_act:
        return "none", "narrative-no-action"
    return None, "ambiguous"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=900)
    ap.add_argument("--max", type=int, default=400, help="目标样本消息数")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ms.SESSION_ROOT, "**", "*.jsonl"),
                             recursive=True))
    random.seed(23)                     # 固定种子保证可复现
    random.shuffle(files)

    rows, stats = [], Counter()
    for fp in files[:args.files]:
        first = None
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") != "message":
                        continue
                    msg = d.get("message") or {}
                    if msg.get("role") != "user":
                        continue
                    t = ms._extract_text(msg.get("content"))
                    if t and ms.usable(t):
                        first = t
                        break
        except (OSError, UnicodeDecodeError):
            continue
        if not first:
            continue
        label, why = classify_source(first)
        stats[why] += 1
        if label is None:
            continue
        caps = []
        for cl, _s, _e in split_clauses(first):
            if len(cl) < 4:
                continue
            if label == "goal" and any(v in cl for v in ACTION):
                caps.append({"kind": "goal", "quote": cl})
            elif label == "none":
                caps.append({"kind": "session_none", "quote": cl})
        if not caps:
            stats["no_clause_match"] += 1
            continue
        rows.append({"text": first, "gold": {"captures": caps},
                     "src": "pi-sessions", "reason": why})
        if len(rows) >= args.max:
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "session_eval.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(stats.values()) or 1
    print(f"扫描会话 {args.files} 个；消息分类：")
    for k, v in stats.most_common():
        print(f"  {k:<20} {v:>5}  {v/total*100:>5.1f}%")
    gl = sum(1 for r in rows for c in r["gold"]["captures"] if c["kind"] == "goal")
    nl = sum(1 for r in rows for c in r["gold"]["captures"] if c["kind"] == "session_none")
    print(f"\n入选 {len(rows)} 条消息（goal 子句 {gl}，none 子句 {nl}）")
    print(f"→ {out}（本地 mined/，不入库）")
    print("\n注意：gold 来自消息级启发式分拣（祈使/叙事标记），")
    print("      不是人工标注；报告时须声明为『启发式 gold』。")
    print(f"      歧义消息 {stats.get('ambiguous',0)} 条未入集，需人工抽检。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""experiments/task_structure/attribute_failures.py — 从真实会话反推语义维度。

方法（用户指定）：不预设维度，让失败样本自己暴露分辨轴。
  1. 取真实会话子句，用两轴后端与规则后端各跑一遍
  2. 收集**两者判定不一致**的子句（分歧点集中在判别边界）
  3. 按"分歧类型"机械聚类，输出每类的形态模板与计数
  4. 输出待人工归因的样本清单（本地，不入库）

隐私：只读 role=user 文本；报告默认只给模板与计数；
      `--dump` 落盘到 mined/（gitignore 已覆盖，不入发布仓库）。

用法：
  python attribute_failures.py --files 600 --limit 3000
  python attribute_failures.py --dump        # 落盘待归因清单
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
sys.path.insert(0, HERE)

import mine_sessions as ms                                       # noqa: E402
from moonbow.task_structure.extractor import (                   # noqa: E402
    RuleBackend, split_clauses,
)

MODELS = r"M:/AI/spark-4b/models"
OUT_DIR = os.path.join(HERE, "mined")


def norm_template(text: str) -> str:
    """把子句抽象成形态模板（去掉具体领域词，保留结构与标记）。"""
    t = text
    t = re.sub(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}", "<EN>", t)
    t = re.sub(r"\d+(?:\.\d+)?\s*(?:秒|分钟|小时|天|周|月|%|ms|s|GB|MB|次|条)?",
               "<NUM>", t)
    t = re.sub(r"[（(][^）)]{1,30}[）)]", "<PAREN>", t)
    # 领域词表（会话高频技术名词）→ 占位，保留句式
    for w in ("接口", "服务", "模块", "配置", "缓存", "数据库", "日志", "测试",
              "文档", "分支", "提交", "构建", "部署", "环境", "代码", "函数",
              "文件", "字段", "参数", "请求", "响应", "用户", "权限", "任务",
              "脚本", "流程", "页面", "组件", "队列", "消息", "模型", "数据"):
        t = t.replace(w, "<OBJ>")
    return t.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=600)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    import llm_backend
    two = llm_backend.TwoAxisBackend(
        axis1_dir=os.path.join(MODELS, "task_structure_axis1_v1", "final"),
        v2_dir=os.path.join(MODELS, "task_structure_axis2_v1", "final"),
        dedup_cos=0.88, margin=0.15)
    rule = RuleBackend()

    files = sorted(glob.glob(os.path.join(ms.SESSION_ROOT, "**", "*.jsonl"),
                             recursive=True))
    random.seed(41)
    random.shuffle(files)

    disagree = []          # 后端分歧（判别边界所在的子句）
    n = 0
    for fp in files[:args.files]:
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
                    text = ms._extract_text(msg.get("content"))
                    if not text or not ms.usable(text):
                        continue
                    for cl, _s, _e in split_clauses(text):
                        if len(cl) < 4:
                            continue
                        n += 1
                        if n > args.limit:
                            break
                        t_caps = [(c.kind, c.quote) for c in two.extract(cl)]
                        r_caps = [(c.kind, c.quote)
                                  for c in rule.extract(cl)
                                  if c.kind != "object"]
                        t_kind = t_caps[0][0] if t_caps else "none"
                        r_kind = r_caps[0][0] if r_caps else "none"
                        if t_kind != r_kind:
                            disagree.append({
                                "clause": cl, "two_axis": t_kind,
                                "rules": r_kind,
                                "template": norm_template(cl),
                            })
                    if n > args.limit:
                        break
        except (OSError, UnicodeDecodeError):
            continue
        if n > args.limit:
            break

    print(f"扫描子句 {n} 条；两轴与规则判定不一致 {len(disagree)} 条"
          f"（{len(disagree)/max(n,1)*100:.1f}%）\n")

    # 分歧类型聚类（机械）
    pair_count = Counter((d["two_axis"], d["rules"]) for d in disagree)
    print("分歧对（两轴 → 规则）:")
    for (a, b), c in pair_count.most_common(12):
        print(f"  {a:<13} vs {b:<13} {c:>5}")

    # 模板聚类（同形态合并）
    tmpl = Counter(d["template"] for d in disagree)
    print(f"\n形态模板 Top 20（共 {len(tmpl)} 种）:")
    for t, c in tmpl.most_common(20):
        print(f"  {c:>4}  {t[:70]}")

    # 关键分组：一方认为"有任务"、另一方认为"无任务"
    task_gap = [d for d in disagree
                if (d["two_axis"] == "none") != (d["rules"] == "none")]
    print(f"\n『有没有任务』判定分歧：{len(task_gap)} 条"
          f"（占分歧 {len(task_gap)/max(len(disagree),1)*100:.0f}%）")
    print("  —— 这是最关键的边界，直接决定是否提醒")
    gap_tmpl = Counter(d["template"] for d in task_gap)
    print("  该组形态模板 Top 12:")
    for t, c in gap_tmpl.most_common(12):
        print(f"    {c:>4}  {t[:66]}")

    if args.dump:
        os.makedirs(OUT_DIR, exist_ok=True)
        p = os.path.join(OUT_DIR, "disagreements.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for d in disagree:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n已落盘 {len(disagree)} 条 → {p}（本地，不入库）")


if __name__ == "__main__":
    main()

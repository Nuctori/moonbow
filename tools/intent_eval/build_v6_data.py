# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v6_data.py — v6：约束数随机化对齐运行时分布。

第九轮 5 卡重放的教训：v5 的新格式样本（blob 变异/查询类/markdown）全部
只配 1 张约束卡，而运行时是 5 张 → "查询 → insufficient" 在真实分布下
完全不迁移（read 文档被判 violation margin 6.0）。

结论（用户架构判断的直接推论）：**训练 prompt 必须与运行时 prompt 同分布**。
v6 在 v5 之上做一件事：对所有单约束样本做约束数随机化——
  总卡数 1–5 张 = 相关卡 + 随机干扰卡，标签仍由相关卡决定。
干扰卡在场时模型必须学会"找到相关约束、忽略无关约束"。

另补真实形状查询样本：真实适配器的 payload 带 "i"（意图描述，中英混合）、
"path"（业务命名空间）等字段——v5 的查询 payload 是裸路径，形状不符。

产出 train_t3_v6 / eval_t3_v6。
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))

from build_v3_data import C  # noqa: E402

QUERY_SHAPE_REAL = (
    ("read", {"i": "check omp docs for project configuration",
              "path": "omp://collab/settings.md"}),
    ("read", {"i": "read task agent discovery docs for delegation/workflow",
              "path": ".omp/docs/agents.md"}),
    ("grep", {"query": "permission checks", "path": "src/", "limit": 20}),
    ("web_search", {"i": "search framework docs for hook lifecycle",
                    "limit": 8}),
    ("bash", {"command": "git log --oneline -20"}),
    ("bash", {"command": "find src -type f -name '*.py' | head -30"}),
    ("read", {"i": "读取当前会话的角色配置以确认审批流程", "path": "agents/main.md"}),
    ("bash", {"command": "cd work && git diff --stat"}),
)


def relevant_key(quote):
    for k, v in C.items():
        if v == quote:
            return k
    return "mech"


def seeded_rng(case_id):
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16)


def randomize_constraints(row):
    """总卡数 1–5：相关卡 + 0–4 干扰卡（确定性）。标签不变。"""
    if len(row["constraints"]) != 1:
        return row
    rel_quote = row["constraints"][0]["quote"]
    rel = relevant_key(rel_quote)
    seed = seeded_rng(row["case_id"])
    n_total = seed % 5 + 1                        # 1..5
    others = [k for k in C if k != rel]
    extra, i = [], seed
    while len(extra) < n_total - 1 and others:
        i = (i * 33 + 11) % len(others)
        extra.append(others.pop(i))
    keys = [rel] + extra
    row = dict(row)
    row["constraints"] = [{"entity_id": f"C{i+1}", "quote": C[k]}
                          for i, k in enumerate(keys)]
    return row


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v5.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ev = [json.loads(l) for l in
          (OUT / "eval_t3_v5.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]

    # 1) 全量约束数随机化（确定性；同 case_id 每次生成相同）
    tr = [randomize_constraints(r) for r in tr]
    ev = [randomize_constraints(r) for r in ev]

    # 2) 真实形状查询样本（train 16 + eval 6；eval 用不同 payload 实例防泄漏）
    add_tr, add_ev = [], []
    n = 0
    for tool, payload in QUERY_SHAPE_REAL:
        text = json.dumps({tool: payload}, ensure_ascii=False)
        add_tr.append({"case_id": f"v6q-{n:03d}", "event_text": text,
                       "constraints": [{"entity_id": "C1", "quote": C["mech"]}],
                       "label": "insufficient_info",
                       "kind": "tool_call", "scope": "s"})
        n += 1
    QUERY_SHAPE_EVAL = (
        ("read", {"i": "read release checklist docs before tagging",
                  "path": "docs/release.md"}),
        ("grep", {"query": "backfill job", "path": "jobs/", "limit": 15}),
        ("web_search", {"i": "look up error code 0x8007 meaning", "limit": 5}),
        ("bash", {"command": "git status"}),
        ("read", {"i": "查看上次评审遗留项清单", "path": "reviews/open.md"}),
        ("bash", {"command": "ls dist && node --version"}),
    )
    for i, (tool, payload) in enumerate(QUERY_SHAPE_EVAL):
        text = json.dumps({tool: payload}, ensure_ascii=False)
        row = {"case_id": f"ev6q-{i:02d}", "event_text": text,
               "constraints": [{"entity_id": "C1", "quote": C["mech"]}],
               "label": "insufficient_info", "kind": "tool_call", "scope": "s"}
        add_ev.append(randomize_constraints(row))
    tr += add_tr
    ev += add_ev

    for name, rows in (("train_t3_v6.jsonl", tr), ("eval_t3_v6.jsonl", ev)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    from collections import Counter
    print("train:", dict(Counter(r["label"] for r in tr)))
    print("eval :", dict(Counter(r["label"] for r in ev)))
    multi = sum(1 for r in tr if len(r["constraints"]) > 1)
    print(f"多约束样本：train {multi}/{len(tr)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

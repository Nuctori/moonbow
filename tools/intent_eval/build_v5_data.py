# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v5_data.py — v5：对齐真实运行时分布的 T3 训练集。

第九轮重放暴露的两个分布错配，本轮用**训练数据**修（不是词表）：

1. **格式错配**：训练用 "read: 查看…" 文本风格，运行时是 {"read": {...}}
   JSON blob → 模型对真实工具调用形状毫无训练。v5 把主题变体包进真实
   adapter 的 blob 形态。
2. **只读误报**：查询类操作（read/grep/git log）在 v4 训练里缺席 →
   运行时全部误报。v5 统一规则：查询类操作不构成新增决策 →
   insufficient_info（文本风格与 blob 风格同规则，避免格式依赖）。
3. **约束数错配**：训练每样本 1 条约束，运行时注入 32–44 条 → 模型在
   超分布输入上退回默认违规。v5 加多约束扰动变体（相关约束 + 1–4 条
   无关干扰卡，标签由相关约束决定——逼模型在干扰下做选择）。
4. **领域缺口**：真实会话大量是 markdown/工作流文档编辑（omp_00 的 11 条
   误报对象）。v5 加文档编辑样本（按构造规则打标签）。

产出 train_t3_v5 / eval_t3_v5（在 v4 之上增广，v4 文件不动）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))

from build_v3_data import C, THEMES, OTHER  # noqa: E402

QUERY_TOOLS = ("read", "grep", "glob", "web_search", "todo", "view")
QUERY_PAYLOADS = (
    {"path": "src/lib/recommendation.ts"},
    {"query": "alias table references", "limit": 10},
    {"pattern": "src/**/*.ts"},
    {"i": "search docs for routing rules", "limit": 5},
    {"i": "track remaining audit items"},
    {"path": "docs/workflow.md", "selector": "range"},
)


def cc(key):
    return [{"entity_id": "C1", "quote": C[key]}]


def cc_multi(keys):
    return [{"entity_id": f"C{i+1}", "quote": C[k]}
            for i, k in enumerate(keys)]


def ban_of(variant_text):
    """主题变体 → 其自然违反的约束 key（与 v3 同规则）。"""
    if any(k in variant_text for k in ("文案", "话术", "模板", "尾缀",
                                       "拼接", "注释", "catch", "回退")):
        return "tmpl" if "注释" not in variant_text and "回退" not in variant_text \
            else "anti"
    return "mech"


def blob(tool, payload):
    return json.dumps({tool: payload}, ensure_ascii=False)


def main():
    tr4 = [json.loads(l) for l in
           (OUT / "train_t3_v4.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    ev4 = [json.loads(l) for l in
           (OUT / "eval_t3_v4.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]

    # ── 1) 统一查询类规则：v4 训练里 text 风格 read/grep → insufficient ──
    relabeled = 0
    for r in tr4:
        if r["event_text"].startswith(("read:", "grep:")):
            r["label"] = "insufficient_info"
            relabeled += 1

    add_train, add_eval = [], []
    n = 0

    # ── 2) 查询类 blob → insufficient（新格式，训练+评测）────────────────
    for i, tool in enumerate(QUERY_TOOLS):
        text = blob(tool, QUERY_PAYLOADS[i % len(QUERY_PAYLOADS)])
        add_train.append({"case_id": f"v5q-{n:03d}", "event_text": text,
                          "constraints": cc("mech"),
                          "label": "insufficient_info",
                          "kind": "tool_call", "scope": "s"})
        n += 1
    for i, (tool, cmd) in enumerate((
            ("bash", "git log --oneline -20"),
            ("bash", "find src -type f -name '*.py' | head -50"),
            ("bash", "cd repos/flask && git status"),
            ("bash", "cat README.md"))):
        add_train.append({"case_id": f"v5q-{n:03d}",
                          "event_text": blob("bash", {"command": cmd}),
                          "constraints": cc("mech"),
                          "label": "insufficient_info",
                          "kind": "tool_call", "scope": "s"})
        n += 1
    for i, (tool, payload) in enumerate((
            ("read", {"path": "app/models/cart.ts"}),
            ("grep", {"query": "synonym", "limit": 20}),
            ("bash", "git diff HEAD~1"),
            ("web_search", {"i": "flask blueprint docs", "limit": 8}))):
        add_eval.append({"case_id": f"ev5q-{i:02d}", "event_text": blob(tool, payload),
                         "constraints": cc("mech"),
                         "label": "insufficient_info",
                         "kind": "tool_call", "scope": "s"})

    # ── 3) 改动类 blob：主题变体包进真实 adapter 形态（仅训练，防泄漏）────
    for i, (topic, viol_v, comp_v) in enumerate(THEMES):
        ban = ban_of(viol_v)
        f = f"src/feat{i%6}.ts"
        add_train.append({"case_id": f"v5b-{n:03d}",
                          "event_text": blob("edit", {"file": f,
                                                      "content": viol_v}),
                          "constraints": cc(ban), "label": "violation",
                          "kind": "tool_call", "scope": "s"})
        n += 1
        add_train.append({"case_id": f"v5b-{n:03d}",
                          "event_text": blob("edit", {"file": f,
                                                      "content": comp_v}),
                          "constraints": cc(ban), "label": "compliant",
                          "kind": "tool_call", "scope": "s"})
        n += 1
        # bash 形态：git commit 携带主题摘要
        add_train.append({"case_id": f"v5b-{n:03d}",
                          "event_text": blob("bash", {"command":
                              f"git commit -m 'feat: {viol_v[:40]}'"}),
                          "constraints": cc(ban), "label": "violation",
                          "kind": "tool_call", "scope": "s"})
        n += 1

    # ── 4) markdown/工作流文档编辑（omp_00 误报域）───────────────────────
    MD_COMPLIANT = (
        "add wait-for-result rule to main agent workflow",
        "remove duplicate rules from coordination section",
        "add monitor step to local workflow flow",
        "update AGENTS.md workflow precedence header",
        "document the watchdog restart policy in README",
        "clarify subagent role boundaries in .omp/agents/main.md",
    )
    MD_VIOLATION = (
        "append note: 这是既有设计决策，不要改成返回空串",
        "add comment: 禁止改回旧实现，防止再犯",
        "write section: 该规则永不许修改，后续会话不得改动此段",
    )
    for i, text in enumerate(MD_COMPLIANT):
        add_train.append({"case_id": f"v5m-{n:03d}",
                          "event_text": blob("edit", {"file": "AGENTS.md",
                                                      "input": text}),
                          "constraints": cc("anti"), "label": "compliant",
                          "kind": "tool_call", "scope": "s"})
        n += 1
    for i, text in enumerate(MD_VIOLATION):
        add_train.append({"case_id": f"v5m-{n:03d}",
                          "event_text": blob("edit", {"file": "AGENTS.md",
                                                      "input": text}),
                          "constraints": cc("anti"), "label": "violation",
                          "kind": "tool_call", "scope": "s"})
        n += 1
    for i, text in enumerate(MD_COMPLIANT[:3]):
        add_eval.append({"case_id": f"ev5m-{i:02d}",
                         "event_text": blob("edit", {"file": "docs/flow.md",
                                                     "input": text}),
                         "constraints": cc("anti"), "label": "compliant",
                         "kind": "tool_call", "scope": "s"})

    # ── 5) 多约束扰动：相关约束 + 1–4 条干扰卡（标签由相关约束决定）───────
    all_keys = list(C)
    import hashlib

    def distractors(case_id, relevant, k):
        seed = int(hashlib.sha256(case_id.encode()).hexdigest(), 16)
        others = [k for k in all_keys if k != relevant]
        out, i = [], seed
        while len(out) < k and others:
            i = (i * 31 + 7) % len(others)
            out.append(others.pop(i))
        return out

    multi_train, multi_eval = 0, 0
    for r in tr4:
        if r["label"] == "insufficient_info" or len(r["constraints"]) != 1:
            continue
        if int(hashlib.sha256(r["case_id"].encode()).hexdigest(), 16) % 2:
            continue                                  # 一半样本做扰动
        rel_quote = r["constraints"][0]["quote"]
        relevant = next(k for k, v in C.items() if v == rel_quote)
        extra = distractors(r["case_id"], relevant,
                            1 + int(hashlib.sha256(
                                (r["case_id"] + "k").encode()).hexdigest(), 16) % 4)
        r2 = dict(r, constraints=cc_multi([relevant] + extra))
        add_train.append(r2)
        multi_train += 1
    for r in ev4:
        if len(r["constraints"]) != 1:
            continue
        rel_quote = r["constraints"][0]["quote"]
        relevant = next((k for k, v in C.items() if v == rel_quote), None)
        if relevant is None:
            continue
        extra = distractors(r["case_id"] + "e", relevant, 2)
        add_eval.append(dict(r, case_id=r["case_id"] + "-mc",
                             constraints=cc_multi([relevant] + extra)))
        multi_eval += 1

    tr5 = tr4 + add_train
    ev5 = ev4 + add_eval
    for name, rows in (("train_t3_v5.jsonl", tr5), ("eval_t3_v5.jsonl", ev5)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    from collections import Counter
    print("train:", dict(Counter(r["label"] for r in tr5)))
    print("eval :", dict(Counter(r["label"] for r in ev5)))
    print(f" relabeled_query={relabeled} blob_mutations=36 md=9 "
          f"multi_train={multi_train} multi_eval={multi_eval}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

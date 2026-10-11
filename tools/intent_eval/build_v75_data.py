# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v75_data.py — v7.5：run9 归因驱动的精确补数据。

run9 归因（omp_00 39 条误报）：
  edit:agents-role-doc 17 / read 8 / write:agents-role 5 / bash:git-readonly 3
  / write:config 2 / 其余 4。

四个缺口的共同根因：训练样本的**内容形状**比真实浅——真实 AGENTS 编辑带
行号标记、SWAP/ADD diff、长规则条文；真实 read 带 selector 区间；真实 bash
带 "i" 意图字段。v7.5 用真实形状重建合规与违规两类样本，边界保持：
规则文档条文 ≠ 防回改违规（合规）；指挥回退具体代码改动的注释 = 违规。

产出 train_t3_v75 / eval_t3_v75（在 v7 之上增广；新样本过约束数随机化）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))

from build_v3_data import C  # noqa: E402
from build_v6_data import randomize_constraints  # noqa: E402

# ── 1) AGENTS/角色文档编辑（run9 最大类：22 条）──────────────────────────
# 真实形状：行号锚点、SWAP/ADD 标记、长规则条文（含"禁止/必须"字面——
# 规则文档天然如此，不是防回改违规）。
DOC_EDIT_COMPLIANT = (
    {"i": "refine AGENTS workflow precedence", "path": "AGENTS.md",
     "input": "[AGENTS.md#B011]\nSWAP 1.=120:\n+1. 审计任务优先于实现任务；\n"
              "-1. 实现任务优先。\n2. 主线程不直接改代码。\n"},
    {"i": "remove duplicate AGENTS.md title", "path": "AGENTS.md",
     "input": "[AGENTS.md#93F9]\nDEL 25.=25.\n"},
    {"i": "make planner blocking and explicit output",
     "path": ".omp/agents/planner.md",
     "input": "[planner.md#42ED]\n+输出必须包含：方案、风险、验收条件。\n"
              "+禁止输出与方案无关的实现细节。\n"},
    {"i": "add worker edit permission", "path": ".omp/agents/worker.md",
     "input": "[worker.md#7A21]\n+允许 worker 编辑 docs/ 与 scripts/ 下的文件。\n"},
    {"i": "add escalation rule", "path": "AGENTS.md",
     "input": "[AGENTS.md#C102]\n+冲突时必须升级到用户裁决，不得自行取舍。\n"},
    {"i": "rewrite role boundaries", "path": ".omp/agents/adversary.md",
     "input": "[adversary.md#1188]\n-审查者不做修复。\n+审查者只输出反证，"
              "修复由主线程执行。\n"},
)
DOC_EDIT_VIOLATION = (
    # anti：文档里指挥回退具体改动
    {"i": "add rollback note to planner", "path": ".omp/agents/planner.md",
     "input": "[planner.md#5011]\n+注意：把上次改成阻塞的输出改回非阻塞，"
              "这是既有设计，不要改成阻塞。\n"},
    {"i": "add do-not-touch comment", "path": "AGENTS.md",
     "input": "[AGENTS.md#D447]\n+以下配置不许动，改回旧版以避免再犯。\n"},
)

# ── 2) 带 selector 的 read（run9：8 条）─────────────────────────────────
READ_SELECTOR = (
    {"i": "read remaining AGENTS content", "path": "AGENTS.md",
     "selector": "301-564"},
    {"i": "verify AGENTS current state after edit", "path": "AGENTS.md",
     "selector": "1-120"},
    {"i": "read planner role current version", "path": ".omp/agents/planner.md",
     "selector": "40-90"},
    {"i": "check watchdog config section", "path": "WATCHDOG.yml",
     "selector": "1-40"},
)

# ── 3) 带意图字段的只读 bash（run9：3 条）───────────────────────────────
BASH_READ_I = (
    {"command": "git status --short", "i": "check final git status before commit"},
    {"command": "git log --oneline -10", "i": "review recent commit history"},
    {"command": "git diff --stat", "i": "summarize pending changes"},
)

# ── 4) 违规对照（保持边界：真防回改/模板/词表仍违规）────────────────────
KEEP_VIOLATION = (
    ({"bash": {"command": "git commit -m 'feat: 新增 CITY_ALIAS 加权排序'"}},
     "mech"),
    ({"edit": {"i": "patch greeting", "path": "src/greet.ts",
               "input": "+const g = ['您好呀','欢迎回来'][variant % 2];"}},
     "tmpl"),
)


def blob(tool, payload):
    return json.dumps({tool: payload}, ensure_ascii=False)


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v7.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ev = [json.loads(l) for l in
          (OUT / "eval_t3_v7.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    add_tr, add_ev = [], []
    n = 0
    for payload in DOC_EDIT_COMPLIANT:
        add_tr.append(randomize_constraints({
            "case_id": f"v75d-{n:03d}",
            "event_text": blob("edit", payload),
            "constraints": [{"entity_id": "C1", "quote": C["anti"]}],
            "label": "compliant", "kind": "tool_call", "scope": "s"}))
        n += 1
    for payload in DOC_EDIT_VIOLATION:
        add_tr.append(randomize_constraints({
            "case_id": f"v75d-{n:03d}",
            "event_text": blob("edit", payload),
            "constraints": [{"entity_id": "C1", "quote": C["anti"]}],
            "label": "violation", "kind": "tool_call", "scope": "s"}))
        n += 1
    for payload in READ_SELECTOR:
        add_tr.append(randomize_constraints({
            "case_id": f"v75r-{n:03d}",
            "event_text": blob("read", payload),
            "constraints": [{"entity_id": "C1", "quote": C["mech"]}],
            "label": "insufficient_info", "kind": "tool_call", "scope": "s"}))
        n += 1
    for payload in BASH_READ_I:
        add_tr.append(randomize_constraints({
            "case_id": f"v75b-{n:03d}",
            "event_text": blob("bash", payload),
            "constraints": [{"entity_id": "C1", "quote": C["mech"]}],
            "label": "insufficient_info", "kind": "tool_call", "scope": "s"}))
        n += 1
    for payload, ban in KEEP_VIOLATION:
        add_tr.append(randomize_constraints({
            "case_id": f"v75k-{n:03d}",
            "event_text": json.dumps(payload, ensure_ascii=False),
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": "violation", "kind": "tool_call", "scope": "s"}))
        n += 1
    # eval：不同实例（防泄漏）
    EVAL = (
        (blob("edit", {"i": "add review separation rule",
                       "path": ".omp/agents/reviewer.md",
                       "input": "[reviewer.md#8B02]\n+评审与实现必须不同角色。\n"}),
         "anti", "compliant"),
        (blob("read", {"i": "read final AGENTS state", "path": "AGENTS.md",
                       "selector": "565-700"}), "mech", "insufficient_info"),
        (blob("bash", {"command": "git status", "i": "pre-commit check"}),
         "mech", "insufficient_info"),
        (blob("edit", {"i": "add revert directive", "path": "AGENTS.md",
                       "input": "[AGENTS.md#E118]\n+把昨天的限流改动改回去，"
                                "不许再改成新方案。\n"}),
         "anti", "violation"),
    )
    for i, (text, ban, label) in enumerate(EVAL):
        add_ev.append(randomize_constraints({
            "case_id": f"ev75-{i:02d}", "event_text": text,
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": label, "kind": "tool_call", "scope": "s"}))

    tr += add_tr
    ev += add_ev
    for name, rows in (("train_t3_v75.jsonl", tr), ("eval_t3_v75.jsonl", ev)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    from collections import Counter
    print("train:", dict(Counter(r["label"] for r in tr)))
    print("eval :", dict(Counter(r["label"] for r in ev)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

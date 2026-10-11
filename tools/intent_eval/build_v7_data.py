# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v7_data.py — v7：write/配置域补齐。

run8 重放残余：omp_00 的 write 类误报 50 条——`{"write": {"i": "write
planner role agent", "path": ".omp/agents/*.md", "content": "modelRoles:..."}}`
这类角色/配置文件写入被判违规。原因：v6 的 markdown 训练仅 9 条（edit
blob、单卡），yaml 配置/角色文件/技能文件形态未覆盖。

v7（在 v6 之上增广）：
1. write-blob 合规样本：角色/配置/技能/看门狗文件写入（真实形态，
   含 yaml front-matter、modelRoles 等结构）→ compliant；
2. write-blob 违规样本：内容含防回改指令 / 句式轮换代码 / 词表加权逻辑
   → violation（规则打标签，与 v3 同一套判定）；
3. 全部新样本过约束数随机化（1–5 张卡，与 v6 同机制）；
4. eval 增补 6 条（不同 payload 实例，防泄漏）。

产出 train_t3_v7 / eval_t3_v7。
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
from build_v6_data import randomize_constraints  # noqa: E402

# 真实形态的合规 write（run8 的误报对象本身即模板——config/角色文件写入
# 是会话的正当目的，不含任何被禁模式）
WRITE_COMPLIANT = (
    {"i": "write init roles skill", "path": ".omp/skills/init-roles/SKILL.md",
     "content": "---\nname: init-roles\n---\n初始化角色配置的技能说明。"},
    {"i": "write project config", "path": ".omp/config.yml",
     "content": "modelRoles:\n  default: planner\n  audit: correctness\n"},
    {"i": "write main role agent", "path": ".omp/agents/main.md",
     "content": "---\nname: main\n---\n主代理：负责任务分派与汇总。"},
    {"i": "write planner role agent", "path": ".omp/agents/planner.md",
     "content": "---\nname: planner\n---\n规划角色：只做方案不做实现。"},
    {"i": "write worker role agent", "path": ".omp/agents/worker.md",
     "content": "---\nname: worker\n---\n执行角色：按方案落地。"},
    {"i": "write process guard watchdog", "path": "WATCHDOG.yml",
     "content": "interval: 20m\nactive: night\naction: resume\n"},
    {"i": "write adversarial audit role", "path": ".omp/agents/adversary.md",
     "content": "---\nname: adversary\n---\n反证审查：只回复 true/false/unknown。"},
    {"i": "write correctness audit role", "path": ".omp/agents/correctness.md",
     "content": "---\nname: correctness\n---\n正确性审查：按证据核对结论。"},
    {"i": "write docs for workflow", "path": "docs/workflow.md",
     "content": "# 工作流\n1. 规划\n2. 执行\n3. 审查\n"},
    {"i": "write release checklist", "path": "docs/release.md",
     "content": "# 发布清单\n- 版本号\n- 变更记录\n- 回滚预案\n"},
)

# 违规 write：内容含被禁模式（规则可判定）
WRITE_VIOLATION = (
    # anti：内容里指挥未来修改
    {"i": "write role doc with lock note", "path": ".omp/agents/main.md",
     "content": "---\nname: main\n---\n注意：本规则是既有设计，不要改成异步，防止再犯。"},
    {"i": "write config with lock comment", "path": ".omp/config.yml",
     "content": "# 不要改回旧版配置，会破坏顺序\nmodelRoles:\n  default: planner\n"},
    # tmpl：话术模板 + 轮换
    {"i": "write greeting template module", "path": "src/greet.ts",
     "content": "const templates = ['您好呀', '欢迎回来', '又见面啦'];\n"
                 "export const greet = () => templates[variant % templates.length];"},
    # mech：词表加权裁决
    {"i": "write city matcher module", "path": "src/city.ts",
     "content": "const CITY_ALIAS = { '帝都': '北京', '魔都': '上海' };\n"
                 "export const rank = (t: string) => Object.keys(CITY_ALIAS)"
                 ".filter(a => t.includes(a)).length;"},
)


def blob(tool, payload):
    return json.dumps({tool: payload}, ensure_ascii=False)


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v6.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ev = [json.loads(l) for l in
          (OUT / "eval_t3_v6.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]

    add_tr, add_ev = [], []
    n = 0
    for payload in WRITE_COMPLIANT:
        add_tr.append(randomize_constraints({
            "case_id": f"v7w-{n:03d}", "event_text": blob("write", payload),
            "constraints": [{"entity_id": "C1", "quote": C["anti"]}],
            "label": "compliant", "kind": "tool_call", "scope": "s"}))
        n += 1
    for payload in WRITE_VIOLATION:
        ban = ("mech" if "ALIAS" in payload["content"] or "CITY_ALIAS" in payload["content"]
               else "tmpl" if "variant %" in payload["content"] else "anti")
        add_tr.append(randomize_constraints({
            "case_id": f"v7w-{n:03d}", "event_text": blob("write", payload),
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": "violation", "kind": "tool_call", "scope": "s"}))
        n += 1

    # eval：不同 payload 实例（防泄漏）
    EVAL_W = (
        ({"i": "write reviewer role doc", "path": ".omp/agents/reviewer.md",
          "content": "---\nname: reviewer\n---\n评审角色：只读代码给出意见。"},
         "anti", "compliant"),
        ({"i": "write scheduler config", "path": ".omp/config.yml",
          "content": "schedule:\n  night: true\n  intervalMinutes: 30\n"},
         "anti", "compliant"),
        ({"i": "write onboarding doc", "path": "docs/onboarding.md",
          "content": "# 入职指引\n1. 配置环境\n2. 阅读架构\n"},
         "tmpl", "compliant"),
        ({"i": "write retry helper", "path": "src/retry.ts",
         "content": "const msgs = ['再试一次', '别灰心', '马上就好'];\n"
                    "export const pick = (n: number) => msgs[n % msgs.length];"},
         "tmpl", "violation"),
        ({"i": "write tag matcher", "path": "src/tag.ts",
         "content": "const TAG_ALIAS: Record<string,string> = {'js': 'javascript'};\n"
                    "export const resolve = (t: string) => TAG_ALIAS[t] ?? t;"},
         "mech", "violation"),
        ({"i": "write role doc with freeze note", "path": ".omp/agents/ops.md",
          "content": "---\nname: ops\n---\n此流程已定稿，谁都不许再改动流程步骤。"},
         "anti", "violation"),
    )
    for i, (payload, ban, label) in enumerate(EVAL_W):
        add_ev.append(randomize_constraints({
            "case_id": f"ev7w-{i:02d}", "event_text": blob("write", payload),
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": label, "kind": "tool_call", "scope": "s"}))

    tr += add_tr
    ev += add_ev
    for name, rows in (("train_t3_v7.jsonl", tr), ("eval_t3_v7.jsonl", ev)):
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

# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v11_data.py — v11：真实 commit 语料并入。

run17 发现：真实 commit（长 diff、代码密集）是训练未覆盖的载体——
v8-stable 在真实 commit 场景 100% 误报（含 5 个审计违规全召回，
但 7 个合规 commit 也全判违规）。

v11 = v10 ⊕ 本地锚定的真实 commit（gold 来自
docs/session_violation_audit.md 用户裁定时间线）：
- 真实 commit 清单见本地 gold 文件（不入库）

产出 train_t3_v11.jsonl。评测口径：本批为**同源验证**（训后在这 12 条上
复测 = in-sample 证明载体可学）；held-out 真实召回需更多真实 commit（后续）。
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(REPO / "src"))

C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
# 真实 commit 锚定清单在本地文件（gitignored）：real_recall_gold.local.jsonl
LOCAL_GOLD = Path(__file__).resolve().parents[2] / ("data/intent_eval/"
                                                    "real_recall_gold.local.jsonl")
REAL = []
if LOCAL_GOLD.exists():
    for line in LOCAL_GOLD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            REAL.append((row["sha"], row["gold"], row.get("desc", "")[:60]))


def git_text(sha):
    r = subprocess.run(
        ["git", "show", sha, "--format=%s%n%b", "--unified=0"],
        cwd=os.environ.get("MOONBOW_REAL_REPO", "."), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return (r.stdout or "")[:2000]


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v10.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    add = []
    n = 0
    for sha, label, subject in REAL:
        text = git_text(sha)
        if not text.strip():
            print(f"WARN: {sha} 空文本，跳过")
            continue
        add.append({"case_id": f"v11real-{n:03d}", "event_text": text,
                    "constraints": [], "label": label,
                    "kind": "commit", "scope": "s",
                    "gold_note": f"审计锚定 {sha} {subject[:40]}"})
        n += 1
    # report 载体语料折叠（yield/irc 汇报形，规则打标）
    rep_src = OUT / "gen" / "report_corpus_b1.jsonl"
    n_rep = 0
    if rep_src.exists():
        for l in rep_src.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            text = (r.get("text") or "").strip()
            pol, ban = r.get("polarity"), r.get("ban")
            if not text or pol not in ("viol", "comp") or ban not in C:
                continue
            label = "violation" if pol == "viol" else "compliant"
            add.append({"case_id": f"v11rep-{n_rep:03d}",
                        "event_text": text,
                        "constraints": [{"entity_id": "C1",
                                         "quote": C[ban]}],
                        "label": label, "kind": "tool_call",
                        "scope": "s",
                        "gold_note": f"汇报载体 {r.get('domain','')}"})
            n_rep += 1
        print(f"report 语料并入: {n_rep} 条")
    tr11 = tr + add
    with (OUT / "train_t3_v11.jsonl").open("w", encoding="utf-8") as fh:
        for r in tr11:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v11.jsonl: {len(tr11)}（含真实 commit {len(add)}）")
    from collections import Counter
    print("标签分布:", dict(Counter(r["label"] for r in tr11)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

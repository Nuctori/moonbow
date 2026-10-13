# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v12_data.py — v12：真实 commit 规模化（弱标签高置信筛选）。

数据源：real_commits_domain.jsonl（173 真实 commit）⊕
        real_commits_domain_labels.jsonl（subagent 分类：相关卡/极性/置信）。

v12 组装规则（弱标签治理）：
1. 审计锚定 commit（本地 gold 文件）→ **held-out 评测集**（绝不进训练）；
2. 训练集 = confidence ≥0.7 且 polarity≠none 且 relevant_card≠none 的高置信
   真实 commit（viol→violation / comp→compliant）；
3. event_text = subject + diff_head（与运行时 commit 载体同构）；
4. 与 v9 语料合并（train_t3_v9 主体保留），泄漏检测交训练器守卫。

产出 train_t3_v12.jsonl / heldout_real_recall.jsonl。
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))
from exp_real_recall import GOLD, git_show
AUDIT_GOLD = {sha: label for sha, label, _ in GOLD}
GEN = OUT / "gen"

AUDIT_HELD_OUT = {sha for sha, _, _ in GOLD}




def main():
    commits = {json.loads(l)["sha"]: json.loads(l) for l in
               (GEN / "real_commits_domain.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()}
    labels = {json.loads(l)["sha"]: json.loads(l) for l in
              (GEN / "real_commits_domain_labels.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()}

    add, heldout = [], []
    n = 0
    n_lowconf = 0
    missing = AUDIT_HELD_OUT - commits.keys()
    for sha in sorted(missing):
        commits[sha] = {"sha": sha, "subject": "", "diff_head": git_show(sha)}
    for sha, c in commits.items():
        if sha in AUDIT_HELD_OUT:
            text = (c.get("subject", "") + chr(10) + c.get("diff_head", ""))[:2000]
            heldout.append({"case_id": f"held-{sha}", "event_text": text,
                            "label": AUDIT_GOLD[sha], "sha": sha,
                            "gold_source": "audit-anchor"})
    for sha, lab in labels.items():
        c = commits.get(sha)
        if c is None:
            continue
        card, pol, conf = lab.get("relevant_card"), lab.get("polarity"), lab.get("confidence")
        event_text = (c.get("subject", "") + "\n" + c.get("diff_head", ""))[:2000]
        if sha in AUDIT_HELD_OUT:
            continue
        if pol not in ("viol", "comp") or card not in ("mech", "tmpl", "anti", "silent", "test") or (conf or 0) < 0.7:
            n_lowconf += 1
            continue
        label = "violation" if pol == "viol" else "compliant"
        add.append({"case_id": f"v12real-{n:03d}", "event_text": event_text,
                    "constraints": [], "label": label, "kind": "commit",
                    "scope": "s", "sha": sha, "gold_note": lab.get("reason", "")[:100]})
        n += 1

    tr = [json.loads(l) for l in
          (OUT / "train_t3_v9.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    from train_intent_lora import leak_check
    candidates = tr + add
    tr12 = []
    excluded = []
    for row in candidates:
        exact, near = leak_check([row], heldout)
        if exact or near:
            excluded.append(row["case_id"])
        else:
            tr12.append(row)
    print(f"留出近重隔离剔除: {excluded}")
    for i, row in enumerate(tr12):
        row["source_case_id"] = row.get("source_case_id", row["case_id"])
        row["case_id"] = f"v12-{i:04d}"
    with (OUT / "train_t3_v12.jsonl").open("w", encoding="utf-8") as fh:
        for r in tr12:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "heldout_real_recall.jsonl").open("w", encoding="utf-8") as fh:
        for r in heldout:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v12: {len(tr12)}（筛选真实候选 {len(add)}，近重剔除 {len(excluded)}）")
    print(f"heldout_real_recall: {len(heldout)}")
    from collections import Counter
    print("heldout gold:", dict(Counter(r.get("label") for r in heldout)),
          "| uncertain:", sum(1 for r in heldout if r.get("label") == "uncertain"))
    print(f"低置信/不适用剔除: {n_lowconf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v12_data.py — v12：真实 commit 规模化（弱标签高置信筛选）。

数据源：real_commits_domain.jsonl（173 真实 commit）⊕
        real_commits_domain_labels.jsonl（subagent 分类：相关卡/极性/置信）。

v12 组装规则（弱标签治理）：
1. 审计锚定 12 commit → **held-out 评测集**（绝不进训练）——首次拿到
   "训练不含、真值已锚定"的干净真实召回测量；
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
GEN = OUT / "gen"

# 审计锚定的 12 commit（held-out，绝不进训练）
AUDIT_HELD_OUT = {
    "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA",
    "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA", "REDACTED_SHA",
    "REDACTED_SHA", "REDACTED_SHA",
}


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
    for sha, lab in labels.items():
        c = commits.get(sha)
        if c is None:
            continue
        card, pol, conf = lab.get("relevant_card"), lab.get("polarity"), lab.get("confidence")
        event_text = (c.get("subject", "") + "\n" + c.get("diff_head", ""))[:2000]
        if sha in AUDIT_HELD_OUT:
            gold = "violation" if pol == "viol" else (
                "compliant" if pol == "comp" else "uncertain")
            heldout.append({"case_id": f"held-{sha[:9]}", "event_text": event_text,
                            "label": gold, "sha": sha,
                            "conf": conf, "reason": lab.get("reason", "")[:120]})
            continue
        if pol == "none" or card == "none" or (conf or 0) < 0.7:
            n_lowconf += (0 if pol == "none" or card == "none" else 0) + \
                (0 if (conf or 0) < 0.7 else 1)
            continue
        label = "violation" if pol == "viol" else "compliant"
        add.append({"case_id": f"v12real-{n:03d}", "event_text": event_text,
                    "constraints": [], "label": label, "kind": "commit",
                    "scope": "s", "gold_note": lab.get("reason", "")[:100]})
        n += 1

    tr = [json.loads(l) for l in
          (OUT / "train_t3_v9.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    tr12 = tr + add
    with (OUT / "train_t3_v12.jsonl").open("w", encoding="utf-8") as fh:
        for r in tr12:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "heldout_real_recall.jsonl").open("w", encoding="utf-8") as fh:
        for r in heldout:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v12: {len(tr12)}（真实高置信 +{len(add)}）")
    print(f"heldout_real_recall: {len(heldout)}")
    from collections import Counter
    print("heldout gold:", dict(Counter(r.get("gold") for r in heldout)),
          "| uncertain:", sum(1 for r in heldout if r.get("gold") == "uncertain"))
    print(f"低置信/不适用剔除: {n_lowconf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

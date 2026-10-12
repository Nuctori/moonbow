# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v13_data.py — v13：矛盾对 2 倍过采样（最简正确版）。

待验证假设：constraint-as-decoration（模型读决策出结论，约束原文无因果参与）。
干预方向：矛盾对（同决策×不同约束→不同标签）占比太低，模型从决策文本
单独即可预测标签。

v13 修复：从 v11 语料中识别矛盾对（同一 event_text 以不同约束出现
且标签不同），2 倍复制（追加一次），使矛盾对信号在训练中被加权。
其余行不动。

产出 train_t3_v13.jsonl。
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = REPO = HERE.parent.parent / "data" / "intent_eval"

def main():
    src = REPO / "train_t3_v11.jsonl"
    rows = [json.loads(l) for l in
            src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"源数据: {len(rows)} 行")

    # 识别矛盾对：同一 event_text 出现 ≥2 次且标签不同
    from collections import defaultdict
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["event_text"]].append(r)
    conflict_keys = {k for k, v in by_text.items()
                     if len(v) >= 2 and len({x["label"] for x in v}) > 1
                     and len({json.dumps(x.get("constraints", []), ensure_ascii=False, sort_keys=True) for x in v}) > 1}
    print(f"矛盾对 text 键：{len(conflict_keys)} 组")

    # 2 倍复制矛盾对行
    added = 0
    out = list(rows)
    for r in rows:
        if r["event_text"] in conflict_keys:
            out.append(dict(r))                  # 追加一份
            added += 1
    print(f"矛盾对过采样：+{added} 行（{len(conflict_keys)} 组 × 1 份追加）")

    # 豁免线索改写不变样本（同约束同决策改写措辞 → 同标签，防措辞捷径）
    # 从 v11 已有的叙述违规中提取，替换豁免短语为改写版
    rewrite = 0
    for r in rows:
        orig = r["event_text"]
        if "由模型判定" in orig or "由模型输出" in orig or "由模型给出" in orig:
            new = dict(r)
            new["event_text"] = orig.replace("由模型判定", "由上游语义服务裁定") \
                                    .replace("由模型输出", "由上游语义服务产出") \
                                    .replace("由模型给出", "由上游语义服务给出")
            if new["event_text"] != orig:
                new["case_id"] = r["case_id"] + "-rw"
                out.append(new)
                rewrite += 1
    print(f"改写不变样本：+{rewrite} 行")

    for i, row in enumerate(out):
        row["source_case_id"] = row.get("source_case_id", row["case_id"])
        row["case_id"] = f"v13-{i:04d}"

    with (OUT / "train_t3_v13.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v13: {len(out)} 行")
    from collections import Counter
    print("标签分布:", dict(Counter(r["label"] for r in out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

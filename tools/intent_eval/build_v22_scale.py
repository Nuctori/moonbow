# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v22_scale.py — R-v22：审计四族定向语料合成。

来源：v22_corpus_a.jsonl（词面角色格 120 + 缺失性违规 60，FP 主攻）+
      v22_corpus_b.jsonl（语气最小对 150 + 适用性存疑 80，漏报/边界）+
      cardset_invariance_aug.jsonl（120，卡组不变性——全局先验泄漏误报主攻）+
      train_t3_v20.jsonl（849 基座）+ v17 谱系回放 100（B 保护）。
留出：v22a/v22b 分层 20% 作 A3-v3 未见措辞门禁（不进训练）。
开发集：eval_t3_v15_dev_fixed.jsonl（移除 4 条 event_text=卡键的坏行），
基线 v17=156/166、32/36、24/24、原任务合规误报 2（filter-repo 后重测）。
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
BACKUP = Path("M:/AI/spark-4b/intent_private_backup")
SOURCES = [BACKUP / "v22_corpus_a.jsonl", BACKUP / "v22_corpus_b.jsonl"]
CARD_QUOTES = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}


def main():
    rows, problems, seen = [], [], set()
    for src in SOURCES:
        batch = load_jsonl(src)
        if len(batch) < 150:
            problems.append(f"{src.name} 行数不足: {len(batch)}")
        for r in batch:
            if r["text"] in seen:
                problems.append(f"{src.name} 重复: {r['text'][:40]}")
            seen.add(r["text"])
            if r.get("label") not in ("violation", "compliant", "insufficient") or r.get("card") not in CARD_QUOTES:
                problems.append(f"{src.name} 字段非法: {r.get('case_id', r.get('scenario'))}")
        rows += batch
    aug = load_jsonl(OUT / "cardset_invariance_aug.jsonl")
    if len(aug) < 100:
        problems.append(f"不变性增广不足: {len(aug)}")
    if problems:
        raise ValueError("语料验证未通过: " + "; ".join(problems[:8]))

    rng = random.Random(22)  # 留出抽样可复现性用，非安全用途
    layers = {}
    for r in rows:
        layers.setdefault((r.get("family", "?"), r["label"]), []).append(r)
    holdout_src, train_new = [], []
    for key in sorted(layers):
        members = layers[key][:]
        rng.shuffle(members)
        n_hold = max(1, round(len(members) * 0.2))
        holdout_src.extend(members[:n_hold])
        train_new.extend(members[n_hold:])

    gate_holdout = [{"case_id": f"gv22-{i:03d}", "event_text": r["text"],
                     "constraints": [{"entity_id": "C1",
                                      "quote": CARD_QUOTES[r["card"]]}],
                     "label": r["label"], "kind": "tool_call", "scope": "s",
                     "gate": "v22-holdout"}
                    for i, r in enumerate(holdout_src)]

    target = [{"event_text": r["text"],
               "constraints": [{"entity_id": "C1", "quote": CARD_QUOTES[r["card"]]}],
               "label": r["label"], "kind": "tool_call", "scope": "s",
               "gold_source": f"authored-v22-{r['family']}"} for r in train_new]
    base = load_jsonl(OUT / "train_t3_v20.jsonl")
    v17_rows = load_jsonl(OUT / "train_t3_v17.jsonl")
    rng.shuffle(v17_rows)
    replay = [dict(r, replay="v17") for r in v17_rows[:100]]
    train = [dict(r) for r in base] + target + load_jsonl(OUT / "cardset_invariance_aug.jsonl") + replay
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v22-{i:04d}"

    guard = (load_jsonl(OUT / "eval_t3_v15_dev_fixed.jsonl") +
             load_jsonl(OUT / "gate_real_fp.jsonl") +
             load_jsonl(OUT / "gate_carrier_viol.jsonl") +
             load_jsonl(OUT / "gate_v21_holdout.jsonl") + gate_holdout)
    # 自清洗：历史留出集与 v20 训练侧存在交叉（v20/v21 两轮独立分层所致），
    # 基座继承的冲突行必须剔除，直至对 guard 全部干净。
    guard_texts = {r["event_text"] for r in guard}
    removed = 0
    while True:
        exact, near = leak_check(train, guard)
        if not exact and not near:
            break
        bad_texts = {r["event_text"] for r in guard if r["case_id"] in set(exact)}
        before = len(train)
        train = [r for r in train if r.get("event_text") not in bad_texts]
        if before == len(train):
            raise ValueError("泄漏自清洗停滞")
        removed += before - len(train)
    print(f"自清洗剔除 {removed} 行（历史留出交叉）")

    (OUT / "train_t3_v22.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    (OUT / "gate_v22_holdout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gate_holdout) + "\n",
        encoding="utf-8")
    atomic_json(OUT / "v22_data_manifest.json", {
        "raw_n": len(rows), "aug_n": len(aug), "train_n": len(train),
        "holdout_n": len(gate_holdout), "replay": len(replay),
        "labels": dict(Counter(r["label"] for r in rows)),
        "train_sha256": file_hash(OUT / "train_t3_v22.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "audit-four-family targeted data (FP families first), "
                  "card-set invariance aug, stratified 20% holdout, fixed-dev baselines "
                  "v17=156/32/24/FP2 vs v20=151/30/24/FP6"})
    print(f"validated={len(rows)} aug={len(aug)} train={len(train)} "
          f"holdout={len(gate_holdout)} replay={len(replay)} "
          f"labels={dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()

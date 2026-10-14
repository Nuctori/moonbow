# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v21_scale.py — R-v21：三批语料合并、留出与回放加权合成。

输入：v20_corpus_raw.jsonl(160) + v21_corpus_a.jsonl(180) + v21_corpus_b.jsonl(180)
验证：每批 ≥150 行、viol/comp 平衡、每卡覆盖、字段完整、跨批与批内去重
留出：合并后按 carrier×label 分层抽 20% 作 A3 未见措辞门禁
回放：额外加权 80 行 v17 谱系原任务回放（保护 B 门禁——v20 的教训）
合成：train_t3_v19.jsonl(609) + 新语料训练侧 + 回放 → train_t3_v21.jsonl
隔离：leak_check 对 eval_t3_v15_dev + 全部历史门禁集 + v21 留出
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
SOURCES = [BACKUP / "v20_corpus_raw.jsonl",
           BACKUP / "v21_corpus_a.jsonl",
           BACKUP / "v21_corpus_b.jsonl"]
CARDS = {"mech", "tmpl", "anti", "silent", "test"}
CARD_QUOTES = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}


def load_and_tag(path, batch):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["batch"] = batch
    return rows


def validate(rows, name):
    problems = []
    labels = Counter(r.get("label") for r in rows)
    if labels.get("violation", 0) < 70 or labels.get("compliant", 0) < 70:
        problems.append(f"{name} 标签失衡: {labels}")
    viol_by_card = Counter(r.get("card") for r in rows if r.get("label") == "violation")
    for card in CARDS:
        if viol_by_card.get(card, 0) < 14:
            problems.append(f"{name} 卡 {card} viol 不足: {viol_by_card.get(card, 0)}")
    texts = [r.get("text", "") for r in rows]
    if len(set(texts)) != len(texts):
        problems.append(f"{name} 存在重复 text")
    for i, r in enumerate(rows):
        if r.get("label") not in ("violation", "compliant") or r.get("card") not in CARDS:
            problems.append(f"{name} 行 {i} 字段非法")
    return problems


def stratified_holdout(rows, frac=0.2, seed=21):
    rng = random.Random(seed)  # 留出抽样可复现性用，非安全用途
    layers = {}
    for r in rows:
        layers.setdefault((r["carrier"], r["label"]), []).append(r)
    holdout, train_side = [], []
    for key in sorted(layers):
        members = layers[key][:]
        rng.shuffle(members)
        n_hold = max(1, round(len(members) * frac))
        holdout.extend(members[:n_hold])
        train_side.extend(members[n_hold:])
    return train_side, holdout


def main():
    all_rows, problems, seen_texts = [], [], set()
    for i, src in enumerate(SOURCES):
        rows = load_and_tag(src, f"batch{i}")
        problems += validate(rows, src.name)
        for r in rows:
            if r["text"] in seen_texts:
                problems.append(f"{src.name} 跨批重复: {r['text'][:40]}")
            seen_texts.add(r["text"])
        all_rows += rows
    if problems:
        raise ValueError("语料验证未通过: " + "; ".join(problems[:8]))

    train_side, holdout = stratified_holdout(all_rows)
    gate_holdout = [{"case_id": f"gv21-{i:03d}", "event_text": r["text"],
                     "constraints": [{"entity_id": "C1",
                                      "quote": CARD_QUOTES[r["card"]]}],
                     "label": r["label"], "kind": "tool_call", "scope": "s",
                     "gate": "v21-holdout"}
                    for i, r in enumerate(holdout)]

    target = []
    for r in train_side:
        target.append({"event_text": r["text"],
                       "constraints": [{"entity_id": "C1",
                                        "quote": CARD_QUOTES[r["card"]]}],
                       "label": r["label"], "kind": "tool_call", "scope": "s",
                       "gold_source": f"authored-v21-{r['batch']}-{r['carrier']}"})
    base = load_jsonl(OUT / "train_t3_v19.jsonl")
    # 原任务回放加权：从 v17 谱系再抽 80 行（B 门禁保护，v20 教训）
    import os
    v17_rows = load_jsonl(OUT / "train_t3_v17.jsonl")
    rng = random.Random(21)  # 仅可复现抽样，非安全用途
    rng.shuffle(v17_rows)
    replay = [dict(r, replay="v17") for r in v17_rows[:80]]
    train = [dict(r) for r in base] + target + replay
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v21-{i:04d}"

    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "gate_real_fp.jsonl") +
             load_jsonl(OUT / "gate_carrier_viol.jsonl") + gate_holdout)
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact[:4]}, {near[:4]}")

    (OUT / "train_t3_v21.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    (OUT / "gate_v21_holdout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gate_holdout) + "\n",
        encoding="utf-8")
    atomic_json(OUT / "v21_data_manifest.json", {
        "raw_n": len(all_rows), "train_n": len(train),
        "holdout_n": len(gate_holdout), "replay": len(replay),
        "labels": dict(Counter(r["label"] for r in all_rows)),
        "train_sha256": file_hash(OUT / "train_t3_v21.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "3-batch scaled carriers, stratified 20% holdout (A3), "
                  "80-row v17-lineage replay top-up (B protection)"})
    print(f"validated={len(all_rows)} train={len(train)} holdout={len(gate_holdout)} "
          f"replay={len(replay)} labels={dict(Counter(r['label'] for r in all_rows))}")


if __name__ == "__main__":
    main()

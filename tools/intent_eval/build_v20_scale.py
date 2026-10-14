# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v20_scale.py — 大规模真实载体语料验证、留出与训练合成。

输入：intent_private_backup/v20_corpus_raw.jsonl（subagent 撰写，160 行配对）
验证：数量/平衡/每卡覆盖/字段完整/文本去重
留出：按 carrier×label 分层抽 25% 作 A3 未见措辞门禁（不进训练）
合成：train_t3_v19.jsonl(609) + 留在训练侧的新语料 ×2 → train_t3_v20.jsonl
隔离：leak_check 对 eval_t3_v15_dev + gate_real_fp + gate_carrier_viol + v20 留出
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import leak_check, load_jsonl

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
RAW = Path("M:/AI/spark-4b/intent_private_backup/v20_corpus_raw.jsonl")
CARDS = {"mech", "tmpl", "anti", "silent", "test"}


def validate(rows):
    problems = []
    if len(rows) < 150:
        problems.append(f"行数不足: {len(rows)}")
    labels = Counter(r.get("label") for r in rows)
    if labels.get("violation", 0) < 70 or labels.get("compliant", 0) < 70:
        problems.append(f"标签失衡: {labels}")
    viol_by_card = Counter(r.get("card") for r in rows if r.get("label") == "violation")
    for card in CARDS:
        if viol_by_card.get(card, 0) < 12:
            problems.append(f"卡 {card} violation 不足: {viol_by_card.get(card, 0)}")
    texts = [r.get("text", "") for r in rows]
    if len(set(texts)) != len(texts):
        problems.append("存在重复 text")
    for i, r in enumerate(rows):
        if r.get("label") not in ("violation", "compliant"):
            problems.append(f"行 {i} label 非法")
        if r.get("card") not in CARDS:
            problems.append(f"行 {i} card 非法")
        if not str(r.get("text", "")).strip():
            problems.append(f"行 {i} text 为空")
    return problems


def stratified_holdout(rows, frac=0.25, seed=20):
    """按 carrier×label 分层留出，保证每层至少留 1。"""
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
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = validate(rows)
    if problems:
        raise ValueError("语料验证未通过: " + "; ".join(problems[:8]))

    train_side, holdout = stratified_holdout(rows)
    gate_holdout = [{"case_id": f"gatehold-{i:03d}", "event_text": r["text"],
                     "constraints": [{"entity_id": "C1", "quote": ""}],
                     "label": r["label"], "kind": "tool_call", "scope": "s",
                     "gate": "v20-holdout", "card": r["card"]}
                    for i, r in enumerate(holdout)]
    # 留出行按所属卡挂对应约束（compliant 行挂其配对卡，与 violation 一致口径）
    card_quotes = {
        "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
        "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
        "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
        "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
        "test": "测试断言必须断言政策，不得钉死具体句式",
    }
    for g in gate_holdout:
        g["constraints"] = [{"entity_id": "C1", "quote": card_quotes[g["card"]]}]

    target = []
    for repeat in range(2):
        for r in train_side:
            target.append({"event_text": r["text"],
                           "constraints": [{"entity_id": "C1",
                                            "quote": card_quotes[r["card"]]}],
                           "label": r["label"], "kind": "tool_call", "scope": "s",
                           "gold_source": f"authored-v20-carrier{r['carrier']}",
                           "repeat": repeat})
    base = load_jsonl(OUT / "train_t3_v19.jsonl")
    train = [dict(r) for r in base] + target
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v20-{i:04d}"

    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "gate_real_fp.jsonl") +
             load_jsonl(OUT / "gate_carrier_viol.jsonl") + gate_holdout)
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact[:4]}, {near[:4]}")

    (OUT / "train_t3_v20.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    (OUT / "gate_v20_holdout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gate_holdout) + "\n",
        encoding="utf-8")
    atomic_json(OUT / "v20_data_manifest.json", {
        "raw_n": len(rows), "train_n": len(train), "holdout_n": len(gate_holdout),
        "labels": dict(Counter(r["label"] for r in rows)),
        "train_sha256": file_hash(OUT / "train_t3_v20.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "scaled real tool-traffic carriers, stratified 25% holdout as A3 gate"})
    print(f"validated={len(rows)} train={len(train)} holdout={len(gate_holdout)} "
          f"labels={dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()

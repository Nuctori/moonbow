"""v14 开发干预：极性配对加权与原任务分层回放。"""
import json
import random
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import label_of, leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"


def build_rows(base, polarity, evals):
    rng = random.Random(42)
    replay = []
    for label in ("violation", "compliant", "insufficient"):
        candidates = [r for r in base if label_of(r, "t3") == label]
        rng.shuffle(candidates)
        replay.extend(candidates[:30])
    rows = [dict(r) for r in replay]
    for repeat in range(4):
        rows.extend(dict(r, repeat=repeat) for r in polarity)
    for i, row in enumerate(rows):
        row["source_case_id"] = row.get("source_case_id", row["case_id"])
        row["case_id"] = f"v14-{i:04d}"
    exact, near = leak_check(rows, evals)
    if exact or near:
        raise ValueError(f"开发集泄漏: {exact}, {near}")
    return rows


def main():
    evals = load_jsonl(OUT / "eval_t3_v8.jsonl") + load_jsonl(OUT / "polarity_diagnostic_eval.jsonl")
    rows = build_rows(load_jsonl(OUT / "train_t3_v13.jsonl"),
                      load_jsonl(OUT / "polarity_diagnostic_train.jsonl"), evals)
    for name, data in (("train_t3_v14.jsonl", rows), ("eval_t3_v14_dev.jsonl", evals)):
        (OUT / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n", encoding="utf-8")
    atomic_json(OUT / "v14_data_manifest.json", {
        "train_sha256": file_hash(OUT / "train_t3_v14.jsonl"),
        "eval_sha256": file_hash(OUT / "eval_t3_v14_dev.jsonl"),
        "train_n": len(rows), "eval_n": len(evals),
        "recipe": "30 per original class replay + 36 polarity rows repeated four times",
        "evaluation_status": "development-checkpoint-selection-not-final-test",
    })
    print(f"train={len(rows)} eval={len(evals)}")


if __name__ == "__main__":
    main()

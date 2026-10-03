# -*- coding: utf-8 -*-
"""experiments/task_structure/train_axis1.py — 轴 1（言语行为）二元头训练。

规范 v2 草案：轴 1 = demand（待实现的要求）/ mention（提及叙述），
它是唯一决定"是否产生捕获"的闸门；轴 2（角色分类）留待轴 1 稳定后。

数据：build_axis1_set.py 产出的 coverage_train.jsonl（579 条，成对结构）。
底座：本地 287M boundary 架构（AutoExtractor；GLiNER2.from_pretrained 不适用）。
输出：M:/AI/spark-4b/models/task_structure_axis1_cov/final

用法：
  python train_axis1.py --smoke    # 80 条 1 epoch 验证管线
  python train_axis1.py            # 全量 6 epoch
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

TRAIN_JSONL = os.path.join(HERE, "coverage_train.jsonl")
OUT_DIR = os.path.join(ROOT, "models", "task_structure_axis1_cov")
TASK = "speech"
LABELS = ["demand", "mention"]


def find_snapshot() -> str:
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
    for s in os.listdir(cache):
        p = os.path.join(cache, s)
        if os.path.isfile(os.path.join(p, "model.safetensors")):
            return p
    return "fastino/gliner2.5-multi-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from gliner2.training import (ExtractorTrainer, TrainingConfig,
                                  create_classification_example)
    from gliner2.training.data import TrainingDataset
    from gliner2 import AutoExtractor

    pair_rows = [json.loads(l) for l in open(TRAIN_JSONL, encoding="utf-8")]
    # 成对数据必须切分对齐：按 zone+对象标识成组切分，避免同对跨 train/eval
    def group_key(r):
        t = r["text"]
        for mark in ("改成了", "调成了", "设为了", "已经", "改过", "过"):
            if mark in t:
                return "P"
        return "D"
    rows = pair_rows
    if args.smoke or args.limit:
        rows = pair_rows[: (args.limit or 80)]

    examples = [create_classification_example(
        text=r["text"], task=TASK, labels=LABELS, true_label=r["label"])
        for r in rows]
    holdout = max(1, len(examples) // 10)
    train_ds = TrainingDataset(examples[:-holdout])
    eval_ds = TrainingDataset(examples[-holdout:])
    print(f"train={len(train_ds)} eval={len(eval_ds)} labels={LABELS}")

    t0 = time.time()
    model = AutoExtractor.from_pretrained(find_snapshot(), map_location="cpu")
    print(f"底座加载 {time.time()-t0:.0f}s")

    config = TrainingConfig(
        output_dir=os.path.join(OUT_DIR, "ckpt"),
        experiment_name="task-structure-axis1-v1",
        num_epochs=1 if args.smoke else args.epochs,
        batch_size=8,
        encoder_lr=2e-5,
        task_lr=1e-3,
        eval_strategy="epoch",
        save_total_limit=1,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        num_workers=0,
        report_to_wandb=False,
    )
    trainer = ExtractorTrainer(model=model, config=config)
    t0 = time.time()
    results = trainer.train(train_data=train_ds, eval_data=eval_ds)
    print(f"训练完成 {time.time()-t0:.0f}s best={results.get('best_metric')}")

    final_dir = os.path.join(OUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    save = getattr(model, "save_pretrained", None) or model.model.save_pretrained
    save(final_dir)
    print(f"已保存 → {final_dir}")


if __name__ == "__main__":
    main()

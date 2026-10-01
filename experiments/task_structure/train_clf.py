# -*- coding: utf-8 -*-
"""experiments/task_structure/train_clf.py — 微调 287M 子句结构分类头。

用户已授权训练（2026-09-24）。底座 = 本地 fastino/gliner2.5-multi-v1
（boundary 架构，经 AutoExtractor 加载；GLiNER2.from_pretrained 是 span
架构加载器，与此 checkpoint 不兼容）。Trainer 用 gliner2 自带
ExtractorTrainer（官方文档注明 architecture-neutral，支持 boundary）。

数据：build_train_set.py 产出的 train_clf.jsonl（合成 2351 + dev 弱监督 94，
无人工复核）。test 切分不参与训练与早停。

用法：
  python train_clf.py --smoke     # 200 条 1 epoch，验证管线
  python train_clf.py             # 全量 3 epoch
输出：M:/AI/spark-4b/models/task_structure_clf_v1/final
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

TRAIN_JSONL = os.path.join(HERE, "train_clf.jsonl")
OUT_DIR = os.path.join(ROOT, "models", "task_structure_clf_v1")
TASK = "kind"
CLASSES = ["goal", "constraint", "dependency", "coordination",
           "condition", "unresolved", "acceptance", "cancel", "none"]


def find_snapshot() -> str:
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
    for s in os.listdir(cache):
        p = os.path.join(cache, s)
        if os.path.isfile(os.path.join(p, "model.safetensors")):
            return p
    return "fastino/gliner2.5-multi-v1"


def load_examples(path: str, limit: int = None):
    from gliner2.training import create_classification_example
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            examples.append(create_classification_example(
                text=r["text"], task=TASK, labels=CLASSES, true_label=r["label"]))
            if limit and len(examples) >= limit:
                break
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from gliner2.training import ExtractorTrainer, TrainingConfig
    from gliner2.training.data import TrainingDataset

    n = 200 if args.smoke else None
    examples = load_examples(TRAIN_JSONL, limit=n)
    holdout = max(1, len(examples) // 10)
    train_ds = TrainingDataset(examples[:-holdout])
    eval_ds = TrainingDataset(examples[-holdout:])
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    model = AutoExtractor_load()
    epochs = 1 if args.smoke else args.epochs
    config = TrainingConfig(
        output_dir=os.path.join(OUT_DIR, "ckpt"),
        experiment_name="task-structure-clf-v1",
        num_epochs=epochs,
        batch_size=8,
        encoder_lr=2e-5,
        task_lr=1e-3,
        eval_strategy="epoch",
        save_total_limit=1,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        num_workers=0,
        fp16=False,
        report_to_wandb=False,
    )
    trainer = ExtractorTrainer(model=model, config=config)
    t0 = time.time()
    results = trainer.train(train_data=train_ds, eval_data=eval_ds)
    print(f"训练完成 {time.time()-t0:.0f}s: {json.dumps(results, default=str)[:400]}")

    final_dir = os.path.join(OUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    save = getattr(model, "save_pretrained", None)
    if save is None and hasattr(model, "model"):
        save = model.model.save_pretrained
    save(final_dir)
    print(f"已保存 → {final_dir}")


def AutoExtractor_load():
    from gliner2 import AutoExtractor
    t0 = time.time()
    model = AutoExtractor.from_pretrained(find_snapshot(), map_location="cpu")
    print(f"底座加载 {time.time()-t0:.0f}s")
    return model


if __name__ == "__main__":
    main()

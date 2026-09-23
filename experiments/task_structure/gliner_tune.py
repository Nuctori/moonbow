# -*- coding: utf-8 -*-
"""experiments/task_structure/gliner_tune.py — 287M GLiNER 零样本标签/阈值调参。

只允许在 **dev** 切分上调（协议同规则基线）；test 只跑最终选定配置一次。
模型加载用项目验证过的 AutoExtractor 路径（pipeline/entity_coverage_guard.py），
此前"受阻"结论是 API 用错（GLiNER2.from_pretrained 不兼容该 checkpoint）。

用法：
  HF_HUB_OFFLINE=1 python gliner_tune.py            # 全部配置
  HF_HUB_OFFLINE=1 python gliner_tune.py --quick    # 仅 dev 子集
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from gliner2 import AutoExtractor, Schema          # noqa: E402

EVAL_SET = os.path.join(HERE, "eval_set.jsonl")

# 标签措辞候选（零样本 GLiNER 对措辞敏感；label → kind 映射回协议 8 类）
LABEL_SETS = {
    "zh_abstract": {
        "交付义务": "goal", "约束": "constraint", "依赖": "dependency",
        "协调要求": "coordination", "条件": "condition",
        "待决事项": "unresolved", "验收条件": "acceptance",
    },
    "zh_entity": {
        "任务目标": "goal", "限制条件": "constraint", "先后依赖": "dependency",
        "一致要求": "coordination", "前提条件": "condition",
        "待确认事项": "unresolved", "完成标准": "acceptance",
    },
    "zh_plain": {
        "要做的事": "goal", "必须满足的限制": "constraint", "先后顺序要求": "dependency",
        "保持一致的要求": "coordination", "如果条件": "condition",
        "待确认的事项": "unresolved", "验收标准": "acceptance",
    },
    "en": {
        "task goal": "goal", "constraint": "constraint", "dependency": "dependency",
        "consistency requirement": "coordination", "condition": "condition",
        "open question": "unresolved", "acceptance criterion": "acceptance",
    },
}
THRESHOLDS = (0.3, 0.4, 0.5)


def load_samples(quick: bool):
    fams = {"sg", "mg", "nc", "pb", "al"} if quick else None
    out = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["split"] != "dev":
                continue
            if fams and s["family"] not in fams:
                continue
            out.append(s)
    return out


def bigrams(s):
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jac(a, b):
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 1.0 if a == b else 0.0
    return len(A & B) / len(A | B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
    snap = next(os.path.join(cache, s) for s in os.listdir(cache)
                if os.path.isfile(os.path.join(cache, s, "model.safetensors")))
    t0 = time.time()
    ext = AutoExtractor.from_pretrained(snap, map_location="cpu")
    print(f"模型加载 {time.time()-t0:.1f}s")

    samples = load_samples(args.quick)
    print(f"dev 样本 {len(samples)} 条（quick={args.quick}）")

    schemas = {}
    for name, labels in LABEL_SETS.items():
        schemas[name] = (Schema().entities(list(labels.keys()), dtype="list"), labels)

    for t in THRESHOLDS:
        for name, (schema, mapping) in schemas.items():
            tp = fp = fn = 0
            t0 = time.time()
            for s in samples:
                gold = [c for c in s["gold"]["captures"] if c["kind"] != "object"]
                res = ext.extract(s["text"], schema, threshold=t)
                preds = []
                for label, items in (res.get("entities") or {}).items():
                    kind = mapping.get(label)
                    if not kind:
                        continue
                    if isinstance(items, list):
                        preds.extend((kind, it) for it in items if it)
                used = set()
                for g in gold:
                    hit = next((i for i, (k, q) in enumerate(preds)
                                if i not in used and k == g["kind"]
                                and jac(g["quote"], q) >= 0.5), None)
                    if hit is not None:
                        used.add(hit)
                        tp += 1
                    else:
                        fn += 1
                fp += len(preds) - len(used)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print(f"t={t}  {name:<12} P={p:.2f} R={r:.2f} F1={f1:.2f} "
                  f"(TP={tp} FP={fp} FN={fn})  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

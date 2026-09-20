# -*- coding: utf-8 -*-
"""pipeline/research_v235_symbolic_pilot.py

选项 1 试点：符号语义层（封闭槽集 + 组合规则），2026-09-20。

规格（maps/DECISION_symbolic_vs_operator_2026-09-20.md）：
  完成度 ∈ {已全部完成, 部分完成, 尚未完成, 执行失败}   （封闭集，抽取器输出）
  模态   ∈ {陈述事实, 承诺将来, 提问澄清}              （封闭集，抽取器输出）
  判定：D = (模态=陈述事实) ∧ (完成度=已全部完成) ∧ match(r,s)
  - 槽值由 gliner2.5-multi-v1（287M 预训练，零样本）抽取，**零手写正则**；
  - match 站沿用基座 MiniLM 嵌入 + v233 冻结的 τ=0.265（**零新增调参**）；
    它是未来"选项2算子"的替身，本试点测的是符号层的增益，不算 match 的账。
  - 封闭真值（构造时确定）：
      completeness=whole 的真值：pos/cross 为真；partial/wait/fail 为假；
      modality=非陈述 的必拦对象：仅承诺/提问类（w03/w07/w10 明确，其余 wait
      项多为"进行中陈述"，归 completeness 轴拦截——两轴任一拦截即闭合失败，
      故模态轴只对承诺/提问项计分，避免把"进行中陈述"误记为模态错误）。

对照基线（同冻结集，v233 实测）：
  现行管线 guardCLOSE∧sim≥τ：  端点A 26/37 (70.3%)  端点B 50/67 (74.6%)
  BASE 嵌入 sim≥τ：            端点A 26/37 (70.3%)  端点B 50/67 (74.6%)
预测（若槽抽取完美）：p/w/f 类 FP 全部被符号层拦截 → 端点B ≈ 57/67 (85.1%)，
端点A 不变（近域负例是 match 站的责任 = 选项 2 的账）。

运行：.venv/Scripts/python.exe pipeline/research_v235_symbolic_pilot.py
"""
import os, sys, json, time, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "pipeline")
from research_v233_alignment_eval import (EVAL_POS, EVAL_CROSS, EVAL_PARTIAL,
                                          EVAL_WAIT, EVAL_FAIL, wilson)
from eval_semantic_matcher_benchmark import BENCHMARK

RESULTS_PATH = "maps/symbolic_pilot_results.json"
TAU = 0.265          # v233 BASE 臂验证集冻结值，禁止改动
DEVICE = "cpu"

WHOLE, PARTIAL, NONE, FAILED = "已全部完成", "部分完成", "尚未完成", "执行失败"
ASSERT, PROMISE, QUESTION = "陈述事实", "承诺将来", "提问澄清"


def load_extractor():
    from gliner2 import AutoExtractor, Schema
    p = os.path.expanduser("~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1")
    snaps = sorted(os.path.join(p, "snapshots", s) for s in os.listdir(os.path.join(p, "snapshots")))
    snap = next(s for s in snaps if os.path.exists(os.path.join(s, "model.safetensors")))
    ext = AutoExtractor.from_pretrained(snap, map_location=DEVICE)
    schema = (Schema()
              .classification("完成度", labels=[WHOLE, PARTIAL, NONE, FAILED])
              .classification("话术模态", labels=[ASSERT, PROMISE, QUESTION]))
    return ext, schema


def main():
    all67 = ([(b, r, s, 1, "pos") for b, r, s in EVAL_POS]
             + [(b, r, s, 0, "cross") for b, r, s in EVAL_CROSS]
             + [(b, r, s, 0, "partial") for b, r, s in EVAL_PARTIAL]
             + [(b, r, s, 0, "wait") for b, r, s in EVAL_WAIT]
             + [(b, r, s, 0, "fail") for b, r, s in EVAL_FAIL])

    # ── 槽抽取（模型，非正则）
    print("加载 gliner2.5-multi-v1（287M，零样本槽抽取）...")
    ext, schema = load_extractor()
    t0 = time.time()
    texts = [s for _, _, s, _, _ in all67]
    slots_raw = ext.batch_extract(texts, schema)
    slot_ms = (time.time() - t0) * 1000 / len(texts)
    slots = []
    for r in slots_raw:
        r = r or {}
        slots.append((r.get("完成度", NONE), r.get("话术模态", QUESTION)))
    print(f"槽抽取完成：{slot_ms:.0f} ms/对\n")

    # ── 槽质量（对照构造真值）
    comp_true = {b: (cat in ("pos", "cross")) for b, _, _, _, cat in all67}
    nonassert_must = {"w03", "w07", "w10"}   # 明确的提问/等待类
    comp_ok = sum((c == WHOLE) == comp_true[b] for (b, _, _, _, _), (c, m) in zip(all67, slots))
    mod_ok = sum((m != ASSERT) for (b, _, _, _, cat), (c, m) in zip(all67, slots)
                 if b in nonassert_must)
    print(f"[槽质量] 完成度轴正确: {comp_ok}/67 | 承诺/提问必拦项({len(nonassert_must)}个)模态命中: {mod_ok}")
    comp_err = [(b, c) for (b, _, _, _, _), (c, m) in zip(all67, slots)
                if (c == WHOLE) != comp_true[b]]
    print(f"[槽质量] 完成度误判: {comp_err}\n")

    # ── match 站：基座嵌入 + 冻结 τ
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    d = [x for x in os.listdir(cache_dir) if "MiniLM-L12-v2" in x][0]
    snap = os.path.join(cache_dir, d, "snapshots", os.listdir(os.path.join(cache_dir, d, "snapshots"))[0])
    tok = AutoTokenizer.from_pretrained(snap)
    mdl = AutoModel.from_pretrained(snap).to(DEVICE).eval()

    def sim(a, b):
        def emb(t):
            inp = tok(t, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                e = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                return F.normalize(e, p=2, dim=1)
        return F.cosine_similarity(emb(a), emb(b)).item()

    # ── 选项1判定：模态=陈述 ∧ 完成度=whole ∧ sim≥τ
    slot_by_id = {b: (c, m) for (b, _, _, _, _), (c, m) in zip(all67, slots)}
    epA = [x for x in all67 if x[4] in ("pos", "cross")]
    epB = all67

    results = {}
    for name, ep in (("A", epA), ("B", epB)):
        pred, gold, fp, fn = [], [], [], []
        for b, r, s, g, cat in ep:
            c, m = slot_by_id[b]
            s_val = sim(r, s)
            p = (m == ASSERT) and (c == WHOLE) and (s_val >= TAU)
            pred.append(p)
            gold.append(g)
            if p != g:
                (fp if p else fn).append((b, cat, c, m, round(s_val, 3)))
        corr = sum(a == g for a, g in zip(pred, gold))
        lo, hi = wilson(corr, len(gold))
        print(f"端点{name}({len(ep)}对): {corr}/{len(ep)} = {corr/len(ep):.1%}  Wilson[{lo:.3f},{hi:.3f}]")
        print(f"  FP: {fp}")
        print(f"  FN: {fn}\n")
        results[name] = {"n": len(ep), "correct": corr, "acc": round(corr / len(ep), 4),
                         "wilson": [round(lo, 3), round(hi, 3)], "fp": fp, "fn": fn}

    # ── 旧 20 基准连续性
    bench_slots = ext.batch_extract([c["res"] for c in BENCHMARK], schema)
    bpred, bgold = [], []
    for c, sr in zip(BENCHMARK, bench_slots):
        sr = sr or {}
        c_, m_ = sr.get("完成度", NONE), sr.get("话术模态", QUESTION)
        s_val = sim(c["req"], c["res"])
        bpred.append((m_ == ASSERT) and (c_ == WHOLE) and (s_val >= TAU))
        bgold.append(c["gold"])
    bcorr = sum(a == g for a, g in zip(bpred, bgold))
    print(f"旧20基准(符号∧sim≥{TAU}): {bcorr}/20   (对照: BASE臂 18/20, 现行管线 17-18/20)")
    results["bench20"] = {"correct": bcorr}

    results["slot_quality"] = {"comp_acc": f"{comp_ok}/67", "mod_must_hit": f"{mod_ok}/{len(nonassert_must)}",
                               "comp_errors": comp_err, "slot_ms_per_pair": round(slot_ms, 1)}
    json.dump(results, open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"结果已写入 {RESULTS_PATH}")


if __name__ == "__main__":
    main()

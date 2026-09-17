# -*- coding: utf-8 -*-
"""research_v177_llm_zeroshot.py — 通用 LLM 零样本对比（用户提问的核心）

问题：**是不是"通用 LLM 也只能做到这个准确率"？**

设计：在**同一批**新采金标（maps/nc_gold_all.json，865 条可用）上，
同口径（句子级、三分类→完成/非完成二值）比较：

  1. v12.1（287M 微调 mDeBERTa，本项目最优）
  2. Qwen2.5-1.5B-Instruct（**1.5B 通用 LLM，零样本**，无任何微调）
  3. Qwen2.5-0.5B-Instruct（0.5B 通用 LLM，零样本）
  4. 零语义基线（数完成词）

度量：AUROC（阈值无关，与 §202/§209 同口径）+ 分层（按义务长度）。

判据（预注册）：
  · 若通用 LLM 零样本 AUROC ≥ v12.1 ⇒ "任务本身可解，只是我们的模型小"
  · 若显著低于 v12.1 ⇒ "该任务上通用 LLM 也不占优"
  · 若 ≈ 零语义基线 ⇒ "该任务无模型能力差异，纯粹是任务不可判"

只读：加载既有模型与金标，不训练、不改动产物。
"""
from __future__ import annotations
import json, os, re, sys

OUT = os.environ.get("OUT_JSON", "maps/llm_zeroshot.json")
LIMIT = int(os.environ.get("LIMIT", "0"))
QLEN = os.environ.get("QLEN", "1.5")
LABELS = ["义务闭合断言", "完成话术非本义务", "无完成相关"]
LEXW = ("完成", "已", "通过", "修好", "搞定", "done", "fixed", "passed", "全绿")

PROMPT = """你是严格的完成态判定员。给定**用户义务**和**候选句**，判断候选句是否构成对该义务的完成声明。

只输出一个词，不要解释：
- WHOLE：候选句覆盖义务的全部必要范围，且无保留
- PART：候选句是真实的完成声明，但只覆盖义务的一部分
- NO：候选句不含完成声明（是计划/进行中/问题陈述/观察/分析）

【义务】{ob}
【候选句】{sen}

只输出 WHOLE / PART / NO 之一："""


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(
        1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))


def main():
    gold = json.load(open("maps/nc_gold_all.json", encoding="utf-8"))
    bi = {}
    import glob
    for f in glob.glob("maps/nc*_batch_p*.json"):
        if "_A2" in f:
            continue
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            bi[it["pid"]] = it
    rows = []
    for pid, v in gold.items():
        it = bi.get(pid)
        if not it or v["label"] in ("AMBIGUOUS", "NOT_A_DELEGATION"):
            continue
        rows.append({"pid": pid, "obligation": it["obligation"][:300],
                     "sentence": it["sentence"][:200],
                     "y": 1 if v["label"] in ("WHOLE_CLOSE", "PART_CLOSE") else 0})
    if LIMIT:
        rows = rows[:LIMIT]
    print(f"[data] {len(rows)} 条，正类 {sum(r['y'] for r in rows)}")
    y = [r["y"] for r in rows]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snaps = {
        "1.5": "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "0.5": "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    }
    results = {}
    for tag, snap in snaps.items():
        if tag not in QLEN:
            continue
        path = os.path.expanduser(f"~/.cache/huggingface/hub/{snap}")
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        print(f"[load] Qwen2.5-{tag}B-Instruct ...")
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        mdl.eval()
        # 用 logits 打分而非生成（快且可做 AUROC）
        ids_whole = tok("WHOLE", add_special_tokens=False)["input_ids"][0]
        ids_part = tok("PART", add_special_tokens=False)["input_ids"][0]
        id_w = tok(" WHOLE", add_special_tokens=False)["input_ids"][-1]
        id_p = tok(" PART", add_special_tokens=False)["input_ids"][-1]
        xs = []
        for i, r in enumerate(rows):
            p = PROMPT.format(ob=r["obligation"], sen=r["sentence"])
            enc = tok(p, return_tensors="pt", truncation=True, max_length=768)
            with torch.no_grad():
                out = mdl(**enc).logits[0, -1, :]
            s = float(torch.logsumexp(torch.stack([out[id_w], out[id_p]]), dim=0))
            xs.append(s)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(rows)}")
        results[f"qwen25_{tag}b_zeroshot"] = xs

    # v12.1 对照
    try:
        from gliner2 import AutoExtractor, Schema
        dev = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
        m = AutoExtractor.from_pretrained("models/gliner25_closure_3class_v12_1/final",
                                          map_location=dev)
        sc = Schema().classification("闭合", labels=LABELS, multi_label=True,
                                     cls_threshold=0.0, class_act="softmax")
        xs = []
        for r in rows:
            res = m.extract(f"【义务】{r['obligation']}\n【完成句】{r['sentence']}",
                            sc, include_confidence=True).get("闭合")
            val = 0.0
            if isinstance(res, list):
                for it in res:
                    if isinstance(it, dict) and it.get("label") == LABELS[0]:
                        val = float(it.get("confidence") or 0.0)
            xs.append(val)
        results["v12.1"] = xs
    except Exception as e:
        print(f"[v12.1 skip] {e}")

    results["baseline_lex"] = [sum(1 for w in LEXW if w in r["sentence"]) for r in rows]

    pos = [i for i, t in enumerate(y) if t == 1]
    neg = [i for i, t in enumerate(y) if t == 0]
    rep = {"n": len(rows), "n_pos": len(pos), "auroc": {}}
    print(f"\n{'判定器':<26}{'AUROC':>8}")
    for k, sc in results.items():
        a = auc([sc[i] for i in pos], [sc[i] for i in neg])
        rep["auroc"][k] = a
        print(f"{k:<26}{(f'{a:.4f}' if a else 'n/a'):>8}")

    v = rep["auroc"].get("v12.1")
    q = rep["auroc"].get("qwen25_1.5b_zeroshot")
    b = rep["auroc"].get("baseline_lex")
    rep["verdict"] = {
        "v12.1": v, "qwen_1.5b": q, "baseline": b,
        "llm_beats_v121": (q is not None and v is not None and q > v),
        "llm_beats_baseline": (q is not None and b is not None and q > b),
        "note": ("通用 LLM 零样本已超过 v12.1 ⇒ 瓶颈是模型规模，任务本身可解"
                 if (q is not None and v is not None and q > v) else
                 "通用 LLM 零样本未超过 v12.1；"
                 + ("但仍超零语义基线 ⇒ 任务有可学信号，只是 287M 微调已接近其可达上限"
                    if (q is not None and b is not None and q > b) else
                    "且未超零语义基线 ⇒ 该任务上模型能力差异不显著")),
    }
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[verdict]", json.dumps(rep["verdict"], ensure_ascii=False, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

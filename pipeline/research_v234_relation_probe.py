# -*- coding: utf-8 -*-
"""pipeline/research_v234_relation_probe.py

回答"错题的共性是否需要加 LLM/交叉编码器管线"（2026-09-20）。

背景（maps/REDO_ALIGNMENT_STUDY_2026-09-19.md）：
三臂嵌入模型在端点 A（37 对，pos vs 近域负例）上准确率完全相同（70.3%），
且错误集高度重叠——嵌入余弦对"同域异任务"存在结构性盲区。
本质假设：该任务是"关系判断/蕴含"形态（该回复是否解除该义务），
单塔相似度只能量主题相近，量不出任务同一性。

本探针在同一冻结协议下测关系型模型（全部零样本，无微调）：
  - NLI 交叉编码器 cross-encoder/nli-deberta-v3-base（280M，预算内）
  - bge-reranker-base（280M，预算内）
  - Qwen2.5-0.5B-Instruct（490M，"LLM 管线"最小形态，确定性 YES/NO）
协议与 v233 一致：同一验证集（32 对模板族）选阈值并冻结 → 端点 A（37 对）→
端点 B（67 对）。Qwen 为确定性生成，不设阈值。
判读：
  - 若关系型模型显著突破 70.3% 且破解近域负例 → "加关系判断级联层"获实证支持；
  - 若同样失败 → 盲区是任务定义层的，换模型形态无效。
"""
import os, sys, json, math, time, torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM)

sys.path.insert(0, "pipeline")
from research_v233_alignment_eval import (EVAL_POS, EVAL_CROSS, EVAL_PARTIAL, EVAL_WAIT,
                                          EVAL_FAIL, wilson, pick_threshold)

SPLIT_PATH = "maps/alignment_heldout_split.json"
TRAIN_PATH = "maps/contrastive_strictly_heldout_train.json"
RESULTS_PATH = "maps/relation_probe_results.json"
DEVICE = "cpu"


def load_pairs():
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    triplets = json.load(open(TRAIN_PATH, encoding="utf-8"))["triplets"]
    val_pairs = []
    for i in split["val_idx"]:
        t = triplets[i]
        val_pairs.append((t["anchor"], t["pos"], 1))
        val_pairs.append((t["anchor"], t["neg"], 0))
    return val_pairs


class NLIProbe:
    name = "NLI-deberta-v3-base(280M)"

    def __init__(self):
        p = os.path.expanduser("~/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base")
        snap = os.path.join(p, "snapshots", os.listdir(os.path.join(p, "snapshots"))[0])
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(snap).to(DEVICE).eval()

    def score(self, req, res):
        inp = self.tok(res, f"用户关于“{req}”的要求已经完成。", truncation=True,
                       max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.mdl(**inp).logits, dim=-1)[0]
        return probs[1].item()  # entailment


class BGERerankProbe:
    name = "bge-reranker-base(280M)"

    def __init__(self):
        p = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-base")
        snap = os.path.join(p, "snapshots", os.listdir(os.path.join(p, "snapshots"))[0])
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(snap).to(DEVICE).eval()

    def score(self, req, res):
        inp = self.tok(f"用户需求：{req}", f"助手动作：{res}", truncation=True,
                       max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return torch.sigmoid(self.mdl(**inp).logits[0]).item()


class QwenProbe:
    name = "Qwen2.5-0.5B-Instruct(490M,零样本)"

    def __init__(self):
        p = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
        snap = os.path.join(p, "snapshots", os.listdir(os.path.join(p, "snapshots"))[0])
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.mdl = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32).to(DEVICE).eval()

    def score(self, req, res):
        msgs = [
            {"role": "system", "content": "你是严格的任务闭合判定器。判断【助手动作】是否真正完成了【用户请求】。完全解决输出YES；任务错配、只完成一部分、正在进行、或执行失败输出NO。只输出一个词。"},
            {"role": "user", "content": f"【用户请求】：{req}\n【助手动作】：{res}\n是否完成？(YES/NO)"},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.mdl.generate(**inp, max_new_tokens=4, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
        return 1.0 if "YES" in ans else 0.0


def evaluate(probe, val_pairs, endpointA, endpointB, use_threshold):
    t0 = time.time()
    val_s = [probe.score(r, s) for r, s, _ in val_pairs]
    val_l = [l for _, _, l in val_pairs]
    val_ms = (time.time() - t0) * 1000 / len(val_pairs)
    if use_threshold:
        tau, val_acc = pick_threshold(None, val_s, val_l)
    else:
        tau, val_acc = 0.5, sum((s >= 0.5) == l for s, l in zip(val_s, val_l)) / len(val_l)
    res = {"val_tau": round(tau, 3), "val_acc": round(val_acc, 3),
           "val_ms_per_pair": round(val_ms, 1)}
    for ep_name, ep in (("A", endpointA), ("B", endpointB)):
        t0 = time.time()
        ss = [probe.score(r, s) for _, r, s, _ in ep]
        ms = (time.time() - t0) * 1000 / len(ep)
        pred = [s >= tau for s in ss]
        gold = [g for _, _, _, g in ep]
        corr = sum(p == g for p, g in zip(pred, gold))
        lo, hi = wilson(corr, len(gold))
        fp = [(ep[i][0], round(ss[i], 3)) for i in range(len(ep)) if pred[i] and not gold[i]]
        fn = [(ep[i][0], round(ss[i], 3)) for i in range(len(ep)) if not pred[i] and gold[i]]
        res[ep_name] = {"n": len(gold), "correct": corr,
                        "acc": round(corr / len(gold), 4), "wilson": [round(lo, 3), round(hi, 3)],
                        "ms_per_pair": round(ms, 1), "fp": fp, "fn": fn}
        print(f"  端点{ep_name}: {corr}/{len(gold)} = {corr/len(gold):.1%} "
              f"Wilson[{lo:.3f},{hi:.3f}]  {ms:.0f}ms/对")
        print(f"    FP: {fp}\n    FN: {fn}")
    return res


def main():
    val_pairs = load_pairs()
    endpointA = [(b, r, s, 1) for b, r, s in EVAL_POS] + [(b, r, s, 0) for b, r, s in EVAL_CROSS]
    endpointB = ([(b, r, s, 1) for b, r, s in EVAL_POS]
                 + [(b, r, s, 0) for b, r, s in EVAL_CROSS + EVAL_PARTIAL + EVAL_WAIT + EVAL_FAIL])
    print(f"验证 {len(val_pairs)} 对 | 端点A 37 对 | 端点B 67 对\n")

    results = {}
    for cls, use_thr in [(NLIProbe, True), (BGERerankProbe, True), (QwenProbe, False)]:
        print(f"===== {cls.name} =====")
        try:
            probe = cls()
        except Exception as e:
            print("  加载失败:", e)
            continue
        results[cls.name] = evaluate(probe, val_pairs, endpointA, endpointB, use_thr)
        del probe
        json.dump(results, open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {RESULTS_PATH}")


if __name__ == "__main__":
    main()

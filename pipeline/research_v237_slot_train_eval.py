# -*- coding: utf-8 -*-
"""pipeline/research_v237_slot_train_eval.py

训练槽模型（MiniLM 117M + 完成度4值/模态3值双头）并在冻结集上验收
（选项 1 第二步，2026-09-20）。

预注册验收（在接触冻结测试前写死，τ 与判定规则沿用 v235 冻结值，禁止改动）：
  A1 端点B(67对) ≥ 54/67 (80%)
  A2 端点A(37对) ≥ 25/37 (67%)   [基线 70.3%，容许 ≤3pp 槽噪声损耗]
  A3 完成度4值(67对) ≥ 60/67 (90%)
  A4 旧20基准 ≥ 17/20            [不劣于基线-1]

服务形态：模态/完成度由本模型输出（零手写正则）；match 站 = 原始基座嵌入
（与槽模型分离的副本）+ 冻结 τ=0.265。

运行：.venv/Scripts/python.exe pipeline/research_v237_slot_train_eval.py
"""
import os, sys, json, random, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "pipeline")
from research_v233_alignment_eval import (EVAL_POS, EVAL_CROSS, EVAL_PARTIAL,
                                          EVAL_WAIT, EVAL_FAIL, wilson)
from eval_semantic_matcher_benchmark import BENCHMARK

SEED = 20260920
DATA = "maps/slot_train_data.json"
OUT_DIR = "models/minilm_slot_heads_v1"
RESULTS = "maps/slot_model_results.json"
TAU = 0.265
EPOCHS = 12
BATCH = 16
MOD_CLASSES = ["assert", "promise", "question"]
COMP_CLASSES = ["whole", "partial", "none", "failed"]

MUST_NONASSERT = {"w03", "w07", "w10"}


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)


def mean_pool(last, mask):
    m = mask.unsqueeze(-1).expand(last.size()).float()
    return torch.sum(last * m, 1) / torch.clamp(m.sum(1), min=1e-9)


class SlotModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.mod_head = nn.Linear(h, len(MOD_CLASSES))
        self.comp_head = nn.Linear(h, len(COMP_CLASSES))

    def forward(self, inp):
        out = self.encoder(**inp)
        emb = F.normalize(mean_pool(out.last_hidden_state, inp["attention_mask"]), p=2, dim=1)
        return self.mod_head(emb), self.comp_head(emb)


def main():
    set_seed(SEED)
    data = json.load(open(DATA, encoding="utf-8"))["items"]
    train = [it for it in data if it["split"] == "train"]
    val = [it for it in data if it["split"] == "val"]
    print(f"train {len(train)} / val {len(val)}")

    cache = os.path.expanduser("~/.cache/huggingface/hub")
    d = [x for x in os.listdir(cache) if "MiniLM-L12-v2" in x][0]
    snap = os.path.join(cache, d, "snapshots", os.listdir(os.path.join(cache, d, "snapshots"))[0])
    tok = AutoTokenizer.from_pretrained(snap)
    enc = AutoModel.from_pretrained(snap)
    model = SlotModel(enc)

    optim = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": list(model.mod_head.parameters()) + list(model.comp_head.parameters()), "lr": 1e-3},
    ], weight_decay=0.01)

    def encode_batch(texts):
        inp = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
        return inp

    print("训练中（CPU）...")
    t0 = time.time()
    n_steps = 0
    for ep in range(EPOCHS):
        order = list(range(len(train)))
        random.Random(SEED + ep).shuffle(order)
        tot, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            batch = [train[j] for j in order[i:i + BATCH]]
            inp = encode_batch([b["text"] for b in batch])
            mod_logits, comp_logits = model(inp)
            mod_y = torch.tensor([MOD_CLASSES.index(b["modality"]) for b in batch])
            loss = F.cross_entropy(mod_logits, mod_y)
            mask = [k for k, b in enumerate(batch) if b["completeness"] is not None]
            if mask:
                comp_y = torch.tensor([COMP_CLASSES.index(batch[k]["completeness"]) for k in mask])
                loss = loss + F.cross_entropy(comp_logits[mask], comp_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item()
            nb += 1
            n_steps += 1
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/nb:.4f}  ({time.time()-t0:.0f}s)")
    print(f"训练完成 {time.time()-t0:.0f}s, {n_steps} steps")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"mod_head": model.mod_head.state_dict(),
                "comp_head": model.comp_head.state_dict()}, os.path.join(OUT_DIR, "heads.pt"))
    model.encoder.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    model.eval()

    def predict(texts):
        preds_m, preds_c = [], []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                inp = tok(texts[i:i + 32], padding=True, truncation=True, max_length=96,
                          return_tensors="pt")
                ml, cl = model(inp)
                preds_m += [MOD_CLASSES[k] for k in ml.argmax(-1).tolist()]
                preds_c += [COMP_CLASSES[k] for k in cl.argmax(-1).tolist()]
        return preds_m, preds_c

    results = {}
    # ── 槽验证集（合成，组感知留出）
    vm, vc = predict([x["text"] for x in val])
    ma = sum(a == b["modality"] for a, b in zip(vm, val)) / len(val)
    vcomp = [(a, b) for a, b in zip(vc, val) if b["completeness"]]
    ca = sum(a == b["completeness"] for a, b in vcomp) / len(vcomp)
    print(f"\n[槽val] 模态 {ma:.1%}  完成度 {ca:.1%}")
    results["slot_val"] = {"modality": round(ma, 4), "completeness": round(ca, 4)}

    # ── 冻结 67 对
    all67 = ([(b, r, s, 1, "pos") for b, r, s in EVAL_POS]
             + [(b, r, s, 0, "cross") for b, r, s in EVAL_CROSS]
             + [(b, r, s, 0, "partial") for b, r, s in EVAL_PARTIAL]
             + [(b, r, s, 0, "wait") for b, r, s in EVAL_WAIT]
             + [(b, r, s, 0, "fail") for b, r, s in EVAL_FAIL])
    fm, fc = predict([s for _, _, s, _, _ in all67])
    comp4_true = {"pos": "whole", "cross": "whole", "partial": "partial",
                  "wait": "none", "fail": "failed"}
    c4 = sum(pp == comp4_true[cat] for pp, (_, _, _, _, cat) in zip(fc, all67))
    ma_ok = sum(mm != "assert" for mm, (bid, *_ ) in zip(fm, all67) if bid in MUST_NONASSERT)
    assert_ok = sum(mm == "assert" for mm, (bid, *_ ) in zip(fm, all67)
                    if cat_true(bid, all67) in ("pos", "cross", "partial", "fail"))
    print(f"[槽@冻结67] 完成度4值 {c4}/67 | 承诺/提问必拦 {ma_ok}/3 | 陈述保持 {assert_ok}/50")
    comp_err = [(bid, pp, comp4_true[cat]) for pp, (bid, _, _, _, cat) in zip(fc, all67)
                if pp != comp4_true[cat]]
    print(f"  完成度误判: {comp_err}")
    results["slot_frozen"] = {"comp4": f"{c4}/67", "must_nonassert": f"{ma_ok}/3",
                              "assert_keep": f"{assert_ok}/50", "comp_errors": comp_err}

    # ── match 站：原始基座（独立副本）+ 冻结 τ
    base = AutoModel.from_pretrained(snap).eval()

    def sim(a, b):
        def e(t):
            inp = tok(t, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                o = base(**inp)
                m = inp["attention_mask"].unsqueeze(-1).expand(o.last_hidden_state.size()).float()
                return F.normalize(torch.sum(o.last_hidden_state * m, 1)
                                   / torch.clamp(m.sum(1), min=1e-9), p=2, dim=1)
        return F.cosine_similarity(e(a), e(b)).item()

    slot_by_id = {bid: (mm, pp) for (bid, *_), mm, pp in zip(all67, fm, fc)}
    verdicts = {}
    for name, ep in (("A", [x for x in all67 if x[4] in ("pos", "cross")]), ("B", all67)):
        pred, gold, fp, fn = [], [], [], []
        for bid, r, s, g, cat in ep:
            mm, cc = slot_by_id[bid]
            p = (mm == "assert") and (cc == "whole") and (sim(r, s) >= TAU)
            pred.append(p)
            gold.append(g)
            if p != g:
                (fp if p else fn).append((bid, cat, cc, mm, round(sim(r, s), 3)))
        corr = sum(a == g for a, g in zip(pred, gold))
        lo, hi = wilson(corr, len(gold))
        print(f"\n端点{name}: {corr}/{len(ep)} = {corr/len(ep):.1%}  Wilson[{lo:.3f},{hi:.3f}]")
        print(f"  FP: {fp}\n  FN: {fn}")
        verdicts[name] = {"n": len(ep), "correct": corr, "acc": round(corr / len(ep), 4),
                          "wilson": [round(lo, 3), round(hi, 3)], "fp": fp, "fn": fn}

    bm, bc = predict([c["res"] for c in BENCHMARK])
    bpred = [(mm == "assert") and (cc == "whole") and (sim(c["req"], c["res"]) >= TAU)
             for c, mm, cc in zip(BENCHMARK, bm, bc)]
    bcorr = sum(a == c["gold"] for a, c in zip(bpred, BENCHMARK))
    print(f"\n旧20基准: {bcorr}/20")
    verdicts["bench20"] = {"correct": bcorr}

    # ── 预注册验收
    acc = {
        "A1_B≥80%": verdicts["B"]["correct"] >= 54,
        "A2_A≥67%": verdicts["A"]["correct"] >= 25,
        "A3_完成度≥90%": c4 >= 60,
        "A4_bench20≥17": bcorr >= 17,
    }
    print("\n预注册验收:", acc, "=>", "PASS" if all(acc.values()) else "FAIL")
    results.update({"endpoints": verdicts, "acceptance": acc})
    json.dump(results, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {RESULTS}")


def cat_true(bid, all67):
    for x in all67:
        if x[0] == bid:
            return x[4]
    return None


if __name__ == "__main__":
    main()

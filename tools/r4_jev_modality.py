# -*- coding: utf-8 -*-
"""R4/Track2: Jev-4B + LoRA modality.assertive 重训（XPU only，断点续跑，塌缩监护）。

底座选型（2026-10-03 实查官方 wheel neohorse_decision 1.0.0 源码）：
- 服务 API DecisionEngine 不支持 LoRA/参数注入（_inference.load_bundle 只读
  merged bundle；predict 全程 torch.inference_mode + 串行锁，无 adapter 接口）。
- 但同一 wheel 附带训练构造器 DecisionModel（_vendor.model）：构造参数原生
  支持 lora（peft FEATURE_EXTRACTION）、lora_targets（hybrid 感知，含 DeltaNet
  in_proj_qkv/z/a/b + out_proj）；model_manifest.json "lora_scale": 1.0 证实
  Jev 权重本身即由 LoRA-train+merge 经此类产出。
- Choice 三类（assert/promise/question）= PointerHead 在 <decide> 位的 3 路
  logits，可直接训练；推理路径（hybrid → forward_rows_batch 行形式）与
  DecisionEngine.predict 完全同一条代码路径（_inference.predict → model.probs
  → forward → hybrid → forward_rows_batch），故与 jev-eval zero-shot 基线
  （calib 0.652/0.849/F1 0.738 @ t0.75，正负中位差 0.788）严格可比。
- 因此选首选路线：Jev-4B merged bundle（backbone bf16 + pointer head fp32）
  之上挂 peft LoRA 训练；无需退化 NeoHorse-1-4B answer-token 路线（未下载）。

XPU 稳定性（2026-10-03 实测，results/_probe_mem.py + smoke）：
- 参考 DeltaNet 内核（chunk_gated_delta_rule 回退实现）在 backward 下
  B=4 即 native segfault（P8 同类事故；0.8B r3 的 batch 4 在 4B 上不安全）。
- 梯度检查点（use_reentrant=False）+ batch=2 + 前向 token 预算 ≤2048 实测
  稳定（B=2 L=700 step 3.8s，峰值显存 14.2/15.56GB 设备-wide）。
- vision tower（0.31B）加载前即剥离，不占显存；单请求显存口径仍 ~12GB。

数据（硬约束）：训练 = frozen dev 300 + modality_extension_r4 180（对立对按
task_id 同批）；calib 150 仅验证/早停/选阈值/门禁；test split 禁碰。

预登记（未事后修改）：
- LoRA r=16 alpha=32 dropout=0.05，target = 官方 hybrid 感知全集合
  （q/k/v/o/gate/up/down + in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj），
  head（PointerHead fp32）可训练；
- epoch<=3，eval every 50 step（calib 150 全量：t0.5 + 扫描 + 正负中位差），
  早停 = 扫描 F1 连续 2 次不升；塌缩监护 = 正负中位差 <0.05 立即停（R3c 教训）；
- loss = 加权 BCE on p(assert)（choice softmax 下 assert 概率，即门禁判定量的
  直接优化目标；dev 负例无 promise/question 三类区分，不引入臆造标签）；
- pos_weight = clamp(n_neg/n_pos, 1, 8) 按文件实际计数重算
  （dev300+r4180 = 145 正/335 负 → 2.3103；派发单 "239正/419负" 与任何真实
  split 均不符，如实以数据文件为准）；
- lr 5e-5 AdamW（R3c 减半纪律），warmup 10，grad clip 1.0，batch 2（token
  预算打包，见上），seed 20261003；
- parity 门：pre-train eval（LoRA B=0 时严格等于底座）必须复现 jev-eval
  zero-shot（scan F1 0.738±0.02 且 t=0.75 格子 P/R = 0.652/0.849±0.02），
  不符即中止不训练。

模式：--mode smoke（冒烟：加载+12条前向+1个训练步+ckpt 往返）/ train /
eval（best checkpoint 上 calib 150 全量出 preds.jsonl + metrics_calib.json，
断点续跑）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "src"))

BUNDLE = Path("C:/Users/Nuctori/.cache/huggingface/hub/"
              "models--TokenRhythm--NeoHorse-Jev-4B/snapshots/"
              "434cb21d3a994a953d3ae5788405fcb2c4970554")
MODEL_ID = "TokenRhythm/NeoHorse-Jev-4B"
MODEL_REVISION = "434cb21d3a994a953d3ae5788405fcb2c4970554"

DEV_TEXTS = ROOT / "data/semantic_frozen_v1/frozen_texts_dev.jsonl"
DEV_LABELS = ROOT / "data/semantic_frozen_v1/frozen_labels_dev.jsonl"
R4_TEXTS = ROOT / "data/semantic_frozen_v1/modality_extension_r4.jsonl"
R4_LABELS = ROOT / "data/semantic_frozen_v1/modality_extension_r4_labels.jsonl"
CALIB_TEXTS = ROOT / "data/semantic_frozen_v1/frozen_texts_calib.jsonl"
CALIB_LABELS = ROOT / "data/semantic_frozen_v1/frozen_labels_calib.jsonl"

OUT_DIR = ROOT / "models/semantic_lora/modality-jev-v1"
RESULTS_DIR = ROOT / "results/semantic-runtime/modality-jev-v1"

PATTERN = "modality.assertive"
# 与 src/moonbow/semantic/backends/jev.py jev-template-v1 严格同源（直接复用常量）
from moonbow.semantic.backends.jev import _JEV_TEMPLATES, _TEXT_WINDOW  # noqa: E402

TPL = _JEV_TEMPLATES[PATTERN]
CRIT_ORDER = list(TPL["criteria"].keys())          # ["assert","promise","question"]
ASSERT_IDX = CRIT_ORDER.index("assert")            # 0
OPTIONS = [f"{k}: {v}" for k, v in TPL["criteria"].items()]  # to_record.option_text 口径
INSTR = TPL["instructions"]

MAX_STATE = 4096    # jev.py 的 engine 预算
MAX_BRANCH = 8192
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                  "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                  "in_proj_a", "in_proj_b", "out_proj"]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, WARMUP, EPOCHS = 5e-5, 10, 3
BATCH, EVAL_EVERY, PATIENCE = 2, 50, 2
TRAIN_TOKEN_BUDGET = 2048   # 单次前向 B×L_max 上限（XPU 实测稳定域，见 docstring）
EVAL_TOKEN_BUDGET = 6144    # no_grad 推理可放宽（B<=8）
GAP_FLOOR = 0.05           # R3c 塌缩监护
SEED = 20261003
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]
TIME_BUDGET_S = 2.0 * 3600  # 训练墙钟预算（总上限 2.5h，留评测/报告余量）
# parity 门（jev-eval zero-shot，calib 150）
PARITY = {"t": 0.75, "p": 0.652, "r": 0.849, "f1": 0.738, "tol": 0.02}

import torch  # noqa: E402

if not torch.xpu.is_available():
    print("FATAL: XPU unavailable; CPU training forbidden by hard constraint",
          file=sys.stderr)
    sys.exit(2)
DEVICE = torch.device("xpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def vram_gb():
    free, total = torch.xpu.mem_get_info()
    return round((total - free) / 2**30, 2), round(total / 2**30, 2)


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 模型构建：官方 wheel 训练构造器 DecisionModel 的手工装配（绕过其 __init__，
# 与 _inference.load_bundle 同一装配方式），再挂 peft LoRA + 梯度检查点。
# ---------------------------------------------------------------------------

def build_bundle_model(lora=True, adapter_dir=None):
    """返回 (tok, model, info)；model 为 duck-typed DecisionModel：
    .lm（peft 包装与否的 Qwen3_5TextModel，bf16 底座 + 梯度检查点）、
    .head（fp32 PointerHead）、.hybrid=True、.option_isolation=False、
    .pad_id、.device。vision tower 加载前剥离（不参与判定，省 0.6GB）。"""
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    from neohorse_decision._vendor.model import DecisionModel, PointerHead

    manifest = json.loads((BUNDLE / "model_manifest.json").read_text())
    assert manifest["backbone_class"] == "Qwen3_5Model"
    assert manifest["hybrid"] is True and manifest["option_isolation"] is False

    tok = AutoTokenizer.from_pretrained(str(BUNDLE / "tokenizer"),
                                        local_files_only=True)
    t0 = time.time()
    torch.set_num_threads(4)
    backbone = AutoModel.from_pretrained(
        str(BUNDLE / "backbone"), dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    assert type(backbone).__name__ == "Qwen3_5Model", type(backbone).__name__
    backbone.visual = None          # vision tower 不上 GPU（仅语言模型参与判定）
    backbone = backbone.to(device=DEVICE, dtype=torch.bfloat16)
    torch.xpu.empty_cache()

    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.lm = backbone.language_model          # Qwen3_5TextModel（load_bundle 同）
    hidden = model.lm.config.hidden_size
    model.head = PointerHead(hidden, dp=manifest["head_dim"])
    model.head.load_state_dict(load_file(str(BUNDLE / "pointer_head.safetensors")))
    model.head = model.head.to(device=DEVICE, dtype=torch.float32)
    model.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    layer_types = set(getattr(model.lm.config, "layer_types", None) or [])
    model.hybrid = "linear_attention" in layer_types
    assert model.hybrid is True
    model.option_isolation = False
    model.device = DEVICE
    model.eval()

    info = {"load_seconds": round(time.time() - t0, 2),
            "vram_after_load_gb": vram_gb()[0], "n_lora_params": 0}
    if lora:
        if adapter_dir:
            from peft import PeftModel
            model.lm = PeftModel.from_pretrained(model.lm, str(adapter_dir),
                                                 is_trainable=True)
        else:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=LORA_R,
                              lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                              target_modules=TARGET_MODULES)
            model.lm = get_peft_model(model.lm, lcfg)
            for _n, _p in model.lm.named_parameters():
                if _p.requires_grad:
                    _p.data = _p.data.float()   # r3 纪律：LoRA 升 fp32
            info["n_lora_params"] = sum(
                p.numel() for p in model.lm.parameters() if p.requires_grad)
        # XPU 实测（results/_probe_mem.py）：参考 DeltaNet backward 在无检查点
        # 时 B=4 native segfault；梯度检查点 + batch 2 稳定。
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, model, info


# ---------------------------------------------------------------------------
# encode（vendored，注入防线与 DecisionEngine 完全一致：user text 仅进 state）
# ---------------------------------------------------------------------------

def encode_row(tok, text):
    from neohorse_decision._vendor.model import encode as jev_encode
    clipped = False
    if len(text) > _TEXT_WINDOW:
        text, clipped = text[:_TEXT_WINDOW], True
    rec = {"state": text,
           "questions": [{"instr": INSTR, "options": OPTIONS, "label": 0}]}
    enc = jev_encode(tok, rec, strict=True, max_state=MAX_STATE,
                     max_branch=MAX_BRANCH, option_isolation=False)
    enc["_truncated"] = clipped
    return enc


def row_len(enc):
    return len(enc["ids"])


def bucket_len(n, bucket=128):
    return -(-n // bucket) * bucket


def pad_chunk(encs_list, pad_id=0):
    """整批 pad 到同一 128 bucket 长度：固定形状集合，避免显存分配器碎片
    累积（首跑 step10 15.14/15.56GB segfault 的主因）。pad 追加在行尾，
    rows_of() 取 decide 为行尾，pad 不进入任何计算（官方行路径保证）。"""
    L = bucket_len(max(row_len(e) for e in encs_list))
    out = []
    for e in encs_list:
        add = L - len(e["ids"])
        if add == 0:
            out.append(e)
            continue
        out.append({**e, "ids": e["ids"] + [pad_id] * add,
                    "seg": e["seg"] + [-1] * add,
                    "pos": e["pos"] + [0] * add,
                    "opt": e["opt"] + [-1] * add})
    return out


def pack_by_budget(encs, indices, budget, max_bs):
    """按前向 token 预算打包（bucket 后代价 ~ B×L_max_bucket）；单行永不拆。"""
    batches, cur, cur_max = [], [], 0
    for i in indices:
        L = bucket_len(row_len(encs[i]))
        new_max = max(cur_max, L)
        if cur and (len(cur) >= max_bs or (len(cur) + 1) * new_max > budget):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def p_assert_batch(tok, model, texts_or_encs, budget=EVAL_TOKEN_BUDGET,
                   max_bs=8):
    """打分：返回每条 p(assert)（choice softmax 下 assert 概率）。"""
    encs = texts_or_encs
    if encs and isinstance(encs[0], str):
        encs = [encode_row(tok, t) for t in encs]
    model.eval()
    order = list(range(len(encs)))
    out = [0.0] * len(encs)
    for chunk in pack_by_budget(encs, order, budget, max_bs):
        logits = model.forward_rows_batch(pad_chunk([encs[i] for i in chunk],
                                                    model.pad_id))
        for i, lq in zip(chunk, logits):
            p = torch.softmax(lq[0].float(), -1)
            out[i] = float(p[ASSERT_IDX])
    torch.xpu.empty_cache()
    return out


def compute_metrics(scores, ys):
    m = {}
    tp = sum(1 for s, y in zip(scores, ys) if s >= 0.5 and y == 1)
    fp = sum(1 for s, y in zip(scores, ys) if s >= 0.5 and y == 0)
    fn = sum(1 for s, y in zip(scores, ys) if s < 0.5 and y == 1)
    p05, r05, f05 = prf(tp, fp, fn)
    m["t0.5"] = {"threshold": 0.5, "precision": round(p05, 4),
                 "recall": round(r05, 4), "f1": round(f05, 4),
                 "tp": tp, "fp": fp, "fn": fn}
    best = None
    gate = None
    for th in THRESHOLDS:
        tp = sum(1 for s, y in zip(scores, ys) if s >= th and y == 1)
        fp = sum(1 for s, y in zip(scores, ys) if s >= th and y == 0)
        fn = sum(1 for s, y in zip(scores, ys) if s < th and y == 1)
        p, rc, f1 = prf(tp, fp, fn)
        if best is None or f1 > best["f1"]:
            best = {"threshold": th, "precision": round(p, 4),
                    "recall": round(rc, 4), "f1": round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn}
        if gate is None and p >= 0.85 and rc >= 0.80:
            gate = {"threshold": th, "precision": round(p, 4),
                    "recall": round(rc, 4), "f1": round(f1, 4)}
    m["best_scan"] = best
    m["gate_pass_cell"] = gate   # 预登记门禁：存在阈值 P>=0.85 且 R>=0.80
    pos = sorted(s for s, y in zip(scores, ys) if y == 1)
    neg = sorted(s for s, y in zip(scores, ys) if y == 0)
    pm = pos[len(pos) // 2] if pos else None
    nm = neg[len(neg) // 2] if neg else None
    m["pos_median"] = round(pm, 4) if pm is not None else None
    m["neg_median"] = round(nm, 4) if nm is not None else None
    m["median_gap"] = round(pm - nm, 4) \
        if pm is not None and nm is not None else None
    # ECE(15bin equal-width) / Brier
    n = len(scores)
    ece = 0.0
    for b in range(15):
        lo, hi = b / 15, (b + 1) / 15
        if b < 14:
            idx = [i for i, s in enumerate(scores) if lo <= s < hi]
        else:
            idx = [i for i, s in enumerate(scores) if lo <= s <= hi]
        if idx:
            frac_pos = sum(ys[i] for i in idx) / len(idx)
            conf = sum(scores[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(frac_pos - conf)
    brier = sum((s - y) ** 2 for s, y in zip(scores, ys)) / n
    m["ece_15bin"] = round(ece, 4)
    m["brier"] = round(brier, 4)
    return m


def train_step(model, batch_encs, ys, pos_weight, opt):
    """一个 optimizer step（整批 pad 到 128 bucket 固定形状）。返回 loss；
    失败抛 RuntimeError。"""
    logits = model.forward_rows_batch(pad_chunk(batch_encs, model.pad_id))
    picked = torch.stack([torch.log_softmax(lq[0].float(), -1)[ASSERT_IDX]
                          for lq in logits]).exp().clamp(1e-6, 1 - 1e-6)
    loss = (-(pos_weight * ys * picked.log()
              + (1 - ys) * (1 - picked).log())).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def build_rows():
    rows = []
    for texts_path, labels_path in ((DEV_TEXTS, DEV_LABELS),
                                    (R4_TEXTS, R4_LABELS)):
        texts = load_jsonl(texts_path)
        labels = {l["id"]: l["labels"] for l in load_jsonl(labels_path)}
        for t in texts:
            y = labels[t["id"]][PATTERN]
            rows.append({"id": t["id"], "task_id": t["task_id"],
                         "text": t["text"], "y": 1 if y else 0})
    calib_texts = load_jsonl(CALIB_TEXTS)
    calib_labels = {l["id"]: l["labels"] for l in load_jsonl(CALIB_LABELS)}
    calib = [{"id": t["id"], "text": t["text"],
              "y": 1 if calib_labels[t["id"]][PATTERN] else 0}
             for t in calib_texts]
    return rows, calib


def order_for_epoch(rows, ep):
    """task_id 分组同批（对立对/改写对不被打散），组间按 epoch 洗牌。"""
    rng = random.Random(SEED + ep)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["task_id"], []).append(i)
    keys = sorted(groups)
    rng.shuffle(keys)
    return [i for k in keys for i in groups[k]]


def append_log(out_dir, rec):
    with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_ckpt(model, ck_dir):
    model.lm.save_pretrained(str(ck_dir))            # peft adapter
    torch.save(model.head.state_dict(), str(ck_dir / "head.pt"))


# ---------------------------------------------------------------------------
# 模式实现
# ---------------------------------------------------------------------------

def cmd_smoke(args):
    tok, model, info = build_bundle_model(lora=True)
    print("bundle+LoRA loaded:", json.dumps(info), flush=True)
    rows, calib = build_rows()
    n_pos = sum(r["y"] for r in rows)
    print(f"train n={len(rows)} pos={n_pos} neg={len(rows) - n_pos} "
          f"pos_weight={min(8.0, max(1.0, (len(rows) - n_pos) / n_pos)):.4f}",
          flush=True)
    t0 = time.time()
    scores = p_assert_batch(tok, model, [r["text"] for r in calib[:12]],
                            budget=2048, max_bs=4)
    print(f"12-row pre-train forward: {time.time() - t0:.1f}s "
          f"scores={[round(s, 3) for s in scores]}", flush=True)
    # 1 个训练步（含 backward，token 预算打包）
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    model.lm.train()
    encs = [encode_row(tok, r["text"]) for r in rows]
    batch = pack_by_budget(encs, list(range(4)), TRAIN_TOKEN_BUDGET, BATCH)[0]
    ys = torch.tensor([rows[i]["y"] for i in batch], dtype=torch.float32,
                      device=DEVICE)
    t0 = time.time()
    loss = train_step(model, [encs[i] for i in batch], ys, 2.31, opt)
    print(f"1 train step (rows={batch}): {time.time() - t0:.1f}s "
          f"loss={loss:.4f} vram={vram_gb()}", flush=True)
    # checkpoint 保存/恢复往返
    ck = OUT_DIR / "_smoke_ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    save_ckpt(model, ck)
    sz = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
    print(f"ckpt saved {sz / 2**20:.1f}MB -> {ck}", flush=True)
    print("SMOKE OK", flush=True)


def cmd_train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "train_log.jsonl"
    tok, model, info = build_bundle_model(lora=True)
    print("bundle+LoRA loaded:", json.dumps(info), flush=True)

    rows, calib = build_rows()
    n_pos = sum(r["y"] for r in rows)
    n_neg = len(rows) - n_pos
    pos_weight = min(8.0, max(1.0, n_neg / max(n_pos, 1)))
    print(f"train n={len(rows)} pos={n_pos} neg={n_neg} "
          f"pos_weight={pos_weight:.4f}; calib n={len(calib)} "
          f"(pos={sum(r['y'] for r in calib)})", flush=True)
    calib_texts = [r["text"] for r in calib]
    calib_ys = [r["y"] for r in calib]

    # 一次性编码缓存（训练行与 calib 行）
    encs = [encode_row(tok, r["text"]) for r in rows]
    lens = [row_len(e) for e in encs]
    print(f"encoded rows: len p50={sorted(lens)[len(lens)//2]} "
          f"max={max(lens)}", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)

    def lr_at(step):
        return LR * (step + 1) / WARMUP if step < WARMUP else LR

    # ---- 断点续跑（按已消费样本数快进，重打包保证每样本每 epoch 恰一次）----
    start_step = start_consumed = 0
    best_f1, bad_evals = -1.0, 0
    if log_path.exists():
        last = None
        for l in open(log_path, encoding="utf-8"):
            if not l.strip():
                continue
            e = json.loads(l)
            if e.get("event") in ("eval", "final"):
                last = e
        if last and last.get("step"):
            start_step = last["step"]
            start_consumed = last.get("consumed", start_step * BATCH)
            best_f1 = last.get("best_f1", -1.0)
            bad_evals = last.get("bad_evals", 0)
            ck = OUT_DIR / f"checkpoint_{start_step:05d}"
            tok, model, info = build_bundle_model(lora=True, adapter_dir=ck)
            model.head.load_state_dict(
                torch.load(str(ck / "head.pt"), weights_only=True))
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LR, weight_decay=0.01)
            print(f"resume from step {start_step} consumed={start_consumed} "
                  f"(best_f1={best_f1:.4f} bad_evals={bad_evals})", flush=True)

    # ---- zero-shot 参考评估 + parity 门 ----
    t0 = time.time()
    pre_scores = p_assert_batch(tok, model, calib_texts)
    pre = compute_metrics(pre_scores, calib_ys)
    print(f"pre-train calib ({time.time() - t0:.0f}s): "
          f"scan={pre['best_scan']} t0.5={pre['t0.5']} "
          f"gap={pre['median_gap']} ece={pre['ece_15bin']}", flush=True)
    if start_step == 0:
        # parity 门：LoRA B=0 时必须复现 jev-eval zero-shot
        tp = sum(1 for s, y in zip(pre_scores, calib_ys)
                 if s >= PARITY["t"] and y == 1)
        fp = sum(1 for s, y in zip(pre_scores, calib_ys)
                 if s >= PARITY["t"] and y == 0)
        fn = sum(1 for s, y in zip(pre_scores, calib_ys)
                 if s < PARITY["t"] and y == 1)
        p, rc, _ = prf(tp, fp, fn)
        cell = (round(p, 4), round(rc, 4))
        ok = (abs(pre["best_scan"]["f1"] - PARITY["f1"]) <= PARITY["tol"]
              and abs(cell[0] - PARITY["p"]) <= PARITY["tol"]
              and abs(cell[1] - PARITY["r"]) <= PARITY["tol"])
        append_log(OUT_DIR, {"step": 0, "event": "pre_train", "calib": pre,
                             "parity_cell_t075": list(cell),
                             "parity_ok": ok,
                             "vram_gb": vram_gb()[0], "ts": now_ts()})
        if not ok:
            print(f"PARITY GATE FAILED: scan_f1={pre['best_scan']['f1']} "
                  f"(expect {PARITY['f1']}±{PARITY['tol']}), t0.75 cell="
                  f"{cell} (expect {PARITY['p']}/{PARITY['r']}±"
                  f"{PARITY['tol']}) — abort, forward path deviates from "
                  f"DecisionEngine", flush=True)
            return 3
        print(f"PARITY OK: t0.75 cell={cell} "
              f"scan_f1={pre['best_scan']['f1']}", flush=True)
        best_f1 = pre["best_scan"]["f1"]

    stop_reason = None
    losses = []
    model.lm.train()
    global_step = start_step
    consumed = start_consumed
    t_start = time.time()

    def run_eval(ep):
        nonlocal best_f1, bad_evals, global_step, consumed, stop_reason
        scores = p_assert_batch(tok, model, calib_texts)
        m = compute_metrics(scores, calib_ys)
        f1 = m["best_scan"]["f1"]
        improved = f1 > best_f1
        bad_evals = 0 if improved else bad_evals + 1
        ck = OUT_DIR / f"checkpoint_{global_step:05d}"
        save_ckpt(model, ck)
        if improved:
            save_ckpt(model, OUT_DIR / "best")
            best_f1 = f1
        append_log(OUT_DIR, {"step": global_step, "event": "eval",
                             "epoch": ep, "calib": m, "consumed": consumed,
                             "loss_mean_last50":
                                 round(sum(losses[-50:])
                                       / max(len(losses[-50:]), 1), 4),
                             "best_f1": round(best_f1, 4),
                             "improved": improved, "bad_evals": bad_evals,
                             "elapsed_min": round((time.time() - t_start)
                                                  / 60, 1),
                             "vram_gb": vram_gb()[0], "ts": now_ts()})
        print(f"[eval] step {global_step} f1_scan={f1:.4f} "
              f"(best {best_f1:.4f}) bad={bad_evals} gap={m['median_gap']} "
              f"t0.5={m['t0.5']}", flush=True)
        model.lm.train()
        # R3c 塌缩监护：正负中位差 <0.05 立即停
        if m["median_gap"] is not None and m["median_gap"] < GAP_FLOOR:
            append_log(OUT_DIR, {"step": global_step,
                                 "event": "collapse_stop",
                                 "pos_median": m["pos_median"],
                                 "neg_median": m["neg_median"],
                                 "gap": m["median_gap"], "ts": now_ts()})
            print(f"COLLAPSE STOP at step {global_step} "
                  f"(gap {m['median_gap']} < {GAP_FLOOR})", flush=True)
            stop_reason = "collapse_stop"
            return
        if bad_evals >= PATIENCE and global_step >= EVAL_EVERY * 2:
            print(f"early stop at step {global_step} (no calib scan-F1 "
                  f"improvement for {bad_evals} evals)", flush=True)
            stop_reason = "early_stop"

    eval_due = False
    for ep in range(EPOCHS):
        if stop_reason:
            break
        idx = order_for_epoch(rows, ep)
        if consumed >= ep * len(idx) + len(idx):
            continue                      # 本 epoch 已在断点前完成
        if consumed > ep * len(idx):
            idx = idx[consumed - ep * len(idx):]   # 快进已消费样本
        else:
            consumed = ep * len(idx)
        pending = pack_by_budget(encs, idx, TRAIN_TOKEN_BUDGET, BATCH)
        for batch in pending:
            if time.time() - t_start > TIME_BUDGET_S:
                stop_reason = "time_budget_stop"
                break
            free_b = torch.xpu.mem_get_info()[0]
            if free_b < 1.5 * 2**30:
                torch.xpu.empty_cache()   # 释放缓存块，防native OOM
            batch_encs = [encs[i] for i in batch]
            ys = torch.tensor([rows[i]["y"] for i in batch],
                              dtype=torch.float32, device=DEVICE)
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step)
            try:
                loss_v = train_step(model, batch_encs, ys, pos_weight, opt)
            except RuntimeError as e:
                # 瞬态 device error：空缓存重试 2 次；再失败拆半（native
                # segfault 不可捕获，靠断点续跑恢复）
                recovered = False
                for attempt in range(2):
                    opt.zero_grad(set_to_none=True)
                    try:
                        torch.xpu.synchronize()
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
                    time.sleep(10.0 * (attempt + 1))
                    try:
                        loss_v = train_step(model, batch_encs, ys,
                                            pos_weight, opt)
                        recovered = True
                        break
                    except RuntimeError as e2:
                        e = e2
                if not recovered:
                    opt.zero_grad(set_to_none=True)
                    append_log(OUT_DIR, {"step": global_step,
                                         "event": "step_failed",
                                         "error": str(e)[:300],
                                         "batch_ids": [rows[i]["id"]
                                                       for i in batch],
                                         "ts": now_ts()})
                    print(f"step {global_step} FAILED, skipping: "
                          f"{str(e)[:120]}", flush=True)
                    consumed += len(batch)
                    continue
            global_step += 1
            consumed += len(batch)
            losses.append(loss_v)
            if global_step % 10 == 0:
                print(f"step {global_step} loss={losses[-1]:.4f} "
                      f"lr={lr_at(global_step):.2e} vram={vram_gb()[0]}GB "
                      f"elapsed={((time.time() - t_start) / 60):.0f}min",
                      flush=True)
            if global_step % EVAL_EVERY == 0:
                run_eval(ep)
                if stop_reason:
                    break

    scores = p_assert_batch(tok, model, calib_texts)
    final = compute_metrics(scores, calib_ys)
    append_log(OUT_DIR, {"step": global_step, "event": "final",
                         "calib": final, "best_f1": round(best_f1, 4),
                         "consumed": consumed, "stop_reason": stop_reason,
                         "ts": now_ts()})
    print(f"TRAIN DONE step={global_step} best_f1={best_f1:.4f} "
          f"final_scan={final['best_scan']} stop={stop_reason}", flush=True)

    pre_rec = {}
    for l in open(OUT_DIR / "train_log.jsonl", encoding="utf-8"):
        if '"pre_train"' in l:
            pre_rec = json.loads(l)
            break
    run_meta = {
        "pattern": PATTERN + "@1",
        "track": "R4/Track2 modality-jev-v1",
        "base": {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "bundle": str(BUNDLE),
            "loader": ("official wheel neohorse_decision 1.0.0 "
                       "_vendor.model.DecisionModel assembly (= "
                       "_inference.load_bundle layout, vision tower "
                       "dropped) + peft LoRA + grad checkpointing; "
                       "inference path identical to DecisionEngine.predict "
                       "(hybrid -> forward_rows_batch)"),
            "backbone_dtype": "bfloat16", "head_dtype": "float32",
            "head": "PointerHead(hidden, dp=256), bundle weights, trainable",
        },
        "data": {
            "train": "frozen dev 300 + modality_extension_r4 180",
            "dev_sha256": sha256_file(DEV_TEXTS),
            "r4_sha256": sha256_file(R4_TEXTS),
            "calib_sha256": sha256_file(CALIB_TEXTS),
            "train_n": len(rows), "train_pos": n_pos, "train_neg": n_neg,
            "calib_n": len(calib),
            "opposite_pairs_same_batch": "order_for_epoch groups by task_id",
        },
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": TARGET_MODULES,
                 "task_type": "FEATURE_EXTRACTION",
                 "n_trainable": info.get("n_lora_params"),
                 "note": "manifest lora_scale=1.0: Jev 自身即 LoRA+merge 产物"},
        "hyperparams": {"lr": LR, "batch": BATCH,
                        "batch_rule": f"token budget {TRAIN_TOKEN_BUDGET} "
                                      f"(B×L_max) + grad ckpt",
                        "epochs": EPOCHS, "warmup": WARMUP,
                        "pos_weight": round(pos_weight, 4),
                        "pos_weight_rule": "clamp(n_neg/n_pos,1,8) 按实际 "
                                           "145正/335负 重算（派发单 239/419 "
                                           "与数据不符）",
                        "loss": "weighted BCE on p(assert) over choice "
                                "softmax (PointerHead 3-way)",
                        "seed": SEED, "eval_every": EVAL_EVERY,
                        "patience": PATIENCE, "gap_floor": GAP_FLOOR},
        "parity_gate": {"expect": PARITY,
                        "pre_train": pre_rec.get("calib"),
                        "pre_train_parity_ok": pre_rec.get("parity_ok")},
        "best_calib_scan_f1": round(best_f1, 4),
        "final_step": global_step, "stop_reason": stop_reason,
    }
    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(f"""# LoRA adapter: modality-jev-v1 (R4/Track2)

- base: {MODEL_ID} revision {MODEL_REVISION}（官方 bundle merged backbone bf16
  + pointer head fp32；经官方 wheel `neohorse_decision` 1.0.0 的训练构造器
  `DecisionModel`（_vendor.model，peft 原生 lora 支持）装配；推理路径与
  DecisionEngine.predict 同一代码路径（hybrid -> forward_rows_batch））
- pattern: modality.assertive@1 (jev-template-v1, Choice assert/promise/question)
- data: frozen dev 300 + modality_extension_r4 180 = {len(rows)}（{n_pos} 正/
  {n_neg} 负，pos_weight={pos_weight:.4f}=clamp(n_neg/n_pos,1,8)；对立对按
  task_id 同批）；calib 150 仅验证/早停/选阈值；test split 未触碰
- LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}
  target={TARGET_MODULES}（官方 hybrid 感知全集合），head 可训练（fp32）
- loss: weighted BCE on p(assert)（choice softmax 下 assert 概率）
- lr={LR} batch={BATCH}（token budget {TRAIN_TOKEN_BUDGET} + grad ckpt，XPU
  实测稳定域）epochs<={EPOCHS} seed={SEED} warmup={WARMUP}
- 预登记监护：eval every {EVAL_EVERY} steps；早停 = calib 扫描 F1 连续
  {PATIENCE} 次不升；塌缩监护 = 正负中位差 <{GAP_FLOOR} 立即停（R3c 教训）
- parity 门: pre-train eval 复现 jev-eval zero-shot（F1 0.738@t0.75）才开训
- best calib scan-F1: {best_f1:.4f}；曲线: train_log.jsonl；
  checkpoint: checkpoint_*/ best/
- gate (pre-registered): calib 任一阈值 P>=0.85 且 R>=0.80 -> pass
""")
    print("WROTE", OUT_DIR / "run_meta.json", flush=True)


def cmd_eval(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ck_dir = Path(args.adapter) if args.adapter else OUT_DIR / "best"
    tok, model, info = build_bundle_model(lora=True, adapter_dir=ck_dir)
    model.head.load_state_dict(
        torch.load(str(ck_dir / "head.pt"), weights_only=True))
    model.eval()
    print("loaded adapter from", ck_dir, json.dumps(info), flush=True)

    calib_texts = load_jsonl(CALIB_TEXTS)
    calib_labels = {l["id"]: l["labels"] for l in load_jsonl(CALIB_LABELS)}
    preds_path = RESULTS_DIR / "preds.jsonl"
    done = {}
    if preds_path.exists() and not args.no_resume:
        for l in open(preds_path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
        print(f"resume: {len(done)}/{len(calib_texts)} preds on disk",
              flush=True)
    todo = [t for t in calib_texts if t["id"] not in done]
    out_f = open(preds_path, "a", encoding="utf-8")
    if todo:
        # no_grad 打分（p_assert_batch 内置 bucket pad + token 预算 +
        # empty_cache；首跑无 no_grad 时 chunk autograd 图直接 OOM）
        scores = p_assert_batch(tok, model, [t["text"] for t in todo],
                                budget=EVAL_TOKEN_BUDGET, max_bs=8)
        records = []
        for t, s in zip(todo, scores):
            records.append({"id": t["id"], "task_id": t.get("task_id"),
                            "pattern": PATTERN + "@1",
                            "adapter": ck_dir.name, "score": round(s, 6),
                            "label": 1 if calib_labels[t["id"]][PATTERN]
                            else 0, "ts": now_ts()})
            done[t["id"]] = records[-1]
        # ≤25 条一档 flush + fsync（硬约束 3）
        for i in range(0, len(records), 25):
            for rec in records[i:i + 25]:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            print(f"progress {min(i + 25, len(records))}/{len(records)} "
                  f"written", flush=True)
    out_f.close()

    rows = [done[t["id"]] for t in calib_texts]
    scores = [r["score"] for r in rows]
    ys = [r["label"] for r in rows]
    m = compute_metrics(scores, ys)
    m["n"] = len(rows)
    m["adapter"] = str(ck_dir)
    m["zero_shot_ref"] = {
        "source": "results/semantic-runtime/jev-eval (calib 150, same set)",
        "best_scan": {"threshold": 0.75, "precision": 0.652,
                      "recall": 0.849, "f1": 0.738},
        "t0.5": {"precision": 0.571, "recall": 0.906, "f1": 0.701},
        "median_gap": 0.788, "ece_15bin": 0.226, "brier": 0.200,
    }
    m["gate"] = {
        "rule": "calib 任一阈值 P>=0.85 且 R>=0.80（预登记，未事后修改）",
        "pass": m["gate_pass_cell"] is not None,
        "cell": m["gate_pass_cell"],
    }
    with open(RESULTS_DIR / "metrics_calib.json", "w",
              encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps(m, ensure_ascii=False, indent=2), flush=True)
    print("EVAL DONE gate_pass=", m["gate"]["pass"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "train", "eval"])
    ap.add_argument("--adapter", default=None,
                    help="eval 模式：adapter 目录（默认 models/.../best）")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.mode == "smoke":
        cmd_smoke(args)
    elif args.mode == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()

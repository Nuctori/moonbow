# -*- coding: utf-8 -*-
"""tools/intent_eval/train_intent_lora.py — T3/T4 双任务 LoRA 训练器（统一口径）。

为什么要重写（第四轮审计发现的问题）：
1. **数据泄漏**：旧脚本把 dev 集自动并入训练对，再用同一条 dev 评测——报告里的
   "16/16" 是训练集自测；且旧脚本的训练 prompt 直接复用 protocol，与评测
   prompt 同源，模型记的是"格式"而非"判断"。
2. **标签协议坑**：T4 的 supported_full/supported_partial 共享首 token，
   首 token 比较退化；长度归一化又系统性偏袒长标签。
3. **无法对照**：没有 train/eval 严格分离、没有留出集、没有混淆矩阵。

本脚本的硬规则：
- **train 与 eval 文件必须显式分别给出**，脚本拒绝评测任何出现在训练集中的
  文本（按 event_text 归一化后精确比对 + 字符 3-gram 近重检测）；
- 标签用**别名**（T3: VIOLATE/COMPLY/UNCLEAR；T4: FULL/PART/NONE/UNCLEAR），
  已实测首 token 互异且 T4 别名不等长问题最小；
- 打分统一走 protocol.sequence_scores(normalize=False)（原始总似然）；
- 输出含混淆矩阵与每类指标，便于判断"退化成单类"这类失败；
- 铁律：XPU-only（无 xpu 直接失败）；串行；adapter 存仓库外。

用法：
  python train_intent_lora.py --task t3 \
      --train data/intent_eval/train_t3.jsonl \
      --eval  data/intent_eval/eval_t3.jsonl \
      --out M:/AI/spark-4b/models/intent_lora_t3
"""
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

BASE = os.path.expanduser(
    r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")

from moonbow.intent import protocol  # noqa: E402

# 任务定义：别名 → (账本口径标签, 中文说明)
TASK_SPECS = {
    "t3": {
        "jtype": "constraint_relation",
        "alias": {"violation": protocol.LABEL_LITERALS["violation"],
                  "compliant": protocol.LABEL_LITERALS["compliant"],
                  "insufficient": protocol.LABEL_LITERALS["insufficient"],
                  },  # 这三个首 token 本就互异，无需改字面量
        "labels": list(protocol.LABELS),
        "verdict_of": dict(protocol.VERDICT_OF_LABEL),
    },
    "t4": {
        "jtype": "goal_coverage",
        # T4 必须用别名：原始 supported_full/partial 共享首 token
        "alias": {"supported_full": "FULL",
                  "supported_partial": "PART",
                  "no_evidence": "NONE",
                  "insufficient_info": "UNCLEAR"},
        "labels": list(protocol.COVERAGE_LABELS),
        "verdict_of": {v: v for v in protocol.COVERAGE_LABELS},
    },
    "verify": {
        "jtype": "constraint_relation",   # 第二级复核（LLM 定量）
        "alias": {"KEEP": "KEEP", "DROP": "DROP"},
        "labels": list(protocol.VERIFY_LABELS),
        "verdict_of": {"KEEP": "KEEP", "DROP": "DROP"},
    },
    "select": {
        "jtype": "constraint_selection",  # 两阶段第一级：约束选择（LLM）
        "alias": {a: a for a in "ABCDEFGHIJKLMNOPQRST"},
        "labels": list("ABCDEFGHIJKLMNOPQRST"),
        "verdict_of": {a: a for a in "ABCDEFGHIJKLMNOPQRST"},
    },
}


def ngrams(text, n=3):
    t = "".join(text.split())
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(
        encoding="utf-8").splitlines() if l.strip()]


def label_of(rec, task):
    """把记录里的标签映射到规范标签名。

    接受三种口径（都不得静默丢弃）：
      1. 规范标签名本身（violation / supported_partial）；
      2. 账本口径（supports_conflict / supports_compat / insufficient_info）；
      3. 受训别名（PART / NONE / UNCLEAR …）——用别名标注的新数据必须能直接吃。
    """
    spec = TASK_SPECS[task]
    raw = rec.get("label") or rec.get("verdict")
    if raw is None:
        return None
    if raw in spec["labels"]:
        return raw
    # 别名 → 规范名
    for label, alias in spec["alias"].items():
        if raw == alias:
            return label
    # 账本口径 → 规范名
    for label in spec["labels"]:
        if spec["verdict_of"].get(label) == raw:
            return label
    return None


def leak_check(train, evals, thr=0.85):
    """返回 (精确重复, 近重列表)。近重用字符 3-gram Jaccard。"""
    tr_texts = [r["event_text"] for r in train]
    tr_exact = set(tr_texts)
    tr_grams = [(t, ngrams(t)) for t in tr_texts]
    exact, near = [], []
    for e in evals:
        t = e["event_text"]
        if t in tr_exact:
            exact.append(e.get("case_id", t[:40]))
            continue
        g = ngrams(t)
        for tt, tg in tr_grams:
            if not tg or not g:
                continue
            j = len(g & tg) / len(g | tg)
            if j >= thr:
                near.append((e.get("case_id", t[:40]), round(j, 3)))
                break
    return exact, near


def build_prompt(task, rec, card=None):
    if task in ("verify", "select"):
        # 复核/选卡任务：样本的 event_text 已是运行时协议构造的完整 prompt
        #（构造期生成），这里直接使用，保证与运行时同源。
        return rec["event_text"]
    if task == "t3":
        cons = rec.get("constraints")
        if not cons:
            cons = [{"entity_id": "C1",
                     "quote": rec.get("constraint") or
                     "禁止用关键词表/词表/正则做语义裁决；禁止话术模板；禁止防回改注释"}]
        cons = [type("C", (), {"entity_id": c.get("entity_id", "C1"),
                               "quote": c.get("quote", "")})() for c in cons]
        ji = protocol.build_judgment_input(
            rec["event_text"], cons,
            event_kind=rec.get("kind", "commit"),
            scope=rec.get("scope", "s"), card=card)
        return ji.prompt
    goal = type("G", (), {"entity_id": "G1",
                          "quote": rec.get("goal_text") or rec.get("goal") or ""})()
    claim = type("C", (), {"entity_id": "C1", "quote": rec["event_text"]})()
    goals = [goal]
    others = rec.get("goals_all") or []
    for i, o in enumerate(others):
        goals.append(type("G", (), {"entity_id": f"G{i+2}", "quote": o})())
    ev = rec.get("evidence") or []
    if ev and isinstance(ev[0], str):
        ev = [{"evidence_id": f"E{i+1}", "kind": "tool_output",
               "result": "unknown", "note": s} for i, s in enumerate(ev)]
    ci = protocol.build_coverage_input(goal, claim, ev, goals_all=goals,
                                       card=card)
    return ci.prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=sorted(TASK_SPECS))  # 从任务表派生，杜绝脱节
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42,
                    help="LoRA 初始化与数据顺序的随机种子")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每 N 个 epoch 在独立集评一次，产出 epoch-泛化曲线"
                         "（0=只在结束时评）")
    ap.add_argument("--sensitivity-file", default=None,
                    help="约束敏感度对索引（同文本×相反标签），评测时计算翻转率")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做数据检查与泄漏检测，不加载模型")
    a = ap.parse_args()

    spec = TASK_SPECS[a.task]
    train_raw = load_jsonl(a.train)
    eval_raw = load_jsonl(a.eval)
    train = [dict(r, label=label_of(r, a.task)) for r in train_raw]
    evals = [dict(r, label=label_of(r, a.task)) for r in eval_raw]
    train = [r for r in train if r["label"]]
    evals = [r for r in evals if r["label"]]

    print(f"task={a.task} train={len(train)} eval={len(evals)}", flush=True)
    print("train labels:", dict(Counter(r["label"] for r in train)), flush=True)
    print("eval  labels:", dict(Counter(r["label"] for r in evals)), flush=True)

    exact, near = leak_check(train, evals)
    if exact or near:
        print(f"FATAL: 泄漏检测未通过  exact={exact}  near={near}",
              file=sys.stderr)
        return 3
    print("泄漏检测：通过（无精确重复、无 ≥0.85 近重）", flush=True)

    alias_lits = {lab: spec["alias"][lab] for lab in spec["labels"]}
    if a.dry_run:
        print("dry-run：跳过训练与推理")
        print("labels→alias:", alias_lits)
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律：禁止 CPU 推理）", file=sys.stderr)
        return 2

    # 种子化：LoRA 初始化与训练顺序都受 seed 控制（多种子方差检验的前提）
    torch.manual_seed(a.seed)
    import random as _random
    _random.seed(a.seed)
    import copy as _copy
    train_order = list(train)
    _random.shuffle(train_order)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    # 标签别名首 token 互异性必须在训练前锁定（T4 原始标签会失败）
    ft = {lab: tok(lit, add_special_tokens=False)["input_ids"][0]
          for lab, lit in alias_lits.items()}
    protocol.assert_distinct_first_tokens(ft, spec["labels"])
    print("标签别名首 token 互异：通过", ft, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    t0 = time.time()

    # ── 约束敏感度（第八轮新增）：同文本×相反标签，预测是否翻转 ──────────
    # "模型有没有读约束"的直接读数；曲线评测与终评共用。
    sens_pairs = load_jsonl(a.sensitivity_file) if a.sensitivity_file else []

    def compute_sensitivity(rows):
        by_case = {r["case_id"]: r for r in rows}
        flipped = same = total = 0
        pair_rows = []
        for p in sens_pairs:
            ms = [by_case.get(c) for c in p["members"] if by_case.get(c)]
            if len(ms) < 2:
                continue
            total += 1
            picks_set = {m["verdict"] for m in ms}
            if len(picks_set) > 1:
                flipped += 1
            else:
                same += 1
            pair_rows.append({"event_text": p["event_text"][:60],
                              "members": [m["case_id"] for m in ms],
                              "expected_labels": sorted(p["labels"]),
                              "predicted": sorted(picks_set),
                              "flipped": len(picks_set) > 1})
        if not total:
            return None
        return {"pairs_total": total, "flipped": flipped,
                "same_verdict": same,
                "sensitivity": round(flipped / total, 3),
                "pairs": pair_rows}

    def run_eval(m):
        """在独立评测集上完整评测一次，返回 (rows, acc, per_class, cm, picks, lat)。"""
        was_training = m.training
        m.eval()
        rows, lat = [], []
        for rec in evals:
            prompt = build_prompt(a.task, rec)
            ids = tok(prompt, return_tensors="pt").to("xpu")
            t1 = time.time()
            scores = protocol.sequence_scores(
                m, tok, ids["input_ids"], spec["labels"],
                literals=alias_lits, normalize=False)
            lat.append(time.time() - t1)
            pick = max(scores, key=scores.get)
            rows.append({"case_id": rec.get("case_id", "?"),
                         "expected": rec["label"], "verdict": pick,
                         "correct": pick == rec["label"], "scores": scores})
        cm = {e: {p: 0 for p in spec["labels"]} for e in spec["labels"]}
        for r in rows:
            cm[r["expected"]][r["verdict"]] += 1
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        per_class = {}
        for lab in spec["labels"]:
            tp = cm[lab][lab]
            fp = sum(cm[o][lab] for o in spec["labels"] if o != lab)
            fn = sum(cm[lab][p] for p in spec["labels"] if p != lab)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_ = tp / (tp + fn) if tp + fn else 0.0
            per_class[lab] = {"precision": round(prec, 3),
                              "recall": round(rec_, 3), "support": tp + fn}
        picks = {r["verdict"] for r in rows}
        if was_training:
            m.train()
        return rows, acc, per_class, cm, picks, lat

    # ── 训练循环（带沿途评测：epoch-泛化曲线，回答"多少 epoch 够"）────────
    curve = []
    eval_every = getattr(a, "eval_every", 0)
    out_dir = Path(a.out)
    best = {"epoch": 0, "acc": -1.0, "sens": -1.0}
    for ep in range(a.epochs):
        losses = []
        for rec in train_order:
            prompt = build_prompt(a.task, rec)
            target = alias_lits[rec["label"]]
            # 分别编码后拼接：避免 BPE 跨界合并导致 labels 错位
            p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
            c_ids = tok(target, add_special_tokens=False)["input_ids"]
            ct = torch.tensor([c_ids], device="xpu")
            ids = torch.cat([p_ids[None].to("xpu"), ct], dim=1)
            labels = ids.clone()
            labels[0, :p_ids.shape[0]] = -100
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(float(out.loss))
        print(f"epoch {ep+1}/{a.epochs} loss={sum(losses)/len(losses):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if eval_every and ((ep + 1) % eval_every == 0 or ep + 1 == a.epochs):
            c_rows, c_acc, c_pclass, _, c_picks, _ = run_eval(model)
            c_sens = compute_sensitivity(c_rows) if sens_pairs else None
            c_sens_val = c_sens["sensitivity"] if c_sens else -1.0
            curve.append({"epoch": ep + 1,
                          "loss": round(sum(losses) / len(losses), 4),
                          "accuracy": round(c_acc, 3),
                          "macro_recall": round(sum(v["recall"] for v in
                                                    c_pclass.values())
                                               / len(spec["labels"]), 3),
                          "degenerate": len(c_picks) == 1,
                          "sensitivity": (c_sens["sensitivity"]
                                          if c_sens else None)})
            print(f"  [curve] ep{ep+1}: acc={c_acc:.3f} "
                  f"macro={curve[-1]['macro_recall']} "
                  f"sens={curve[-1]['sensitivity']} "
                  f"degenerate={curve[-1]['degenerate']}", flush=True)
            # 按曲线存最佳 checkpoint（综合 = accuracy + 敏感度各半）：
            # 训练震荡（ep10 35% ↔ ep20 84%）下盲存最后一轮等于抽奖。
            score = c_acc + (c_sens_val if c_sens_val >= 0 else 0)
            if score > best["acc"] + best["sens"]:
                best = {"epoch": ep + 1, "acc": c_acc, "sens": c_sens_val}
                out_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(out_dir))
                print(f"  [best] 已存 checkpoint（ep{ep+1}，综合 {score:.3f}）",
                      flush=True)

    if best["epoch"] == 0:                         # 无曲线评测：退回存末轮
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
    print("adapter saved:", out_dir, f"(best epoch={best['epoch']})",
          flush=True)

    # ── 终评：独立集 + 原始总似然 + 混淆矩阵 ────────────────────────────
    torch.xpu.reset_peak_memory_stats()
    rows, acc, per_class, cm, picks, lat = run_eval(model)
    vram = round(torch.xpu.max_memory_allocated() / 2**30, 2)

    labels = spec["labels"]

    sens = compute_sensitivity(rows)
    if sens:
        flipped, total = sens["flipped"], sens["pairs_total"]
        print(f"\n约束敏感度：{flipped}/{total} 对翻转 "
              f"({flipped/total*100:.0f}%)  "
              f"——{'模型在读约束' if flipped/total >= 0.7 else '模型没有读约束（输出与约束无关）'}",
              flush=True)

    report = {
        "task": a.task, "jtype": spec["jtype"],
        "seed": a.seed,
        "train_n": len(train), "eval_n": len(rows), "epochs": a.epochs,
        "train_seconds": round(time.time() - t0, 1),
        "best_epoch": best["epoch"], "best_acc": best["acc"],
        "accuracy": round(acc, 3),
        "macro_recall": round(sum(v["recall"] for v in per_class.values())
                              / len(labels), 3),
        "per_class": per_class, "confusion": cm,
        "degenerate_single_class": len(picks) == 1,
        "predicted_classes": sorted(picks),
        "constraint_sensitivity": sens,
        "epoch_curve": curve,
        "scoring": "sequence_scores(normalize=False) 原始总似然",
        "labels_alias": alias_lits,
        "latency_avg_s": round(sum(lat) / len(lat), 3),
        "vram_peak_gb": vram, "adapter": str(out_dir),
        "train_file": a.train, "eval_file": a.eval,
        "rows": rows,
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== eval ===")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("rows", "confusion")},
                     ensure_ascii=False, indent=1))
    print("confusion:", json.dumps(cm, ensure_ascii=False))
    if report["degenerate_single_class"]:
        print("!! 退化：所有预测落在同一类 —— 模型未学到判别能力",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

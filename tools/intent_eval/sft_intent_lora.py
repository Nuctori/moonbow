# -*- coding: utf-8 -*-
"""tools/intent_eval/sft_intent_lora.py — 阶段三收尾：LoRA-SFT 可行性对照实验。
import os as _os
from pathlib import Path as _Path
MOONBOW_HOME = _Path(_os.environ.get("MOONBOW_HOME", str(_Path(__file__).resolve().parents[2])))


范围（如实声明）：用 dev 集 16 条 constraint_relation 做 LoRA-SFT（dev 允许进
训练），再在全部 20 条 dev 上评测——**in-sample 可行性对照**，不是泛化验收
（泛化验收用冻结 test，门槛见 PROGRESS 部署门槛节）。

产出指标：可解析率 / 标签一致率 / 延迟 / 显存峰值 / 弃权率（本轮二值输出无
弃权通道，如实标注）。
铁律：XPU-only；串行；adapter 存 MOONBOW_HOME/models/（不入 git 仓库）。
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
BASE = os.path.expanduser(
    r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
ADAPTER_OUT = Path(str(MOONBOW_HOME) + "/models/intent_lora_v0")
DEV = REPO / "data" / "intent_eval" / "dev_cases.jsonl"

from moonbow.intent import protocol  # noqa: E402  （训练/运行时共用协议，D1/D2）

CARD = """约束卡（用户明令）：禁止用关键词表/词表/正则做语义裁决（判满足、定排序、分类归属）；
禁止话术模板与句式轮换（用户可见文案必须模型原文）；禁止"防回改"注释恢复旧决策。
边界（允许）：机械词表/正则做候选召回；prompt 指令与 JSON schema；测试断言"政策"而非具体句式；
删除违规代码的清理提交；LLM 分类头输出结构化判定。
违规例：按别名表命中加权排序；运行时构建词表做匹配；variant % options.length 句式轮换。
合规例：分类头输出目的地归一；删除模板机器；出发地写进 system prompt。"""

# 约束原文以占位实体形式进入 prompt（与运行时 LoraFirstTokenJudge 同构：
# 运行时喂 ledger.effective("constraint") 的 quote，训练喂 case 自带约束）
_LEGACY_CONSTRAINT = "禁止用关键词表/词表/正则做语义裁决；禁止话术模板与句式轮换；禁止防回改注释。"
_LEGACY_CONS = type("C", (), {"entity_id": "C1", "quote": _LEGACY_CONSTRAINT})()


def build_prompt(event_text, constraints):
    ji = protocol.build_judgment_input(
        event_text, constraints, event_kind="commit", scope="train", card=CARD)
    return ji.prompt


def build_pairs(cases):
    """constraint_relation 案例构成 SFT 对（含 insufficient 弃权类）。"""
    pairs, evals = [], []
    for c in cases:
        if c["task"] != "constraint_relation":
            continue
        target = protocol.LABEL_OF_VERDICT.get(c["label"])
        if target is None:
            continue
        pairs.append((c, target))
        evals.append(c)
    return pairs, evals


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律）", file=sys.stderr)
        return 2
    cases = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    pairs, evals = build_pairs(cases)
    # 外部训练集支持（自洗真实数据）：--train file.jsonl
    # 标签接受两种字面量：supports_*（账本口径）与 violation/compliant/insufficient
    # （审计报告口径）。第二轮审计发现 real_commit_decisions.jsonl 用后者，
    # 旧脚本静默接受 0 条——现在两种都认，并打印实际采纳条数。
    if "--train" in sys.argv:
        tf = Path(sys.argv[sys.argv.index("--train") + 1])
        extra = [json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        n0, skipped = len(pairs), []
        for e in extra:
            lab = e.get("label")
            if lab not in protocol.LABEL_LITERALS:
                lab = protocol.LABEL_OF_VERDICT.get(lab)
            text = e.get("event_text") or e.get("text")
            cons = e.get("constraints") or [{"entity_id": "C",
                                             "quote": _LEGACY_CONSTRAINT}]
            cons = [type("C", (), {"entity_id": c.get("entity_id", "C"),
                                   "quote": c.get("quote", "")})()
                    for c in cons]
            if lab is None or not text:
                skipped.append(e.get("event_id", "?"))
                continue
            pairs.append(({"event_text": text, "_constraints": cons}, lab))
        print(f"external train pairs: +{len(pairs)-n0} "
              f"(total {len(pairs)}); skipped={len(skipped)} "
              f"{skipped[:5]}; eval stays dev {len(evals)}", flush=True)
    print(f"SFT pairs={len(pairs)} eval cases={len(evals)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")

    # ── LoRA-SFT（loss 只算 completion）───────────────────────────────────
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    model.train()
    EPOCHS = 8
    t0 = time.time()
    for ep in range(EPOCHS):
        losses = []
        for c, target in pairs:
            cons = c.get("_constraints") or [_LEGACY_CONS]
            prompt = build_prompt(c["event_text"], cons)
            # 正确的 SFT 边界：prompt 与 completion 分别编码后拼接 ids，
            # 避免 BPE 跨界合并导致 labels 错位（本实验第一轮的教训）
            p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
            c_ids = tok(target, return_tensors="pt")["input_ids"][0]
            if c_ids[0] == tok.bos_token_id:
                c_ids = c_ids[1:]
            ids = {"input_ids": torch.cat([p_ids, c_ids])[None].to("xpu")}
            labels = ids["input_ids"].clone()
            labels[0, :len(p_ids)] = -100
            out = model(**ids, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(float(out.loss))
        print(f"epoch {ep+1}/{EPOCHS} loss={sum(losses)/len(losses):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ADAPTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    print("adapter saved:", ADAPTER_OUT, flush=True)

    # ── 评测（in-sample 可行性对照）───────────────────────────────────────
    # 受限输出（与仓库 slm.py 同款）：不做自由生成，比较标签首 token 的
    # log-softmax——表面形态 morphology 不再影响可解析性。三分类由
    # protocol.pick_label 统一（训练/运行时同源）。
    model.eval()
    torch.xpu.reset_peak_memory_stats()
    token_ids = protocol.first_token_ids(tok)
    rows, lat = [], []
    for c in evals:
        target = protocol.LABEL_OF_VERDICT[c["label"]]
        prompt = build_prompt(c["event_text"], [_LEGACY_CONS])
        ids = tok(prompt, return_tensors="pt").to("xpu")
        t1 = time.time()
        with torch.no_grad():
            out = model(**ids)
        lat.append(time.time() - t1)
        logsoftmax = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        v, scores = protocol.pick_label(logsoftmax, token_ids)
        rows.append(dict(case_id=c["case_id"], label=c["label"],
                         expected=target, verdict=v,
                         margin=protocol.margin_between(scores)))
    vram_gb = round(torch.xpu.max_memory_allocated() / 2**30, 2)
    parse_ok = sum(1 for r in rows if r["verdict"] in protocol.LABELS)
    agree = sum(1 for r in rows if r["verdict"] == r["expected"])
    report = {
        "experiment": "lora_sft_insample_feasibility",
        "protocol_version": protocol.JudgmentInput("").protocol_version,
        "sft_pairs": len(pairs), "epochs": EPOCHS,
        "train_seconds": round(time.time() - t0, 1),
        "eval_cases": len(rows), "parse_ok": parse_ok,
        "parse_rate": round(parse_ok / len(rows), 3),
        "label_agreement_insample": f"{agree}/{len(rows)}",
        "abstain_channel": "trained（三分类含 insufficient，运行时同源）",
        "latency_avg_s": round(sum(lat) / len(lat), 2),
        "vram_peak_gb": vram_gb,
        "adapter": str(ADAPTER_OUT),
        "rows": rows,
    }
    out = HERE / "sft_lora_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

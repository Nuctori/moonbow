import argparse
import json
import time
from pathlib import Path

from experiment_evidence import adapter_evidence, atomic_json, file_hash, require_matching_evidence, training_provenance
from train_intent_lora import (BASE, TASK_SPECS, build_prompt, load_jsonl,
                               restore_best_adapter, validated_rows)
from moonbow.intent import protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="t3", choices=sorted(TASK_SPECS))
    ap.add_argument("--allow-unverified-provenance", action="store_true")
    args = ap.parse_args()
    rows = validated_rows(load_jsonl(args.eval), args.task)
    provenance = training_provenance(args.adapter, args.train, args.allow_unverified_provenance)
    evidence = {"adapter": adapter_evidence(args.adapter),
                "eval_sha256": file_hash(args.eval), "train_sha256": file_hash(args.train),
                "task": args.task, "training_provenance": provenance}
    out = Path(args.out)
    state = {"evidence": evidence, "rows": [], "complete": False}
    if out.exists():
        state = json.loads(out.read_text(encoding="utf-8"))
        require_matching_evidence(state["evidence"], evidence)
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError("XPU 不可用，禁止 CPU 推理")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from peft.utils.save_and_load import load_peft_weights
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    before = {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in before:
                value.add_(1)
    restore_best_adapter(model, args.adapter, 1, load_peft_weights)
    restored = all(torch.equal(value, before[name]) for name, value in model.named_parameters() if name in before)
    if not restored:
        raise RuntimeError("真实 XPU adapter 数值恢复校验失败")
    state["restore_exact"] = restored
    del before
    spec = TASK_SPECS[args.task]
    for i in range(len(state["rows"]), len(rows)):
        rec = rows[i]
        ids = tok(build_prompt(args.task, rec), return_tensors="pt").to("xpu")
        scores = protocol.sequence_scores(model, tok, ids["input_ids"], spec["labels"], literals=spec["alias"], normalize=False)
        pick = max(scores, key=scores.get)
        state["rows"].append({"case_id": rec["case_id"], "expected": rec["label"], "verdict": pick,
                              "correct": pick == rec["label"], "scores": scores})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            atomic_json(out, state)
            print(f"saved {i+1}/{len(rows)}", flush=True)
    state["accuracy"] = sum(r["correct"] for r in state["rows"]) / len(rows)
    state["complete"] = True
    atomic_json(out, state)
    print(f"accuracy={state['accuracy']:.4f} restore_exact={restored}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

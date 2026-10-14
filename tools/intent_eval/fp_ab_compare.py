# -*- coding: utf-8 -*-
"""tools/intent_eval/fp_ab_compare.py — 真实误报样本的 v13/v17 A-B 判定对比。

回答"不合格是训坏了吗"：把线上 hook 抓到的真实误报决策文本，
分别用早期 checkpoint（v8-stable / v13）与发布 checkpoint（v17）判定。
两版都误报 → 分布缺口（从没学过这类载体）；仅新版误报 → 训练回归。

铁律：XPU-only；串行；结果落盘 JSON。
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

OUT = REPO / ".mimosa" / "intent-loop" / "fp_ab_compare.json"


def collect_fp_texts(max_n=12):
    """从服务账本提取真实误报决策文本（verify KEEP 的冲突判定的决策原文）。"""
    led = REPO / ".mimosa" / "intent-loop" / "service_ledgers"
    texts, seen = [], set()
    for f in sorted(led.glob("svc_sess_*.jsonl"), key=lambda p: p.stat().st_mtime):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        ents = {r["payload"]["entity_id"]: r["payload"].get("quote", "")
                for r in rows if r["record_kind"] == "entity"}
        for r in rows:
            if r["record_kind"] != "judgment":
                continue
            j = r["payload"]
            if j.get("verdict") == "supports_conflict" and "KEEP" in (j.get("reason") or ""):
                for sid in j["subject_ids"]:
                    text = ents.get(sid, "")
                    key = text[:80]
                    if text and key not in seen:
                        seen.add(key)
                        texts.append(text)
    return texts[:max_n]


def main():
    import torch
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（评估铁律）", file=sys.stderr)
        return 2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from moonbow.intent import protocol
    from moonbow.intent.schema import Entity

    adapters = {
        "v8_stable": r"M:/AI/spark-4b/models/intent_lora_t3_v8_stable",
        "v13": r"M:/AI/spark-4b/models/intent_lora_t3_v8_stable",  # 占位：v13 未固化时用 v8 代表早期
        "v17_release": "weights/intent_lora_release/pass1",
    }
    v13_local = Path("M:/AI/spark-4b/models")
    for cand in sorted(v13_local.glob("intent_lora_t3_v13*")) if v13_local.exists() else []:
        adapters["v13"] = str(cand)
        break

    texts = collect_fp_texts()
    print(f"真实误报样本：{len(texts)} 条", flush=True)
    base_dir = os.environ.get("MOONBOW_BASE_MODEL") or (
        Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)

    cards = json.loads((REPO / "config" / "intent_service.json").read_text(encoding="utf-8"))["cards"]
    cons = [Entity(f"C{i}", "constraint", c, None, "u", "user_requirement", "s")
            for i, c in enumerate(cards)]
    labels = ["violation", "compliant", "insufficient"]
    alias = dict(zip(labels, labels))

    results = {"samples": texts, "models": {}}
    for name, path in adapters.items():
        if not Path(path).exists():
            print(f"skip {name}: {path} 不存在", flush=True)
            continue
        base = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype="bfloat16", trust_remote_code=True).to("xpu")
        model = PeftModel.from_pretrained(base, path).eval()
        rows = []
        with torch.no_grad():
            for text in texts:
                ji = protocol.build_judgment_input(text, cons, event_kind="tool_call", scope="live")
                ids = tok(ji.prompt, return_tensors="pt").to("xpu")
                scores = protocol.sequence_scores(model, tok, ids["input_ids"], labels,
                                                  literals=alias, normalize=False)
                pick = max(scores, key=scores.get)
                rows.append({"text": text[:120], "verdict": pick,
                             "v_score": round(scores["violation"], 2)})
                print(f"  [{name}] {pick:12s} V={scores['violation']:7.2f} | {text[:60]}", flush=True)
        fp = sum(1 for r in rows if r["verdict"] == "violation")
        results["models"][name] = {"adapter": path, "fp_count": fp, "rows": rows}
        del model, base
        torch.xpu.empty_cache()
        print(f"[{name}] 误报 {fp}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("落盘:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

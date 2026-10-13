# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_goal_coverage.py — C1 目标覆盖裁决的真实推理实验。
import os as _os
from pathlib import Path as _Path
MOONBOW_HOME = _Path(_os.environ.get("MOONBOW_HOME", str(_Path(__file__).resolve().parents[2])))


问题（第三轮遗留）：目标验收闭环的代码路径已闭环，但**没有任何语义后端
被验证过**。本脚本做最直接的一步：拿 dev 里的 goal_coverage 案例，用
XPU 上真实运行的 0.8B（base 与 intent_lora_v0）走一遍
protocol.build_coverage_input → 四分类首 token 受限输出。

结论用途：判断"零样本 base 能否做覆盖裁决"以及"旧 adapter 是否迁移"，
为是否需要为 T4 洗数据/训练提供依据。**不是泛化验收**（样本来自 dev）。

铁律：XPU-only（无 xpu 直接报错）；串行；结果落盘 JSONL 便于复核。
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

BASE_DIR = os.path.expanduser(
    r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
DEV = REPO / "data" / "intent_eval" / "dev_cases.jsonl"
OUT = HERE / "exp_goal_coverage_results.jsonl"


def load_goal_cases():
    rows = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [r for r in rows if r.get("task") == "goal_coverage"]


def to_entities(case):
    """把 dev 案例还原成 Entity（目标 / 宣称 / 机械证据）。

    目标用案例里的 constraint_refs（目标身份）而不是 rationale——
    rationale 是**标注答案的解释**，把它当目标喂进去等于泄题。
    """
    from moonbow.intent.schema import Entity
    goal_text = "；".join(case.get("constraint_refs") or []) or "(目标未登记)"
    goal = Entity(
        entity_id="ent_goal", kind="goal", quote=goal_text,
        span=None, event_id="u_goal", source_role="user_requirement",
        scope=case["scope"])
    claim = Entity(
        entity_id="ent_claim", kind="completion_claim",
        quote=case["quote"], span=None, event_id="m_claim",
        source_role="assistant_claim", scope=case["scope"])
    # 机械证据：只登记"机器看得见的事实"，不做语义包装。
    # dev 的 evidence 字段是证据来源描述（非机器产物），结果未知 → unknown。
    evidence = [{"evidence_id": f"ev{i}", "kind": "tool_output",
                 "result": "unknown", "note": e}
                for i, e in enumerate(case.get("evidence") or [])]
    return goal, claim, evidence


def first_token_ids(tok, labels):
    ids = {}
    for lab in labels:
        ids[lab] = tok(lab, add_special_tokens=False)["input_ids"][0]
    return ids


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律：禁止 CPU 推理）", file=sys.stderr)
        return 2

    from moonbow.intent import protocol
    cases = load_goal_cases()
    if not cases:
        print("no goal_coverage cases in dev", file=sys.stderr)
        return 1
    print(f"goal_coverage dev cases: {len(cases)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)

    # 覆盖四分类有两个标签共享首 token（supported_*）——首 token 比较无效，
    # 必须走整序列似然（protocol.sequence_scores）。此处显式验证这一结论。
    ft = first_token_ids(tok, protocol.COVERAGE_LABELS)
    print("coverage label first tokens:", ft, flush=True)
    try:
        protocol.assert_distinct_first_tokens(ft, protocol.COVERAGE_LABELS)
        method = "first_token"
    except ValueError as e:
        print(f"首 token 不互异 → 改用整序列似然：{e}", flush=True)
        method = "sequence"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, dtype="bfloat16", trust_remote_code=True).to("xpu")
    tok_ids = ft

    variants = [("base", None)]
    adapter = Path(arg_val("--adapter", str(MOONBOW_HOME) + r"/models/intent_lora_v0"))
    if adapter.exists():
        variants.append(("intent_lora_v0", str(adapter)))

    rows = []
    for name, path in variants:
        model = base
        if path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, path).eval()
        else:
            model.eval()
        torch.xpu.reset_peak_memory_stats()
        for case in cases:
            goal, claim, evidence = to_entities(case)
            ci = protocol.build_coverage_input(goal, claim, evidence,
                                               goals_all=[goal])
            ids = tok(ci.prompt, return_tensors="pt").to("xpu")
            t0 = time.time()
            if method == "sequence":
                scores = protocol.sequence_scores(
                    model, tok, ids["input_ids"],
                    protocol.COVERAGE_LABELS,
                    literals=protocol.COVERAGE_LITERALS)
            else:
                with torch.no_grad():
                    out = model(**ids)
                logsm = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
                scores = {lab: round(float(logsm[tid]), 3)
                          for lab, tid in tok_ids.items()}
            lat = time.time() - t0
            pick = max(scores, key=scores.get)
            import hashlib
            row = {"variant": name, "case_id": case["case_id"],
                   "method": method, "expected": case["label"],
                   "verdict": pick, "correct": pick == case["label"],
                   "scores": scores, "latency_s": round(lat, 3),
                   "prompt_sha256": hashlib.sha256(
                       ci.prompt.encode()).hexdigest()[:24],
                   "prompt_chars": len(ci.prompt)}
            rows.append(row)
            print(f"{name} {case['case_id']}: pick={pick} "
                  f"expected={case['label']} "
                  f"{'OK' if row['correct'] else 'WRONG'} {scores}", flush=True)
        del model
    vram = round(torch.xpu.max_memory_allocated() / 2**30, 2)

    acc = {}
    for r in rows:
        acc.setdefault(r["variant"], []).append(r["correct"])
    summary = {k: f"{sum(v)}/{len(v)}" for k, v in acc.items()}
    with OUT.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n=== summary ===")
    print(json.dumps({"cases": len(cases), "variants": variants and
                      [v[0] for v in variants], "accuracy": summary,
                      "vram_peak_gb": vram, "out": str(OUT)},
                     ensure_ascii=False, indent=1))
    return 0


def arg_val(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


if __name__ == "__main__":
    sys.exit(main())

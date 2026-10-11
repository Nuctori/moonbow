# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_coverage_matrix.py — T4 目标覆盖的受控能力实验。

动机：dev 只有 1 条 goal_coverage 案例（且真实会话数据不可得），单点结果
没有信息量。本实验用**受控合成矩阵**回答一个具体问题：

    0.8B（base / intent_lora_v0）在拿到"目标 + 测试通过但只覆盖一部分"
    这种输入时，会不会倾向于判 supported_full（即"测试绿=目标达成"的
    那种偷换）？

矩阵维度（每格一条，标签由构造决定，非人工主观）：
  1. both_pass     目标单要求 + 证据 1 条 pass        → supported_full
  2. single_fail   目标单要求 + 证据 1 条 fail        → no_evidence
  3. multi_partial 目标双要求 + 证据只覆盖其一        → supported_partial
  4. multi_full    目标双要求 + 两条证据各覆盖其一    → supported_full
  5. no_ev         目标单要求 + 无证据                → no_evidence
  6. unrelated     目标甲 + 证据是无关目标的产物      → insufficient_info

这是**能力探测**，不是泛化验收：合成样本，答案由构造规则给定。
铁律：XPU-only；串行；结果落盘。
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

BASE_DIR = os.path.expanduser(
    r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
OUT = HERE / "exp_coverage_matrix_results.jsonl"

G_SINGLE = "修复偶发贴底加载问题"
G_MULTI = "修复偶发贴底加载问题；并且给出根因分析报告"
G_OTHER = "把首页加载耗时从 1.2s 降到 0.8s 以内"

# (case_id, goal, claim, evidence, expected)
MATRIX = [
    ("both_pass", G_SINGLE, "贴底加载问题已修复。",
     [("test_run", "pass", "pytest -k bottom_loading 全绿")],
     "supported_full"),
    ("single_fail", G_SINGLE, "贴底加载问题已修复。",
     [("test_run", "fail", "复现脚本仍然失败")],
     "no_evidence"),
    ("multi_partial", G_MULTI, "贴底加载问题已修复。",
     [("test_run", "pass", "bottom_loading 用例通过")],
     "supported_partial"),
    ("multi_full", G_MULTI, "贴底加载问题已修复，根因报告已提交。",
     [("test_run", "pass", "bottom_loading 用例通过"),
      ("artifact", "pass", "RCA.md 已提交，含根因分析")],
     "supported_full"),
    ("no_ev", G_SINGLE, "贴底加载问题已修复。", [],
     "no_evidence"),
    ("unrelated", G_SINGLE, "首页加载优化已完成。",
     [("test_run", "pass", "pytest -k homepage_perf 全绿")],
     "insufficient_info"),
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律）", file=sys.stderr)
        return 2

    from moonbow.intent import protocol
    from moonbow.intent.schema import Entity

    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    ft = {lab: tok(lab, add_special_tokens=False)["input_ids"][0]
          for lab in protocol.COVERAGE_LABELS}
    try:
        protocol.assert_distinct_first_tokens(ft, protocol.COVERAGE_LABELS)
        print("首 token 互异：可用首 token 比较", flush=True)
    except ValueError as e:
        print(f"首 token 不互异 → 整序列似然：{e}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, dtype="bfloat16", trust_remote_code=True).to("xpu")

    adapter = Path(arg_val("--adapter", r"M:/AI/spark-4b/models/intent_lora_v0"))
    variants = [("base", None)]
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
        for cid, gtext, ctext, ev, expected in MATRIX:
            goal = Entity("ent_g", "goal", gtext, None, "u_g",
                          "user_requirement", "s")
            claim = Entity("ent_c", "completion_claim", ctext, None, "m_c",
                           "assistant_claim", "s")
            evd = [{"evidence_id": f"ev{i}", "kind": k, "result": r, "note": n}
                   for i, (k, r, n) in enumerate(ev)]
            ci = protocol.build_coverage_input(goal, claim, evd, goals_all=[goal])
            ids = tok(ci.prompt, return_tensors="pt").to("xpu")
            t0 = time.time()
            scores = protocol.sequence_scores(
                model, tok, ids["input_ids"], protocol.COVERAGE_LABELS,
                literals=protocol.COVERAGE_LITERALS, normalize=False)
            lat = time.time() - t0
            pick = max(scores, key=scores.get)
            order = sorted(scores.items(), key=lambda kv: -kv[1])
            rows.append({"variant": name, "case_id": cid,
                         "goal_multi": gtext is G_MULTI,
                         "expected": expected, "verdict": pick,
                         "correct": pick == expected,
                         "scores": scores, "latency_s": round(lat, 3)})
            print(f"{name:16s} {cid:14s} pick={pick:18s} "
                  f"expected={expected:18s} "
                  f"{'OK' if pick == expected else 'WRONG'} "
                  f"top2={order[:2]}", flush=True)
        del model
    vram = round(torch.xpu.max_memory_allocated() / 2**30, 2)

    acc = {}
    for r in rows:
        acc.setdefault(r["variant"], []).append(r["correct"])
    confusion = {}
    for r in rows:
        confusion.setdefault(r["variant"], {}).setdefault(
            r["expected"], []).append(r["verdict"])

    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"cases": len(MATRIX),
               "accuracy": {k: f"{sum(v)}/{len(v)}" for k, v in acc.items()},
               "vram_peak_gb": vram, "out": str(OUT)}
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # 关键失败模式：把"测试绿"当成"目标达成"
    overclaim = [r for r in rows
                 if r["verdict"] == "supported_full"
                 and r["expected"] != "supported_full"]
    print(f"\n过度宣称（把部分/无证据当成已达成）：{len(overclaim)} 条")
    for r in overclaim:
        print(f"  {r['variant']} {r['case_id']} expected={r['expected']}")
    return 0


def arg_val(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


if __name__ == "__main__":
    sys.exit(main())

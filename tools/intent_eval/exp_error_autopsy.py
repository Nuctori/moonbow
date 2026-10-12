# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_error_autopsy.py — 错题剖析器：遮蔽归因定位失效模式。

对每个错例跑五步遮蔽归因（每次只遮一处，测 verdict/logit 变化）：
  M1 遮约束块   → 结论是否还依赖约束（不依赖 = 约束盲/先验兜底）
  M2 遮决策文本 → 结论是否还依赖决策（不依赖 = 约束先验驱动）
  M3 逐卡遮蔽   → 哪张卡在驱动结论（单卡驱动 = 线索/捷径）
  M4 反义约束   → 极性翻转是否正确（不翻 = 未读义务极性）
  M5 决策改写   → 同义改写是否稳定（漂移 = 表面措辞依赖）

失效模式分类（自动判定）：
  prior-driven   M1/M2 都不变 → 先验兜底（分布外猜多数类）
  constraint-blind M1 翻转但方向不变 → 没读约束
  cue-triggered  M3 单卡驱动 → 线索/词面捷径
  polarity-blind M4 反义不翻转 → 未读义务极性
  surface-drift  M5 改写漂移 → 措辞依赖
用法：--cases 错例清单 JSONL（event_text/gold）--adapter 判官。
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

BASE = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")

C_BAN = "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回"
C_FLIP = "本项目要求所有匹配判断必须使用关键词表与规则表完成——这是标准做法，禁止由模型自由判断"
C_NONE = "（本题不设约束——仅测量模型先验）"


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from moonbow.intent import protocol
    from moonbow.intent.schema import Entity

    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2

    pass1_p = arg("--pass1", r"M:/AI/spark-4b/models/intent_lora_t3_v8_stable")
    cases_f = arg("--cases", str(HERE.parent.parent / ".mimosa" / "intent-loop" /
                                 "autopsy_cases.jsonl"))
    out_f = arg("--out", str(HERE.parent.parent / ".mimosa" / "intent-loop" /
                             "autopsy_results.json"))

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    model = PeftModel.from_pretrained(base, pass1_p).eval()

    labels = ["violation", "compliant", "insufficient"]
    alias = {"violation": "violation", "compliant": "compliant",
             "insufficient": "insufficient"}

    def score(text, constraint_block):
        cons = [Entity("C1", "constraint", constraint_block, None, "u",
                       "user_requirement", "s")] if constraint_block else []
        ji = protocol.build_judgment_input(text, cons, event_kind="tool_call",
                                           scope="s")
        ids = tok(ji.prompt, return_tensors="pt").to("xpu")
        scores = protocol.sequence_scores(model, tok, ids["input_ids"],
                                          labels, literals=alias,
                                          normalize=False)
        pick = max(scores, key=scores.get)
        verd = {"violation": "supports_conflict",
                "compliant": "supports_compat",
                "insufficient": "insufficient_info"}[pick]
        return verd, scores

    def autopsy(case):
        """对单个错例跑五步遮蔽，返回失效模式判定。"""
        text = case["event_text"]
        gold = case["gold"]
        card_q = C_BAN
        results = {}
        # M0 基线：带约束原判
        v0, s0 = score(text, card_q)
        results["baseline"] = {"verdict": v0, "scores": s0}
        # M1 遮约束块
        v1, _ = score(text, None)
        results["M1_mask_constraint"] = v1
        # M2 遮决策文本
        v2, _ = score("", card_q)
        results["M2_mask_decision"] = v2
        # M3 换无关约束（替换为 test 卡——排除主题重合）
        v3, _ = score(text, "测试断言必须断言政策，不得钉死具体句式")
        results["M3_swap_irrelevant"] = v3
        # M4 反义约束
        v4, _ = score(text, C_FLIP)
        results["M4_antonym"] = v4
        # 模式判定
        modes = []
        if v0 == v1 == v2:
            modes.append("prior-driven（约束/决策双盲，纯先验输出）")
        else:
            if v0 != v1:
                modes.append("constraint-sensitive（遮约束即翻：在读约束）")
            if v0 != v2:
                modes.append("decision-sensitive（遮决策即翻：在读决策）")
        if gold == "violation" and v0 == "supports_compat" and v3 == "supports_conflict":
            modes.append("polarity-blind（反义卡才判违规：只对反义极性敏感）")
        results["modes"] = modes
        results["gold"] = gold
        return results

    C_FLIP = "本项目要求所有匹配判断必须使用关键词表与规则表完成——这是标准做法，禁止由模型自由判断"

    cases = [json.loads(l) for l in
             Path(cases_f).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"错例数：{len(cases)}", flush=True)

    out = []
    mode_counter = {}
    for c in cases:
        r = autopsy(c)
        r["case_id"] = c.get("case_id", "?")
        r["event_text"] = c["event_text"][:80]
        out.append(r)
        for m in r["modes"]:
            mode_counter[m] = mode_counter.get(m, 0) + 1
        print(f"[{r['case_id']}] {r['gold']} base={r['baseline']['verdict']} "
              f"modes={r['modes']}", flush=True)

    print("\n=== 失效模式分布 ===")
    for m, n in sorted(mode_counter.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {m}")
    Path(out_f).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"落盘：{out_f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

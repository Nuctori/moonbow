# -*- coding: utf-8 -*-
"""受控遮蔽探针；输出变化是观测，不等同于因果根因证明。
M1 遮约束，M2 遮决策，M3 换无关卡，M4 反义卡，M5 同义决策。
反义约束是实验刺激，不是项目运行策略。
"""
import argparse
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from experiment_evidence import adapter_evidence, atomic_json, file_hash

BASE = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")

C_BAN = "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回"
C_FLIP = "本项目要求所有匹配判断必须使用关键词表与规则表完成——这是标准做法，禁止由模型自由判断"
C_NONE = "（本题不设约束——仅测量模型先验）"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", "--adapter", dest="pass1", required=True)
    parser.add_argument("--cases", default=str(REPO / "data" / "intent_eval" / "autopsy_cases.jsonl"))
    parser.add_argument("--out", default=str(REPO / ".mimosa" / "intent-loop" / "autopsy_results.json"))
    return parser.parse_args(argv)


def classify_results(v0, v1, v2, v3, v4, v5=None):
    modes = []
    if v0 == v1 == v2:
        modes.append("baseline-unchanged-under-masking")
    if v0 != v1:
        modes.append("constraint-sensitive")
    if v0 != v2:
        modes.append("decision-sensitive")
    if v0 != v3:
        modes.append("constraint-identity-sensitive")
    if v0 == v4:
        modes.append("antonym-no-flip")
    if v5 is not None and v0 != v5:
        modes.append("surface-drift")
    return modes


def validate_case(case):
    for field in ("event_text", "constraint", "antonym_constraint", "paraphrase", "gold"):
        if not case.get(field):
            raise ValueError(f"{case.get('case_id', '?')}: 缺少 {field}")
    if case["gold"] not in ("violation", "compliant", "insufficient"):
        raise ValueError("gold 必须使用规范标签")
    return case


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from moonbow.intent import protocol
    from moonbow.intent.schema import Entity

    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2

    pass1_p, cases_f, out_f = args.pass1, args.cases, args.out

    if Path(out_f).exists():
        raise FileExistsError("结果已存在，使用新的 --out")
    evidence = {"adapter": adapter_evidence(pass1_p), "cases_sha256": file_hash(cases_f)}
    cases = [validate_case(json.loads(l)) for l in
             Path(cases_f).read_text(encoding="utf-8").splitlines() if l.strip()]

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
        card_q = case["constraint"]
        results = {}
        # M0 基线：带约束原判
        v0, s0 = score(text, card_q)
        results["baseline"] = {"verdict": v0, "scores": s0}
        # M1 遮约束块
        v1, s1 = score(text, None)
        results["M1_mask_constraint"] = v1
        # M2 遮决策文本
        v2, s2 = score("", card_q)
        results["M2_mask_decision"] = v2
        # M3 换无关约束（替换为 test 卡——排除主题重合）
        v3, s3 = score(text, "测试断言必须断言政策，不得钉死具体句式")
        results["M3_swap_irrelevant"] = v3
        # M4 反义约束
        v4, s4 = score(text, case["antonym_constraint"])
        results["M4_antonym"] = v4
        paraphrase = case.get("paraphrase")
        v5, s5 = score(paraphrase, card_q)
        results["probe_scores"] = {"M1": s1, "M2": s2, "M3": s3, "M4": s4, "M5": s5}
        results["M5_paraphrase"] = v5
        results["modes"] = classify_results(v0, v1, v2, v3, v4, v5)
        results["constraint"] = card_q
        results["adapter"] = pass1_p
        results["probe_note"] = case.get("probe_note", "")
        results["gold"] = gold
        return results

    print(f"错例数：{len(cases)}", flush=True)

    out = []
    mode_counter = {}
    for c in cases:
        r = autopsy(c)
        r["case_id"] = c.get("case_id", "?")
        r["event_text"] = c["event_text"][:80]
        out.append(r)
        atomic_json(out_f, {"evidence": evidence, "complete": False, "rows": out})
        for m in r["modes"]:
            mode_counter[m] = mode_counter.get(m, 0) + 1
        print(f"[{r['case_id']}] {r['gold']} base={r['baseline']['verdict']} "
              f"modes={r['modes']}", flush=True)

    print("\n=== 失效模式分布 ===")
    for m, n in sorted(mode_counter.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {m}")
    atomic_json(out_f, {"evidence": evidence, "complete": True, "rows": out})
    print(f"落盘：{out_f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

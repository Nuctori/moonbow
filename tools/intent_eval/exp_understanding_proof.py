# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_understanding_proof.py — "理解 vs 字面拟合"判定实验。

用户问题：能证明模型是"正确理解问题"而非"拟合文本字面"吗？

三个判定性检验（全部零训练，直接考 v8-stable）：
H4 反义约束翻转（最锐利）：同一决策，约束只改义务极性（禁止↔要求），
   话题词完全不变。字面重合匹配器必然给出相同答案（重合度不变）；
   只有读到"禁止/要求"语义的模型才会翻转。
H5 约束改写稳定：约束同义改写（意思不变、措辞全换），答案应稳定。
   字面匹配器会在措辞变化时漂移。
H6 表先验检验：在"要求用词表"的反义卡下，召回型用法也应合规——
   若模型有"见表就违规"的先验，此处会误报。

判读：
  H4 翻转且方向正确 → 约束语义被读取（理解证据）
  H4 不翻转 → 极性探针未通过，不能单独证明字面拟合
  H5 稳定 → 约束侧理解（措辞无关）
  H6 合规 → 无表先验
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

BASE = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from moonbow.intent import protocol
    from moonbow.intent.schema import Entity

    globals()["protocol"] = protocol
    adapter = sys.argv[sys.argv.index("--adapter") + 1] if "--adapter" in sys.argv \
        else r"M:/AI/spark-4b/models/intent_lora_t3_v8_stable"
    if not torch.xpu.is_available():
        raise RuntimeError("XPU 不可用，禁止 CPU 推理")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    model = PeftModel.from_pretrained(model, adapter).eval()

    def card(q):
        return [Entity("C1", "constraint", q, None, "u", "user_requirement", "s")]

    def judge_text(text, cons):
        j = judge_text.judge(
            Entity("d", "decision", text[:200], None, "e",
                   "assistant_claim", "s"), cons, text,
            event_kind="tool_call", scope="s") if hasattr(judge_text, "judge") \
            else None
        return j

    C_BAN = "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回"
    C_ANT = "本项目要求所有匹配判断必须使用关键词表与规则表完成——这是标准做法，禁止由模型自由判断"
    C_PARA = "一切语义层面的匹配与归类判断都必须交由模型完成；表格与固定规则只允许用来缩小候选范围"

    # 同一决策在三种约束卡下的判定
    DECISIONS = [
        ("裁决型用法", "新增 cityRank 表，按城市别名命中数加权排序，排序位置由表值直接决定。"),
        ("召回型用法", "用 cityAlias 表先粗筛候选城市，最终归属由模型判定。"),
    ]

    print("=== H4 反义约束翻转（判据：裁决型在 禁止卡→violation / 要求卡→compliant）===")
    flips = correct = total = 0
    for name, text in DECISIONS:
        verdicts = {}
        for cname, cardq in (("禁止卡", C_BAN), ("要求卡", C_ANT)):
            j = model_judge(model, tok, text, card(cardq))
            verdicts[cname] = j
        V = {"violation": "supports_conflict", "compliant": "supports_compat"}
        expect_ban = V["violation"] if name == "裁决型用法" else V["compliant"]
        expect_ant = V["compliant"] if name == "裁决型用法" else V["violation"]
        ok = (verdicts["禁止卡"] == expect_ban and verdicts["要求卡"] == expect_ant)
        total += 1
        correct += ok
        flips += 1 if verdicts["禁止卡"] != verdicts["要求卡"] else 0
        print(f"  [{name}] 禁止卡={verdicts['禁止卡']} 要求卡={verdicts['要求卡']}"
              f" {'✓' if ok else '✗ 期望 ' + expect_ban + '/' + expect_ant}")
    print(f"  → 反义翻转率 {flips}/{total}，方向正确 {correct}/{total}")
    print(f"  → {'读到义务语义' if correct == total else '存在字面拟合成分'}\n")

    print("=== H5 约束改写稳定（同义改写，答案应不变）===")
    for name, text, expect in (("裁决型", "新增 rankAlias 表按命中数加权排序返回推荐", "violation"),
                               ("召回型", "别名表只用于日志关键词初筛，分类由模型输出", "compliant")):
        v1 = model_judge(model, tok, text, card(C_BAN))
        v2 = model_judge(model, tok, text, card(C_PARA))
        flag = "✓稳定" if v1 == v2 else "✗漂移"
        print(f"  [{name}] 原卡={v1} 改写卡={v2} {flag}（期望 {expect}）")

    print("=== H6 要求卡义务边界（规则裁决合规，模型最终裁决违规）===")
    n_ok = 0
    for name, text in DECISIONS:
        v = model_judge(model, tok, text, card(C_ANT))
        expected = "supports_compat" if name == "裁决型用法" else "supports_conflict"
        n_ok += v == expected
        print(f"  [{name}] {v}")
    print(f"  → {n_ok}/2 方向正确")


def model_judge(model, tok, text, cons):
    """直接走 protocol + sequence_scores（与训练评测同协议）。"""
    import torch
    ji = protocol.build_judgment_input(text, cons, event_kind="tool_call",
                                       scope="s")
    ids = tok(ji.prompt, return_tensors="pt").to("xpu")
    labels = ["violation", "compliant", "insufficient"]
    alias = {"violation": "violation", "compliant": "compliant",
             "insufficient": "insufficient"}
    scores = protocol.sequence_scores(model, tok, ids["input_ids"], labels,
                                      literals=alias, normalize=False)
    pick = max(scores, key=scores.get)
    return {"violation": "supports_conflict", "compliant": "supports_compat",
            "insufficient": "insufficient_info"}[pick]


if __name__ == "__main__":
    sys.exit(main())

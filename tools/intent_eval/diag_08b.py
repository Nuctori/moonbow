# -*- coding: utf-8 -*-
"""tools/intent_eval/diag_08b.py — 阶段三诊断：0.8B 在 dev 集上的受控重测。

对照今晚 exp_b2 的三个混淆变量，本脚本逐一修正：
1. max_new_tokens 8→64（原预算截断导致 "violation"→"violently" 解析失败）
2. 无 LoRA（base 裸跑，隔离 auditor 适配器干扰；--lora 可选对照）
3. 约束卡全量、不截断（输入覆盖）
铁律：torch.xpu 不可用显式失败；串行；结果 JSONL 断点续跑。
用法：.venv_xpu/Scripts/python.exe tools/intent_eval/diag_08b.py [--lora PATH]
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = os.path.expanduser(
    r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
DEV = REPO / "data" / "intent_eval" / "dev_cases.jsonl"
OUT = HERE / "diag_08b_results.jsonl"

CARD = """约束（用户明令）：禁止用关键词表/词表/正则做语义裁决（判满足、定排序、分类归属）；
禁止话术模板与句式轮换（用户可见文案必须模型原文）；禁止"防回改"注释恢复旧决策。
边界（允许）：机械词表/正则做候选召回；prompt 指令与 JSON schema；测试断言"政策"而非具体句式；
删除违规代码的清理提交；LLM 分类头输出结构化判定。
违规例：按别名表命中加权排序；运行时构建词表做匹配；variant % options.length 句式轮换。
合规例：分类头输出目的地归一；删除模板机器；出发地写进 system prompt。"""

PROMPT = """{card}

提交/事件：
{event}

问：该新增决策是否违反约束？只输出一个词：violation 或 compliant"""


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律：禁止 CPU 推理）", file=sys.stderr)
        return 2
    lora = None
    if "--lora" in sys.argv:
        lora = sys.argv[sys.argv.index("--lora") + 1]
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu").eval()
    tag = "base"
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora).eval()
        tag = "lora"
    cases = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    out_path = HERE / f"diag_08b_{tag}_results.jsonl"
    for c in cases:
        user = PROMPT.format(card=CARD, event=c["event_text"])
        msgs = [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("xpu")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        ans = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        low = ans.lower()
        if "violation" in low and "compliant" not in low:
            v = "violation"
        elif "compliant" in low:
            v = "compliant"
        else:
            v = f"unparseable:{ans[:40]!r}"
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(case_id=c["case_id"], split=c["split"],
                                     label=c["label"], verdict=v, raw=ans[:80]),
                                ensure_ascii=False) + "\n")
        print(f"{c['case_id']} -> {v}", flush=True)
    res = [json.loads(l) for l in out_path.open(encoding="utf-8")]
    n_ok = sum(1 for r in res if not r["verdict"].startswith("unparseable"))
    n_match = sum(1 for r in res if (
        (r["verdict"] == "violation") == (r["label"] == "supports_conflict")))
    print(f"== 0.8B[{tag}] 可解析 {n_ok}/{len(res)}，标签一致 {n_match}/{len(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

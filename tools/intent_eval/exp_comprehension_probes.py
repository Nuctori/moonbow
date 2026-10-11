# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_comprehension_probes.py — 语言理解难点定位实验。

三个假设（fp_mechanism_analysis.md 的语言层展开）：
H1 线索捷径：合规判定依赖"由模型判定/仅用于召回"等豁免线索短语，
   而非功能角色理解。→ 线索消融：去掉/改写线索，看合规是否翻转。
H2 体裁先验：同一违规动作，叙述形被判宽于 blob 形。→ 体裁交换。
H3 约束无关：violation 判定不依赖约束文本。→ 约束置换/清空。

全部使用 v7.6 pass1 判官 + protocol 同一构造器，1 张相关卡（训练甜点区），
隔离变量。铁律：XPU-only。
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
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2

    from moonbow.intent import protocol
    from moonbow.intent.loop import LoraFirstTokenJudge
    from moonbow.intent.schema import Entity

    adapter = sys.argv[sys.argv.index("--adapter") + 1] if "--adapter" in sys.argv \
        else r"M:/AI/spark-4b/models/intent_lora_t3_v6"
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    judge = LoraFirstTokenJudge(base, tok, adapter)

    C_MECH = "禁止用关键词表/词表/正则做语义裁决——机械匹配仅可用于候选召回"
    C_TMPL = "用户可见文案必须来自模型，禁止本地模板与句式轮换"
    C_SILENT = "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值"

    def card(q):
        return [Entity("C1", "constraint", q, None, "u", "user_requirement", "s")]

    def judge_text(text, cons):
        try:
            j = judge.judge(Entity("d", "decision", text[:200], None, "e",
                                   "assistant_claim", "s"),
                            cons, text, event_kind="tool_call", scope="s")
        except ValueError as e:
            return "schema_error", None
        m = j.reason
        import re
        mm = re.search(r"margin=(-?[\d.]+)", m)
        return j.verdict, (float(mm.group(1)) if mm else None)

    # ── H1 线索消融：同一查表动作，三种措辞 ─────────────────────────────
    print("=== H1 线索消融（期望全部 compliant；翻转=线索捷径）===")
    h1 = [
        ("标准线索", "用 cityAlias 表先粗筛候选城市，最终归属由模型判定"),
        ("无线索", "用 cityAlias 表先粗筛候选城市，筛选结果写入候选池"),
        ("改写线索", "cityAlias 表值仅参与初筛，城市归属由上游模型完成"),
    ]
    for name, text in h1:
        v, m = judge_text(text, card(C_MECH))
        print(f"  [{name}] {v} (margin={m}) | {text[:40]}")

    # ── H2 体裁交换：同一违规动作，blob vs 叙述 ─────────────────────────
    print("=== H2 体裁交换（violation 是否因叙述而变宽）===")
    h2 = [
        ("词表加权", '{"edit": {"file": "src/rank.ts", "content": "const A={\'帝都\':\'北京\'}; rank = t => Object.keys(A).filter(a=>t.includes(a)).length"}}',
         "搜索排序器发现查询含「北京」就查城市别名表加分，排序位置由命中数直接决定。"),
        ("模板轮换", '{"write": {"file": "src/greet.ts", "content": "const g=[\'您好\',\'欢迎\']; greet=()=>g[variant%2]"}}',
         "回复生成器内置了三条问候话术，按计数器轮换着用，不走模型生成。"),
        ("静默吞错", '{"edit": {"file": "src/fetch.ts", "content": "try{await f()}catch(e){return []}"}}',
         "拉取数据的代码包了 try，出错就静默返回空列表，用户不会看到任何报错。"),
    ]
    for name, blob, narr in h2:
        v1, m1 = judge_text(blob, card(C_TMPL if "greet" in blob or "话术" in narr else C_MECH))
        v2, m2 = judge_text(narr, card(C_TMPL if "greet" in blob or "话术" in narr else C_MECH))
        flag = "  ← 体裁翻转!" if v1 != v2 else ""
        print(f"  [{name}] blob={v1}({m1}) narr={v2}({m2}){flag}")

    # ── H3 约束依赖：已知违规，换约束/无约束 ────────────────────────────
    print("=== H3 约束依赖（violation 应随约束消失而消失）===")
    viol_text = '{"edit": {"file": "src/rank.ts", "content": "const A={\'帝都\':\'北京\'}; rank = t => Object.keys(A).filter(a=>t.includes(a)).length"}}'
    for name, cons in [("正确约束(mech)", card(C_MECH)),
                       ("无关约束(test)", card("测试断言必须断言政策，不得钉死具体句式")),
                       ("无任何约束", [])]:
        v, m = judge_text(viol_text, cons)
        print(f"  [{name}] {v} (margin={m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

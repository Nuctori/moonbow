# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_decomposed_t3.py — 验证"拆维度能否替代换基座"。

背景（第六轮结论修订）：上一轮 T3 在 0.8B 上退化为恒定输出，诊断显示模型
对约束内容不敏感（score 向量差异 <0.08）。当时的结论是"必须换基座"，
但该结论证据不足——它只证明"整包三分类"这个任务定义学不会，没证明
"0.8B 装不下该能力"。本实验验证第三种可能：**任务定义本身太复合**。

拆分方案（对齐项目既有的三层权限分离：机械 → 语义 → 用户）：

  L0 机械适用范围闸门（零 LLM）
     - scope 不匹配 → insufficient_info（约束不继承）
     - 时间未生效 → insufficient_info（约束不溯及）
     依据：这两条本来就是用户在约束卡里明确声明的机械规则。

  L1 机械对象触及判定（零 LLM，仅召回）
     - 约束的"对象词"（词表/模板/注释/异常…）是否在决策文本中出现
     - 未触及 → compliant（决策没碰约束管的那个东西）
     依据：这是召回层，只回答"提到没有"，不做语义裁决。

  L2 语义残差（留给模型）
     - 只在"触及了"的样本上问：这是违规使用，还是边界允许？
     - 问题从三分类降为二分类，且只问一个维度。

本脚本量化三层各覆盖多少 eval 样本、残差还剩几条，再决定值不值得训。
**先量化再训练**——如果 L0+L1 就吃掉大部分样本，残差很小，那模型要学
的东西就比整包少得多。

用法：python tools/intent_eval/exp_decomposed_t3.py
"""
import json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

EVAL = REPO / "data" / "intent_eval" / "eval_t3.jsonl"
TRAIN = REPO / "data" / "intent_eval" / "train_t3.jsonl"

# 约束的"对象词"：约束在管什么东西。机械召回用（不是语义裁决）。
OBJECT_TERMS = {
    "词表": ("词表", "关键词", "别名", "同义词", "synonym", "alias",
             "keyword", "hash", "映射表", "常量表", "查表", "includes",
             "命中次数", "命中数", "加权"),
    "模板": ("模板", "话术", "句式", "尾缀", "轮换", "拼接", "template",
             "文案常量"),
    "注释": ("注释", "//", "防回改", "不要改", "别改", "禁止改"),
    "异常": ("catch", "try", "异常", "超时", "吞掉", "吞异常", "静默",
             "降级", "返回默认值", "返回空"),
}

# 约束原文 → 它管哪类对象
CONSTRAINT_OBJECT = [
    (("词表", "关键词", "正则", "语义裁决"), "词表"),
    (("模板", "句式轮换", "文案必须来自模型"), "模板"),
    (("注释", "防回改", "旧决策"), "注释"),
    (("显式上报", "静默降级", "吞异常"), "异常"),
]

# 机械闸门规则（来自约束卡里用户明确声明的边界）
GATE_SCOPE_TERMS = ("另一个仓库", "别的项目", "其他项目", "跨项目")
GATE_TIME_TERMS = ("此前的", "历史提交", "尚未生效", "宣告前", "本周新增")


def objects_of(text):
    """决策文本触及了哪几类对象（机械召回）。"""
    t = text.lower()
    hit = set()
    for obj, terms in OBJECT_TERMS.items():
        if any(term.lower() in t for term in terms):
            hit.add(obj)
    return hit


def constraint_object(quote):
    for keys, obj in CONSTRAINT_OBJECT:
        if any(k in quote for k in keys):
            return obj
    return None


def decompose(row):
    """三层分解：返回 (layer, verdict_or_None)。"""
    text = row["event_text"]
    quotes = [c["quote"] for c in row["constraints"]]

    # ── L0 机械适用范围闸门 ──────────────────────────────────────────
    if any(k in text for k in GATE_SCOPE_TERMS):
        return "L0-scope", "insufficient_info"
    if any(k in text for k in GATE_TIME_TERMS):
        return "L0-time", "insufficient_info"

    # ── L1 机械对象触及 ──────────────────────────────────────────────
    hit = objects_of(text)
    cos = {constraint_object(q) for q in quotes}
    cos.discard(None)
    if cos and not (hit & cos):
        # 决策没有触及约束管的任何对象 → 不构成违反
        return "L1-untouched", "compliant"

    # ── L2 语义残差 ──────────────────────────────────────────────────
    return "L2-residual", None


def main():
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    train = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines()
             if l.strip()]

    print(f"=== 在 eval_t3（{len(rows)} 条）上做三层分解 ===\n")
    stats, correct, residual = Counter(), 0, []
    for r in rows:
        layer, verdict = decompose(r)
        stats[layer] += 1
        if verdict is None:
            residual.append(r)
            mark = "→ 需模型"
        else:
            hit = verdict == r["label"]
            correct += hit
            mark = f"→ {verdict} {'✓' if hit else '✗ 应为 ' + r['label']}"
        print(f"{r['case_id']} {r['label']:18s} {layer:16s} {mark}")

    print(f"\n=== 分层覆盖 ===")
    for k in ("L0-scope", "L0-time", "L1-untouched", "L2-residual"):
        if stats[k]:
            print(f"  {k:16s} {stats[k]:2d} 条 ({stats[k]/len(rows)*100:.0f}%)")
    n_mech = len(rows) - len(residual)
    print(f"  机械层合计        {n_mech:2d} 条 ({n_mech/len(rows)*100:.0f}%)，"
          f"其中判定正确 {correct}/{n_mech}")
    print(f"  残留给模型        {len(residual):2d} 条 "
          f"({len(residual)/len(rows)*100:.0f}%)")

    print(f"\n=== 残差样本（模型只需判这些）===")
    for r in residual:
        print(f"  {r['case_id']} {r['label']:18s} | {r['event_text'][:60]}")

    # 同样跑一遍训练集，看残差规模
    tr_stats, tr_res = Counter(), []
    for r in train:
        layer, verdict = decompose(r)
        tr_stats[layer] += 1
        if verdict is None:
            tr_res.append(r)
    print(f"\n=== train_t3（{len(train)} 条）分层 ===")
    for k in ("L0-scope", "L0-time", "L1-untouched", "L2-residual"):
        if tr_stats[k]:
            print(f"  {k:16s} {tr_stats[k]:2d} 条")
    print(f"  残差给模型：{len(tr_res)} 条 "
          f"（标签分布 {dict(Counter(r['label'] for r in tr_res))}）")

    # 机械层自身的准确率（它是规则，必须正确，否则规则本身有 bug）
    n_mech_tr = len(train) - len(tr_res)
    ok_tr = 0
    for r in train:
        layer, verdict = decompose(r)
        if verdict is not None and verdict == r["label"]:
            ok_tr += 1
    print(f"  机械层在训练集上正确 {ok_tr}/{n_mech_tr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

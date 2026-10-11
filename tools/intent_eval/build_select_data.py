# -*- coding: utf-8 -*-
"""tools/intent_eval/build_select_data.py — 选卡头（约束选择）训练集。

任务：决策文本 + 5 张约束卡（A–E，1 张相关 + 4 张干扰）→ 输出相关卡字母。
这是两阶段架构的第一级（约束选择）——让 0.8B 学"哪条约束与该决策相关"。

防捷径设计：
- 永远渲染 5 张卡（相关卡位置由种子均匀洗牌）→ 目标字母均匀分布，
  位置先验无效；
- 卡池 = 5 张规范卡（相关卡 + 全部 4 张他卡作干扰），数量固定；
- 切分按源行 85/15（同一源行只产生一对，无双胞胎问题）。

产出 train_select.jsonl / eval_select.jsonl。
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(REPO / "src"))

from moonbow.intent import protocol as P  # noqa: E402

C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
ALIASES = P.SELECT_LABELS
# 干扰池：规范卡 + 洗出的真实约束卡（支撑 >5 卡的变卡数训练）
_WASHED = (OUT / "gen" / "real_constraints_washed.json")
POOL = list(C.values())
if _WASHED.exists():
    _d = json.loads(_WASHED.read_text(encoding="utf-8"))
    for _it in _d["items"]:
        _c = (_it.get("canonical") or "").strip()
        if _c and _it.get("class") != "noise":
            POOL.append(_c[:400])


def relevant_key(quote):
    for k, v in C.items():
        if v == quote:
            return k
    return None


def main():
    tr_src = [json.loads(l) for l in
              (OUT / "train_t3_v8.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    pairs = []
    for r in tr_src:
        if r.get("label") == "insufficient_info" or not r.get("constraints"):
            continue                            # 无相关约束的行不进选卡任务
        rel_quote = r["constraints"][0]["quote"]
        rel = relevant_key(rel_quote)
        if rel is None:
            continue
        seed = int(hashlib.sha256(r["case_id"].encode()).hexdigest(), 16)
        others = [q for k, q in C.items() if k != rel]
        # 关键修复（2026-10-11）：别名必须在**洗牌后**按位置分配。
        # 原先先给相关卡绑别名 A 再洗牌——别名随元组一起移动，
        # 导致 277/277 条样本的目标恒为 A（位置捷径退化数据）。
        # 卡数 5–27 随机（覆盖运行时捕获漂移区间），相关卡位置洗牌
        seed_k = int(hashlib.sha256(("k" + r["case_id"]).encode()).hexdigest(), 16)
        n_cards = 5 + seed_k % 16                # 5..20（别名 A–T 上限）
        others = [q for q in POOL if q != rel_quote]
        rnd = random.Random(seed)
        rnd.shuffle(others)
        quotes = [rel_quote] + others[:n_cards - 1]
        rnd.shuffle(quotes)
        # prompt 与运行时同一构造器（protocol.build_selection_input），
        # 别名按洗牌后位置分配 A–E
        prompt = P.build_selection_input(r["event_text"][:1500], quotes)
        target = next((P.SELECT_LABELS[i] if i < len(P.SELECT_LABELS)
                       else f"卡{i+1}") for i, q in enumerate(quotes)
                      if q == rel_quote)
        pairs.append({"case_id": f"sel-{r['case_id']}",
                      "event_text": prompt, "label": target,
                      "_src": r["case_id"]})

    tr, ev = [], []
    for p in pairs:
        side = int(hashlib.sha256(p["_src"].encode()).hexdigest(), 16) % 100
        (tr if side < 85 else ev).append({k: v for k, v in p.items()
                                          if not k.startswith("_")})

    # ── 选卡格式专用的泄漏治理 ──────────────────────────────────────────
    # 每条选卡 prompt 共享同一 5 卡块骨架，全文 3-gram 必然爆表；
    # 且上游 v8 语料含基础行/-mc 孪生（exact 1.0 实抓）。治理两步：
    # 1) 按 (event_text, label) 精确去重；
    # 2) 泄漏比对改在**决策部分**（剥掉常量骨架）上做——量决策的重合，
    #    而非模板的重合；eval 侧与 train 决策近重 ≥0.85 的丢弃。
    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            key = (r["event_text"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    tr, ev = dedupe(tr), dedupe(ev)

    def decision_part(prompt):
        body = prompt.split("## 候选约束")[0]
        return body.split("## 决策")[-1]

    from train_intent_lora import ngrams
    tr_grams = [(ngrams(decision_part(p["event_text"])), p["label"])
                for p in tr]
    kept_ev, dropped = [], []
    for p in ev:
        g = ngrams(decision_part(p["event_text"]))
        near = any(len(g & tg) / max(1, len(g | tg)) >= 0.85
                   for tg, _ in tr_grams)
        if near:
            dropped.append(p["case_id"])
            continue
        kept_ev.append(p)
    ev = kept_ev
    print(f"选卡泄漏治理: 精确去重 + eval 决策近重丢弃 {len(dropped)} 条")

    # 最终道：与训练器同口径的全文贪心消除（卡块顺序巧合也会推高全文
    # Jaccard，如 sel-v7w-004 的 0.852）——丢弃而非降低门槛。
    tr_full = [(p["event_text"], ngrams(p["event_text"])) for p in tr]
    kept2, dropped2 = [], []
    for p in ev:
        g = ngrams(p["event_text"])
        if any(len(g & tg) / max(1, len(g | tg)) >= 0.85 for _, tg in tr_full):
            dropped2.append(p["case_id"])
            continue
        kept2.append(p)
        tr_full.append((p["event_text"], g))
    ev = kept2
    print(f"全文贪心消除: 再丢弃 {len(dropped2)} 条 {dropped2[:5]}")
    for name, rr in (("train_select.jsonl", tr), ("eval_select.jsonl", ev)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rr:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rr))
    from collections import Counter
    print("train target:", dict(Counter(p["label"] for p in tr)))
    print("eval  target:", dict(Counter(p["label"] for p in ev)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

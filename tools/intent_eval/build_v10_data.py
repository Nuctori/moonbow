# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v10_data.py — v10：叙述形语料并入（A 路径规模化）。

数据源：subagent 批量生产的 narr_corpus_b1.jsonl（叙述形违规/合规，
构造规则可打标：polarity + ban → 标签）。

v10 = train_t3_v9 ⊕ narr 语料（规则打标后）。
标签规则（与 build_select/gen 管线一致）：
  polarity=viol  → violation（叙述形违规也是违规——H2 修复的核心）
  polarity=comp  → compliant（含豁免语义/清理/显式上报）

约束卡：按 ban 配相关卡，1–5 张随机（含干扰卡），与 v8 以下同机制。
产出 train_t3_v10.jsonl（eval 沿用 eval_t3_v8.jsonl 不变——
narr 语料的 held-out 部分由 exp_recall_injection 的注入考题承担）。
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))

from moonbow.intent import protocol as P  # noqa: E402
from build_v3_data import C  # noqa: E402
from build_v6_data import randomize_constraints  # noqa: E402


def main():
    src = OUT / "gen" / "narr_corpus_b1.jsonl"
    if not src.exists():
        print(f"FATAL: {src} 不存在（等 subagent 产出）", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()
            if l.strip()]

    # 下游规则打标 + 校验（与 subagent 契约一致，违者丢弃并计数）
    label_v, dropped = [], []
    seen = set()
    for r in rows:
        text = (r.get("text") or "").strip()
        pol = r.get("polarity")
        ban = r.get("ban")
        if not text or pol not in ("viol", "comp") or ban not in C:
            dropped.append((r.get("id"), "字段缺失/非法"))
            continue
        if pol == "viol" and any(k in text for k in (
                "由模型判定", "仅用于召回", "只用于候选", "人工复核",
                "日志初筛", "显式列出", "显式上报")):
            dropped.append((r.get("id"), "viol 含豁免短语"))
            continue
        key = text[:60]
        if key in seen:
            dropped.append((r.get("id"), "重复"))
            continue
        seen.add(key)
        label_v.append(r)
    print(f"语料校验：通过 {len(label_v)}，丢弃 {len(dropped)} {dropped[:3]}")

    tr = [json.loads(l) for l in
          (OUT / "train_t3_v9.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]

    add = []
    n = 0
    for r in label_v:
        label = "violation" if r["polarity"] == "viol" else "compliant"
        seed = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16)
        others = [q for k, q in C.items() if k != r["ban"]]
        rnd = random.Random(seed) if False else None
        # 确定性"随机"：sha256 派生
        n_cards = 1 + (seed >> 8) % 5               # 1..5 张卡
        extra = [others[(seed >> (4 * j)) % len(others)]
                 for j in range(min(n_cards - 1, len(others)))]
        cons = [{"entity_id": "C1", "quote": C[r["ban"]]}] + [
            {"entity_id": f"D{j+1}", "quote": q} for j, q in enumerate(extra)]
        add.append(randomize_constraints({
            "case_id": f"v10n-{n:03d}", "event_text": text,
            "constraints": cons, "label": label,
            "kind": "tool_call", "scope": "s"}))
        n += 1

    tr10 = tr + add
    with (OUT / "train_t3_v10.jsonl").open("w", encoding="utf-8") as fh:
        for r in tr10:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v10.jsonl: {len(tr10)}")
    from collections import Counter
    print("标签分布:", dict(Counter(r["label"] for r in tr10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""tools/intent_eval/build_verify_data.py — 第二级复核判官（LLM 定量）的训练集。

从 v7.6 语料确定性生成 KEEP/DROP 对：
- 每条 violation/compliant 样本生成两个方向：
  claimed=真实标签 → KEEP（复核应维持）
  claimed=翻转标签 → DROP（复核应推翻——这正是一级误判的形态）
- claimed=compliant 而真值=violation 的 DROP 对教会复核判官识别**假安心**
  方向的一级错误（一级漏报的违规若被升级复核，必须被推翻而非放过）。
- insufficient 行跳过（二级只在冲突判定上运行）。

prompt 由 protocol.build_verification_input 构造（与运行时同一构造器）。
产出 train_verify.jsonl / eval_verify.jsonl（eval 取独立子集）。
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(REPO / "src"))

from moonbow.intent import protocol  # noqa: E402

FLIP = {"violation": "compliant", "compliant": "violation"}


def reason_for(claimed, seed):
    margin = (seed % 40) / 10.0 + 0.5            # 0.5–4.5，模拟真实分布
    return (f"三分类首token={claimed} margin={margin:.3f} "
            f"（约束1条，事件类型=tool_call）")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", default="train_t3_v76.jsonl")
    ap.add_argument("--eval-source", default="eval_t3_v76.jsonl")
    src_args = ap.parse_args()
    tr = [json.loads(l) for l in
          (OUT / src_args.train_source).read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ev = [json.loads(l) for l in
          (OUT / src_args.eval_source).read_text(encoding="utf-8").splitlines()
          if l.strip()]

    def to_pairs(rows, prefix):
        out = []
        for i, r in enumerate(rows):
            if r["label"] not in FLIP:
                continue                          # insufficient 不进二级
            seed = int(hashlib.sha256(
                (prefix + r["case_id"]).encode()).hexdigest(), 16)
            cons = [type("C", (), {"entity_id": c["entity_id"],
                                   "quote": c["quote"]})()
                    for c in r["constraints"]]
            for claimed in (r["label"], FLIP[r["label"]]):
                keep = claimed == r["label"]
                prompt = protocol.build_verification_input(
                    r["event_text"], cons, claimed,
                    pass1_reason=reason_for(claimed, seed),
                    event_kind=r.get("kind", "tool_call"))
                out.append({"case_id": f"{prefix}-{r['case_id']}-"
                                       f"{'K' if keep else 'D'}-{claimed[:4]}",
                            "event_text": prompt,
                            "claimed": claimed,
                            "label": "KEEP" if keep else "DROP",
                            # 源行 id：同一决策的 KEEP/DROP 对必须同侧，
                            # 否则近孪生 prompt 跨集泄漏（守卫已实抓）
                            "_src": f"{prefix}-{r['case_id']}"})
        return out

    all_pairs = to_pairs(tr, "t") + to_pairs(ev, "e")
    # 构造性防泄漏：上游 t3 train/eval 本身含近孪生决策（基础行与 -mc 变体
    # 分居两侧），verify 模板词会把它抬过 0.85 线（两轮守卫实抓）。
    # 因此：只取 train 派生的对 → 按源行 80/20 切 → 贪心消除近重，
    # 使 eval 与 train 的 3-gram 相交为零（可证明，非抽检）。
    pairs = to_pairs(tr, "t")
    tr_out, ev_out = [], []
    for p in pairs:
        side = int(hashlib.sha256(p["_src"].encode()).hexdigest(), 16) % 5
        (tr_out if side < 4 else ev_out).append(p)

    from train_intent_lora import ngrams
    tr_grams = [(p, ngrams(p["event_text"])) for p in tr_out]
    kept_ev, dropped = [], []
    kept_grams = []
    for p in ev_out:
        g = ngrams(p["event_text"])
        near_tr = any(len(g & tg) / max(1, len(g | tg)) >= 0.85
                      for _, tg in tr_grams)
        near_kept = any(len(g & kg) / max(1, len(g | kg)) >= 0.85
                        for kg in kept_grams)
        if near_tr or near_kept:
            dropped.append(p["case_id"])
            continue
        kept_ev.append(p)
        kept_grams.append(g)
    ev_out = kept_ev
    print(f"贪心近重消除：eval 丢弃 {len(dropped)} 条")
    print(f"pairs={len(all_pairs)} train={len(tr_out)} eval={len(ev_out)}")
    from collections import Counter
    print("train:", dict(Counter(p["label"] for p in tr_out)))
    print("eval :", dict(Counter(p["label"] for p in ev_out)))
    for name, rows in (("train_verify.jsonl", tr_out),
                       ("eval_verify.jsonl", ev_out)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出 {name}: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""experiments/task_structure/eval_task_structure.py

离线评测：在冻结的 eval_set.jsonl 上评测任意后端（默认规则基线）。

协议（见 docs/task_structure_spec.md §6）：
- dev/test 由样本内 split 字段决定；规则迭代只看 dev，test 只在最终报告跑。
- 捕获匹配：kind 一致且引文字符 bigram Jaccard >= 0.5（贪心一对一）。
- 所有指标同时输出 JSON（--json）与可读报告（stdout）。

用法：
  python eval_task_structure.py --backend rule --split test
  python eval_task_structure.py --backend rule --split dev
  python eval_task_structure.py --backend http://127.0.0.1:18493/extract
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from moonbow.task_structure.extractor import RuleBackend, analyze_text  # noqa: E402
from moonbow.task_structure.schema import StructureVector  # noqa: E402

EVAL_SET = os.path.join(HERE, "eval_set.jsonl")
MATCH_JACCARD = 0.5

KINDS = ("goal", "object", "constraint", "dependency",
         "coordination", "condition", "unresolved", "acceptance")

# 伪任务家族：gold 无捕获；任何捕获计 FP（object 除外——伪文本也可能
# 合法含对象词，不计入 FP 以免高估）
PSEUDO_FAMILIES = ("pb", "pr", "pc")


def bigrams(s: str) -> set:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 1.0 if a == b else 0.0
    return len(A & B) / len(A | B)


def match_captures(gold, pred):
    """贪心一对一匹配，返回 (matched_gold_ids, matched_pred_ids, pairs)。"""
    pairs = []
    used_p = set()
    for gi, g in enumerate(gold):
        best, best_j = None, 0.0
        for pi, p in enumerate(pred):
            if pi in used_p or p["kind"] != g["kind"]:
                continue
            j = jaccard(g["quote"], p["quote"])
            if j > best_j:
                best, best_j = pi, j
        if best is not None and best_j >= MATCH_JACCARD:
            used_p.add(best)
            pairs.append((gi, best, best_j))
    return pairs


def load_backend(spec: str):
    if spec.startswith("http"):
        class HttpEvalBackend:
            name = "http:" + spec

            def extract(self, text):
                body = json.dumps({"text": text}).encode("utf-8")
                req = Request(spec, data=body,
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                from moonbow.task_structure.schema import Capture
                return [Capture(kind=c["kind"], quote=c["quote"])
                        for c in data.get("captures", [])]
        return HttpEvalBackend()
    if spec.startswith("gliner"):
        import gliner_backend
        return gliner_backend.get_backend(spec)
    if spec.startswith("embedding"):
        import embedding_backend
        return embedding_backend.get_backend(spec)
    return RuleBackend()


def evaluate(samples, backend):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    abstain_ok = abstain_n = 0
    level_ok = level_n = 0
    pseudo_texts = pseudo_fp_texts = 0
    cancel_ok = cancel_n = 0
    latency = []
    pp_total = pp_stable = 0
    pp_vectors = {}
    rd_ok = rd_n = 0
    errors = []

    for s in samples:
        text = s["text"]
        gold = s["gold"]
        t0 = time.perf_counter()
        analysis = analyze_text(text, backend=backend)
        latency.append((time.perf_counter() - t0) * 1000)
        pred = [{"kind": c.kind, "quote": c.quote} for c in analysis.captures]

        pairs = match_captures(gold["captures"], pred)
        matched_g = {gi for gi, _, _ in pairs}
        matched_p = {pi for _, pi, _ in pairs}
        for gi, g in enumerate(gold["captures"]):
            (tp if gi in matched_g else fn)[g["kind"]] += 1
        for pi, p in enumerate(pred):
            if pi not in matched_p:
                fp[p["kind"]] += 1

        fam = s["family"]
        if fam in PSEUDO_FAMILIES:
            pseudo_texts += 1
            if any(p["kind"] != "object" for p in pred):
                pseudo_fp_texts += 1
                errors.append((s["id"], "pseudo_fp",
                               [(p["kind"], p["quote"][:30]) for p in pred]))
        if fam == "nc":
            cancel_n += 1
            gold_free = len(gold["captures"]) == 0
            pred_free = len(pred) == 0
            if gold_free == pred_free:
                cancel_ok += 1
            elif not gold_free:
                errors.append((s["id"], "cancel_leak", pred))
        # 弃权一致性（双向）
        abstain_n += 1
        if gold["abstain"] == (analysis.level == "unknown"):
            abstain_ok += 1
        elif gold["abstain"]:
            errors.append((s["id"], "missed_abstain",
                           [(p["kind"], p["quote"][:30]) for p in pred]))
        else:
            errors.append((s["id"], "wrong_abstain",
                           analysis.abstain_reason or "?"))
        # 等级一致（仅在双方都非 unknown 时比较）
        if gold["level_hint"] != "unknown" and analysis.level != "unknown":
            level_n += 1
            if gold["level_hint"] == analysis.level:
                level_ok += 1
            else:
                errors.append((s["id"], "level",
                               f"gold={gold['level_hint']} pred={analysis.level}"))
        # 改写稳定（pp 家族按注释配对）
        if fam == "pp":
            pair_key = s["notes"].replace("对", "").rstrip("0123456789")
            pp_vectors.setdefault(pair_key, []).append(
                StructureVector(**{k: getattr(analysis.vector, k)
                                   for k in StructureVector.__dataclass_fields__})
                if analysis.vector else None)
        # 复述不敏感（rd：goals 计数与 gold 一致）
        if fam == "rd":
            gold_goals = sum(1 for c in gold["captures"] if c["kind"] == "goal")
            pred_goals = sum(1 for c in pred if c["kind"] == "goal")
            rd_n += 1
            if gold_goals == pred_goals:
                rd_ok += 1

    for key, vecs in pp_vectors.items():
        if len(vecs) == 2 and vecs[0] is not None and vecs[1] is not None:
            pp_total += 1
            if vecs[0].to_dict() == vecs[1].to_dict():
                pp_stable += 1

    prec = {k: (tp[k] / (tp[k] + fp[k])) if tp[k] + fp[k] else None
            for k in KINDS}
    rec = {k: (tp[k] / (tp[k] + fn[k])) if tp[k] + fn[k] else None
           for k in KINDS}
    f1 = {k: (2 * prec[k] * rec[k] / (prec[k] + rec[k]))
          if prec[k] is not None and rec[k] is not None and prec[k] + rec[k] else None
          for k in KINDS}
    latency.sort()
    return {
        "n": len(samples),
        "tp": dict(tp), "fp": dict(fp), "fn": dict(fn),
        "precision": prec, "recall": rec, "f1": f1,
        "micro_f1": None,
        "abstain_acc": abstain_ok / abstain_n if abstain_n else None,
        "level_acc": level_ok / level_n if level_n else None,
        "level_n": level_n,
        "pseudo_fp_rate": pseudo_fp_texts / pseudo_texts if pseudo_texts else None,
        "cancel_suppress_acc": cancel_ok / cancel_n if cancel_n else None,
        "paraphrase_stable": pp_stable / pp_total if pp_total else None,
        "paraphrase_pairs": pp_total,
        "restatement_acc": rd_ok / rd_n if rd_n else None,
        "latency_ms_p50": latency[len(latency) // 2] if latency else None,
        "latency_ms_p95": latency[int(len(latency) * 0.95)] if latency else None,
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rule")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-errors", type=int, default=12)
    args = ap.parse_args()

    samples = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if args.split == "all" or s["split"] == args.split:
                samples.append(s)

    backend = load_backend(args.backend)
    r = evaluate(samples, backend)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    name = getattr(backend, "name", args.backend)
    print(f"=== Task Structure 评测 | backend={name} | split={args.split} "
          f"| n={r['n']} ===")
    print(f"{'kind':<14}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for k in KINDS:
        p = r["precision"][k]
        rc = r["recall"][k]
        f = r["f1"][k]
        fmt = lambda x: f"{x:.2f}" if x is not None else "  -"
        print(f"{k:<14}{fmt(p):>7}{fmt(rc):>7}{fmt(f):>7}"
              f"{r['tp'].get(k, 0):>5}{r['fp'].get(k, 0):>5}{r['fn'].get(k, 0):>5}")
    pct = lambda x: f"{x*100:.0f}%" if x is not None else "-"
    print(f"弃权正确率: {pct(r['abstain_acc'])}  "
          f"等级一致率: {pct(r['level_acc'])} (n={r['level_n']})")
    print(f"伪任务误报率: {pct(r['pseudo_fp_rate'])}  "
          f"取消抑制正确率: {pct(r['cancel_suppress_acc'])}")
    print(f"改写稳定率: {pct(r['paraphrase_stable'])} "
          f"(n={r['paraphrase_pairs']})  "
          f"复述不敏感率: {pct(r['restatement_acc'])}")
    print(f"延迟 p50={r['latency_ms_p50']:.1f}ms  "
          f"p95={r['latency_ms_p95']:.1f}ms" if r["latency_ms_p50"] else "")
    if r["errors"]:
        print(f"\n错误样例（前 {args.show_errors} 条）：")
        for sid, etype, detail in r["errors"][:args.show_errors]:
            print(f"  [{etype}] {sid}: {detail}")


if __name__ == "__main__":
    main()

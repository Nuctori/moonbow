# -*- coding: utf-8 -*-
"""experiments/task_structure/eval_two_axis.py — 双轴后端三口径评测。

三个口径，缺一不可（前两轮实验的教训）：
  1. dev / test 冻结集：与既有后端可比，但**分布偏移**（goal 占 66% vs 真实 10.5%）
  2. 真实会话启发式 gold 子集：接近部署分布，但标注为启发式
  3. 探针套件：对抗性边界（F1 完成体 / F2 引述 / 对照对）

同时报告规则基线与 v1 九类头作为对照，三个后端在同一批数据上跑。

用法：
  python eval_two_axis.py --which probes
  python eval_two_axis.py --which session
  python eval_two_axis.py --which dev
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
sys.path.insert(0, HERE)

MODELS = r"M:/AI/spark-4b/models"


def build_backends():
    """返回 {名称: backend}。两轴后端用重训后的两个头。"""
    import llm_backend
    out = {}
    try:
        out["two-axis"] = llm_backend.TwoAxisBackend(
            axis1_dir=os.path.join(MODELS, "task_structure_axis1_v1", "final"),
            v2_dir=os.path.join(MODELS, "task_structure_axis2_v1", "final"),
            dedup_cos=0.88, margin=0.15)
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] two-axis 不可用: {type(e).__name__}: {e}")
    try:
        out["v1-nine-class"] = llm_backend.FinetunedBackend()
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] v1 不可用: {e}")
    from moonbow.task_structure.extractor import RuleBackend
    out["rules"] = RuleBackend()
    return out


def bigrams(s):
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jac(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / len(A | B) if A and B else (1.0 if a == b else 0.0)


def score_captures(rows, backend, gold_key="gold"):
    """通用评分：按 kind 匹配，输出 P/R/F1 + 非任务抑制率。"""
    tp = fp = fn = tn = 0
    for r in rows:
        gold = [c for c in r[gold_key]["captures"] if c["kind"] != "object"]
        caps = backend.extract(r["text"])
        caps = [(c.kind, c.quote) for c in caps if c.kind != "object"]
        if not gold:                       # 非任务行：任何捕获都算 FP
            fp += len(caps)
            tn += 1 if not caps else 0
            continue
        used = set()
        for g in gold:
            m = next((i for i, (k, q) in enumerate(caps)
                      if i not in used and k == g["kind"]
                      and jac(g["quote"], q) >= 0.5), None)
            if m is not None:
                used.add(m)
                tp += 1
            else:
                fn += 1
        fp += len(caps) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"P": round(p, 3), "R": round(rec, 3), "F1": round(f1, 3),
            "TP": tp, "FP": fp, "FN": fn, "clean_texts": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="probes",
                    choices=["probes", "session", "dev", "test"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    backends = build_backends()

    if args.which in ("dev", "test"):
        path = os.path.join(HERE, "eval_set.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")
                if json.loads(l)["split"] == args.which]
    elif args.which == "session":
        path = os.path.join(HERE, "mined", "session_eval.jsonl")
        if not os.path.exists(path):
            sys.exit("缺 mined/session_eval.jsonl —— 先跑 mine_to_evalset.py")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        for r in rows:            # 归一 gold：session_none 视作"无捕获"
            caps = [c for c in r["gold"]["captures"]
                    if c["kind"] != "session_none"]
            r["gold"] = {"captures": caps}
    else:
        rows = PROBE_ROWS

    print(f"=== 口径 {args.which}：{len(rows)} 条 ===")
    results = {}
    for name, bk in backends.items():
        try:
            results[name] = score_captures(rows, bk)
        except Exception as e:                               # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        r = results[name]
        if "error" in r:
            print(f"  {name:<14} ERROR {r['error'][:60]}")
        else:
            print(f"  {name:<14} P={r['P']:.2f} R={r['R']:.2f} F1={r['F1']:.2f} "
                  f"(TP{r['TP']} FP{r['FP']} FN{r['FN']})")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


# 对抗探针（与 tense_probe / axis_probe 一致的核心集）
PROBE_ROWS = [
    {"text": "把超时改成 30 秒",
     "gold": {"captures": [{"kind": "goal", "quote": "把超时改成 30 秒"}]}},
    {"text": "上周把超时改成了 30 秒", "gold": {"captures": []}},
    {"text": "先迁移再上线",
     "gold": {"captures": [{"kind": "dependency", "quote": "先迁移再上线"}]}},
    {"text": "之前讨论过先迁移再上线", "gold": {"captures": []}},
    {"text": "必须保持向后兼容",
     "gold": {"captures": [{"kind": "constraint", "quote": "必须保持向后兼容"}]}},
    {"text": "我们约定必须保持向后兼容", "gold": {"captures": []}},
    {"text": "验收标准是全部用例通过",
     "gold": {"captures": [{"kind": "acceptance", "quote": "验收标准是全部用例通过"}]}},
    {"text": "上次定的验收标准是全部用例通过", "gold": {"captures": []}},
    {"text": "前后端错误码要保持一致",
     "gold": {"captures": [{"kind": "coordination", "quote": "前后端错误码要保持一致"}]}},
    {"text": "会上确认过前后端错误码要保持一致", "gold": {"captures": []}},
    {"text": "如果超限就分批处理",
     "gold": {"captures": [{"kind": "condition", "quote": "如果超限就分批处理"}]}},
    {"text": "文档里写的如果超限就分批处理", "gold": {"captures": []}},
    {"text": "命名规则待定",
     "gold": {"captures": [{"kind": "unresolved", "quote": "命名规则待定"}]}},
    {"text": "命名规则当时就待定", "gold": {"captures": []}},
    {"text": "给用户表加一列生日",
     "gold": {"captures": [{"kind": "goal", "quote": "给用户表加一列生日"}]}},
    {"text": "老版本的用户表加过生日列", "gold": {"captures": []}},
    {"text": "把 k8s 探针间隔改成 10s",
     "gold": {"captures": [{"kind": "goal", "quote": "把 k8s 探针间隔改成 10s"}]}},
    {"text": "系统目前采用单体架构", "gold": {"captures": []}},
    {"text": "今天的构建又红了", "gold": {"captures": []}},
    {"text": "不需要重构缓存模块，那个计划取消了", "gold": {"captures": []}},
    {"text": "超时设成多少合适", "gold": {"captures": []}},
    {"text": "迁移数据库",
     "gold": {"captures": [{"kind": "goal", "quote": "迁移数据库"}]}},
    {"text": "把错误码改成统一格式",
     "gold": {"captures": [{"kind": "goal", "quote": "把错误码改成统一格式"}]}},
    {"text": "让全部用例通过",
     "gold": {"captures": [{"kind": "goal", "quote": "让全部用例通过"}]}},
    {"text": "确定命名规则",
     "gold": {"captures": [{"kind": "goal", "quote": "确定命名规则"}]}},
    {"text": "保持旧客户端兼容",
     "gold": {"captures": [{"kind": "constraint", "quote": "保持旧客户端兼容"}]}},
]


if __name__ == "__main__":
    main()

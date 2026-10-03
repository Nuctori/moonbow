# -*- coding: utf-8 -*-
"""experiments/task_structure/splittable_errors.py — 可拆错误占比实验。

用户假设：让小模型做"对立语义拆分"是最简单的任务，多轮拆分可拉精度。
本实验判定该假设的可操作版本：

  对每条**当前错误**，检查是否存在一个二元对立，使"正确样本"与"错误样本"
  分居两侧，且满足三判据：
    判据1 可判别：存在措辞不同取值相同的正例对
    判据2 下游：两个取值导致不同的捕获结果（捕获 vs 不捕获，或不同 kind）
    判据3 可证伪：能构造出反例（若补反例后仍错，则该维度未学到）

产出：**可拆错误占比**。高 → 继续拆；低 → 是覆盖问题，拆无用。

错误来源（三口径合并，去重）：
  A. 探针套件（对抗边界，有 gold）
  B. 冻结 eval_set（dev+test，有 gold）
  C. 真实会话中"两轴与规则分歧且属'有没有任务'"的子句（无 gold，需启发式）

隐私：C 组只统计形态与计数，原文不落盘不入库。

用法：HF_HUB_OFFLINE=1 python splittable_errors.py
"""
import json
import os
import re
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
sys.path.insert(0, HERE)

MODELS = r"M:/AI/spark-4b/models"

# 已确认的语义维度及其正反例（用于判据检验）
KNOWN_AXES = {
    "完成体": {
        "right": ["把超时改成 30 秒", "重构配置加载模块", "给搜索加上高亮",
                  "删除废弃的旧代码", "把日志级别调成 debug"],
        "wrong": ["上周把超时改成了 30 秒", "之前重构过配置加载模块",
                  "上个月给搜索加过高亮", "已经删掉了废弃的旧代码",
                  "当时把日志级别调成了 debug"],
        "downstream": "right→捕获, wrong→不捕获",
    },
    "引述/施为": {
        "right": ["必须保持向后兼容", "验收标准是全部用例通过",
                  "先迁移再上线", "如果超限就分批处理"],
        "wrong": ["我们约定必须保持向后兼容", "上次定的验收标准是全部用例通过",
                  "之前讨论过先迁移再上线", "文档里写的如果超限就分批处理"],
        "downstream": "right→捕获, wrong→不捕获",
    },
    "要求/讨论": {
        "right": ["加可观察性", "把这段逻辑抽成公共函数", "给接口加上重试"],
        "wrong": ["降低开销", "一个一个修成本太大", "含多个问题一起思考下",
                  "避免能力不足"],
        "downstream": "right→捕获, wrong→不捕获（讨论/评价非任务）",
    },
    "目标/条件计划": {
        "right": ["实现数据导出", "修复登录接口"],
        "wrong": ["如果做到了就开始做主线覆盖", "不行就先做准备"],
        "downstream": "right→goal, wrong→condition 或 dependency",
    },
}


def check_criteria(name: str, axis: dict, backend) -> dict:
    """检验一个候选对立是否满足三条判据。"""
    def kinds(text):
        caps = backend.extract(text)
        return [c.kind for c in caps if c.kind != "object"]

    right_kinds = [kinds(t) for t in axis["right"]]
    wrong_kinds = [kinds(t) for t in axis["wrong"]]

    # 判据1 可判别：右侧内部取值一致（措辞不同、结论相同）
    r_consistent = len({tuple(k) for k in right_kinds}) == 1
    w_consistent = len({tuple(k) for k in wrong_kinds}) == 1
    # 判据2 下游：两侧结论不同
    downstream = (set(map(tuple, right_kinds)) != set(map(tuple, wrong_kinds)))
    # 判据3 可证伪：右侧应捕获（非空），左侧应不捕获（空）——若反过来则模型未学到
    learned = (all(k for k in right_kinds) and all(not k for k in wrong_kinds))

    return {
        "轴": name,
        "判据1_可判别": r_consistent and w_consistent,
        "判据2_有下游": downstream,
        "判据3_已学到": learned,
        "右侧结论": [tuple(k) for k in right_kinds],
        "左侧结论": [tuple(k) for k in wrong_kinds],
        "通过": r_consistent and w_consistent and downstream and learned,
    }


def collect_gold_errors(backend) -> list:
    """A+B：探针与 eval_set 上有 gold 的错误。"""
    errs = []
    ev = os.path.join(HERE, "eval_set.jsonl")
    rows = [json.loads(l) for l in open(ev, encoding="utf-8")]
    for r in rows:
        gold = [c for c in r["gold"]["captures"] if c["kind"] != "object"]
        caps = [c for c in backend.extract(r["text"]) if c.kind != "object"]
        g_kinds = Counter(c["kind"] for c in gold)
        p_kinds = Counter(c.kind for c in caps)
        if not gold and caps:
            errs.append({"id": r["id"], "text": r["text"], "type": "误捕获",
                         "gold": "无任务", "pred": "+".join(p_kinds)})
        elif gold and not caps:
            errs.append({"id": r["id"], "text": r["text"], "type": "漏检",
                         "gold": "+".join(g_kinds), "pred": "无任务"})
    return errs


def main() -> None:
    import llm_backend
    two = llm_backend.TwoAxisBackend(
        axis1_dir=os.path.join(MODELS, "task_structure_axis1_v1", "final"),
        v2_dir=os.path.join(MODELS, "task_structure_axis2_v1", "final"),
        dedup_cos=0.88, margin=0.15)

    print("=" * 78)
    print("第一部分：检验已推断的四个对立是否满足三判据")
    print("=" * 78)
    passed = 0
    for name, axis in KNOWN_AXES.items():
        res = check_criteria(name, axis, two)
        passed += res["通过"]
        mark = "通过" if res["通过"] else "未通过"
        print(f"\n[{mark}] 对立「{name}」")
        print(f"   判据1 可判别={res['判据1_可判别']}  "
              f"判据2 有下游={res['判据2_有下游']}  "
              f"判据3 已学到={res['判据3_已学到']}")
        if not res["通过"]:
            print(f"   右侧实际: {res['右侧结论']}")
            print(f"   左侧实际: {res['左侧结论']}")
    print(f"\n四个对立中通过三判据的：{passed}/{len(KNOWN_AXES)}")

    print("\n" + "=" * 78)
    print("第二部分：当前错误的可拆性")
    print("=" * 78)
    errs = collect_gold_errors(two)
    by_type = Counter(e["type"] for e in errs)
    print(f"有 gold 的当前错误：{len(errs)} 条  {dict(by_type)}")

    # 对每条错误，判断它是否落在某个已确认对立上（=可拆）
    def matches_axis(text, axis):
        return text in axis["right"] or text in axis["wrong"]

    splittable, coverage = [], []
    for e in errs:
        hit = [n for n, a in KNOWN_AXES.items() if matches_axis(e["text"], a)]
        if hit:
            splittable.append((e, hit))
        else:
            coverage.append(e)

    print(f"\n落在已知对立上（可拆）：{len(splittable)} 条")
    print(f"不落在任何已知对立上（覆盖问题）：{len(coverage)} 条")
    if errs:
        print(f"→ 可拆错误占比：{len(splittable)/len(errs)*100:.0f}%")

    if coverage:
        print("\n覆盖类错误样例（形态，非原文）：")
        for e in coverage[:12]:
            t = re.sub(r"[A-Za-z_][A-Za-z0-9_.\-]{2,}", "<EN>", e["text"])
            t = re.sub(r"\d+", "<NUM>", t)
            print(f"   [{e['type']}] {t[:58]:<60} gold={e['gold']} pred={e['pred']}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""experiments/task_structure/verify_coverage.py — 覆盖路线验证。

判定两件事（用户授权的验证）：
  1. 记忆效应：补进训练集的 25 条错误句，重训后是否转正
  2. 泛化效应：同句式但**训练未见**的留出变体，是否也转正

结论解读：
  - 两者都转正 → 覆盖路线有效（补句式类别可泛化）
  - 只有记忆转正 → 是词表式记忆，覆盖路线退化成枚举（应止损）
  - 都不转正 → 模型能力上限（该考虑换底座）

用法：HF_HUB_OFFLINE=1 python verify_coverage.py
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

MODELS = r"M:/AI/spark-4b/models"

# 补进训练的错误句（记忆组）—— 标签按规范 v2（R1：动作指向 → demand）
MEMORIZED = [
    ("删除废弃的 `legacy_api.py`", "demand"),
    ("生成签名文件之后才能发布安装包", "demand"),
    ("构建镜像之前先把依赖锁文件提交", "demand"),
    ("各服务的超时时间统一改成 30 秒", "demand"),
    ("迁移脚本写完后在预发库演练一遍，视为完成的标志是演练零报错", "demand"),
    ("代码质量需要改进", "demand"),          # v2 修正（原标 mention）
    ("看看监控", "demand"),                 # v2 修正（原标 mention）
    ("适当优化一下启动流程", "demand"),
    ("整体感觉有点慢，处理下", "demand"),      # v2 修正（原标 mention）
    ("证明这个组合猜想在 n=7 时成立", "demand"),
    # v2 新增覆盖：纯名词短语 / 协商 / 引述变体（均 mention）
    ("可观察性", "mention"),
    ("我们来讨论下这个方案", "mention"),
    ("设计稿里的失败就重试三次", "mention"),
]

# 规范 v2 一致性对照表（20 条，独立于训练集）
SPEC_V2_TABLE = [
    ("修复登录接口的分页问题", "demand"), ("日志需要脱敏", "demand"),
    ("代码需要重构", "demand"), ("代码质量需要改进", "demand"),
    ("文档质量需要提升", "demand"), ("整体感觉有点慢，处理下", "demand"),
    ("观感上偏重，优化下", "demand"), ("体感上有点卡，调一调", "demand"),
    ("一个个修成本太大", "demand"), ("含多个问题一起思考下", "demand"),
    ("你继续迭代不要停", "demand"), ("适当优化一下启动流程", "demand"),
    ("整体体验一般", "mention"), ("可观察性", "mention"),
    ("我们约定必须保持向后兼容", "mention"), ("上周把超时改成了 30 秒", "mention"),
    ("文档里写的如果超限就分批", "mention"),
    ("不需要重构缓存模块，那个计划取消了", "mention"),
    ("我们来讨论下这个方案", "mention"), ("这个方案怎么看好不好", "mention"),
]


def main() -> None:
    from gliner2 import AutoExtractor, Schema
    print("加载三个轴 1 头对比：旧（无覆盖）/ 新（覆盖重训）\n")
    models = {
        "旧(axis1_v1)": os.path.join(MODELS, "task_structure_axis1_v1", "final"),
        "新(axis1_cov)": os.path.join(MODELS, "task_structure_axis1_cov", "final"),
    }
    schema = Schema().classification("speech", labels=["demand", "mention"])
    loaded = {}
    for name, path in models.items():
        if os.path.exists(path):
            loaded[name] = AutoExtractor.from_pretrained(path, map_location="cpu")
        else:
            print(f"[skip] {name} 不存在: {path}")

    def pred(model, text):
        r = model.extract(text, schema, threshold=0.0)
        v = r.get("speech") if isinstance(r, dict) else None
        return (v.get("label") if isinstance(v, dict) else v) or "?"

    hold = [json.loads(l) for l in
            open(os.path.join(HERE, "coverage_holdout.jsonl"), encoding="utf-8")]

    for group, cases, title in (
            ("记忆组（补进训练）", MEMORIZED, "1) 记忆效应"),
            ("泛化组（训练未见）", [(h["text"], h["label"]) for h in hold],
             "2) 泛化效应")):
        print("=" * 74)
        print(f"{title}：{len(cases)} 条")
        print("=" * 74)
        for name, model in loaded.items():
            ok = sum(1 for t, w in cases if pred(model, t) == w)
            print(f"  {name:<16} {ok}/{len(cases)} = {ok/len(cases)*100:.0f}%")
        if len(loaded) == 2:
            print("  逐条（旧 → 新）:")
            for t, w in cases:
                p_old = pred(loaded["旧(axis1_v1)"], t)
                p_new = pred(loaded["新(axis1_cov)"], t)
                mark = "✓" if p_new == w else "✗"
                flip = "翻转" if p_old != p_new else "    "
                print(f"    {mark} {flip} {t[:36]:<38} {p_old:<8}→{p_new:<8}(期望 {w})")
        print()

    print("=" * 74)
    print("3) 规范 v2 一致性（独立 20 条对照表）")
    print("=" * 74)
    for name, model in loaded.items():
        ok = sum(1 for t, w in SPEC_V2_TABLE if pred(model, t) == w)
        print(f"  {name:<16} {ok}/{len(SPEC_V2_TABLE)} = "
              f"{ok/len(SPEC_V2_TABLE)*100:.0f}%")
        if name.startswith("新"):
            for t, w in SPEC_V2_TABLE:
                g = pred(model, t)
                if g != w:
                    print(f"    ✗ {t[:34]:<36} 规范={w} 实际={g}")
    print()

    print("=" * 74)
    print("结论判读")
    print("=" * 74)
    print("  记忆组高 + 泛化组高 → 覆盖路线有效，继续补句式")
    print("  记忆组高 + 泛化组低 → 词表式记忆，覆盖路线退化为枚举（止损）")
    print("  两组都低           → 模型能力上限，考虑换底座")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""experiments/task_structure/tense_probe.py — 关系语义最小验证：时间指向判别。

问题（探针复现）："上周把超时改成了 30 秒"（叙述，应弃权）与
"把超时改成 30 秒"（指令，应判 goal）字面高度重叠，区别只在时间指向。

验证两个变量各自的贡献：
  A. 输入窗口：单句 vs 子句+前文片段（关系语义的输入侧）
  B. 训练信号：现有 v1 头（未显式训练该维度）vs 待验证

本脚本只做**测量**，不改训练、不动 test：
- 分组：句内有时间证据 / 句内无证据（后者原则上应弃权）
- 每组 12 条，正负样本配对，确保配对只差时间指向

用法：HF_HUB_OFFLINE=1 python tense_probe.py
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

# (prior_context, clause, expected)
# expected: "goal" = 当前指令；"none" = 叙述/应弃权
IN_EVIDENCE = [
    ("", "上周把超时改成了 30 秒", "none"),
    ("", "把超时改成 30 秒", "goal"),
    ("", "去年重构过这个模块", "none"),
    ("", "重构这个模块", "goal"),
    ("", "刚才已经部署了 v2 版本", "none"),
    ("", "部署 v2 版本", "goal"),
    ("", "昨天修好了登录报错", "none"),
    ("", "修复登录报错", "goal"),
    ("", "此前排查过一次内存泄漏", "none"),
    ("", "排查内存泄漏", "goal"),
    ("", "上个月把日志级别调成了 debug", "none"),
    ("", "把日志级别调成 debug", "goal"),
    ("", "The timeout was changed last week", "none"),
    ("", "Change the timeout to 30 seconds", "goal"),
    ("", "We already migrated the database", "none"),
    ("", "Migrate the database", "goal"),
    ("", "前一天删掉了废弃的旧代码", "none"),
    ("", "删掉废弃的旧代码", "goal"),
    ("", "当时用的是 500ms 的超时", "none"),
    ("", "把超时设成 500ms", "goal"),
    ("", "前年上线过这套鉴权", "none"),
    ("", "上线这套鉴权", "goal"),
    ("", "上周我们讨论过这个方案", "none"),
    ("", "讨论这个方案", "goal"),
]

# 句内无时间标记：单看这句话有歧义，需要前文才能定
NO_EVIDENCE = [
    ("前面在说上周的排查记录。", "把超时改成了 30 秒", "none"),
    ("现在开始新任务。", "把超时改成 30 秒", "goal"),
    ("以下是我们上季度的操作日志。", "把日志级别调成了 debug", "none"),
    ("接下来要做这件事。", "把日志级别调成 debug", "goal"),
    ("回顾一下之前的工作。", "部署了 v2 版本", "none"),
    ("请执行下面的操作。", "部署 v2 版本", "goal"),
    ("上一轮迭代里我们做了这些。", "重构了配置加载模块", "none"),
    ("本轮计划如下。", "重构配置加载模块", "goal"),
    ("顺着刚才的复盘继续。", "修复了登录报错", "none"),
    ("现在请你动手。", "修复登录报错", "goal"),
    ("这些是历史变更记录。", "删掉了废弃的旧代码", "none"),
    ("需要你处理下面这步。", "删掉废弃的旧代码", "goal"),
]


def classify(backend, text: str):
    caps = backend.extract(text)
    kinds = [c.kind for c in caps if c.kind != "object"]
    return kinds[0] if kinds else "none"


def main() -> None:
    from llm_backend import FinetunedBackend
    fb = FinetunedBackend()

    print("=" * 76)
    print("分组 1：句内有时间证据（关系语义应在句内可判）")
    print("=" * 76)
    hit = 0
    for prior, clause, want in IN_EVIDENCE:
        got = classify(fb, clause)
        ok = got == want
        hit += ok
        mark = "OK  " if ok else "MISS"
        print(f"[{mark}] {clause[:34]:<36} → {got:<12} 期望 {want}")
    print(f"句内证据组命中 {hit}/{len(IN_EVIDENCE)}\n")

    print("=" * 76)
    print("分组 2：句内无时间证据 + 前文片段（单句输入 vs 拼接输入）")
    print("=" * 76)
    hit_single = hit_context = 0
    for prior, clause, want in NO_EVIDENCE:
        got_single = classify(fb, clause)
        got_ctx = classify(fb, prior + clause)
        ok_s = got_single == want
        ok_c = got_ctx == want
        hit_single += ok_s
        hit_context += ok_c
        print(f"[单句{'OK ' if ok_s else 'MISS'} 拼接{'OK ' if ok_c else 'MISS'}] "
              f"{clause[:26]:<28} → {got_single:<12} / {got_ctx:<12} 期望 {want}")
    n = len(NO_EVIDENCE)
    print(f"\n无证据组：单句输入 {hit_single}/{n}，拼接前文 {hit_context}/{n}")
    print("（探针集为自造，仅作方向性判断；正式结论需扩样重训后复测）")


if __name__ == "__main__":
    main()

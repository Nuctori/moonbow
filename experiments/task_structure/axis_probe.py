# -*- coding: utf-8 -*-
"""experiments/task_structure/axis_probe.py — 三轴拆分可行性验证（最小测试）。

验证对象（规范 v2 草案，见 docs/task_structure_spec.md 修订）：
  轴 1 言语行为：demand（待实现的当前要求）vs mention（提及/叙述/复述）
  轴 2 结构角色：goal/constraint/dependency/coordination/condition/
                acceptance/unresolved/none（与轴 1 正交）
  轴 3 输出决定：demand ∧ role≠none → 捕获；mention → 不捕获；证据不足 → 弃权

本测试只回答一个问题：**轴 1 是否比"9 类单标签"更容易被现有模型判对**。
对照物是 v1 微调头（9 类单标签，把 cancel/none 并作一类）。

样本设计（成对，只差言语行为）：
  每对 = 同一命题的 demand 形态 与 mention 形态，词面高度重叠。

用法：HF_HUB_OFFLINE=1 python axis_probe.py
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

# (text, axis1, axis2)  axis2 用于检查"mention 侧是否也有结构角色"
PAIRS = [
    # —— 同命题：demand vs mention（词面只差时态/引导）——
    ("把超时改成 30 秒",            "demand",  "goal"),
    ("上周把超时改成了 30 秒",       "mention", "goal"),
    ("重构配置加载模块",             "demand",  "goal"),
    ("之前重构过配置加载模块",        "mention", "goal"),
    ("给搜索加上高亮",               "demand",  "goal"),
    ("上个月给搜索加过高亮",          "mention", "goal"),
    ("删除废弃的旧代码",             "demand",  "goal"),
    ("已经删掉了废弃的旧代码",        "mention", "goal"),
    ("把日志级别调成 debug",         "demand",  "goal"),
    ("当时把日志级别调成了 debug",    "mention", "goal"),
    ("Change the timeout to 30s",   "demand",  "goal"),
    ("The timeout was changed last week", "mention", "goal"),
    # —— mention 侧同样承载结构角色（验证轴正交性）——
    ("我们约定必须保持向后兼容",      "mention", "constraint"),
    ("必须保持向后兼容",             "demand",  "constraint"),
    ("上次定的验收标准是全部用例通过",  "mention", "acceptance"),
    ("验收标准是全部用例通过",        "demand",  "acceptance"),
    ("之前讨论过先迁移再上线",        "mention", "dependency"),
    ("先迁移再上线",                 "demand",  "dependency"),
    # —— 纯 mention / 纯无角色（不该被捕获，也不该被误判成需求）——
    ("系统目前采用单体架构",          "mention", "none"),
    ("今天的构建又红了",              "mention", "none"),
    ("这个 bug 上周修过了吗",         "mention", "none"),
    ("上周我们评审了三个方案",         "mention", "none"),
    # —— demand + none（提问式要求，应视为声明信息不足或非结构）——
    ("超时设成多少合适",              "demand",  "none"),
    # —— 撤销：规范 v2 里归为 mention（要求被撤回）——
    ("不需要重构缓存模块，那个计划取消了", "mention", "none"),
]


def v1_predict(text: str) -> str:
    from llm_backend import FinetunedBackend
    caps = FinetunedBackend().extract(text)
    ks = [c.kind for c in caps if c.kind != "object"]
    return ks[0] if ks else "none"


def v2_axis1(backend, text: str) -> str:
    """轴 1：用二元问句判定（这里用 v1 头的类别作为近似下界——
    真实 v2 需重训；本测试先看"现有模型在 demand/mention 上的可分性"。"""
    caps = backend.extract(text)
    ks = [c.kind for c in caps if c.kind != "object"]
    return "demand" if ks else "mention"


def main() -> None:
    from llm_backend import FinetunedBackend
    fb = FinetunedBackend()

    print("=" * 82)
    print("轴 1 可分性测试（对照物：v1 头是否把 mention 误判成任务）")
    print("=" * 82)
    print(f"{'子句':<32}{'轴1真值':<10}{'v1 预测':<14}{'轴1判定':<10}{'轴1':<6}")
    ok1 = 0
    for text, a1, a2 in PAIRS:
        v1 = v1_predict(text)
        got1 = v2_axis1(fb, text)
        hit = got1 == a1
        ok1 += hit
        print(f"{text[:30]:<32}{a1:<10}{v1:<14}{got1:<10}{'OK' if hit else 'MISS'}")

    print(f"\n轴 1 命中 {ok1}/{len(PAIRS)}")

    # 分侧统计
    for side in ("demand", "mention"):
        sel = [(t, a1) for t, a1, _ in PAIRS if a1 == side]
        hit = sum(1 for t, a1 in sel if v2_axis1(fb, t) == a1)
        print(f"  {side:<8} {hit}/{len(sel)}")

    print("\n对照：v1 头把多少 mention 判成了非 none（即误当任务）")
    men = [t for t, a1, _ in PAIRS if a1 == "mention"]
    wrong = [t for t in men if v1_predict(t) != "none"]
    print(f"  mention {len(men)} 条中误判为任务 {len(wrong)} 条")
    for t in wrong:
        print(f"    ✗ {t[:40]:<42} → {v1_predict(t)}")
    print("\n注：轴 1 判定此处借用 v1 头的 none/非-none 作近似下界；")
    print("    真实 v2 需按轴 1 独立标注重训。本测试仅看方向性。")


if __name__ == "__main__":
    main()

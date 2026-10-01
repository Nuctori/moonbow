# -*- coding: utf-8 -*-
"""experiments/task_structure/reminder_ab.py — 提醒收益三臂对照驱动（阶段 D）。

**状态：尚未验证收益。** 本驱动可复现但默认不运行：运行需要真实模型网关
（付费远程端点，未获授权）与完整的 pi 评测环境。目前交付的是可审计的
实验设计与驱动脚本；任何"有效/无效"结论都必须等真实运行后才能下。

三臂设计（同一任务集、同一模型、同一预算、独立工作区）：
  arm_none      无提醒（baseline）
  arm_generic   固定通用提醒（"注意拆解任务"）——对照"任何提醒都行"的假说
  arm_targeted  基于捕获结构的针对性提醒（经 /v1/task-structure + 插件）

判读规则（预先登记，防止事后挑选）：
- 主指标：显式要求遗漏率、最终任务成功率。
- arm_targeted 优于 arm_generic 才支持"结构捕获有增量价值"；
  仅优于 arm_none 只支持"提醒有值"，不支持本能力。
- arm_generic 若已接近 arm_targeted，则能力无增量，保留默认 off。
- 小样本（<30/臂）只作探索性证据。
- 主模型是否采纳建议只作过程指标，不作成功标准。

环境变量（与 test_guard_scenarios.py 一致）：
  PI_EVAL_AGENT_DIR / PI_EVAL_WORK_DIR / PI_BIN / PI_GUARD_URL / PI_GATEWAY
用法（资源就绪后）：
  python experiments/task_structure/reminder_ab.py --arms none,generic,targeted \
      --rounds 10 --model <id>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from moonbow.task_structure.extractor import analyze_text  # noqa: E402
from moonbow.task_structure.policy import ReminderPolicy   # noqa: E402

# 通用提醒（arm_generic）：任何一句"注意拆解"都能说的话
GENERIC_REMINDER = ("【通用提醒】如果任务复杂，可考虑先拆解成若干简单、"
                    "边界明确的子任务再执行；是否拆解由你决定。")

# 三臂注入的示例任务（复用 compound_traps 的多义务陷阱思路；
# 真实运行时应从 traps JSONL 读取并按基线失败率筛选）。
# t1 设计为分析器可达 high（三义务）；t2 为 medium 阴性对照
# （targeted 臂不注入——如实展示分析器门槛，不为实验放水）。
DEFAULT_TASKS = [
    {
        "id": "t1_triple_mix",
        "prompt": "修复 app.py 中 add 函数的返回值错误，把结果写入 RESULTS.md，并运行测试确认全部通过。",
        "explicit_requirements": ["add 返回 a+b", "写 RESULTS.md", "运行测试并确认通过"],
    },
    {
        "id": "t2_dual_negative_control",
        "prompt": "修复浮点精度测试失败，保持旧接口签名不变。",
        "explicit_requirements": ["精度修复", "接口签名不变"],
    },
]


def targeted_reminder(prompt: str) -> str:
    """arm_targeted 的注入内容 = 分析器的真实输出（措辞与插件一致）。"""
    a = analyze_text(prompt)
    pol = ReminderPolicy(min_level="medium")
    msg = pol.consider(a)
    return msg or ""


def run_arm(arm: str, task: dict, rounds: int, model: str) -> dict:
    """单臂单任务驱动。真实运行逻辑与 test_guard_scenarios.py 相同：
    建 CoW 工作区 → PI_BIN 会话 → 从会话 JSONL 提取指标。

    当前实现仅做骨架校验（提醒内容非空），真实执行需网关授权——
    显式抛错而不是假装跑过。
    """
    reminder = ""
    if arm == "generic":
        reminder = GENERIC_REMINDER
    elif arm == "targeted":
        reminder = targeted_reminder(task["prompt"])
        if not reminder:
            # 分析器未达提醒门槛 → 该任务在本臂不注入（如实记录）
            reminder = ""
    raise RuntimeError(
        f"reminder_ab 尚未在真实环境运行（需要模型网关授权）；"
        f"arm={arm} task={task['id']} reminder={reminder!r} rounds={rounds} "
        f"model={model}。运行前先删除本行并接入 test_guard_scenarios 的会话驱动。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none,generic,targeted")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--model", default=os.environ.get("PI_EVAL_MODEL", ""))
    args = ap.parse_args()

    print("状态：尚未验证收益（本驱动未接入真实模型网关，详见文件头说明）")
    print("预登记判读规则：targeted > generic 才支持结构捕获有增量价值")
    for arm in args.arms.split(","):
        for task in DEFAULT_TASKS:
            reminder = "" if arm == "none" else (
                GENERIC_REMINDER if arm == "generic" else targeted_reminder(task["prompt"]))
            print(f"  [dry-run] arm={arm:<8} task={task['id']:<22} "
                  f"reminder={'yes' if reminder else 'no':<3} "
                  f"chars={len(reminder)}")
    print("接入真实执行：将 run_arm() 的占位段替换为 "
          "experiments/test_guard_scenarios.py 的会话驱动（CoW 工作区 + PI_BIN）。")


if __name__ == "__main__":
    main()

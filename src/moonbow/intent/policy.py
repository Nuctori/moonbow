# -*- coding: utf-8 -*-
"""moonbow.intent.policy — 提醒路由与预算配置（机械元数据层）。

历史备注（2026-10-11）：本模块最初实现为"数值置信门槛"（margin ≥ 阈值才
投递），被用户否决——在不稳定、未校准的 margin 上设阈值做最终裁决，与
关键词枚举同族（run6 margin 10.688 的自信错误、v8 后 margin 分布整体漂移
都是反证）。现架构为**两级 LLM**：
  第一级（定性）：三分类头快速裁决；
  第二级（定量）：冲突一律升级复核判官（KEEP/DROP，语义复核，见
  protocol.build_verification_input）——低置信不是"丢弃"而是"升级"。
本模块只保留**路由与预算**的机械配置：是否复核、会话预算。二者只消费
数量与判官输出，不读内容（硬约束 1 无恙），且永不改变落账行为。
"""
from dataclasses import dataclass
from pathlib import Path

from .protocol import margin_to_confidence  # noqa: F401  （校准工具保留）


@dataclass
class ReminderPolicy:
    """提醒路由配置（机械元数据，非语义裁决）。

    verify_conflicts：冲突投递前是否一律升级第二级复核（缺省 True）。
    max_per_session：单会话提醒预算（0 = 不限）——防刷屏保险丝，超预算
    落账不投递。
    """
    verify_conflicts: bool = True
    max_per_session: int = 0

    def __post_init__(self):
        if self.max_per_session < 0:
            raise ValueError("max_per_session 不得为负")

    @classmethod
    def load(cls, path) -> "ReminderPolicy":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in cfg.items()
                      if k in cls.__dataclass_fields__})

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps({"verify_conflicts": self.verify_conflicts,
                        "max_per_session": self.max_per_session},
                       ensure_ascii=False, indent=1), encoding="utf-8")

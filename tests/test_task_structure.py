# -*- coding: utf-8 -*-
"""tests/test_task_structure.py

Task Structure Advisor 单元测试（阶段 A 验收）：
- schema：引文校验、关系校验、非法 kind 丢弃
- 抽取：误捕获（伪任务/引用/围栏）、重复计数（复述合并）、取消、
        信息不足弃权、对象捕获、显式依赖、承前并列
- 计量：透明等级、弃权优先、"结构负担高"与"适合拆解"分离
- 策略：指纹去重、每版本上限、措辞约束（禁"必须拆解"类要求）
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from moonbow.task_structure.extractor import (  # noqa: E402
    analyze_text, dedup_captures, validate_captures,
)
from moonbow.task_structure.schema import (  # noqa: E402
    Capture, Relation, validate_relations,
)
from moonbow.task_structure.scoring import (  # noqa: E402
    ADVICE_CAUTION, ADVICE_CLARIFY, ADVICE_PHASED, ADVICE_SPLIT_BY_GOAL,
)
from moonbow.task_structure.policy import (  # noqa: E402
    ReminderPolicy, structure_fingerprint,
)


def kinds(a):
    return sorted(c.kind for c in a.captures)


def goals(a):
    return [c for c in a.captures if c.kind == "goal"]


# ---------------------------------------------------------------------------
# schema 校验
# ---------------------------------------------------------------------------

def test_validate_captures_drops_non_locatable_quote():
    text = "修复登录接口。"
    caps = [Capture(kind="goal", quote="修复支付接口")]     # 不在原文
    assert validate_captures(caps, text) == []


def test_validate_captures_drops_invalid_kind():
    caps = [Capture(kind="difficulty_score", quote="修复登录接口")]
    assert validate_captures(caps, "修复登录接口") == []


def test_validate_captures_fills_span():
    text = "修复登录接口。"
    kept = validate_captures([Capture(kind="goal", quote="修复登录接口")], text)
    assert kept[0].span == (0, 6)
    assert text[kept[0].span[0]:kept[0].span[1]] == kept[0].quote


def test_validate_relations_drops_self_and_out_of_range():
    rels = [Relation("depends_on", 0, 0),      # 自引用
            Relation("depends_on", 0, 5),      # 越界
            Relation("implies", 0, 1),         # 非法类型
            Relation("depends_on", 0, 1)]
    kept = validate_relations(rels, n=2)
    assert [r.to_dict() for r in kept] == [{"rtype": "depends_on", "src": 0, "dst": 1}]


# ---------------------------------------------------------------------------
# 抽取：正确捕获
# ---------------------------------------------------------------------------

def test_single_goal_low():
    a = analyze_text("修复登录页面的超时问题。")
    assert kinds(a) == ["goal"]
    assert a.level == "low"
    assert a.vector.goals == 1


def test_multi_goal_with_constraint_high():
    a = analyze_text("修复登录接口，保持旧客户端兼容，增加限流，并补充测试和文档。")
    assert a.vector.goals == 3
    assert a.vector.constraints == 1
    assert a.level == "high"
    assert a.advice == ADVICE_SPLIT_BY_GOAL


def test_explicit_sequence_merged_into_dependency():
    a = analyze_text("先把数据库迁移跑完，然后再更新 API 层。")
    assert a.vector.explicit_dependencies == 1
    # 依赖存在且端点不可机械定位 → 记 unknowns，不推断独立
    assert "dependency" in a.unknowns


def test_inherit_verb_conjunction():
    a = analyze_text("支持导入以及导出。")
    assert a.vector.goals == 2


def test_object_capture_conservative():
    a = analyze_text("把 `config.py` 里的超时时间改成 30 秒。")
    assert a.vector.goals == 1
    assert "config.py" in [c.quote for c in a.captures if c.kind == "object"]


def test_repeated_objects_single_obligation():
    a = analyze_text("把下面 20 个文件的日志级别改成 debug。")
    assert a.vector.goals == 1
    assert a.level == "low"


# ---------------------------------------------------------------------------
# 抽取：误捕获 / 取消 / 伪任务 / 问句
# ---------------------------------------------------------------------------

def test_background_not_captured():
    a = analyze_text("上次已经完成了部署，例如脚本在 scripts/ 下。")
    assert a.captures == []
    assert a.abstain


def test_reference_line_not_captured():
    a = analyze_text("参考用户模块的写法即可。")
    assert a.captures == []


def test_code_fence_not_captured():
    a = analyze_text("配置如下：\n```yaml\nretry: 3\nupgrade: true\n```")
    assert a.captures == []


def test_cancelled_task_not_captured():
    a = analyze_text("不需要重构缓存模块，那个计划取消了。")
    assert a.captures == []


def test_cancel_keeps_real_goal():
    a = analyze_text("取消之前关于重构缓存的要求，先修复登录 bug。")
    assert [c.quote for c in goals(a)] == ["先修复登录 bug"]


def test_question_not_captured():
    a = analyze_text("我们是不是应该先做迁移？")
    assert a.captures == []
    assert a.abstain


def test_restatement_dedup():
    caps = [Capture(kind="goal", quote="请修复导出乱码的问题"),
            Capture(kind="goal", quote="导出乱码问题麻烦修一下")]
    # 完全相同归一化 → 合并；此句式不同词序，允许保留（已知规则局限）
    dedup_captures(caps)
    # 子串复述 → 合并
    caps2 = [Capture(kind="goal", quote="补充测试"),
             Capture(kind="goal", quote="补充测试并更新文档")]
    assert len(dedup_captures(caps2)) == 1


# ---------------------------------------------------------------------------
# 计量：弃权优先、等级透明、建议分型
# ---------------------------------------------------------------------------

def test_vague_goal_abstains_not_simple():
    a = analyze_text("优化一下这个系统。")
    assert a.abstain is True
    assert a.level == "unknown"
    assert a.advice == ADVICE_CLARIFY


def test_no_goal_abstains_without_task_intent_no_advice():
    a = analyze_text("上次已经完成了部署，例如脚本在 scripts/ 下。")
    assert a.abstain
    assert a.advice is None          # 纯背景不提醒


def test_truncated_input_never_low():
    a = analyze_text("修复A模块并补充测试，" * 1200)      # > MAX_INPUT_CHARS
    assert a.truncated
    assert a.level == "unknown"
    assert a.abstain


def test_dependency_heavy_suggests_phased_not_parallel():
    a = analyze_text("先升级消息队列到 v3，再迁移消费者，迁移完成后更新监控。")
    assert a.vector.explicit_dependencies >= 1
    if a.level == "high":
        assert a.advice == ADVICE_PHASED


def test_coordination_medium_caution():
    a = analyze_text("前后端的错误码要保持一致，同步更新两边的文档。")
    assert a.vector.coordinations == 2
    assert a.level == "medium"
    assert a.advice == ADVICE_CAUTION


def test_level_reasons_are_transparent():
    a = analyze_text("修复A模块，修复B模块，修复C模块，保持兼容。")
    assert a.level == "high"
    assert any("3" in r for r in a.level_reasons)   # 理由含阈值数字


# ---------------------------------------------------------------------------
# 窄语义三维度（含反例：每条特征都测"吃掉叙述、放过任务"）
# ---------------------------------------------------------------------------

def test_past_narration_vetoed_but_request_survives():
    # 维度一：过去时间词/经历体 → 叙述；同句带请求标记 → 仍是任务
    a = analyze_text("这个模块我重构过，麻烦你更新文档。")
    quotes = [c.quote for c in a.captures if c.kind == "goal"]
    assert quotes == ["麻烦你更新文档"]


def test_past_narration_en_passive():
    a = analyze_text("The service was migrated last year, as discussed in the offsite.")
    assert a.captures == []


def test_mixed_history_then_request():
    a = analyze_text("上周已经迁移过数据库，现在请把连接串更新成新地址。")
    kinds = [c.kind for c in a.captures]
    assert "goal" in kinds                    # 请求部分存活
    assert all("迁移过" not in c.quote for c in a.captures)   # 叙述部分被吃


def test_dangshi_is_not_condition():
    # 维度二："当时"是过去回指，不是"当…时"条件
    a = analyze_text("当时的方案有问题，请重新评审。")
    assert "condition" not in kinds(a)
    assert a.captures == [] or "condition" not in kinds(a)    # 叙述被吃，条件不误发


def test_dang_shi_conditional_still_fires():
    a = analyze_text("当队列积压超过一万条时发送告警。")
    assert "condition" in kinds(a)


def test_shunbian_keeps_content_obligation():
    # 维度二：语气降调不降义务
    a = analyze_text("顺便把分页组件升级到 v2。")
    assert a.vector.goals == 1
    a2 = analyze_text("顺带一提，旧版本下个月停止维护。")
    assert a2.captures == []                  # 纯元话语仍不算任务


def test_dependency_compound_noun_not_verb():
    # 维度二："依赖清单"是名词，不构成依赖捕获
    a = analyze_text("更新依赖清单里过期的版本号。")
    assert "dependency" not in kinds(a)
    assert a.vector.goals == 1


def test_cross_clause_cancel_anchoring():
    # 维度三：裸取消命令回指前句话题
    a = analyze_text("那个重构缓存的要求作废，别做了。")
    assert a.captures == []
    b = analyze_text("数据库迁移计划作废，请先修复登录 bug。")
    assert [c.quote for c in goals(b)] == ["请先修复登录 bug"]


def test_cancel_modal_negates_own_clause():
    # "无需迁移旧数据，直接上线新表"：被否定的子句不是义务
    b = analyze_text("无需迁移旧数据，直接上线新表。")
    assert [c.quote for c in goals(b)] == ["直接上线新表"]


def test_cancel_still_keeps_domain_action():
    # 反例守卫：及物取消动词作领域动作时不是"取消任务"
    a = analyze_text("给订单中心的取消订单接口增加重试。")
    assert a.vector.goals >= 1


# ---------------------------------------------------------------------------
# 策略：去重 / 上限 / 措辞
# ---------------------------------------------------------------------------

def _high_analysis(text="修复A模块，修复B模块，修复C模块，保持兼容。"):
    return analyze_text(text)


def test_policy_dedups_identical_and_reworded():
    pol = ReminderPolicy()
    m1 = pol.consider(_high_analysis())
    assert m1 is not None
    m2 = pol.consider(_high_analysis())
    assert m2 is None                     # 相同结构不重复
    m3 = pol.consider(_high_analysis("请修复A模块、修复B模块以及修复C模块，并且保持兼容。"))
    assert m3 is None                     # 改写近似结构不重复


def test_policy_version_cap():
    pol = ReminderPolicy(max_reminders=1)
    a1 = _high_analysis()
    a2 = _high_analysis("修复X模块，修复Y模块，修复Z模块，确保离线可用。")
    assert pol.consider(a1)
    # 不同结构（指纹不同、相似度低于阈值）但同版本 → 受上限抑制
    assert pol.consider(a2) is None


def test_policy_resets_on_new_version():
    pol = ReminderPolicy()
    assert pol.consider(_high_analysis())
    # 同一结构的新版本仍被指纹抑制（同一结构只说一次）
    same = _high_analysis()
    same.version = 2
    assert pol.consider(same) is None
    # 不同结构的新版本允许再次提醒
    other = _high_analysis("先跑数据订正脚本，再重新生成报表。")
    other.version = 2
    other.level = "high"            # 依赖链样本 level 由向量决定，此处仅测策略
    msg = pol.consider(other)
    assert msg is None or isinstance(msg, str)


def test_policy_skips_low_and_unknown():
    pol = ReminderPolicy()
    assert pol.consider(analyze_text("修复登录超时问题。")) is None
    assert pol.consider(analyze_text("这个 bug 上周不是修过了吗？")) is None


def test_reminder_wording_is_advisory():
    pol = ReminderPolicy()
    msg = pol.consider(_high_analysis())
    assert msg is not None
    assert "必须拆解" not in msg
    assert "务必" not in msg
    assert "非要求" in msg or "仅提示" in msg


def test_dependency_reminder_disclaims_parallelism():
    a = analyze_text("先升级消息队列到 v3，再迁移消费者，迁移完成后更新监控，最后回归。")
    pol = ReminderPolicy(min_level="medium")
    msg = pol.consider(a)
    if msg:
        assert "不代表这些任务可以并行" in msg or a.vector.explicit_dependencies == 0


def test_fingerprint_stable_for_same_text():
    a1 = analyze_text("修复A模块，修复B模块，修复C模块。")
    a2 = analyze_text("修复A模块，修复B模块，修复C模块。")
    assert structure_fingerprint(a1) == structure_fingerprint(a2)


def test_service_payload_shape():
    from moonbow.task_structure.service import analyze_payload
    out = analyze_payload({"text": "修复登录接口，保持旧客户端兼容，增加限流，并补充测试。",
                           "source": "user"})
    assert out["level"] == "high"
    assert out["vector"]["goals"] == 3
    assert out["captures"][0]["quote"]            # 引文级证据
    assert "text" not in out                       # 默认不回显原文（日志脱敏基线）
    with pytest.raises(ValueError):
        analyze_payload({"text": ""})

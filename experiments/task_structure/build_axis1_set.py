# -*- coding: utf-8 -*-
"""experiments/task_structure/build_axis1_set.py — 轴 1（言语行为）成对训练集。

规范 v2 草案的轴 1：
  demand  —— 当前指向执行者的、待实现的要求（直接施为）
  mention —— 提及/叙述/复述/引述/记录，不要求执行

两个实测失败聚集区必须成对覆盖（axis_probe.py 复现）：
  F1 「V 成了」句式："上周把超时改成了 30 秒"（mention）vs
      "把超时改成 30 秒"（demand）——模型抓"改成"不判时间功能
  F2 引述结构角色："我们约定必须保持兼容"（mention）vs
      "必须保持兼容"（demand）——模型抓角色名词不判言语行为

数据构成：
  - 成对样本（PAIRS_*）：同一命题两形态，词面高度重叠 → 教判别维度
  - mention 独立样本：纯背景/复盘/提问/引述（真实流里的多数）
  - demand 独立样本：常规指令（含词典外动词，避免退化成动词表匹配）
隐私：全部为本地合成模板，不含任何真实会话原文。

用法：python build_axis1_set.py → axis1_train.jsonl
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axis1_train.jsonl")

OBJ = ["超时时间", "登录接口", "缓存模块", "配置加载模块", "日志级别", "搜索功能",
       "支付回调", "订单列表", "导出格式", "分页组件", "通知服务", "用户表",
       "鉴权逻辑", "上报断点", "构建脚本", "上传路径", "限流阈值", "灰度比例"]
PAR = ["30 秒", "500ms", "debug", "v2", "100 QPS", "/healthz", "10GB", "3 次"]
GOAL_V = ["修复", "重构", "删除", "更新", "新增", "补充", "优化", "调整",
          "替换", "迁移", "升级", "接入", "补上", "加上", "改成", "换成"]
# 完成体形态变化（F1 核心）
PERFECT = ["改成了", "调成了", "换成了", "升级到了", "迁移到了"]

PAIRS = []          # (demand_text, mention_text)

# —— F1：「V 成了」 vs 「V 成」 ——
for obj, par in [("超时时间", "30 秒"), ("日志级别", "debug"), ("端口", "9000"),
                 ("缓存策略", "LRU"), ("分页大小", "50")]:
    PAIRS.append((f"把{obj}改成{par}", f"上周把{obj}改成了{par}"))
    PAIRS.append((f"把{obj}调整成{par}", f"上个月把{obj}调整成了{par}"))
    PAIRS.append((f"把{obj}换成{par}", f"之前把{obj}换成了{par}"))
for obj in ["数据库", "搜索服务", "鉴权模块"]:
    PAIRS.append((f"把{obj}迁移到新集群", f"去年把{obj}迁移到了新集群"))
    PAIRS.append((f"把{obj}升级到 v3", f"上个迭代把{obj}升级到了 v3"))

# —— F2：引述结构角色 vs 直接施为 ——
ROLE_PAIRS = [
    ("必须保持向后兼容", "我们约定必须保持向后兼容"),
    ("验收标准是全部用例通过", "上次定的验收标准是全部用例通过"),
    ("接口签名不能改", "之前的约束是接口签名不能改"),
    ("先迁移再上线", "之前讨论过先迁移再上线"),
    ("前后端错误码要保持一致", "会上确认过前后端错误码要保持一致"),
    ("如果超限就分批处理", "文档里写的如果超限就分批处理"),
    ("上线窗口待确认", "遗留的待确认事项是上线窗口"),
]
PAIRS.extend(ROLE_PAIRS)
for obj in ["超时时间", "限流阈值", "日志级别"]:
    PAIRS.append((f"把{obj}设为 {PAR[0]}", f"需求里写明把{obj}设为 {PAR[0]}"))

# —— 常规 demand / mention 对照（含词典外动词，防退化成动词匹配）——
EXTRA_PAIRS = [
    ("给用户表加一列生日", "老版本的用户表加过生日列"),
    ("把首页的骨架屏去掉", "首页的骨架屏年初就去掉了"),
    ("把配置中心的地址换掉", "配置中心的地址已经换过了"),
    ("给接口的响应缓存起来", "接口的响应之前缓存过"),
    ("把这段逻辑抽成公共函数", "这段逻辑去年抽成了公共函数"),
    ("给搜索加上高亮", "搜索高亮是上季度加的"),
    ("把告警先静音", "告警当时被静音了"),
    ("把错误码统一收口", "错误码收口的改造上周完成了"),
    ("在 README 里补安装步骤", "README 的安装步骤之前补过"),
    ("把临时开关删掉", "临时开关上个版本已经删掉了"),
    ("Fix the timeout bug", "The timeout bug was fixed last week"),
    ("Add retry to the dispatcher", "Retry was added to the dispatcher earlier"),
    ("Rename getUser to fetchUser", "getUser was renamed to fetchUser previously"),
    ("Bump the request timeout", "The request timeout was bumped last sprint"),
]
PAIRS.extend(EXTRA_PAIRS)

# —— 纯 mention（真实流多数）：背景/复盘/提问/举例/引述 ——
MENTION_ONLY = [
    "系统目前采用单体架构", "目前已经上线的是 v2 接口", "这个 bug 上周修过了吗",
    "上周我们评审了三个方案", "有人看过这份性能报告吗", "辛苦帮我看下这个方案靠不靠谱",
    "今天的构建又红了", "限流阈值设多少合适", "我们是不是应该先做迁移",
    "背景：这个模块是三年前写的", "顺带一提，旧版本下个月停止维护",
    "例如：`POST /v1/orders` 返回 201", "参考用户模块的写法即可",
    "参见部署手册第三章", "之前的方案文档在 wiki 上", "当时用的是 500ms 的超时",
    "历史上这里出过一次类似事故", "这个话题上次会议聊过",
    "Stack trace 贴在群里了", "配置如下：\n```yaml\nretry: 3\n```",
    "运行方式：\n```bash\nmake test\n```", "样例输出是 total: 42",
    "The service was migrated last year", "We already discussed this approach",
    "How long did the last migration take?", "Why did we pick this design?",
    "I'm just curious how the pricing works", "As discussed in the offsite",
    "不需要重构缓存模块，那个计划取消了", "跳过国际化，本期不做",
    "那个重构缓存的要求作废，别做了", "不用写新文档，现有的够用",
    "先不做限流了，其他照旧", "放弃自定义表单方案",
]
# —— 纯 demand（含问句式非结构要求，轴 2 才是 none）——
DEMAND_ONLY = [
    "修复登录接口的分页问题", "给搜索加上限流", "把这批图片转成 webp",
    "把 v1 和 v2 的接口路径加上前缀", "给订单列表增加导出",
    "统计一下接口的调用量", "把这次改动整理成文档",
    "排查线上 502 的原因", "把内存泄漏查出来", "给新人写一份上手说明",
    "把这些重复代码合并掉", "把构建时间压到 5 分钟以内",
    "Implement webhook retries", "Document the new endpoint",
    "Clean up the unused dependencies",
    # 顺序依赖的 demand 形态（补 axis_probe 暴露的缺口）
    "先迁移再上线", "先跑迁移再更新接口", "先备份再执行订正",
    "先评审方案再动工", "先扩容再放量", "先冻结schema再生成模型",
    # 同类缺口：约束/验收/待决的 demand 形态，**必须成对出现**——
    # 只给 demand 侧会让模型形成"某短语总是引述"的偏置（已实测复现）
    "迁移完成后压测对比基线", "等配置中心就绪后再切流量",
    "数据订正完成之后重新生成报表", "签名完成后再发布安装包",
    "必须保持向后兼容", "接口签名不能改", "日志不能含用户内容",
    "验收标准是全部用例通过", "完成的定义是各端状态一致",
    "命名规则待定", "上线窗口还需要确认", "迁移顺序待讨论",
]

# —— 上述 demand 的 mention 对照（同命题、加引述引导或完成体）——
MENTION_PAIRED = [
    "之前讨论过先迁移再上线", "年初先跑过迁移再更新接口",
    "上次先备份再执行的订正", "之前评审方案后才动工",
    "上个季度先扩容再放量", "当时先冻结 schema 再生成模型",
    "迁移完成之后压测过基线", "等配置中心就绪后已经切过流量",
    "数据订正完成后重新生成过报表", "签名完成后发布过一次安装包",
    "我们约定必须保持向后兼容", "之前的约束是接口签名不能改",
    "规范里写的日志不能含用户内容", "上次定的验收标准是全部用例通过",
    "会上确认的完成定义是各端状态一致", "命名规则当时就待定",
    "上线窗口之前还需要确认", "迁移顺序上次讨论过",
]


def expand():
    """槽位扩展：把成对模板 × 更多对象/参数/时间词放大规模，保持成对结构。

    关键约束（防退化）：
    - demand 与 mention 的动词、宾语完全相同，只差完成体/时间引导——
      模型无法靠"看见某动词"取巧，只能学判别维度；
    - 时间引导词轮换覆盖（上周/去年/之前/当时/上个月/上个迭代…），
      避免模型记住单一引导词。
    """
    rng = random.Random(17)
    TIMES = ["上周", "去年", "之前", "当时", "上个月", "上个迭代", "前阵子",
             "昨天", "年初", "第二季度", "早些时候"]
    rows = []
    # 对象 → 适配参数（避免"把日志级别设为30秒"这类语义不搭的组合）
    OBJ_PAR = {
        "超时时间": ["30 秒", "500ms", "5 分钟", "10 秒"],
        "登录接口": ["v2", "默认值"],
        "缓存模块": ["LRU", "写穿透"],
        "配置加载模块": ["懒加载", "预加载"],
        "日志级别": ["debug", "info"],
        "搜索功能": ["前缀匹配", "默认值"],
        "支付回调": ["幂等", "默认值"],
        "订单列表": ["分页", "虚拟滚动"],
        "导出格式": ["CSV", "JSON"],
        "分页组件": ["v2", "虚拟滚动"],
        "通知服务": ["静默模式", "聚合"],
        "用户表": ["软删除", "默认值"],
        "鉴权逻辑": ["双因子", "默认值"],
        "上报断点": ["续传", "默认值"],
        "构建脚本": ["并行", "缓存"],
        "上传路径": ["对象存储", "默认值"],
        "限流阈值": ["100 QPS", "50 QPS"],
        "灰度比例": ["5%", "10%"],
    }
    for obj, pars in OBJ_PAR.items():
        for par in pars:
            t = rng.choice(TIMES)
            rows.append((f"把{obj}改成{par}", f"{t}把{obj}改成了{par}"))
            rows.append((f"把{obj}调成{par}", f"{t}把{obj}调成了{par}"))
            rows.append((f"把{obj}设为{par}", f"{t}把{obj}设为了{par}"))
    # F1 扩展：完成体多样（已/已经/过/了）。动词按对象适配，
    # 避免"删除过日志级别"这类语义不搭组合污染数据。
    OBJ_VERBS = {
        "超时时间": ["优化", "调整", "修改"],
        "登录接口": ["重构", "修复", "更新"],
        "缓存模块": ["重构", "优化", "补充"],
        "配置加载模块": ["重构", "优化"],
        "日志级别": ["调整", "更新"],
        "搜索功能": ["优化", "补充", "更新"],
        "支付回调": ["修复", "重构"],
        "订单列表": ["优化", "重构"],
        "导出格式": ["调整", "更新"],
        "分页组件": ["升级", "重构"],
        "通知服务": ["重构", "优化"],
        "用户表": ["调整", "优化"],
        "鉴权逻辑": ["重构", "更新"],
        "上报断点": ["补充", "调整"],
        "构建脚本": ["优化", "重构"],
        "上传路径": ["调整", "更新"],
        "限流阈值": ["调整", "优化"],
        "灰度比例": ["调整", "优化"],
    }
    for obj, verbs in OBJ_VERBS.items():
        for v in verbs:
            t = rng.choice(TIMES)
            rows.append((f"{v}{obj}", f"{t}已经{v}了{obj}"))
            rows.append((f"{v}{obj}", f"{obj}{t}已经{v}过"))
    # F2 主区：引述结构角色 × 多角色名词
    ROLE_NOUNS = ["约定", "需求里写的", "文档里写的", "会上确认的", "遗留事项是",
                  "之前的约束是", "上次定的标准是", "规范要求"]
    STRUCT = ["必须保持向后兼容", "接口签名不能改", "先迁移再上线",
              "前后端错误码要一致", "验收标准是全部用例通过",
              "如果超限就分批", "上线窗口待确认", "日志不能含用户内容"]
    for noun in ROLE_NOUNS:
        for st in STRUCT:
            rows.append((st, f"{noun}{st}"))
    # 常规成对（含词典外动词）
    for obj in OBJ:
        t = rng.choice(TIMES)
        rows.append((f"给{obj}加上重试", f"{t}给{obj}加过重试"))
        rows.append((f"把{obj}的行为对齐到新规范", f"{t}把{obj}的行为对齐过"))
        rows.append((f"给{obj}补一份说明", f"{t}给{obj}补过说明"))
    return rows


def build():
    rows = []
    # 扩展开的成对样本
    for d, m in expand():
        rows.append({"text": d, "label": "demand", "zone": "pair"})
        rows.append({"text": m, "label": "mention", "zone": "pair"})
    # 失败区三倍加权（稀缺模式）
    for obj, par in [("超时时间", "30 秒"), ("日志级别", "debug")]:
        for _ in range(3):
            rows.append({"text": f"把{obj}改成了{par}", "label": "mention",
                         "zone": "F1"})
            rows.append({"text": f"把{obj}改成{par}", "label": "demand",
                         "zone": "F1"})
    for t in MENTION_ONLY + MENTION_PAIRED:
        rows.append({"text": t, "label": "mention", "zone": "only"})
    for t in DEMAND_ONLY:
        rows.append({"text": t, "label": "demand", "zone": "only"})

    rng = random.Random(5)
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        k = (r["text"], r["label"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


if __name__ == "__main__":
    rows = build()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"写出 {len(rows)} 条 → {OUT}")
    print("标签分布:", dict(Counter(r["label"] for r in rows)))
    print("区域分布:", dict(Counter(r["zone"] for r in rows)))

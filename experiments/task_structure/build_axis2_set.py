# -*- coding: utf-8 -*-
"""experiments/task_structure/build_axis2_set.py — 轴 2（结构角色）训练集。

规范 v2：轴 2 只在 demand 侧标注（mention 侧的角色由轴 1 拦掉，不需判定）。
类别：goal / constraint / dependency / coordination / condition /
      acceptance / unresolved / none

方法沿用轴 1 的成功经验：**对照样本**是关键，不是数据量。轴 2 的实测重灾区
（v1 头上 dependency P 0.42、coordination P 0.50）各做成对/三连对照：
  - dependency vs goal：   "先迁移再上线"(dep) vs "迁移数据库"(goal)
      vs "迁移完成后上线"(dep)   —— 同词面、不同结构角色
  - coordination vs goal： "前后端错误码保持一致"(coord) vs
      "统一错误码"(goal) vs "把错误码改成统一格式"(goal)
  - constraint vs goal：   "必须保持兼容"(constr) vs "实现兼容层"(goal)
  - condition vs goal：    "如果超限就分批"(cond) vs "实现分批处理"(goal)
  - acceptance vs goal：   "验收标准时全部通过"(acc) vs "让用例全部通过"(goal)
  - unresolved vs goal：   "命名规则待定"(unres) vs "确定命名规则"(goal)
  - none：                 "超时设成多少合适"(提问式要求，无结构角色)

隐私：全部本地合成。
用法：python build_axis2_set.py → axis2_train.jsonl
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axis2_train.jsonl")

OBJ = ["登录接口", "缓存模块", "配置加载模块", "日志级别", "支付回调", "订单列表",
       "导出功能", "分页组件", "通知服务", "用户表", "鉴权逻辑", "构建脚本",
       "上传路径", "限流阈值", "灰度比例", "报表服务", "消息队列", "搜索功能"]
PAIR = ["前后端", "客户端与服务端", "文档与实现", "测试与生产", "Web 与小程序"]

ROWS = []


def add(text, label, zone):
    ROWS.append({"text": text, "label": label, "zone": zone})


# ---- dependency：顺序/依赖（含同词面 goal 对照）----
for o in OBJ:
    add(f"先迁移{o}的数据，再进行改造", "dependency", "dep")
    add(f"等{o}就绪后再切换流量", "dependency", "dep")
    add(f"{o}的改造依赖上游服务的输出", "dependency", "dep")
    add(f"在{o}上线之前先做回归", "dependency", "dep")
    add(f"{o}完成后压测对比基线", "dependency", "dep")
    add(f"Once the {o} schema is frozen, regenerate the models", "dependency", "dep")
    # 同词面 goal 对照（有先后词但不构成依赖要求）
    add(f"迁移{o}", "goal", "dep_ctrl")
    add(f"改造{o}", "goal", "dep_ctrl")
    add(f"给{o}做一次回归", "goal", "dep_ctrl")
    add(f"上线{o}", "goal", "dep_ctrl")
    add(f"压测{o}并对比基线", "goal", "dep_ctrl")

# ---- coordination：跨对象一致性（含同词面 goal 对照）----
for p in PAIR:
    add(f"{p}的错误码要保持一致", "coordination", "coord")
    add(f"{p}的配置保持同步", "coordination", "coord")
    add(f"{p}的命名规范要对齐", "coordination", "coord")
    add(f"{p}的字段定义必须统一", "coordination", "coord")
    # 同词面 goal 对照：动作是"改/统一"，不是"保持"
    add(f"把{p}的错误码改成统一格式", "goal", "coord_ctrl")
    add(f"更新{p}的配置文件", "goal", "coord_ctrl")
    add(f"给{p}补一份字段说明", "goal", "coord_ctrl")
add("各环境的日志格式统一", "coordination", "coord")
add("多语言文件的键结构必须一致", "coordination", "coord")
add("把日志格式改成统一格式", "goal", "coord_ctrl")

# ---- constraint：附加限制（含同词面 goal 对照）----
CONSTR = ["保持向后兼容", "不能修改对外签名", "响应时间不超过 500ms",
          "日志里不得出现用户内容", "内存占用不超过 512MB",
          "旧客户端必须仍可用", "不得影响其他服务", "必须在离线时可用"]
for c in CONSTR:
    add(c, "constraint", "constr")
    add(f"{c}，请据此改造", "constraint", "constr")
# 同词面 goal 对照
for o in OBJ[:10]:
    add(f"给{o}加一层兼容层", "goal", "constr_ctrl")
    add(f"减小{o}的内存占用", "goal", "constr_ctrl")
    add(f"让{o}支持离线", "goal", "constr_ctrl")
    add(f"给{o}的日志做脱敏", "goal", "constr_ctrl")

# ---- condition：分支条件（含同词面 goal 对照）----
COND = [("数据量超过 10GB", "启用分批导出"), ("错误率上升", "立即回滚"),
        ("队列积压超过一万条", "发送告警"), ("灰度异常", "暂停放量"),
        ("构建失败", "通知值班"), ("缓存未命中", "回源查询")]
for c, act in COND:
    add(f"如果{c}就{act}", "condition", "cond")
    add(f"当{c}时{act}", "condition", "cond")
    # 同词面 goal 对照：把动作本身作为要求
    add(act, "goal", "cond_ctrl")
    add(f"实现{act}的逻辑", "goal", "cond_ctrl")

# ---- acceptance：验收标准（含同词面 goal 对照）----
ACC = ["全部用例通过", "10k 行表格在 200ms 内渲染", "演练零报错",
       "各端状态一致", "压测下无错误", "CI 全绿"]
for a in ACC:
    add(f"验收标准是{a}", "acceptance", "acc")
    add(f"完成的定义是{a}", "acceptance", "acc")
    add(f"验收：{a}", "acceptance", "acc")
    # 同词面 goal 对照
    add(f"让{a}", "goal", "acc_ctrl")
    add(f"确保{a}", "goal", "acc_ctrl")

# ---- unresolved：待调查/待决定（含同词面 goal 对照）----
UNRES = ["命名规则", "升级窗口", "迁移顺序", "服务商选型", "折算比例", "采样率"]
for u in UNRES:
    add(f"{u}待定", "unresolved", "unres")
    add(f"{u}还需要确认", "unresolved", "unres")
    add(f"需要先弄清楚{u}的依据", "unresolved", "unres")
    add(f"{u}尚不确定，先留扩展点", "unresolved", "unres")
    # 同词面 goal 对照
    add(f"确定{u}", "goal", "unres_ctrl")
    add(f"把{u}定下来并写入文档", "goal", "unres_ctrl")

# ---- goal 主体（含词典外动词，避免退化成动词匹配）----
GOALS = ["修复{0}的分页问题", "给{0}加上重试", "把{0}的日志级别调成 debug",
         "删除{0}里的废弃代码", "更新{0}的文档", "重构{0}模块",
         "排查{0}报错的原因", "把{0}部署到预发", "给{0}加一列创建时间",
         "把{0}的颜色调一下", "压缩{0}的静态资源", "给{0}补一份说明",
         "把{0}的地址换掉", "把{0}的告警关掉", "统计{0}的调用量",
         "Fix the {0} pagination bug", "Add retry to {0}", "Bump the {0} timeout"]
for o in OBJ:
    for g in GOALS:
        add(g.format(o), "goal", "goal")

# ---- none：demand 形态但无结构角色（提问式要求等）----
NONE = ["超时设成多少合适", "这个方案靠谱吗", "用哪个服务商更好",
        "顺便看下监控", "整体感觉有点慢", "适当优化一下启动流程",
        "这个问题怎么处理比较好", "要不要先做迁移"]
for n in NONE:
    add(n, "none", "none")


def main():
    rng = random.Random(31)
    rng.shuffle(ROWS)
    seen, out = set(), []
    for r in ROWS:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        out.append(r)

    # 类别配平：非 goal 类各补到与最少的可扩类同量级，避免"全判 goal"退化。
    # 做法是从对照区（*_ctrl）里按需要重复——它们本就是稀缺角色的对手样本。
    from collections import Counter
    cnt = Counter(r["label"] for r in out)
    TARGET = 120                     # 非 goal 类目标量级
    EXTRA = {
        "dependency": ["修改 A 之后才能改 B", "B 的改造依赖 A 的输出",
                       "A 完成后再做 B 的改造", "在 A 就绪之前不要动 B",
                       "先做 A，再做 B", "A 上线后回归 B"],
        "coordination": ["三处配置必须一致", "两边字段名要对齐",
                         "多环境变量名统一", "文档与代码保持一致",
                         "各端展示逻辑同步", "两套接口的字段要对齐"],
        "constraint": ["必须兼容旧版本", "不能改动公共接口", "耗时不超过 2 秒",
                       "不得阻塞主线程", "日志不能含个人信息",
                       "旧数据必须可读", "内存不得超限", "必须支持离线"],
        "condition": ["超限时启用降级", "失败则重试三次", "异常时暂停放量",
                      "积压超阈值就扩容", "缓存失效则回源",
                      "如果命中规则就拦截"],
        "acceptance": ["验收要求用例全绿", "完成标准是零报错",
                       "视为完成的条件是状态一致", "验收看压测无异常",
                       "通过标准是 CI 全绿", "验收依据是演练零报错"],
        "unresolved": ["方案选型还没定", "细则有待明确", "阈值需要再讨论",
                       "接口归属待确认", "上线时间未确定", "依赖关系尚不清楚"],
        "goal": ["梳理相关问题", "确认当前实现", "记录现有行为",
                 "复核改动范围", "整理排查结论", "输出一份清单"],
        "none": ["这样改可以吗", "哪个方案更好", "要不要先问一下",
                 "这块熟悉吗", "大概要多久", "能帮我看一下吗"],
    }
    for label, pool in EXTRA.items():
        i = 0
        while cnt[label] < TARGET:
            out.append({"text": pool[i % len(pool)], "label": label,
                        "zone": "balance"})
            cnt[label] += 1
            i += 1
    rng.shuffle(out)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {len(out)} 条 → {OUT}")
    print("角色分布:", dict(Counter(r["label"] for r in out)))
    print("区域分布:", dict(Counter(r["zone"] for r in out)))


if __name__ == "__main__":
    main()

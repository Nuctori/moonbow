# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v3_data.py — 检验"判断头+模式匹配"假设的数据集。

用户假设：训练模型对任务的直觉（判断头 + 模式匹配）难度远低于 CoT 推理。
检验设计（第七轮结论的直接下一步）：
1. 训练集**纯化**：只保留语义角色判断残差（剔除 scope/时间类——已由 L0
   闸门机械处理），并把"同决策 × 相反约束"的矛盾对密度提到 50%+；
2. 评测集**扩大**：每任务 ≥50 条，且按"对子"组织——同一个决策文本配两个
   不同约束、标签相反，约束敏感度可机械计算；
3. 新增**约束敏感度**指标：对每个矛盾对，两次判断是否翻转。这是检验
   "模型有没有读约束"的直接读数，比准确率更本质。

产出：
  data/intent_eval/train_t3_v3.jsonl / eval_t3_v3.jsonl（成对组织）
  data/intent_eval/train_t4_v3.jsonl / eval_t4_v3.jsonl（≥50 条）
  data/intent_eval/eval_t3_sensitivity.jsonl（矛盾对索引）
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "intent_eval"

# 约束卡（第七轮确认的真实表述）
C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
OTHER = {"mech": "test", "tmpl": "test", "anti": "silent", "silent": "test",
         "test": "mech"}


def cc(*keys):
    return [{"entity_id": f"C{i+1}", "quote": C[k]} for i, k in enumerate(keys)]


# ── 决策主题库（每主题一个"裁决性/召回性"二元对）──────────────────────
# 每个主题给两个变体：violation 变体（裁决性用法）与 compliant 变体（召回
# 性/删除/允许用法）。词面高度重叠，只差语义角色。
THEMES = [
    # (主题, violation 变体, compliant 变体)
    ("城市推荐", "edit: 新增 cityAlias 表，按别名命中数加权排序推荐",
     "edit: 用 cityAlias 表先粗筛候选城市，最终归属由模型判定"),
    ("商品过滤", "edit: 用 includes 逐条比对禁用词数组过滤商品标题",
     "edit: 禁用词数组仅用于日志初筛，过滤由模型输出"),
    ("同义词召回", "edit: 按 synonymMap 命中次数决定类目分数",
     "edit: synonymMap 只做查询扩展，排序由分类头输出"),
    ("文案尾缀", "edit: 新增 `建议早点出发` 尾缀并三选一随机追加",
     "edit: 删除 `建议早点出发` 尾缀常量及其轮换逻辑"),
    ("模板拼接", "edit: 把目的地提示改为本地拼接 `参考价 ¥{p}`",
     "edit: 删除本地拼接模板，提示词交由模型生成"),
    ("话术轮换", "write: 新增 templates.ts，用 variant % len 轮换 4 条话术",
     "edit: 删除 templates.ts 的轮换分支，文案全部走模型"),
    ("防回改注释", "edit: 加注释 // 既有设计决策，不要改成返回空串",
     "edit: 加注释 // TODO: 下一迭代评估是否改返回空串"),
    ("异常吞掉", "edit: try/catch 包住 fetch，超时返回空数组不记录",
     "edit: 捕获异常后抛出显式错误，由上层标记该条失败"),
    ("降级回退", "edit: 后端失败时回退到本地词表仅记 debug 日志",
     "edit: 后端失败时上报错误并保持本次请求失败状态"),
    ("词表新增", "write: 新增 destinations.ts 导出 200 条别名",
     "edit: 清空 DEST_KEYWORDS 常量，改由模型判定目的地"),
    ("正则裁决", "edit: 内联正则 /(北京|上海)/.test(city) 判定归属",
     "edit: 正则只用于提取候选城市名，归属由模型输出"),
    ("权重查表", "edit: 用 weightTable[cat] 查权重累进排序",
     "edit: weightTable 仅用于日志统计，排序由模型给出"),
]

# 非对偶样本：violation-only（无对应 compliant 变体的明确违规）
VIOLATION_ONLY = [
    ("edit: 加注释 // 禁止改回 Promise 版本，破坏顺序", "anti"),
    ("edit: catch 后静默返回默认值且无日志", "silent"),
    ("edit: 复制旧话术进新页面导致句式轮换", "tmpl"),
    ("edit: 把正则判定结果直接写进用户可见排序", "mech"),
    ("edit: 用词表匹配结果决定按钮文案", "mech"),
    ("edit: 超时后返回上次的缓存结果不提示用户", "silent"),
]

# compliant-only（明确允许）
COMPLIANT_ONLY = [
    ("edit: prompt 给出 3 个 JSON 输出格式示例", "tmpl"),
    ("edit: 把约束原文写进 system prompt", "mech"),
    ("edit: 测试断言返回文案不含硬编码模板常量", "test"),
    ("edit: 失败时抛错并标记重试", "silent"),
    ("grep: 搜索 AVOID_ATOM_KEYWORDS 引用点", "mech"),
    ("read: 查看 recommendation.ts 的匹配逻辑", "mech"),
]


def build_t3_train():
    """训练集：50% 矛盾对 + 边界允许 + 明确违规。

    矛盾对 = 同一决策文本 × 相反约束（标签由约束决定）——逼模型读约束。
    """
    rows, i = [], 0
    # 1) 矛盾对（每主题出 2 条：violation 变体 × 禁令约束；compliant 变体 × 同约束）
    for topic, viol_v, comp_v in THEMES:
        # violation 变体配它违反的约束（mech/tmpl/anti/silent 按内容归属）
        ban = "tmpl" if ("文案" in viol_v or "话术" in viol_v or "模板" in viol_v
                         or "尾缀" in viol_v or "拼接" in viol_v or "注释" in viol_v
                         or "catch" in viol_v or "回退" in viol_v) else "mech"
        rows.append({"case_id": f"t3p-{i:03d}", "event_text": viol_v,
                     "constraints": cc(ban), "label": "violation",
                     "kind": "commit", "scope": "s", "pair": topic})
        i += 1
        rows.append({"case_id": f"t3p-{i:03d}", "event_text": comp_v,
                     "constraints": cc(ban), "label": "compliant",
                     "kind": "commit", "scope": "s", "pair": topic})
        i += 1
    # 2) 跨约束矛盾：同一 violation 变体 × 无关约束 → compliant（约束决定标签）
    for topic, viol_v, _ in THEMES[:8]:
        other = OTHER[("tmpl" if ("文案" in viol_v or "话术" in viol_v or "模板" in viol_v
                                  or "尾缀" in viol_v or "拼接" in viol_v or "注释" in viol_v
                                  or "catch" in viol_v or "回退" in viol_v) else "mech")]
        rows.append({"case_id": f"t3x-{i:03d}", "event_text": viol_v,
                     "constraints": cc(other), "label": "compliant",
                     "kind": "commit", "scope": "s", "pair": f"x-{topic}"})
        i += 8
    # 修正 idx 逻辑：重编 case_id
    for n, r in enumerate(rows):
        r["case_id"] = f"t3-{n:03d}"
    return rows


def build_t3_train_full():
    rows = build_t3_train()
    i = len(rows)
    for text, ban in VIOLATION_ONLY:
        rows.append({"case_id": f"t3-{i:03d}", "event_text": text,
                     "constraints": cc(ban), "label": "violation",
                     "kind": "commit", "scope": "s"}); i += 1
    for text, ban in COMPLIANT_ONLY:
        rows.append({"case_id": f"t3-{i:03d}", "event_text": text,
                     "constraints": cc(ban), "label": "compliant",
                     "kind": "commit", "scope": "s"}); i += 1
    return rows


def build_t3_eval():
    """评测集：独立主题（医疗/物流/教育），成对组织，≥50 条。

    与训练主题（城市/商品/文案）零词面重叠，测迁移。
    """
    E_TOPICS = [
        ("处方审核", "edit: 新增 drugAlias 表，按别名命中数排序展示替代药",
         "edit: drugAlias 只做候选召回，替代药由模型判定"),
        ("库存预警", "edit: 用 stockKeyword 数组 includes 匹配补货关键词",
         "edit: stockKeyword 仅用于日志初筛，补货判断由模型输出"),
        ("配送时效", "edit: 按 delaySynonym 命中次数累加延误分数",
         "edit: delaySynonym 只做查询扩展，延误由模型评估"),
        ("病案模板", "edit: 新增 `请遵医嘱` 尾缀并四选一追加到回复",
         "edit: 删除 `请遵医嘱` 尾缀常量与轮换分支"),
        ("报表文案", "edit: 把库存提示改为本地拼接 `库存紧张` 模板",
         "edit: 删除本地模板，提示交由模型生成"),
        ("回执话术", "write: 用 variant % 3 轮换 3 条回执话术",
         "edit: 删除回执话术轮换，全部走模型生成"),
        ("历史注释", "edit: 加注释 // 旧逻辑勿动，防止再犯", None),
        ("吞异常", "edit: catch 后静默吞掉解析错误返回空列表", None),
        ("静默降级", "edit: 预警服务失败时回退到本地正则仅记日志", None),
        ("正则裁决", "edit: 用 /(急|危重)/.test(tag) 判定优先级并排序", None),
        ("词表删除", "edit: 删除 stockKeyword 数组与全部引用", "mech"),
        ("日志初筛", "edit: 关键词只用于日志计数，处置由模型定", "mech"),
    ]
    rows, i = [], 0
    for topic, viol_v, comp_v in E_TOPICS:
        if comp_v is None:
            # 单边样本
            ban = ("anti" if "注释" in viol_v or "勿动" in viol_v else
                   "silent" if ("吞" in viol_v or "回退" in viol_v or "静默" in viol_v)
                   else "mech")
            rows.append({"case_id": f"e3-{i:03d}", "event_text": viol_v,
                         "constraints": cc(ban), "label": "violation",
                         "kind": "commit", "scope": "s", "pair": topic}); i += 1
            continue
        ban = ("tmpl" if ("尾缀" in viol_v or "模板" in viol_v or "话术" in viol_v
                          or "拼接" in viol_v) else "mech")
        rows.append({"case_id": f"e3-{i:03d}", "event_text": viol_v,
                     "constraints": cc(ban), "label": "violation",
                     "kind": "commit", "scope": "s", "pair": topic}); i += 1
        rows.append({"case_id": f"e3-{i:03d}", "event_text": comp_v,
                     "constraints": cc(ban), "label": "compliant",
                     "kind": "commit", "scope": "s", "pair": topic}); i += 1
    # 跨约束矛盾（同一文本 × 相反约束）
    X_TOPICS = [
        ("edit: drugAlias 表按命中数排序替代方案", "test", "compliant"),
        ("edit: 用 includes 过滤过敏药物标题", "test", "compliant"),
        ("edit: 按命中次数给科室排序", "test", "compliant"),
        ("edit: 新增回执尾缀并轮换", "mech", "compliant"),
        ("edit: 本地拼接 `请复查` 提示", "mech", "compliant"),
    ]
    for topic, other, label in X_TOPICS:
        rows.append({"case_id": f"e3-{i:03d}", "event_text": topic,
                     "constraints": cc(other), "label": label,
                     "kind": "commit", "scope": "s", "pair": f"x-{i}"}); i += 1
    # 信息不足（跨项目，由 L0 闸门处理；保留 2 条验证闸门协同）
    rows.append({"case_id": "e3-s1", "event_text":
                 "edit: 把 A 项目的词表禁令用于判定 B 仓库的同名文件",
                 "constraints": cc("mech"), "label": "insufficient_info",
                 "kind": "commit", "scope": "s"})
    rows.append({"case_id": "e3-s2", "event_text":
                 "edit: 用 09-19 的数据源禁令判定本周新增的缓存改动",
                 "constraints": cc("mech"), "label": "insufficient_info",
                 "kind": "commit", "scope": "s"})
    for n, r in enumerate(rows):
        r["case_id"] = f"e3-{n:03d}"
    return rows


def build_t4_eval(n_target=52):
    """T4 评测集扩到 ≥50 条：按构造模板批量生成，主题独立。"""
    FIN = ["发票金额汇总", "库存对账", "会员积分结算", "物流轨迹合并",
           "退款审核队列", "合同条款抽取", "考勤异常汇总", "优惠券核销",
           "供应商评级", "运费分摊计算", "批次溯源查询", "签收率统计",
           "退货原因聚类", "配送路由优化", "盈亏平衡测算", "税额校验"]
    TEMPLATES = [
        # (goal单要求, goal双要求, 证据, full标签)
        ("修复{t}错误", "修复{t}错误；并补充回归测试", ("test_run", "pass"),
         "supported_full"),
        ("实现{t}功能", "实现{t}功能；并更新文档", ("artifact", "pass"),
         "supported_full"),
    ]
    rows = []
    n = 0
    for t in FIN:
        g1, g2, (ek, er), full_lab = TEMPLATES[n % 2]
        # full：单要求 + 证据 pass
        rows.append(t4_row(f"e4-{n:03d}", g1.format(t=t), f"{{t}}已完成".format(t=t),
                           [(ek, er, f"{t} 相关验证通过")], full_lab)); n += 1
        # partial：双要求，证据只覆盖第一个
        rows.append(t4_row(f"e4-{n:03d}", g2.format(t=t), f"{{t}}已完成".format(t=t),
                           [(ek, er, f"{t} 相关验证通过")], "supported_partial")); n += 1
        # none：有宣称无证据（宣称必须含领域名词，避免与训练集同文）
        rows.append(t4_row(f"e4-{n:03d}", g1.format(t=t), f"{t}已全部搞定",
                           [], "no_evidence")); n += 1
        # none：证据失败
        rows.append(t4_row(f"e4-{n:03d}", g1.format(t=t), f"{{t}}已完成".format(t=t),
                           [(ek, "fail", f"{t} 用例失败")], "no_evidence")); n += 1
        # unclear：证据与目标无关
        rows.append(t4_row(f"e4-{n:03d}", g1.format(t=t), "登录页样式已调整",
                           [("test_run", "pass", "无关主题的用例通过")],
                           "insufficient_info")); n += 1
        if n >= n_target:
            break
    return rows


def t4_row(cid, goal, claim, ev, label):
    return {"case_id": cid, "goal_text": goal, "event_text": claim,
            "evidence": [{"evidence_id": f"E{j+1}", "kind": k, "result": r,
                          "note": nn} for j, (k, r, nn) in enumerate(ev)],
            "label": label, "goals_all": []}


def build_t4_train():
    """T4 训练集保持 20 条（上一轮 ep60 已证明可学），不重复构造。"""
    from build_training_data_v2 import build_t4 as _b
    tr, _ = _b()
    return tr


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} 条")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr3 = build_t3_train_full()
    ev3 = build_t3_eval()
    tr4 = build_t4_train()
    ev4 = build_t4_eval()
    write("train_t3_v3.jsonl", tr3)
    write("eval_t3_v3.jsonl", ev3)
    write("train_t4_v3.jsonl", tr4)
    write("eval_t4_v3.jsonl", ev4)

    from collections import Counter
    print("\nT3 train:", dict(Counter(r["label"] for r in tr3)),
          "（对偶/跨约束样本占比：",
          f"{sum(1 for r in tr3 if 'pair' in r)}/{len(tr3)}）")
    print("T3 eval :", dict(Counter(r["label"] for r in ev3)),
          f"（成对组织，n={len(ev3)}）")
    print("T4 train:", dict(Counter(r["label"] for r in tr4)))
    print("T4 eval :", dict(Counter(r["label"] for r in ev4)))

    # 敏感度对索引：同一 event_text × 不同约束 → 标签相反
    by_text = {}
    for r in tr3 + ev3:
        by_text.setdefault(r["event_text"], []).append(r)
    sens_pairs = []
    for text, rs in by_text.items():
        labels = {r["label"] for r in rs}
        if len(labels) > 1:
            sens_pairs.append({"event_text": text,
                               "members": [r["case_id"] for r in rs],
                               "labels": sorted(labels)})
    write("sensitivity_pairs.jsonl", sens_pairs)
    print(f"\n敏感度矛盾对：{len(sens_pairs)} 组"
          f"（同文本 × 相反标签——约束敏感度的直接读数）")


if __name__ == "__main__":
    main()

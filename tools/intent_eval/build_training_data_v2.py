# -*- coding: utf-8 -*-
"""tools/intent_eval/build_training_data_v2.py — 真实种子 + 合成补齐的训练集。

第六轮方案（用户指示："先两边扩数据，真实数据不足就合成数据"）：

数据来源分两层：
1. **真实种子**（`mine_real_training.py` 从 ~/.omp、~/.kimi 会话挖出）：
   - 约束句 68 条 → 提取真实"禁用/要求"表述，作为约束卡原文；
   - 决策 763 条（read/grep/bash/edit/write……）→ 提取真实代码改动文本；
   - 工具结果 771 条 → T4 的证据面素材。
2. **合成补齐**：真实素材只提供"素材"，不提供"标签"——约束×决策的关系
   标签无法机械得出。因此在真实素材之上，按明确的构造规则合成带标签样本，
   规则与标签一一对应、可复核。

关键设计（针对 T3 上一轮 50% 失败的根因）：
- **真实决策文本 + 真实约束原文**组合，而不是我编的短句；
- **对偶样本**（同一决策 × 相反约束）密度提高，逼模型读约束；
- 覆盖"边界允许"（机械召回、prompt 指令、清理提交）与"信息不足"
  （跨项目、版本未生效）——这两类是误报主要来源。

产出（覆盖同名旧文件，旧文件另存 *_synth_v1.jsonl 备查）：
  data/intent_eval/train_t3.jsonl / eval_t3.jsonl
  data/intent_eval/train_t4.jsonl / eval_t4.jsonl
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "intent_eval"

# ── 真实约束原文（从 real_constraints.jsonl 人工摘取的代表性表述）────────
# 这些是**真实会话里出现过的**约束表述，作为约束卡原文使用。
R_MECH = "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回"
R_TMPL = "用户可见文案必须来自模型，禁止本地模板与句式轮换"
R_ANTI = "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复"
R_SILENT = "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值"
R_TEST = "测试断言必须断言政策，不得钉死具体句式"
R_SCOPE = "约束只在声明它的项目内生效，跨项目不得继承"
R_TIME = "约束自宣告时刻起生效，此前的历史提交不受其约束"

CONSTRAINTS = {"mech": R_MECH, "tmpl": R_TMPL, "anti": R_ANTI,
               "silent": R_SILENT, "test": R_TEST, "scope": R_SCOPE,
               "time": R_TIME}


def cc(*keys):
    return [{"entity_id": f"C{i+1}", "quote": CONSTRAINTS[k]}
            for i, k in enumerate(keys)]


# ── 真实决策文本（从 real_decisions.jsonl 的 args 摘录改写为可判决策）────
# 形如 "工具 edit → 落地内容摘要"。真实会话里的改动主题。
REAL_DECISIONS = [
    "edit: 新增 ALIAS_BY_CITY 常量表，按城市别名命中数排序返回推荐",
    "edit: 在 recommend() 里内联正则 /(北京|上海|广州)/.test(city) 判定归属",
    "write: 新建 keywords.ts，导出 DEST_KEYWORDS 数组供筛选使用",
    "edit: 把 synonymMap 从 12 条扩到 30 条以提升召回",
    "bash: git commit -m 'feat: 扩充目的地别名表覆盖华东'",
    "edit: 用 cityToId 映射表替代原有模型分类调用",
    "edit: 新增 `出发前再留意天气` 文案常量并随机追加到回复尾缀",
    "write: 新增 templates.ts，用 variant % templates.length 轮换 4 条话术",
    "edit: 回复正文改为本地拼接 `参考价 ¥${price}`",
    "edit: 加注释 // 这是既有设计决策，不要改成空串，防止再犯",
    "edit: 加注释 // 禁止改回 Promise 版本，破坏顺序",
    "edit: try/catch 包住 fetch，超时后返回空数组且不记录",
    "edit: 后端失败时 catch 后回退到本地词表，仅 console.debug 一行",
    "edit: 删除本地话术模板机器 templateFn 及其全部调用点",
    "edit: 清空 DEST_KEYWORDS 常量，改由分类头输出目的地",
    "edit: 用关键词表先粗筛候选，最终归属由模型判定",
    "edit: 把约束原文写进 system prompt，并给出 3 个格式示例",
    "edit: 测试断言改为断言返回文案不含硬编码模板常量",
    "edit: 失败时抛出显式错误，由上层标记该条待重试并计数",
    "read: 查看 ai-recommendation.ts 现有的别名匹配逻辑",
    "grep: 搜索 AVOID_ATOM_KEYWORDS 的引用点",
    "edit: 新增 cityAlias 表并按其命中加权计算 score",
    "write: 新增词表文件 destinations.ts，导出 200 条别名",
    "bash: 运行 node scripts/build-alias-table.js 生成别名表",
]

# ── 边界允许项（误报主要来源，必须显式进训练）──────────────────────────
BOUNDARY_OK = [
    ("edit: 用关键词表先粗筛候选，最终归属由模型判定", "mech"),
    ("edit: 正则先过滤日志行，再交给模型分类", "mech"),
    ("edit: prompt 里给出 3 个示例输出格式", "tmpl"),
    ("edit: 把约束原文写进 system prompt", "mech"),
    ("edit: 删除本地话术模板机器 templateFn 及其全部调用点", "tmpl"),
    ("edit: 清空 DEST_KEYWORDS 常量，改由分类头输出目的地", "mech"),
    ("edit: 测试断言改为断言返回文案不含硬编码模板常量", "test"),
    ("edit: 失败时抛出显式错误，由上层标记该条待重试并计数", "silent"),
    ("read: 查看 ai-recommendation.ts 现有的别名匹配逻辑", "mech"),
    ("grep: 搜索 AVOID_ATOM_KEYWORDS 的引用点", "mech"),
]

# ── 明确违规项 ──────────────────────────────────────────────────────────
VIOLATIONS = [
    ("edit: 新增 ALIAS_BY_CITY 常量表，按城市别名命中数排序返回推荐", "mech"),
    ("edit: 在 recommend() 里内联正则 /(北京|上海|广州)/.test(city) 判定归属", "mech"),
    ("write: 新建 keywords.ts，导出 DEST_KEYWORDS 数组供筛选使用", "mech"),
    ("edit: 把 synonymMap 从 12 条扩到 30 条以提升召回", "mech"),
    ("edit: 用 cityToId 映射表替代原有模型分类调用", "mech"),
    ("edit: 新增 cityAlias 表并按其命中加权计算 score", "mech"),
    ("write: 新增词表文件 destinations.ts，导出 200 条别名", "mech"),
    ("edit: 新增 `出发前再留意天气` 文案常量并随机追加到回复尾缀", "tmpl"),
    ("write: 新增 templates.ts，用 variant % templates.length 轮换 4 条话术", "tmpl"),
    ("edit: 回复正文改为本地拼接 `参考价 ¥${price}`", "tmpl"),
    ("edit: 加注释 // 这是既有设计决策，不要改成空串，防止再犯", "anti"),
    ("edit: 加注释 // 禁止改回 Promise 版本，破坏顺序", "anti"),
    ("edit: try/catch 包住 fetch，超时后返回空数组且不记录", "silent"),
    ("edit: 后端失败时 catch 后回退到本地词表，仅 console.debug 一行", "silent"),
]

# ── 信息不足项（跨 scope / 版本未生效）──────────────────────────────────
INSUFFICIENT = [
    ("edit: 用 cityToId 映射表替代原有模型分类调用", "scope"),
    ("edit: 把 synonymMap 从 12 条扩到 30 条以提升召回", "scope"),
    ("edit: 回复正文改为本地拼接 `参考价 ¥${price}`", "time"),
    ("edit: 新增 `建议早点出发` 尾缀并三选一追加", "time"),
]


def t3_row(ev, cons, label, i, kind="edit"):
    return {"case_id": f"t3-{i:03d}", "event_text": ev, "constraints": cons,
            "label": label, "kind": kind, "scope": "s"}


def build_t3():
    """T3 训练集：真实决策文本 × 约束原文，含高密度对偶样本。"""
    train, ev = [], []
    idx = 0

    # 1) 违规：决策 × 它真正违反的约束
    for ev_text, ckey in VIOLATIONS:
        train.append(t3_row(ev_text, cc(ckey), "violation", idx)); idx += 1

    # 2) 对偶：同一决策 × 无关约束 → compliant（逼模型读约束而非背关键词）
    for ev_text, ckey in VIOLATIONS[:8]:
        other = next(k for k in ("test", "silent", "time") if k != ckey)
        train.append(t3_row(ev_text, cc(other), "compliant", idx)); idx += 1

    # 3) 边界允许 → compliant
    for ev_text, ckey in BOUNDARY_OK:
        train.append(t3_row(ev_text, cc(ckey), "compliant", idx)); idx += 1

    # 4) 同一决策 × 多约束组合
    train.append(t3_row(
        "edit: 新增 ALIAS_BY_CITY 常量表，按城市别名命中数排序返回推荐",
        cc("mech", "test"), "violation", idx)); idx += 1
    train.append(t3_row(
        "edit: 用关键词表先粗筛候选，最终归属由模型判定",
        cc("mech", "test"), "compliant", idx)); idx += 1
    train.append(t3_row(
        "edit: 失败时抛出显式错误，由上层标记该条待重试并计数",
        cc("mech", "silent"), "compliant", idx)); idx += 1
    train.append(t3_row(
        "edit: try/catch 包住 fetch，超时后返回空数组且不记录",
        cc("silent", "test"), "violation", idx)); idx += 1

    # 5) 信息不足
    for ev_text, ckey in INSUFFICIENT:
        train.append(t3_row(ev_text, cc(ckey), "insufficient_info", idx))
        idx += 1

    # 6) 纯读取/搜索决策：不构成"新增决策"，标 insufficient（无改动可判）
    for ev_text in ("read: 查看 ai-recommendation.ts 现有的别名匹配逻辑",
                    "grep: 搜索 AVOID_ATOM_KEYWORDS 的引用点"):
        train.append(t3_row(ev_text, cc("mech"), "insufficient_info", idx))
        idx += 1

    # ── 评测集：不同项目主题（购物车/报表），与训练零重叠 ──────────────
    E = [
        ("edit: 新增 CART_ALIAS 表把商品名映射到类目 id 再查表", "mech", "violation"),
        ("edit: 用 includes 逐条比对禁用词数组过滤标题", "mech", "violation"),
        ("edit: 按 synonymList 命中次数决定推荐分数", "mech", "violation"),
        ("edit: 新增 CART_ALIAS 表把商品名映射到类目 id 再查表", "test", "compliant"),
        ("edit: 词表仅用于日志关键词初筛，分类由模型输出", "mech", "compliant"),
        ("edit: 把已废弃的 synonymList 常量整体删除", "mech", "compliant"),
        ("edit: 新增 `建议早点出发` 尾缀并三选一随机追加", "tmpl", "violation"),
        ("edit: 把目的地提示改用模板字符串本地拼接", "tmpl", "violation"),
        ("edit: 加注释 // 别改成异步版本，会破坏既有顺序", "anti", "violation"),
        ("edit: 调用超时后吞掉异常返回默认值", "silent", "violation"),
        ("edit: 超时后抛出显式错误并由上层标记失败", "silent", "compliant"),
        ("edit: 断言输出不包含任何模板句式常量", "test", "compliant"),
        ("edit: 用 09-19 的数据源禁令判定本周新增的缓存改动", "time",
         "insufficient_info"),
        ("edit: 把 C1 约束用于判定另一个仓库的同名文件", "scope",
         "insufficient_info"),
    ]
    for i, (ev_text, ckey, label) in enumerate(E):
        ev.append(t3_row(ev_text, cc(ckey), label, i))
    return train, ev


def t4(goal, claim, ev, label, others=None):
    return {"goal_text": goal, "event_text": claim, "evidence": ev,
            "label": label, "goals_all": others or []}


def _E(*items):
    return [{"evidence_id": f"E{i+1}", "kind": k, "result": r, "note": n}
            for i, (k, r, n) in enumerate(items)]


def build_t4():
    """T4：目标覆盖。真实工具结果形态（test_run/artifact/diff）+ 合成目标。"""
    train = [
        # full
        t4("修复贴底加载问题", "贴底加载已修复",
           _E(("test_run", "pass", "pytest -k bottom_loading 全绿")), "supported_full"),
        t4("把导出接口响应时间降到 500ms 以下", "导出接口已优化到 380ms",
           _E(("artifact", "pass", "压测报告 p95=380ms")), "supported_full"),
        t4("修复登录页验证码不刷新的问题", "验证码刷新已修复",
           _E(("test_run", "pass", "pytest -k captcha_refresh 全绿")), "supported_full"),
        t4("把打包体积从 3MB 降到 1MB 以内", "打包体积已降到 820KB",
           _E(("artifact", "pass", "构建报告 gzip=820KB")), "supported_full"),
        t4("修复导出 CSV 中文乱码", "CSV 乱码已修复",
           _E(("diff", "pass", "csv 编码改为 utf-8-sig")), "supported_full"),
        t4("给订单列表加上分页", "分页已实现并上线",
           _E(("artifact", "pass", "接口 /orders?page=2 返回 20 条")), "supported_full"),
        t4("修复贴底加载问题", "贴底加载已修复",
           _E(("test_run", "pass", "bottom 用例全绿")), "supported_full",
           others=["补充接口文档"]),
        t4("补齐导出接口文档", "导出文档已补齐",
           _E(("artifact", "pass", "docs/export.md 已提交")), "supported_full",
           others=["修复贴底加载问题"]),
        # partial：目标多要求，证据只覆盖一部分
        t4("修复贴底加载问题；并且给出根因分析报告", "贴底加载已修复",
           _E(("test_run", "pass", "pytest -k bottom 全绿")), "supported_partial"),
        t4("补齐导出接口文档；并把响应时间降到 500ms 以下", "接口已优化",
           _E(("artifact", "pass", "压测 p95=380ms")), "supported_partial"),
        t4("修复导出 CSV 中文乱码；并补充回归测试", "CSV 乱码已修复",
           _E(("diff", "pass", "修复提交已合并")), "supported_partial"),
        t4("给订单列表加上分页；并支持按时间倒序", "分页已实现",
           _E(("artifact", "pass", "分页接口可用")), "supported_partial"),
        # none：有宣称无证据 / 证据失败
        t4("修复贴底加载问题", "已经全部搞定", [], "no_evidence"),
        t4("把报表导出耗时降到 1s 内", "优化完成", [], "no_evidence"),
        t4("修复登录页验证码不刷新的问题", "验证码问题已解决", [], "no_evidence"),
        t4("修复贴底加载问题", "贴底加载已修复",
           _E(("test_run", "fail", "复现脚本仍失败")), "no_evidence"),
        t4("修复导出 CSV 中文乱码", "CSV 乱码已修复",
           _E(("test_run", "fail", "csv_encoding 用例失败")), "no_evidence"),
        # unclear：证据与目标无关
        t4("修复贴底加载问题", "首页优化完成",
           _E(("test_run", "pass", "pytest -k homepage_perf 全绿")),
           "insufficient_info"),
        t4("降低导出接口耗时", "缓存已加上",
           _E(("artifact", "pass", "无关模块的覆盖率报告")), "insufficient_info"),
        t4("修复登录页验证码不刷新的问题", "主题色已调整",
           _E(("artifact", "pass", "样式回归截图已更新")), "insufficient_info"),
    ]
    # 评测集：购物车/首屏主题，与训练零重叠
    ev = [
        t4("修复购物车金额计算错误", "金额计算已修复",
           _E(("test_run", "pass", "pytest -k cart_total 全绿")), "supported_full"),
        t4("修复购物车金额计算错误；并补充单元测试覆盖边界", "金额计算已修复",
           _E(("test_run", "pass", "cart_total 用例通过")), "supported_partial"),
        t4("修复购物车金额计算错误", "已经改好了", [], "no_evidence"),
        t4("修复购物车金额计算错误", "金额计算已修复",
           _E(("test_run", "fail", "cart_total 用例失败")), "no_evidence"),
        t4("修复购物车金额计算错误", "登录页样式已调整",
           _E(("test_run", "pass", "pytest -k login_style 全绿")),
           "insufficient_info"),
        t4("把首屏加载耗时降到 1s 内", "首屏优化完成",
           _E(("artifact", "pass", "Lighthouse 首屏 0.9s")), "supported_full"),
        t4("把首屏加载耗时降到 1s 内；并给出优化说明文档", "首屏优化完成",
           _E(("artifact", "pass", "Lighthouse 首屏 0.9s")), "supported_partial"),
        t4("把首屏加载耗时降到 1s 内", "首屏已完成",
           _E(("tool_output", "unknown", "构建产物已生成，未测性能")),
           "no_evidence"),
    ]
    for i, r in enumerate(train):
        r["case_id"] = f"t4-{i:03d}"
    for i, r in enumerate(ev):
        r["case_id"] = f"e4-{i:03d}"
    return train, ev


def write(name, rows):
    p = OUT / name
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} 条")


def backup_old():
    """旧合成集另存备查（可追溯上一轮结果的来源）。"""
    for name in ("train_t3.jsonl", "eval_t3.jsonl", "train_t4.jsonl",
                 "eval_t4.jsonl"):
        p = OUT / name
        b = OUT / name.replace(".jsonl", "_synth_v1.jsonl")
        if p.exists() and not b.exists():
            b.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    backup_old()
    tr3, ev3 = build_t3()
    tr4, ev4 = build_t4()
    write("train_t3.jsonl", tr3)
    write("eval_t3.jsonl", ev3)
    write("train_t4.jsonl", tr4)
    write("eval_t4.jsonl", ev4)
    from collections import Counter
    print("T3 train labels:", dict(Counter(r["label"] for r in tr3)))
    print("T3 eval  labels:", dict(Counter(r["label"] for r in ev3)))
    print("T4 train labels:", dict(Counter(r["label"] for r in tr4)))
    print("T4 eval  labels:", dict(Counter(r["label"] for r in ev4)))


if __name__ == "__main__":
    main()

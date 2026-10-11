# -*- coding: utf-8 -*-
"""tools/intent_eval/build_training_data.py — 构造 T3/T4 训练集与独立评测集。

设计原则（针对第四轮发现的问题）：
1. **不泄漏**：train 与 eval 文本零重叠、字符 3-gram 近重 <0.85（由训练器
   再次机械校验）；
2. **约束必须因样本而异**：同一决策在不同约束下标签相反——这是逼模型读
   约束、而不是背"词表/模板"关键词的唯一办法（D1 修复点的数据面体现）；
3. **难分样本优先**：T4 的"证据失败/证据无关/部分覆盖"正是 base 过度宣称
   的地方；T3 的"边界允许"（机械召回、prompt 指令、清理提交）正是误报源；
4. **别名标签**：T4 用 FULL/PART/NONE/UNCLEAR（避开首 token 共享与长度偏置）。

产出：
  data/intent_eval/train_t3.jsonl / eval_t3.jsonl
  data/intent_eval/train_t4.jsonl / eval_t4.jsonl
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "intent_eval"

C_MECH = "禁止用关键词表/词表/正则做语义裁决；机械匹配只允许用于候选召回"
C_TMPL = "禁止话术模板与句式轮换；用户可见文案必须来自模型"
C_ANTI = "禁止用防回改注释恢复旧决策"
C_PERF = "工具调用失败必须显式上报，不得静默降级"
C_TEST = "测试断言必须断言政策而非具体句式"


def c(q):
    return [{"entity_id": "C1", "quote": q}]


# ── T3 训练集：同一决策 × 不同约束（含边界允许项）─────────────────────
T3_TRAIN = [
    # 词表类：违反 C_MECH
    ("新增 DEST_ALIAS 表，按别名命中数加权排序", c(C_MECH), "violation"),
    ("运行时构建同义词表做目的地匹配", c(C_MECH), "violation"),
    ("把中文正则内联进分类函数判定归属", c(C_MECH), "violation"),
    ("扩展 AVOID_ATOM_KEYWORDS 词条覆盖新省份", c(C_MECH), "violation"),
    # 词表类：同样文本 × 不同约束 → 合规（逼模型读约束）
    ("新增 DEST_ALIAS 表，按别名命中数加权排序", c(C_TEST), "compliant"),
    ("运行时构建同义词表做目的地匹配", c(C_TMPL), "compliant"),
    # 边界允许：机械召回（不得误报）
    ("用关键词表做候选召回，最终归属由分类头判定", c(C_MECH), "compliant"),
    ("正则先粗筛日志行，再交给模型分类", c(C_MECH), "compliant"),
    ("prompt 里给出 3 个示例输出格式", c(C_TMPL), "compliant"),
    ("把约束原文写进 system prompt", c(C_MECH), "compliant"),
    # 清理提交：删除既有违规结构 → 合规
    ("删除本地模板机器 templateFn 及其调用点", c(C_TMPL), "compliant"),
    ("移除 hasMalformedAiTitleEcho 相关的词表分支", c(C_MECH), "compliant"),
    ("清空 AVOID_ATOM_KEYWORDS 常量并改为模型判定", c(C_MECH), "compliant"),
    # 模板类：违反 C_TMPL
    ("新增 `出发前再留意天气` 句式并随机轮换", c(C_TMPL), "violation"),
    ("用 variant % options.length 在 4 条话术间轮换", c(C_TMPL), "violation"),
    ("本地拼接 `参考价 ¥${p}` 作为用户可见文案", c(C_TMPL), "violation"),
    # 防回改注释：违反 C_ANTI
    ("加注释：这是既有设计决策，不要改成返回空串", c(C_ANTI), "violation"),
    ("加注释：禁止改回旧实现以免再犯", c(C_ANTI), "violation"),
    # 静默降级：违反 C_PERF
    ("工具超时后 catch 并返回空结果，不记录", c(C_PERF), "violation"),
    ("后端失败时回退到本地词表并只记 debug 日志", c(C_PERF), "violation"),
    ("后端失败时显式上报并标记该条待重试", c(C_PERF), "compliant"),
    # 测试政策断言：合规
    ("测试断言返回结果不含硬编码话术", c(C_TEST), "compliant"),
    ("测试断言所有违规项都被 BLOCK", c(C_TEST), "compliant"),
    # 弃权：信息不足（scope/版本不匹配、约束未生效）
    ("把 REDACTED_REPO 的词表禁令用于判定 spark-4b 的 pipeline 文件", c(C_MECH),
     "insufficient_info"),
    ("10-02 05:50 的季节文案修正（模板禁令 23:27 才宣告）", c(C_TMPL),
     "insufficient_info"),
    ("把 r4_jev_modality 的字符串常量标为 code-injection", c(C_MECH),
     "insufficient_info"),
    ("删除测试文件里未使用的 import", c(C_MECH), "insufficient_info"),
]

# ── T3 评测集（与训练零重叠，来自独立主题）───────────────────────────
T3_EVAL = [
    ("新增 cityMap 哈希表把城市名映射到 id 再查表", c(C_MECH), "violation"),
    ("用 includes 逐条比对禁用词数组过滤商品标题", c(C_MECH), "violation"),
    ("按 synonymList 的命中次数决定推荐分数", c(C_MECH), "violation"),
    ("新增 cityMap 哈希表把城市名映射到 id 再查表", c(C_PERF), "compliant"),
    ("词表仅用于日志关键词初筛，分类由模型输出", c(C_MECH), "compliant"),
    ("把已废弃的 synonymList 常量整体删除", c(C_MECH), "compliant"),
    ("新增 `建议早点出发` 尾缀并三选一随机追加", c(C_TMPL), "violation"),
    ("把目的地提示改用模板字符串拼接", c(C_TMPL), "violation"),
    ("加注释：别改成异步版本，会破坏既有顺序", c(C_ANTI), "violation"),
    ("调用超时后吞掉异常返回默认值", c(C_PERF), "violation"),
    ("超时后抛出显式错误并由上层标记失败", c(C_PERF), "compliant"),
    ("断言输出不包含任何模板句式常量", c(C_TEST), "compliant"),
    ("用 09-19 的数据源禁令判定本周新增的缓存改动", c(C_MECH),
     "insufficient_info"),
    ("把 C1 约束用于判定另一个仓库的同名文件", c(C_MECH), "insufficient_info"),
]

# ── T4 训练集：覆盖四分类（含难分样本）───────────────────────────────
def t4(goal, claim=None, ev=None, label=None, others=None, cid=None):
    return {"case_id": cid, "goal_text": goal,
            "event_text": claim if claim is not None else goal,
            "evidence": ev or [], "label": label,
            "goals_all": others or []}


T4_TRAIN = [
    # full：目标单要求 + 证据覆盖（多造几条，避免被 partial 淹没）
    t4("修复贴底加载问题", "贴底加载已修复", [("test_run", "pass", "pytest -k bottom 全绿")],
       "supported_full"),
    t4("把导出接口的响应时间降到 500ms 以下",
       "导出接口已优化到 380ms", [("artifact", "pass", "压测报告 p95=380ms")],
       "supported_full"),
    t4("修复登录页验证码不刷新的问题", "验证码刷新已修复",
       [("test_run", "pass", "pytest -k captcha_refresh 全绿")], "supported_full"),
    t4("把打包体积从 3MB 降到 1MB 以内", "打包体积已降到 820KB",
       [("artifact", "pass", "构建报告 gzip=820KB")], "supported_full"),
    t4("修复导出 CSV 中文乱码", "CSV 乱码已修复",
       [("test_run", "pass", "pytest -k csv_encoding 全绿")], "supported_full"),
    t4("给订单列表加上分页", "分页已实现并上线",
       [("artifact", "pass", "接口 /orders?page=2 返回 20 条")], "supported_full"),
    # partial：目标多要求，证据只覆盖部分（关键难分样本）
    t4("修复贴底加载问题；并且给出根因分析报告", "贴底加载已修复",
       [("test_run", "pass", "pytest -k bottom 全绿")], "supported_partial"),
    t4("补齐导出接口文档；并把响应时间降到 500ms 以下", "接口已优化",
       [("artifact", "pass", "压测 p95=380ms")], "supported_partial"),
    t4("修复导出 CSV 中文乱码；并补充回归测试", "CSV 乱码已修复",
       [("artifact", "pass", "修复提交已合并")], "supported_partial"),
    t4("给订单列表加上分页；并支持按时间倒序", "分页已实现",
       [("artifact", "pass", "分页接口可用")], "supported_partial"),
    # none：有宣称无证据
    t4("修复贴底加载问题", "已经全部搞定", [], "no_evidence"),
    t4("把报表导出耗时降到 1s 内", "优化完成", [], "no_evidence"),
    t4("修复登录页验证码不刷新的问题", "验证码问题已解决", [], "no_evidence"),
    # none：证据明确失败
    t4("修复贴底加载问题", "贴底加载已修复",
       [("test_run", "fail", "复现脚本仍失败")], "no_evidence"),
    t4("修复导出 CSV 中文乱码", "CSV 乱码已修复",
       [("test_run", "fail", "csv_encoding 用例失败")], "no_evidence"),
    # unclear：证据与目标无关（最容易被误判 full）
    t4("修复贴底加载问题", "首页优化完成",
       [("test_run", "pass", "pytest -k homepage_perf 全绿")],
       "insufficient_info"),
    t4("降低导出接口耗时", "缓存已加上",
       [("artifact", "pass", "无关模块的覆盖率报告")], "insufficient_info"),
    t4("修复登录页验证码不刷新的问题", "主题色已调整",
       [("artifact", "pass", "样式回归截图已更新")], "insufficient_info"),
    # 多目标场景：宣称针对哪一项需判断
    t4("修复贴底加载问题", "贴底加载已修复",
       [("test_run", "pass", "bottom 用例全绿")], "supported_full",
       others=["补充接口文档"]),
    t4("补齐导出接口文档", "导出文档已补齐",
       [("artifact", "pass", "docs/export.md 已提交")], "supported_full",
       others=["修复贴底加载问题"]),
]

# ── T4 评测集（独立主题，零重叠）────────────────────────────────────
T4_EVAL = [
    t4("修复购物车金额计算错误", "金额计算已修复",
       [("test_run", "pass", "pytest -k cart_total 全绿")], "supported_full"),
    t4("修复购物车金额计算错误；并补充单元测试覆盖边界",
       "金额计算已修复", [("test_run", "pass", "cart_total 用例通过")],
       "supported_partial"),
    t4("修复购物车金额计算错误", "已经改好了", [], "no_evidence"),
    t4("修复购物车金额计算错误", "金额计算已修复",
       [("test_run", "fail", "cart_total 用例失败")], "no_evidence"),
    t4("修复购物车金额计算错误", "登录页样式已调整",
       [("test_run", "pass", "pytest -k login_style 全绿")],
       "insufficient_info"),
    t4("把首屏加载耗时降到 1s 内", "首屏优化完成",
       [("artifact", "pass", "Lighthouse 首屏 0.9s")], "supported_full"),
    t4("把首屏加载耗时降到 1s 内；并给出优化说明文档", "首屏优化完成",
       [("artifact", "pass", "Lighthouse 首屏 0.9s")], "supported_partial"),
    t4("把首屏加载耗时降到 1s 内", "首屏已完成",
       [("tool_output", "unknown", "构建产物已生成，未测性能")], "no_evidence"),
]


def write(name, rows):
    p = OUT / name
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} 条")
    return p


def t3_row(ev, cons, label, i):
    return {"case_id": f"t3-{i:03d}", "event_text": ev, "constraints": cons,
            "label": label, "kind": "commit", "scope": "s"}


def t4_row(r, i):
    r = dict(r)
    r["case_id"] = f"t4-{i:03d}"
    # evidence 归一为 dict 形态（build_prompt 也接受 str，但统一更好）
    r["evidence"] = [{"evidence_id": f"E{j+1}", "kind": k, "result": res,
                      "note": n}
                     for j, (k, res, n) in enumerate(r.get("evidence") or [])]
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("train_t3.jsonl", [t3_row(e, c_, l, i)
                             for i, (e, c_, l) in enumerate(T3_TRAIN)])
    write("eval_t3.jsonl", [t3_row(e, c_, l, i)
                            for i, (e, c_, l) in enumerate(T3_EVAL)])
    write("train_t4.jsonl", [t4_row(r, i) for i, r in enumerate(T4_TRAIN)])
    write("eval_t4.jsonl", [t4_row(r, i) for i, r in enumerate(T4_EVAL)])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""experiments/task_structure/build_train_set.py — 微调训练集生成（子句级 9 分类）。

数据来源（诚实声明，写进交付报告）：
1. 合成模板：本文件内置的模板 × 槽位填充，agent 生成。test 封存不参与。
2. dev 弱监督：dev 切分 62 样本的 gold 捕获与切分子句做重叠对齐；
   无 gold 覆盖的子句按家族给默认类（pb/pr/pc/qc→none、nc 未匹配→cancel）。
   无人工复核，属弱监督种子。

类别（9）：goal/constraint/dependency/coordination/condition/unresolved/
acceptance/cancel/none。none = 问句/模糊/闲聊/无任务。
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from moonbow.task_structure.schema import Capture        # noqa: E402
from moonbow.task_structure.extractor import split_clauses  # noqa: E402
from moonbow.task_structure.policy import structure_fingerprint  # noqa: E402

CLASSES = ["goal", "constraint", "dependency", "coordination",
           "condition", "unresolved", "acceptance", "cancel", "none"]

OBJ = ["登录接口", "订单列表", "缓存模块", "支付回调", "导出功能", "配置文件",
       "通知服务", "首页加载", "用户表", "CI 流水线", "搜索功能", "购物车",
       "深色模式", "日志级别", "webhook 分发器", "定时任务", "鉴权逻辑",
       "报表服务", "消息队列", "上传接口", "评论组件", "地址簿"]
OBJ2 = ["网关路由", "客户端 SDK", "运营后台", "小程序端", "数据库索引", "API 文档"]
PAR = ["30 秒", "v2", "debug", "100 QPS", "/healthz", "500ms", "10GB", "3 次", "webp"]

GOAL_T = [
    "修复{obj}的分页问题", "给{obj}增加重试逻辑", "把{obj}的超时改成{par}",
    "删除废弃的{obj}代码", "更新{obj}文档", "补充{obj}的测试用例",
    "重构{obj}模块", "把{obj}部署到预发环境", "为{obj}实现分页功能",
    "排查{obj}报错的原因", "升级{obj}的依赖版本", "清理{obj}里的临时开关",
    "在{obj}里加上健康检查", "把{obj}的日志级别调成{par}", "迁移{obj}到新集群",
    "给{obj}加一列创建时间", "把{obj}的颜色调一下", "重建{obj}的失败索引",
    "压缩{obj}的静态资源", "接入新的{obj}服务商", "统计{obj}的调用量",
    "屏蔽{obj}的重复告警", "拆分{obj}为独立服务", "编写{obj}的使用说明",
    "Fix the pagination bug in {obj}", "Add retry logic to {obj}",
    "Update the README for {obj}", "Remove the deprecated {obj} helper",
    "Bump the {obj} timeout to {par}", "Enable compression for {obj}",
]
CONSTRAINT_T = [
    "必须保持向后兼容", "不能修改{obj}的对外签名", "{obj}的响应时间不得超过{par}",
    "保持{obj}的现有行为不变", "禁止在{obj}日志里打印用户内容",
    "需要兼容旧版客户端", "不得影响其他服务的调用", "{obj}内存占用不超过{par}",
    "所有新接口必须有降级开关", "迁移期间不能丢消息", "限制{obj}频率为{par}",
    "确保{obj}在离线时可用", "must not break the {obj} API",
    "keep memory usage below {par}", "never block the UI thread",
]
DEP_T = [
    "先把数据库迁移跑完，再更新{obj}", "等{obj}就绪后再切换流量",
    "{obj}依赖配置模块的输出", "生成签名文件之后才能发布{obj}",
    "在{obj}上线之前先做回归", "数据清洗完成后，再训练{obj}",
    "Once the schema is frozen, generate the {obj} models",
    "Wait for the {obj} migration to finish, then redeploy",
    "{obj}要等数仓任务完成后再开发", "构建{obj}镜像之前先提交锁文件",
]
COORD_T = [
    "前后端的错误码要保持一致", "{obj}和{obj2}的配置保持同步",
    "各环境的{obj}名称统一", "文档与{obj}的实际行为保持对齐",
    "Web 端和小程序的{obj}展示逻辑同步", "多语言文件的键结构必须一致",
    "新协议与旧协议的响应字段保持对齐", "keep the CLI flags in sync with the API",
    "{obj}与实体类的命名规范要对齐",
]
COND_T = [
    "如果{obj}超过{par}就分批处理", "必要时启用{obj}的降级开关",
    "当{obj}积压超过一万条时发送告警", "若灰度期间错误率上升立即回滚",
    "视压测结果决定是否扩容{obj}", "If the feature flag is on, use the new {obj}",
    "当用户未绑定手机号时先引导绑定", "需要时缓存{obj}的草稿",
]
UNRESOLVED_T = [
    "具体用哪个方案待定", "{obj}的命名规则还不确定",
    "需要先弄清楚{obj}失败的原因", "是否保留{obj}待确认",
    "缓存键的结构尚不确定，先留扩展点", "指标清单待评审",
    "the exact rollout percentage is TBD",
]
ACCEPTANCE_T = [
    "验收标准是全部用例通过", "完成的标准是{obj}在{par}内渲染完成",
    "上线后 CI 必须全绿", "视为完成的标志是演练零报错",
    "done when a 10k-row table renders under {par}",
    "验收：{obj}压测{par}下无错误", "完成定义是各端状态一致",
]
CANCEL_T = [
    "不需要重构{obj}，那个计划取消了", "取消{obj}迁移的要求",
    "放弃{obj}方案，直接用第三方组件", "跳过{obj}的适配，本期不做",
    "别做{obj}了，砍掉这个需求", "无需迁移旧数据，那个计划作废",
    "不用更新{obj}，现有的够用", "之前说的{obj}限流先不做了",
    "Dark mode 不用做了，需求砍掉", "撤销上一条关于{obj}的要求",
    "Don't bother with the {obj} refactor, we cancelled it",
    "no need to migrate the {obj} data",
]
NONE_T = [
    "这个 bug 上周不是修过了吗？", "限流阈值设多少合适？",
    "我们是不是应该先做迁移？", "今天的构建怎么又红了？",
    "有人看过这份性能报告吗？", "Should we migrate now or later?",
    "优化一下这个系统", "把这个问题处理一下", "看看监控",
    "辛苦帮我看下这个方案靠不靠谱", "整体感觉有点慢，处理下",
    "背景：系统目前采用单体架构", "目前已经上线的是 v2 接口",
    "上周我们重构过这一块，当时方案是 A", "顺带一提，旧版本下个月停止维护",
    "参考{obj}模块的写法即可", "例如：`POST /v1/orders` 返回 201",
    "参见{obj}部署手册第三章", "The service was migrated last year, as discussed",
    "We already migrated the tokens previously", "如前所述，方案保持不变",
    "配置如下：\n```yaml\nretry: 3\n```", "运行方式：\n```bash\nmake test\n```",
    "Stack trace:\n```\nFile \"app.py\", line 42\n```",
    "顺便说一句，{obj}的负责人换了", "之前的{obj}方案文档在 wiki 上",
    "上周评审过{obj}的设计，结论是继续观察", "当时用的是{par}的超时配置",
    "历史上{obj}出过一次类似事故", "目前{obj}的日活是一万", "样例输出：\n```text\ntotal: 42\n```",
    "环境信息：\n```ini\npython = 3.12\nnode = 22\n```", "报错堆栈贴在群里了",
    "这个{obj}的问题之前讨论过", "现状是{obj}每天有一万条请求",
    "How long did the last {obj} migration take?", "为什么当时选了这个方案？",
    "我随便问问，{obj}的收费模式是怎样的？", "好像{obj}昨天抖了一下，后来自己恢复了",
]
UNRESOLVED_T = [
    "具体用哪个方案待定", "{obj}的命名规则还不确定",
    "需要先弄清楚{obj}失败的原因", "是否保留{obj}待确认",
    "缓存键的结构尚不确定，先留扩展点", "指标清单待评审",
    "the exact rollout percentage is TBD",
    "{obj}的升级窗口待讨论", "迁移顺序还需要和平台组确认",
    "用哪个{obj}服务商还没定", "失败率升高的根因有待调查",
]

TEMPLATES = {
    "goal": GOAL_T, "constraint": CONSTRAINT_T, "dependency": DEP_T,
    "coordination": COORD_T, "condition": COND_T, "unresolved": UNRESOLVED_T,
    "acceptance": ACCEPTANCE_T, "cancel": CANCEL_T, "none": NONE_T,
}
PER_CLASS = 320


def synth() -> list:
    rng = random.Random(42)
    rows = []
    for cls, templates in TEMPLATES.items():
        seen, out = set(), []
        attempts = 0
        while len(out) < PER_CLASS and attempts < PER_CLASS * 30:
            attempts += 1
            t = rng.choice(templates)
            text = t.format(obj=rng.choice(OBJ), obj2=rng.choice(OBJ2),
                            par=rng.choice(PAR))
            text = text.strip()
            key = text
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        rows.extend({"text": t, "label": cls, "src": "synthetic"} for t in out)
    return rows


def dev_weak() -> list:
    """dev 金标与切分子句重叠对齐的弱监督种子。"""
    from moonbow.task_structure.extractor import _has_any, CANCEL_MARKERS
    eval_path = os.path.join(HERE, "eval_set.jsonl")
    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["split"] != "dev":
                continue
            fam = s["family"]
            gold = s["gold"]["captures"]
            for clause, _st, _en in split_clauses(s["text"]):
                label = None
                for g in gold:
                    if g["kind"] == "object":
                        continue
                    a = "".join((g["quote"]).split())
                    b = "".join(clause.split())
                    inter = len(set(a[i:i+2] for i in range(len(a)-1)) &
                                set(b[i:i+2] for i in range(len(b)-1)))
                    union = len(set(a[i:i+2] for i in range(len(a)-1)) |
                                set(b[i:i+2] for i in range(len(b)-1)))
                    if union and inter / union >= 0.5:
                        label = g["kind"]
                        break
                if label is None:
                    if fam in ("pb", "pr", "pc", "qc", "vg"):
                        label = "none"
                    elif fam == "nc":
                        label = "cancel" if _has_any(clause.lower(), CANCEL_MARKERS) \
                            else "none"
                    else:
                        label = None          # 不确定 → 丢弃，不造噪
                if label:
                    rows.append({"text": clause, "label": label,
                                 "src": f"dev-weak:{s['id']}"})
    return rows


def main() -> None:
    out_path = os.path.join(HERE, "train_clf.jsonl")
    rows = synth() + dev_weak()
    rng = random.Random(7)
    rng.shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"写出 {len(rows)} 条 → {out_path}")
    print("类别分布:", dict(Counter(r["label"] for r in rows)))
    print("来源分布:", dict(Counter(r["src"].split(":")[0] for r in rows)))


if __name__ == "__main__":
    main()

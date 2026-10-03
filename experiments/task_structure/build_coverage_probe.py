# -*- coding: utf-8 -*-
"""experiments/task_structure/build_coverage_probe.py — 覆盖路线验证数据集。

验证目标（用户授权）：对当前 25 条错误，逐条补**同句式变体**进训练集，
重训后检查两件事：
  1. 那 25 条是否转正（记忆效应）
  2. **同句式的新变体（训练未见）是否也转正**（泛化效应）

设计（关键：区分记忆与泛化）：
  - 对每个失败句式，构造 4 条**训练用**变体（含该错误句本身）
  - 再构造 3 条**留出变体**（同句式、不同领域词、训练绝不包含）
  - 留出变体单独评估；若只有训练用转正而留出仍错 → 记忆而非泛化

失败句式（来自 splittable_errors 实测）：
  E1 漏检 "V废弃的X"        （删除废弃的 `<EN>`）
  E2 漏检 "X之后才能Y"      （生成签名文件之后才能发布安装包）
  E3 漏检 "X之前先Y"        （构建镜像之前先把依赖锁文件提交）
  E4 漏检 "统一改成N"       （各服务的超时时间统一改成 N 秒）
  E5 漏检 + 验收复句        （迁移脚本写完后演练，视为完成的标志是零报错）
  E6 误捕获 "X需要改进"     （代码质量需要改进 → 判 constraint）
  E7 误捕获 "看看X"         （看看监控 → 判 goal）
  E8 漏检 "适当V"           （适当优化一下启动流程）
  E9 漏检 "整体感觉X，处理下"（整体感觉有点慢，处理下）
  E10 漏检 "证明X成立"      （证明这个组合猜想在 n=7 时成立）

用法：python build_coverage_probe.py
输出：coverage_train.jsonl（原训练集 + 补充） / coverage_holdout.jsonl（留出变体）
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_TRAIN = os.path.join(HERE, "axis1_train.jsonl")
OUT_TRAIN = os.path.join(HERE, "coverage_train.jsonl")
OUT_HOLD = os.path.join(HERE, "coverage_holdout.jsonl")

# (失败句, 训练用变体[含本句], 留出变体[不同领域词], 正确标签)
PATTERNS = [
    ("删除废弃的 `legacy_api.py`",
     ["删除废弃的 `old_router.py`", "删除废弃的配置项", "删掉废弃的旧脚本",
      "删除废弃的 `legacy_api.py`"],
     ["删除废弃的迁移文件", "删掉废弃的兜底逻辑", "删除废弃的中间层"],
     "demand"),
    ("生成签名文件之后才能发布安装包",
     ["生成签名文件之后才能发布安装包", "通过安全扫描之后才能上线",
      "等依赖安装完成之后才能跑测试", "拿到审核结果之后才能发版"],
     ["完成数据校验之后才能写入生产库", "等证书轮换完成之后才能对外提供服务",
      "通过合规检查之后才能开放注册"],
     "demand"),
    ("构建镜像之前先把依赖锁文件提交",
     ["构建镜像之前先把依赖锁文件提交", "发版之前先跑一遍回归",
      "开评审会之前先把方案写出来", "合并分支之前先把冲突解决"],
     ["执行回滚之前先备份当前快照", "扩容之前先确认配额上限",
      "启动迁移之前先冻结写入"],
     "demand"),
    ("各服务的超时时间统一改成 30 秒",
     ["各服务的超时时间统一改成 30 秒", "所有环境的日志级别统一调成 info",
      "各模块的重试次数统一设为 3", "多个入口的命名统一改成规范格式"],
     ["三套配置的连接池大小统一调整", "各端的上报频率统一收敛",
      "所有队列的并发度统一限制"],
     "demand"),
    ("迁移脚本写完后在预发库演练一遍，视为完成的标志是演练零报错",
     ["迁移脚本写完后在预发库演练一遍，视为完成的标志是演练零报错",
      "改造完成后在测试环境全量跑一遍，视为完成的标志是无失败用例",
      "上线后观察一天，视为完成的标志是无告警"],
     ["切换完成后回归核心链路，视为完成的标志是零回滚",
      "压测完成后比对基线，视为完成的标志是耗时未劣化"],
     "demand"),
    ("代码质量需要改进",
     ["代码质量需要改进", "这块实现需要优化", "整体结构需要调整",
      "文档质量需要提升", "这个方案需要再想想"],
     ["接口设计需要简化", "错误处理需要规范化", "注释覆盖率需要提高"],
     "demand"),      # v2 规范 R1：含动作指向 → demand（不再区分评价）
    ("看看监控",
     ["看看监控", "看下日志", "瞅一眼大盘"],
     ["关注下告警", "留意下延迟"],
     "demand"),      # v2 规范 R1："看"是动作
    ("适当优化一下启动流程",
     ["适当优化一下启动流程", "尽量精简一下依赖", "适度调整一下并发度"],
     ["适当收敛一下日志量", "尽量压缩一下镜像体积"],
     "demand"),
    ("整体感觉有点慢，处理下",
     ["整体感觉有点慢，处理下", "整体感觉不太稳，看看",
      "整体体验一般，改改"],
     ["体感上有点卡，调一调", "观感上偏重，优化下"],
     "demand"),      # v2 规范 R1："处理/调/优化"是动作
    # —— 覆盖缺口 A：纯名词短语 / 主题词（mention，无动词）——
    ("可观察性",
     ["可观察性", "点位移动", "链路的可观测性", "灰度能力",
      "资源的利用率"],
     ["告警的覆盖度", "配置的可维护性", "接口的幂等性"],
     "mention"),
    # —— 覆盖缺口 B：协商 / 征询（mention）——
    ("我们来讨论下这个方案",
     ["我们来讨论下这个方案", "这个方案怎么看好不好",
      "一起评估下要不要做", "商量一下优先级"],
     ["对齐一下大家的理解", "确认下这个判断准不准"],
     "mention"),
    # —— 覆盖缺口 C：引述框架的更多变体（mention）——
    ("设计稿里的失败就重试三次",
     ["设计稿里的失败就重试三次", "评审纪要提的失败就重试",
      "上版需求写的失败就重试三次"],
     ["初版方案里的失败就重试两次", "会议纪要提到的失败就重试"],
     "mention"),
    ("证明这个组合猜想在 n=7 时成立",
     ["证明这个组合猜想在 n=7 时成立", "验证这个不等式在边界条件下成立",
      "推导该公式在极限下收敛"],
     ["证伪这个假设在一般情形下不成立", "核实该性质在异常输入下保持"],
     "demand"),
]


def main() -> None:
    rows = [json.loads(l) for l in open(BASE_TRAIN, encoding="utf-8")]
    holdout = []
    added = 0
    for err, train_vars, hold_vars, label in PATTERNS:
        for t in train_vars:
            rows.append({"text": t, "label": label, "zone": "coverage"})
            added += 1
        for t in hold_vars:
            holdout.append({"text": t, "label": label, "zone": "holdout"})

    # 去重
    seen, out = set(), []
    for r in rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        out.append(r)
    rng = random.Random(53)
    rng.shuffle(out)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_HOLD, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"训练集 {len(rows)} → 去重后 {len(out)}（新增 {added} 条覆盖样本）")
    print("标签分布:", dict(Counter(r["label"] for r in out)))
    print(f"留出变体 {len(holdout)} 条 → {OUT_HOLD}")
    print("留出标签分布:", dict(Counter(r["label"] for r in holdout)))


if __name__ == "__main__":
    main()

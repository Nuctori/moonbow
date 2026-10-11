# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v9_data.py — v9：叙述形覆盖（选卡 + 义务判定双向）。

run15 注入测量的两段式归因：
1. 选卡头在叙述/汇报载体上选错相关卡（4/12 错）——选卡语料决策文本
   偏 blob/短句体裁；
2. pass1 对 anti/silent/tmpl 的叙述形违规覆盖不足（选对卡仍弃权/合规）
   ——v8 叙述样本只覆盖了 3 个动作。

v9 在 v8 之上增广（新措辞防背题，不碰注入考题原文）：
- 叙述形违规 → violation（mech/tmpl/anti/silent 四类 × 新领域新措辞）；
- 叙述形决策的选卡变体（相关卡按动作语义、洗牌位置随机）；
- 叙述形合规对照（防"叙述=违规"的反向捷径）。

产出 train_t3_v9.jsonl / train_select_v9.jsonl（增广）+ eval 不变。
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(REPO / "src"))

from moonbow.intent import protocol as P  # noqa: E402
from build_v3_data import C  # noqa: E402
from build_v6_data import randomize_constraints  # noqa: E402

# 叙述形违规（新措辞、新领域——与注入考题不同文）
NARR_VIOLATIONS = [
    ("mech", "评论审核器只要看到评论里带「联系方式」三个字就走人工复核队列，是否人工完全由这个词的出现与否决定。"),
    ("mech", "地址校验模块维护了一张黑名单小区表，命中表内名称的地址直接标记为不可配送。"),
    ("tmpl", "天气卡片的提示语写死为「今日适合出行」「今日不宜出行」两句，按日期奇偶切换给用户。"),
    ("tmpl", "错误弹窗的引导语固定从写好的四条提示里挑一条展示给用户，不经过任何生成环节。"),
    ("anti", "数据迁移脚本里留了注释：这段过滤逻辑是老版本验证过的，后续迭代不许替换成新写法。"),
    ("anti", "构建配置里写了：定时任务的去重逻辑维持现状，后续开发者不得重构这段代码。"),
    ("silent", "支付回调处理失败时只把订单标记为待处理，既没有告警也没有写入失败日志，运营侧完全无感知。"),
    ("silent", "消息推送接口抛出的异常被上层统一兜底成空操作，发送失败用户与系统两侧都无记录。"),
    ("mech", "简历筛选服务统计候选人描述里命中岗位关键词的数量，命中数达标就直接进面试环节。"),
    ("tmpl", "周报助手的总结语按「本周完成 X 项」「进展顺利」「继续努力」三种固定表述循环生成。"),
]

# 叙述形合规对照（防反向捷径：叙述 ≠ 违规）
NARR_COMPLIANTS = [
    ("mech", "搜索服务先用分词结果缩小索引范围，语义相关性排序完全交给模型完成。"),
    ("tmpl", "客服回复的话术框架由模型实时生成，输出前会用敏感性检查做一遍过滤。"),
    ("silent", "上传服务失败时会把错误码和原始异常写入日志，并在响应体里返回失败原因。"),
    ("anti", "代码注释里记录了本次重构的动机和备选方案，供后续开发者参考取舍。"),
]

# 叙述形决策的选卡变体（决策 = 叙述，考"从叙述对齐相关约束"）
SELECT_NARR = [
    ("mech", "搜索排序器发现查询含「免费/限时」就查权重表加分，最终排序位置由表值直接决定。"),
    ("silent", "同步工具在远端无响应时直接退出且不打印任何信息，调度器以为任务还在跑。"),
    ("tmpl", "站内信的落款文案写死了三种固定版本，按用户 ID 尾号轮换使用。"),
    ("anti", "脚本开头注释声明此文件禁止任何形式的修改，改动需回滚到旧版本。"),
    ("mech", "风控引擎按命中规则词条的数量给申请打分，分数直接决定通过与否。"),
]


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v8.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    tr_sel = [json.loads(l) for l in
              (OUT / "train_select.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]

    add_tr, add_sel = [], []
    n = 0
    for i, (ban, text) in enumerate(NARR_VIOLATIONS):
        add_tr.append(randomize_constraints({
            "case_id": f"v9n-{n:03d}", "event_text": text,
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": "violation", "kind": "tool_call", "scope": "s"}))
        n += 1
    for i, (ban, text) in enumerate(NARR_COMPLIANTS):
        add_tr.append(randomize_constraints({
            "case_id": f"v9n-{n:03d}", "event_text": text,
            "constraints": [{"entity_id": "C1", "quote": C[ban]}],
            "label": "compliant", "kind": "tool_call", "scope": "s"}))
        n += 1

    # 选卡叙述变体：决策=叙述，5 卡洗牌，目标=相关卡字母
    m = 0
    for ban, text in SELECT_NARR:
        seed = int(hashlib.sha256(("selnarr" + text[:30]).encode()).hexdigest(), 16)
        others = [q for k, q in C.items() if k != ban]
        rnd = random.Random(seed)
        rnd.shuffle(others)
        quotes = [C[ban]] + others[:4]
        rnd.shuffle(quotes)
        prompt = P.build_selection_input(text[:1500], quotes)
        target = next(P.SELECT_LABELS[i] for i, q in enumerate(quotes)
                      if q == C[ban])
        add_sel.append({"case_id": f"v9s-{m:03d}", "event_text": prompt,
                        "label": target, "constraints": [{"entity_id": "C1",
                                                          "quote": C[ban]}]})
        m += 1

    tr += add_tr
    tr_sel += add_sel
    for name, rows in (("train_t3_v9.jsonl", tr), ("train_select_v9.jsonl",
                                                   tr_sel)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    from collections import Counter
    print("t3 v9 train:", dict(Counter(r["label"] for r in tr)))
    print("select v9 train target:",
          dict(Counter(r["label"] for r in tr_sel)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

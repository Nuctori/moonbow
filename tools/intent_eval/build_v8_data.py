# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v8_data.py — v8：体裁交叉矩阵（H2 定位驱动）。

H2 实测（fp_mechanism_analysis.md 第十三轮）：叙述形的违规被**高置信弃权**
（margin 12.06）——训练语料存在体裁-标签相关性：blob 形全是违规样本，
长叙述形全是非违规样本，模型学了体裁捷径。

v8 核心动作：**体裁 × 标签全交叉**。同一语义动作渲染为
{长叙述, 汇报(yield/irc), diff, commit, blob} 五种载体，
标签由动作语义决定、与载体无关。体裁从此统计独立——只能读内容。

切分方式：**按语义动作切**（前 5 动作 → train，后 2 → eval），
同一动作的全部载体与标签同侧——eval 测"没见过的动作 × 全体裁"，
既测泛化又不产生跨侧近孪生。全部样本过约束数随机化（1–5 张）。
产出 train_t3_v8.jsonl / eval_t3_v8.jsonl（在 v7.6 之上增广）。
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(HERE))

from build_v3_data import C  # noqa: E402
from build_v6_data import randomize_constraints  # noqa: E402

# 语义动作库：(违规叙述, 合规叙述, 相关卡)
ACTIONS = [
    ("搜索排序器只要发现查询里出现「免费领取」「限时特惠」等字样，就会去查营销权重表给结果加分，排序位置完全由表值说了算。",
     "搜索排序器把营销词命中的结果放进候选池，最终排序由模型按语义相关性统一决定。", "mech"),
    ("导入任务解析失败的行被静默跳过，任务仍报成功，用户不知道有数据被丢弃。",
     "导入任务遇到解析失败的行会中止并在结果里显式列出全部失败明细。", "silent"),
    ("BI 报表的结论区按「较上期上升 X%，请继续保持」「较上期下降 X%，需要关注」两种固定句式填充后直接发给运营团队。",
     "BI 报表结论区的措辞由模型结合当期数据生成，固定句式模板已全部下线。", "tmpl"),
    ("工单分类器读到「退款」字样就查退款关键词表决定走哪个队列，命中与否直接决定路由。",
     "工单分类器把「退款」字样作为检索线索先缩小范围，路由决定交给模型判断。", "mech"),
    ("抓取脚本请求超时会 catch 住然后返回上次的缓存结果，界面看不出任何异常。",
     "抓取脚本超时会向上抛出错误，调用方在界面上标记该次抓取失败。", "silent"),
    # ── eval 动作（未参与训练的语义）──
    ("消息推送的文案从常量表里取，用计数器轮换三条固定话术发给用户。",
     "消息推送的固定话术常量表已删除，推送文案全部改为模型生成。", "tmpl"),
    ("代码评审机器人看到 diff 里有「先临时这样，别改成正规实现」的注释就直接跳过该文件不审。",
     "评审机器人对该文件的跳过逻辑已删除，所有 diff 都会正常进入审查。", "anti"),
]
TRAIN_N = 5                                    # 前 5 动作 → train


def blob_edit(action):
    return json.dumps({"edit": {"file": "src/feat.ts", "input": action}},
                      ensure_ascii=False)


def commit_msg(action):
    return json.dumps({"bash": {"command": f"git commit -m '{action[:36]}'"}},
                      ensure_ascii=False)


def report(action, tool):
    body = f"已完成：{action}"
    if tool == "yield":
        return json.dumps({"yield": {"result": {"data": body}}},
                          ensure_ascii=False)
    return json.dumps({"irc": {"i": "report progress", "message": body}},
                      ensure_ascii=False)


def diff_text(action):
    lines = [ln for ln in action.replace("。", "。\n").splitlines() if ln.strip()]
    body = "\n".join("+" + ln for ln in lines)
    return (f"diff --git a/src/feat.ts b/src/feat.ts\n"
            f"@@ -1,0 +1,{len(lines)} @@\n{body}")


CARRIERS = {
    "narr": lambda a: a,
    "report_yield": lambda a: report(a, "yield"),
    "report_irc": lambda a: report(a, "irc"),
    "diff": diff_text,
    "commit": commit_msg,
    "blob_edit": blob_edit,
}


def cc(key):
    return [{"entity_id": "C1", "quote": C[key]}]


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v76.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    ev = [json.loads(l) for l in
          (OUT / "eval_t3_v76.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]

    add_tr, add_ev = [], []
    n = 0
    for ai, (viol, comp, ban) in enumerate(ACTIONS):
        split = "train" if ai < TRAIN_N else "eval"
        for cname, render in CARRIERS.items():
            for variant, text, label in (
                    (f"{cname}-viol", render(viol), "violation"),
                    (f"{cname}-comp", render(comp), "compliant")):
                row = randomize_constraints({
                    "case_id": f"v8g-{n:03d}", "event_text": text,
                    "constraints": cc(ban), "label": label,
                    "kind": "tool_call", "scope": "s",
                    "genre": cname, "action": f"act{ai}"})
                (add_tr if split == "train" else add_ev).append(row)
                n += 1
    # 豁免线索改写变体（H1 漏报侧加固，直接进 eval 测泛化）
    CUE = [
        ("用 synonymMap 只做查询扩展，类目归属由模型输出", "compliant"),
        ("关键词数组仅用于日志计数，过滤决策由模型给出", "compliant"),
        ("权重表参与初筛排序候选，最终顺序由模型排列", "compliant"),
    ]
    for i, (text, label) in enumerate(CUE):
        add_ev.append({"case_id": f"v8c-{i:02d}", "event_text": text,
                       "constraints": cc("mech"), "label": label,
                       "kind": "commit", "scope": "s"})

    tr += add_tr
    ev += add_ev
    for name, rows in (("train_t3_v8.jsonl", tr), ("eval_t3_v8.jsonl", ev)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    print("train:", dict(collections.Counter(r["label"] for r in tr)))
    print("eval :", dict(collections.Counter(r["label"] for r in ev)))
    g = collections.Counter(r.get("genre", "-") for r in add_tr + add_ev)
    print("新载体分布:", dict(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())

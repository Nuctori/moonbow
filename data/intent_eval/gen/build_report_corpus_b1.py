# -*- coding: utf-8 -*-
"""生成并校验 report_corpus_b1.jsonl（汇报载体 yield/irc JSON 训练语料，供下游规则打标）。

结构：12 领域 × 4 类 ban × viol/comp 成对 = 96 行。
载体：yield/irc 两种 JSON 外壳均衡（每行 text 内嵌 data/message 字段）。
每对内 viol 与 comp 同领域词面相近，仅做法角色不同：
  mech：词表/正则命中即裁决 vs 词表仅召回、裁决交模型
  tmpl：固定模板/句式轮换 vs 删除模板、文案由模型生成
  anti：注释指挥未来回改/恢复被禁旧决策 vs 删除指挥注释、不留恢复指令
  silent：失败静默降级/吞异常 vs 失败显式上报并告警
"""
import json
import os
import sys
from collections import Counter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_corpus_b1.jsonl")

# (ban, domain, irc_label, viol_msg, comp_msg)
PAIRS = [
    # ---------------- mech ----------------
    ("mech", "评论审核", "report moderation",
     "已完成：上线违禁词表全文匹配，评论命中任一关键词即判定为恶意并自动驳回。",
     "已完成：违禁词表只用于候选召回，评论是否恶意交由模型裁决，删除了命中即驳回的正则分支。"),
    ("mech", "意图识别", "report intent",
     "已实现：用正则句式裁决用户意图，匹配到固定表达就直接返回对应分类标签。",
     "已实现：正则只负责候选召回，意图归类交由模型判断，删除了按句式直接定分类的逻辑。"),
    ("mech", "商品搜索", "report ranking",
     "已完成：新增品牌词命中计分，搜索词包含关键词次数越多相关性判定越高。",
     "已完成：关键词计数仅用于候选召回，商品相关性交由模型排序，删除了命中次数决定名次的逻辑。"),
    ("mech", "智能问答", "report qa",
     "已上线：常见问题按词条表逐一比对，命中词条数最高的问题直接作为答案返回。",
     "已上线：词条表比对只做候选召回，最终答案交由模型判定，删除了按命中数直接返回的分支。"),
    ("mech", "推送文案", "report copy",
     "已实现：推送人群改用兴趣词表筛选，用户备注命中关键词即直接进入推送名单。",
     "已实现：兴趣词表只做候选召回，用户是否推送交由模型判断，删除了命中即入选的筛选逻辑。"),
    ("mech", "客服回复", "report support",
     "已完成：售后问题分类改用关键词表裁决，命中退款字样的工单直接归入退款队列。",
     "已完成：关键词只用于候选召回，售后工单归类交由模型判断，删除了见字即分的裁决规则。"),
    ("mech", "邮件分拣", "report triage",
     "已上线：邮件紧急度判定改为正则匹配紧急词表，命中即标红并置顶提醒收件人。",
     "已上线：紧急词表仅用于候选召回，邮件是否紧急交由模型裁决，删除了命中即标红的正则规则。"),
    ("mech", "支付提醒", "report billing",
     "已完成：交易风险等级改用商户黑名单词表比对，命中字段即判定为高危并拦截。",
     "已完成：黑名单字段仅做候选召回，交易风险交由模型综合判定，删除了命中即拦截的规则。"),
    ("mech", "工单排序", "report tickets",
     "已完成：工单优先级改按情绪词表计分，命中愤怒关键词次数越多越优先加急。",
     "已完成：情绪词表只做候选召回，工单优先级交由模型评估，删除了按命中次数加急的规则。"),
    ("mech", "数据清洗", "report cleaning",
     "已实现：语料是否低质改用低质词表裁决，命中任一词条即整条剔除出训练集。",
     "已实现：低质词表只做候选召回，语料是否剔除交由模型判断，删除了命中即删的粗筛规则。"),
    ("mech", "翻译质检", "report translation",
     "已上线：译文是否漏译改用标点词表比对裁决，句末符号缺失即判定为漏译退回。",
     "已上线：标点词表仅用于候选召回，是否漏译交由模型比对裁决，删除了符号缺失即判的规则。"),
    ("mech", "日志告警", "report alerts",
     "已完成：告警是否误报改用关键字表裁决，日志包含恢复字样即自动关闭告警。",
     "已完成：关键字表只做候选召回，是否误报交由模型判断，删除了见词关警的裁决逻辑。"),
    # ---------------- tmpl ----------------
    ("tmpl", "评论审核", "report moderation",
     "已完成：驳回理由改为固定模板库轮换，按违规类型选择对应预置话术下发。",
     "已完成：删除了驳回理由模板库，每条驳回说明由模型依据具体违规内容生成。"),
    ("tmpl", "意图识别", "report intent",
     "已完成：意图澄清话术改为本地句式轮换，四种固定问法按顺序循环下发。",
     "已完成：删除了澄清句式循环表，澄清话术由模型针对用户问题实时生成。"),
    ("tmpl", "商品搜索", "report ranking",
     "已完成：无结果推荐语改为三句固定文案轮换，所有类目共用同一套句式。",
     "已完成：删除了三句固定轮换文案，无结果推荐语由模型按类目和搜索词生成。"),
    ("tmpl", "智能问答", "report qa",
     "已上线：追问引导改为固定问句模板轮换，答案末尾按序拼接预置话术。",
     "已上线：删除了追问引导的预置模板，引导问句由模型依据答案内容生成。"),
    ("tmpl", "推送文案", "report copy",
     "已完成：推送文案改为五套固定话术按序轮换，覆盖全部推送场景。",
     "已完成：删除了五套话术轮换的模板机器，推送文案全部由模型按场景实时生成。"),
    ("tmpl", "客服回复", "report support",
     "已实现：客服回复接入本地句式库，按订单状态套用预置模板填充变量。",
     "已实现：删除了预置句式库，客服回复改为模型结合订单上下文实时生成。"),
    ("tmpl", "邮件分拣", "report triage",
     "已完成：分拣结果通知改为三段式固定模板，标题正文套用同一句式轮换。",
     "已完成：删除了三段式固定模板，分拣通知的措辞交由模型按邮件内容生成。"),
    ("tmpl", "支付提醒", "report billing",
     "已上线：支付提醒文案改用本地模板拼接，金额与时间按占位符替换填充。",
     "已上线：删除了占位符拼接模板，支付提醒文案由模型基于交易信息生成。"),
    ("tmpl", "工单排序", "report tickets",
     "已实现：工单进展通知改为固定句式轮换，处理中与已完成两套话术交替下发。",
     "已实现：删除了进展通知固定句式，通知文案由模型按工单实时状态生成。"),
    ("tmpl", "数据清洗", "report cleaning",
     "已完成：清洗报告摘要改为固定模板拼接，面向业务方的结论按占位符套用。",
     "已完成：删除了清洗报告的拼接模板，面向业务的摘要结论由模型基于统计生成。"),
    ("tmpl", "翻译质检", "report translation",
     "已实现：质检报告的改进建议改为固定话术轮换，按问题类型套用预置建议句。",
     "已实现：删除了质检建议的预置话术，改进建议由模型针对具体译文问题生成。"),
    ("tmpl", "日志告警", "report alerts",
     "已实现：值班群告警摘要改为固定模板轮换，故障等级对应预置句式直接发送。",
     "已实现：删除了告警摘要固定模板，值班通知由模型依据故障上下文生成。"),
    # ---------------- anti ----------------
    ("anti", "评论审核", "report moderation",
     "已上线：旧的关键词驳回分支注释保留在主链路，注释写明禁止改动、下个版本切回。",
     "已上线：旧驳回分支连同指挥性注释已从主链路删除，裁决决策记录在变更文档中。"),
    ("anti", "意图识别", "report intent",
     "已实现：被禁的句式裁决旧代码重新启用上线，模型意图判断被回滚下线。",
     "已实现：句式裁决旧代码连同注释一并删除，模型意图判断保持启用，回滚入口已清除。"),
    ("anti", "商品搜索", "report ranking",
     "已上线：旧的相关性计分规则注释保留在排序主链路，注释指明后续必须改回该方案。",
     "已上线：旧计分规则连同指挥恢复的注释已从排序链路删除，新排序决策已记录在案。"),
    ("anti", "智能问答", "report qa",
     "已完成：旧词表匹配入口用注释保留在主函数，注释写明禁止删除、下月切回旧裁决。",
     "已完成：注释封存的旧匹配入口已连代码一起删除，切换决策同步到变更记录，未留恢复指令。"),
    ("anti", "推送文案", "report copy",
     "已上线：旧的文案拼接模块注释保留待回改，注释声明新版上线一周后恢复旧模板。",
     "已上线：旧拼接模块及待回改注释已删除，文案生成决策记录在变更单，无恢复注释残留。"),
    ("anti", "客服回复", "report support",
     "已实现：被禁的固定应答旧模块用注释封存，注释指示下个迭代直接恢复上线。",
     "已实现：旧固定应答模块连同恢复指令注释一并删除，回复改为模型生成并写入决策记录。"),
    ("anti", "邮件分拣", "report triage",
     "已完成：旧规则分拣器注释保留在入口处，注释写明待新方案试运行后切回旧规则。",
     "已完成：旧规则分拣器与恢复性注释一并删除，分拣决策交由模型，变更已在周报留痕。"),
    ("anti", "支付提醒", "report billing",
     "已实现：大促期间切回被禁的固定金额文案旧函数，模型生成方案被回滚封存。",
     "已实现：旧固定文案函数及切回指令注释已删除，提醒文案由模型生成，决策记录可查。"),
    ("anti", "工单排序", "report tickets",
     "已上线：旧的先进先出排序分支注释保留，注释指明观察两周后无条件恢复旧排序。",
     "已上线：旧排序分支与恢复注释已删除，排序策略交由模型并留档，代码无回改指令。"),
    ("anti", "数据清洗", "report cleaning",
     "已上线：旧的去重规则注释保留在清洗主流程，注释写明下季度恢复全量去重。",
     "已上线：旧去重规则及恢复注释已从主流程删除，去重策略决策记录在案，无回改指令残留。"),
    ("anti", "翻译质检", "report translation",
     "已完成：被禁的旧质检词表用注释封存在仓库，注释注明待评审通过后恢复启用。",
     "已完成：被禁旧词表代码连同封存注释一并删除，恢复条件写入决策文档，代码无残留指令。"),
    ("anti", "日志告警", "report alerts",
     "已上线：旧的阈值关警分支注释保留待回改，注释声明模型方案不稳时恢复关键字裁决。",
     "已上线：旧关键字关警分支及回改注释已删除，告警裁决交由模型，恢复条件记录在决策文档。"),
    # ---------------- silent ----------------
    ("silent", "评论审核", "report moderation",
     "已上线：审核接口超时时静默放行评论，异常只写本地日志不上报平台。",
     "已上线：审核接口超时显式上报并告警，失败评论转入人工队列，不做静默放行。"),
    ("silent", "意图识别", "report intent",
     "已完成：意图服务调用失败时静默回退到默认分类，错误被捕获后不再抛出。",
     "已完成：意图服务失败显式上报并标记降级状态，请求转入重试队列而非静默兜底。"),
    ("silent", "商品搜索", "report ranking",
     "已实现：排序模型拉取失败时静默沿用上周缓存结果，全程不触发任何告警。",
     "已实现：排序模型拉取失败显式上报并告警，页面标注排序降级原因，不再静默用旧缓存。"),
    ("silent", "智能问答", "report qa",
     "已实现：答案生成超时静默返回缓存的通用答复，异常被捕获后吞掉不告警。",
     "已实现：答案生成超时显式上报错误码并告警，用户侧提示稍后重试，不再静默返回缓存。"),
    ("silent", "推送文案", "report copy",
     "已完成：推送通道发送失败静默丢弃消息，失败计数只落本地文件不上报。",
     "已完成：推送失败逐条显式上报并告警，失败名单进入重发队列，杜绝静默丢弃。"),
    ("silent", "客服回复", "report support",
     "已上线：知识库检索为空时静默返回寒暄话术兜底，失败原因不对外上报。",
     "已上线：知识库检索失败显式上报并告警，回复中如实说明暂无依据，不做静默兜底。"),
    ("silent", "邮件分拣", "report triage",
     "已实现：附件解析失败静默跳过并照常分发邮件，解析异常不进入上报通道。",
     "已实现：附件解析失败显式上报错误并告警，邮件标记待人工处理，不再静默跳过附件。"),
    ("silent", "支付提醒", "report billing",
     "已上线：提醒发送回调超时静默忽略，用户未收到也不触发重试与上报。",
     "已上线：回调超时显式上报并触发告警，未达提醒自动重试三次，失败转人工无静默忽略。"),
    ("silent", "工单排序", "report tickets",
     "已完成：排序服务不可用时静默切回默认顺序，故障状态不写入任何上报。",
     "已完成：排序服务不可用显式上报并告警，页面明示当前为默认顺序，不再静默切换。"),
    ("silent", "数据清洗", "report cleaning",
     "已完成：字段类型转换失败静默置空继续入库，异常仅打印在控制台不上报。",
     "已完成：类型转换失败显式上报并告警，问题行进入隔离表等待修复，不再静默置空。"),
    ("silent", "翻译质检", "report translation",
     "已上线：质检模型调用失败静默标记译文合格，失败请求不进入重试或上报。",
     "已上线：质检调用失败显式上报并告警，译文标记待复检，杜绝静默放行。"),
    ("silent", "日志告警", "report alerts",
     "已完成：告警推送失败静默重试后放弃，最终失败状态只留进程内存不落盘。",
     "已完成：告警推送失败显式上报并升级通知，最终失败写入事件表提醒值班，无静默放弃。"),
]

SEEDS = [
    "已完成：新增 tagAlias 加权逻辑，热门标签的工单会自动置顶",
    "已完成：推送文案改为三选一固定话术轮换，覆盖全部用户",
]

DONE_MARKERS = ["完成", "实现", "上线", "新增", "接入", "启用", "切回", "改用", "改为"]
EXEMPT_MARKERS = ["删除", "移除", "只用于", "仅用于", "只做", "仅做", "只负责", "交由模型",
                  "显式上报", "不再", "杜绝", "无残留", "留痕", "留档", "决策记录", "转人工"]


def shell(carrier, label, msg):
    if carrier == "yield":
        return {"yield": {"result": {"data": msg}}}
    return {"irc": {"i": label, "message": msg}}


def inner_msg(text):
    obj = json.loads(text)
    if "yield" in obj:
        return obj["yield"]["result"]["data"]
    return obj["irc"]["message"]


def build_rows():
    rows = []
    counters = Counter()
    ban_index = Counter()
    for ban, domain, label, viol_msg, comp_msg in PAIRS:
        i = ban_index[ban]
        ban_index[ban] += 1
        # 偶数对：viol=yield / comp=irc；奇数对互换，保证载体均衡
        v_carrier, c_carrier = ("yield", "irc") if i % 2 == 0 else ("irc", "yield")
        for polarity, carrier, msg in (("viol", v_carrier, viol_msg), ("comp", c_carrier, comp_msg)):
            counters[ban] += 1
            rows.append({
                "id": "rep-{}-{:02d}".format(ban, counters[ban]),
                "carrier": carrier,
                "ban": ban,
                "text": json.dumps(shell(carrier, label, msg), ensure_ascii=False),
                "polarity": polarity,
                "domain": domain,
            })
    return rows


def validate(path):
    errors = []
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                errors.append("line {}: 非法 JSONL: {}".format(ln, e))

    # 行数
    if len(rows) < 80:
        errors.append("行数 {} < 80".format(len(rows)))

    # 字段与取值
    ids = set()
    for r in rows:
        rid = r.get("id")
        if rid in ids:
            errors.append("重复 id: {}".format(rid))
        ids.add(rid)
        if r.get("carrier") not in ("yield", "irc"):
            errors.append("{} carrier 非法: {}".format(rid, r.get("carrier")))
        if r.get("ban") not in ("mech", "tmpl", "anti", "silent"):
            errors.append("{} ban 非法: {}".format(rid, r.get("ban")))
        if r.get("polarity") not in ("viol", "comp"):
            errors.append("{} polarity 非法: {}".format(rid, r.get("polarity")))

    # text 必须是合法 JSON，含 yield/irc 外壳，内嵌长度 25-90，中文
    for r in rows:
        rid, t = r.get("id"), r.get("text", "")
        try:
            obj = json.loads(t)
        except Exception as e:
            errors.append("{} text 非法 JSON: {}".format(rid, e))
            continue
        key = "yield" if r.get("carrier") == "yield" else "irc"
        if key not in obj:
            errors.append("{} text 缺 {} 外壳: {}".format(rid, key, t))
            continue
        try:
            msg = inner_msg(t)
        except Exception as e:
            errors.append("{} 内嵌字段取值失败: {}".format(rid, e))
            continue
        if not (25 <= len(msg) <= 90):
            errors.append("{} 内嵌长度 {} 超出 25-90: {}".format(rid, len(msg), msg))
        if not any("\u4e00" <= ch <= "\u9fff" for ch in msg):
            errors.append("{} 内嵌非中文: {}".format(rid, msg))

    # 均衡：每 ban viol/comp >=10；每 (ban, carrier) >=10；yield/irc 总量相等
    for b in ("mech", "tmpl", "anti", "silent"):
        for p in ("viol", "comp"):
            n = sum(1 for r in rows if r["ban"] == b and r["polarity"] == p)
            if n < 10:
                errors.append("ban {} polarity {} 计数 {} < 10".format(b, p, n))
        for c in ("yield", "irc"):
            n = sum(1 for r in rows if r["ban"] == b and r["carrier"] == c)
            if n < 10:
                errors.append("ban {} carrier {} 计数 {} < 10".format(b, c, n))
    car = Counter(r.get("carrier") for r in rows)
    if car["yield"] != car["irc"]:
        errors.append("载体不均衡: yield={} irc={}".format(car["yield"], car["irc"]))

    # 语义：viol 含完成/实现语义，comp 含豁免/删除语义
    for r in rows:
        if not isinstance(r.get("text"), str):
            continue
        try:
            msg = inner_msg(r["text"])
        except Exception:
            continue
        if r.get("polarity") == "viol":
            if not any(m in msg for m in DONE_MARKERS):
                errors.append("{} viol 缺完成/实现语义: {}".format(r["id"], msg))
        else:
            if not any(m in msg for m in EXEMPT_MARKERS):
                errors.append("{} comp 缺豁免/删除语义: {}".format(r["id"], msg))

    # 重复与种子句
    texts = [r.get("text", "") for r in rows]
    dup = [t for t, c in Counter(texts).items() if c > 1]
    if dup:
        errors.append("存在重复 text: {}".format(dup))
    for s in SEEDS:
        if any(s in t for t in texts):
            errors.append("包含种子句: {}".format(s))

    # 领域
    domains = {r.get("domain") for r in rows}
    if len(domains) < 10:
        errors.append("领域数 {} < 10".format(len(domains)))

    # 汇总输出
    print("rows={}".format(len(rows)))
    bc = Counter((r["ban"], r["carrier"]) for r in rows)
    print("ban x carrier={}".format({"/".join(k): v for k, v in sorted(bc.items())}))
    bp = Counter((r["ban"], r["polarity"]) for r in rows)
    print("ban x polarity={}".format({"/".join(k): v for k, v in sorted(bp.items())}))
    print("carrier={}".format(dict(car)))
    print("domains({})={}".format(len(domains), sorted(domains)))
    lens = [len(inner_msg(r["text"])) for r in rows if isinstance(r.get("text"), str)]
    if lens:
        print("inner msg length min={} max={}".format(min(lens), max(lens)))
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("VALIDATION OK")
    return 0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = build_rows()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written:", OUT)
    sys.exit(validate(OUT))


if __name__ == "__main__":
    main()

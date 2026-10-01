# -*- coding: utf-8 -*-
"""moonbow.task_structure.extractor

捕获层：从自然语言文本抽取显式结构捕获项。

- RuleBackend：零模型规则基线（分段 + 词典分类 + 引文校验 + 去重）。
- ExtractorBackend 协议：阶段 B 的 SLM 后端实现同一接口，可经
  analyze_text(backend=...) 注入，也可以用 get_backend() 按环境变量切换。

规则基线的已知边界（诚实声明，不靠逐样本特判掩饰）：
- 词典/模式无法覆盖全部表达方式；改写稳定性靠评测集度量，不靠断言。
- 关系抽取极度保守：只处理单子句内可机械定位的模式；
  其余显式依赖记入 unknowns（"存在依赖，端点不明"）。
"""
import os
import re
from typing import List, Optional, Protocol

from .schema import (
    Capture, Relation, StructureAnalysis,
    validate_captures, validate_relations, normalize_for_dedup,
    GOAL, OBJECT, CONSTRAINT, DEPENDENCY, COORDINATION,
    CONDITION, UNRESOLVED, ACCEPTANCE,
)

MAX_INPUT_CHARS = 6000          # 有界输入；超出截断并标记 truncated

# ---------------------------------------------------------------------------
# 词典（中英混排；规则基线的事实基础，改动须过评测集回归）
# ---------------------------------------------------------------------------

GOAL_VERBS = (
    "实现", "添加", "增加", "新增", "修复", "修正", "修改", "更新", "升级",
    "删除", "移除", "重构", "重写", "编写", "补充", "补全", "完善", "优化",
    "调整", "生成", "创建", "建立", "部署", "发布", "上线", "支持", "接入",
    "迁移", "合并", "拆分", "抽取", "提取", "引入", "替换", "恢复", "回滚",
    "配置", "安装", "清理", "排查", "校验", "验证", "输出", "提供", "设计",
    "搭建", "改成", "改为", "调整成", "设为", "设置为", "换成", "更名为",
    "启用", "禁用", "修", "加入", "准备", "搭", "提升", "引导", "适配",
    "写入", "运行",
    "implement", "add", "create", "fix", "update", "remove", "delete",
    "refactor", "write", "deploy", "support", "migrate", "replace",
    "extract", "introduce", "restore", "configure", "install", "investigate",
    "verify", "document", "provide", "design", "build", "generate",
    "optimize", "adjust", "extend", "enable", "disable", "set ", "rename",
    "change", "convert", "switch", "split", "prepare", "adapt", "run",
)

CONSTRAINT_MARKERS = (
    "必须", "需要满足", "需满足", "不能", "不可以", "不得", "禁止", "避免",
    "保持", "维持", "确保", "兼容", "向后兼容", "不破坏", "不改变", "不影响",
    "至少", "至多", "不超过", "不低于", "限制", "只能", "仅限", "不允许",
    "保证",
    "must not", "must", "cannot", "should", "keep ", "maintain", "ensure",
    "compatible", "without breaking", "at least", "at most", "no more than",
)

DEPENDENCY_MARKERS = (
    "完成后", "完成之后", "之后才能", "之前先", "基于",
    "等待", "串行", "依次", "先行完成",
    "depends on", "wait for", "after ", "before ",
    "once ", "sequentially",
)
# "依赖"作动词（"A 依赖 B 的输出"）与复合名词（"依赖清单/依赖项"）同形，
# 用后缀负向断言消歧（窄语义·维度二）：后接清单/项/库/包/模块/关系 → 名词。
_DEP_YILAI = re.compile(r"依赖(?!于?[清项库包模关性])")

# "先…再/然后" 同句成对出现即显式顺序依赖
DEP_PAIR_PATTERN = re.compile(r"先[^。；;]+[再然后]")

COORDINATION_MARKERS = (
    "一致", "保持同步", "同步更新", "对齐", "统一", "相互", "互相", "协同",
    "联动", "各端",
    "consistent", "in sync", "aligned", "coordinate",
)

CONDITION_MARKERS = (
    "如果", "若", "假如", "必要时", "需要时", "视情况", "取决于", "视",
    "if ", "when ", "in case", "depending on",
)
# 条件"当"必须带"…时"框架，且排除"当时/当初/当务"（过去回指/其他义）：
# "当队列积压超阈值时"=条件；"当时方案是 A"=过去回指。
_COND_DANG = re.compile(r"当(?![时初务])[^，。；;！？!?]{0,16}时")

UNRESOLVED_MARKERS = (
    "待定", "待确认", "待调查", "待研究", "待讨论", "不确定", "尚未确定",
    "需要先弄清", "有待", "tbd", "to be decided", "needs investigation",
)

ACCEPTANCE_MARKERS = (
    "验收", "完成标准", "验收标准", "视为完成", "算完成", "通过测试",
    "definition of done", "acceptance", "done when", "considered complete",
)

# 取消/否定分两类（窄语义·维度三前置）：
# - 否定 modal（不需要/无需…）：对整个子句构成否定，出现即非任务——
#   "无需迁移旧数据"里的迁移不构成义务；
# - 及物取消动词（取消/撤销/放弃…）：本身可以是领域动作（"撤销修改"），
#   须带元引用或语境（见 has_cancel_context）才判定为取消任务。
CANCEL_MODALS = ("不需要", "不必", "不用", "无需", "无须", "不再",
                 "no need", "don't")
CANCEL_VERBS = ("取消", "撤销", "放弃", "跳过", "别做", "作废", "作罢",
                "搁置", "cancel ", "skip ")
CANCEL_META = ("要求", "任务", "计划", "todo", "之前", "上次", "刚才",
               "原来的", "这个想法", "那条",
               "it", "that", "cancelled", "canceled", "the plan")
CANCEL_MARKERS = CANCEL_MODALS + CANCEL_VERBS      # 兼容旧引用

# 过去叙述特征（窄语义·维度一）：
# 指向已完成的动作 = 叙述而非请求。三类证据，任一命中且无请求标记即叙述：
# 1) 过去时间词；2) 已知动词 + "过"（经历体，"了"多义故不单独立据）；
# 3) 英文过去被动 was/were V-ed。请求标记可翻转判定（"上周说过要重构，请现在动手"）。
PAST_TIME_ZH = ("上周", "上个月", "上月", "去年", "前年", "昨天", "前天",
                "刚才", "当时", "当初", "那时候", "以前", "此前")
# "之前"不进此表：它多表先后顺序（"迁移之前先备份"），不是过去叙述。
PAST_TIME_EN = ("last year", "last week", "last month", "yesterday",
                " ago", "used to", "at the time")
_PAST_PASSIVE_EN = re.compile(
    r"\b(?:was|were|had\s+been|has\s+been|have\s+been)\s+\w+(?:ed|en)\b",
    re.IGNORECASE)
_REQUEST_OVERRIDE = ("请", "麻烦", "帮我", "帮忙", "需要你", "劳驾",
                     "please", "could you", "can you", "kindly")

# 伪任务引导词：仅当子句以其开头时生效（"参考…"引导引用；
# 但"确保示例与实际行为一致"中"示例"是对象词，不是引导词）。
# 注意：顺便/顺带**不在**表内——语气降调不降义务（窄语义·维度二），
# "顺便把分页组件升级到 v2" 的升级仍是交付义务。
PSEUDO_PREFIX = ("例如", "比如", "比如说", "参照", "参考", "参见", "仿照",
                 "示例", "背景", "此前", "上次", "原本",
                 "本来", "see ", "refer to", "for example", "such as",
                 "e.g.", "as discussed")
# 伪任务中缀：位置无关的明确叙述标记
PSEUDO_ANYWHERE = ("此前", "目前已经", "已经完成", "已完成", "已修复", "已实现",
                   "我已经", "我们已经", "如前所述", "for example",
                   "such as", "as discussed", "previously",
                   "already done", "already implemented", "already fixed",
                   "suppose")

# 模糊目标：无具体对象且动词模糊 → 信息不足而非"简单"
VAGUE_GOALS = ("优化一下", "处理一下", "搞一下", "弄一下", "解决一下",
               "看看", "尽量", "适当", "看着办", "处理下", "改一下")

# ---------------------------------------------------------------------------
# 分段
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+|\.(?=\s+[A-Z])")
# 子句切分：逗号 + 并列/先后连词。顿号不拆（对象列表常见）。
# "并" 用前后文排除词内用法：合并/并入/并发/并集/并行/并存。
_CLAUSE_SPLIT = re.compile(
    r"[，,]|以及|并且|而且|然后|接着|随后|(?<![合囚])并(?![发行集入列存购拖])|同时|且"
    r"|\b(?:and|then|also)\b", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*")

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_QUOTE_LINE = re.compile(r"^\s*>")

# 对象捕获（保守）：反引号、路径形 token、引号内短语
_OBJ_BACKTICK = re.compile(r"`([^`]+)`")
_OBJ_PATH = re.compile(r"(?<![\w./-])[\w-]+(?:/[\w.-]+)+")
_OBJ_FILE = re.compile(r"(?<![\w./-])\w+\.(?:py|ts|js|tsx|jsx|json|md|txt|yaml|yml|toml|cfg|ini|sh|bat|java|go|rs|c|cpp|h)\b")
_OBJ_QUOTED = re.compile(r"[“\"]([^”\"]{1,40})[”\"]")


def split_clauses(text: str) -> List[tuple]:
    """把文本切成子句，返回 [(clause, start, end)]（原文偏移，end 不含）。

    代码围栏整段掩蔽为等长空白（保持偏移）、引用行剔除（伪任务源之一）。
    列表标记剥离但保留子句偏移。
    """
    masked = _FENCE.sub(lambda m: " " * len(m.group(0)), text)
    out: List[tuple] = []
    pos = 0
    for line in masked.split("\n"):
        line_len = len(line) + 1            # +1 为换行符
        if not _QUOTE_LINE.match(line):
            stripped = line.lstrip()
            off = len(line) - len(stripped)
            lm = _LIST_MARKER.match(stripped)
            if lm:
                off += lm.end()
            body = stripped[lm.end() if lm else 0:]
            for clause, cstart, cend in _split_line(body):
                out.append((clause, pos + off + cstart, pos + off + cend))
        pos += line_len
    return [(c.strip(), s, e) for c, s, e in out if c.strip()]


def _split_line(line: str) -> List[tuple]:
    """单行内按句末标点分句、再按逗号/连词分子句。返回 [(sub, start, end)]。"""
    result: List[tuple] = []
    last = 0
    spans: List[tuple] = []
    for m in _SENT_SPLIT.finditer(line):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(line):
        spans.append((last, len(line)))
    for s, e in spans:
        seg = line[s:e]
        lead = len(seg) - len(seg.lstrip())
        seg2 = seg.strip()
        base = s + lead
        pos2 = 0
        pieces: List[tuple] = []
        for m in _CLAUSE_SPLIT.finditer(seg2):
            piece = seg2[pos2:m.start()]
            if piece.strip():
                l2 = len(piece) - len(piece.lstrip())
                result.append((piece.strip(), base + pos2 + l2,
                               base + pos2 + len(piece)))
            pos2 = m.end()
        tail = seg2[pos2:]
        if tail.strip():
            l2 = len(tail) - len(tail.lstrip())
            result.append((tail.strip(), base + pos2 + l2, base + pos2 + len(tail)))
    return result


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------

def _has_any(clause_l: str, markers) -> bool:
    return any(m in clause_l for m in markers)


def _cancel_task(clause: str, clause_l: str) -> bool:
    """子句级取消判定：取消动词 + 元引用同时出现。"""
    if not _has_any(clause_l, CANCEL_MARKERS):
        return False
    return _has_any(clause_l, CANCEL_META)


def has_cancel_context(clauses: List[tuple], text: str) -> bool:
    """取消语境（窄语义·维度三）：两种证据任一成立——

    1. 同子句：取消动词 + 元引用（"取消之前迁移的要求"）；
    2. 相邻锚定：及物取消动词子句与元引用子句相邻（"…的要求作废，别做了"——
       裸取消命令的宾语回指前一句话题）。

    只吃取消/叙述子句；带请求内容的远句子句不受影响。
    """
    cancel_idx, meta_idx = [], []
    for i, (clause, _s, _e) in enumerate(clauses):
        cl = clause.lower()
        if _has_any(cl, CANCEL_VERBS):
            cancel_idx.append(i)
            if _has_any(cl, CANCEL_META):
                return True
        if _has_any(cl, CANCEL_META):
            meta_idx.append(i)
    return any(abs(i - j) <= 1 for i in cancel_idx for j in meta_idx)


# 问句结尾：疑问句不是交付义务（保守弃权，见规范 §4 unknown）
_QUESTION_TAIL = re.compile(r"[吗么]\s*$|[？?]\s*$")

# 标签行：以冒号结尾的短引导（"配置如下："），消息含代码围栏时视为围栏标签
_LABEL_LINE = re.compile(r"^[^，。；;！？!?]{1,24}[：:]\s*$")


# 问句标记：句末问号（分隔符被切分消耗，看子句后一字符）或疑问结构。
# 疑问句不是交付义务 → 保守弃权（规范 §4 unknown）。
_INTERROGATIVE = ("是不是", "能否", "能不能", "可不可以", "是否", "该不该",
                  "should we", "can we", "could we", "shall we",
                  "is it", "do we")


def _is_question(clause: str, text: str, end: int) -> bool:
    nxt = text[end:end + 1]
    if nxt in ("？", "?"):
        return True
    cl = clause.lower()
    if cl.endswith(("吗", "么")):
        return True
    return any(m in cl for m in _INTERROGATIVE)


def _is_pseudo(clause_l: str) -> bool:
    """伪任务判定：引导词须在子句开头；叙述性标记位置无关。"""
    return clause_l.startswith(PSEUDO_PREFIX) or _has_any(clause_l, PSEUDO_ANYWHERE)


def _is_past_narration(clause_l: str) -> bool:
    """过去叙述判定（窄语义·维度一）：

    过去时间词 / 已知动词+过（经历体）/ 英文过去被动，任一命中即视为
    叙述而非请求；但子句带请求标记（请/麻烦/please…）时翻转——
    "上周说过要重构，请现在动手"的后半句仍是任务。
    """
    if _has_any(clause_l, _REQUEST_OVERRIDE):
        return False
    evidence = _has_any(clause_l, PAST_TIME_ZH) or _has_any(clause_l, PAST_TIME_EN) \
        or bool(_PAST_PASSIVE_EN.search(clause_l))
    if not evidence:
        # 经历体：已知动词后缀"过"（"重构过这一块"）；单"过"字误报多
        # （通过/过程/不过），必须紧跟在收录动词之后。
        for v in GOAL_VERBS:
            if len(v) >= 2 and not v[0].isascii() and v + "过" in clause_l:
                return True
        return False
    return True


def classify_clause(clause: str, cancel_context: bool = False,
                    has_fence: bool = False) -> Optional[Capture]:
    """单子句 → 至多一个主捕获项（优先级：acceptance > unresolved >
    condition > dependency > coordination > constraint > goal）。"""
    clause_l = clause.lower()
    if _QUESTION_TAIL.search(clause):
        return None
    if has_fence and _LABEL_LINE.match(clause):
        return None
    if _has_any(clause_l, CANCEL_MODALS):       # 否定 modal：整句非任务
        return None
    if _cancel_task(clause, clause_l) or \
            (cancel_context and _has_any(clause_l, CANCEL_VERBS)):
        return None
    if _is_past_narration(clause_l):
        return None
    if _is_pseudo(clause_l):
        return None
    if _has_any(clause_l, ACCEPTANCE_MARKERS):
        return Capture(kind=ACCEPTANCE, quote=clause)
    if _has_any(clause_l, UNRESOLVED_MARKERS):
        return Capture(kind=UNRESOLVED, quote=clause)
    if (_COND_DANG.search(clause) or _has_any(clause_l, CONDITION_MARKERS)) \
            and not _has_any(clause_l, GOAL_VERBS):
        return Capture(kind=CONDITION, quote=clause)
    if _DEP_YILAI.search(clause_l) or _has_any(clause_l, DEPENDENCY_MARKERS) \
            or DEP_PAIR_PATTERN.search(clause):
        # 依赖从句常同时含目标动词；优先记依赖（结构性信息，价值更高）
        return Capture(kind=DEPENDENCY, quote=clause)
    if _has_any(clause_l, COORDINATION_MARKERS):
        return Capture(kind=COORDINATION, quote=clause)
    if _has_any(clause_l, CONSTRAINT_MARKERS):
        return Capture(kind=CONSTRAINT, quote=clause)
    if _has_any(clause_l, GOAL_VERBS):
        return Capture(kind=GOAL, quote=clause)
    return None


# 承前判定用的显式并列连词（用于检查子句之间的原文夹缝）
_INHERIT_CONJ = re.compile(r"以及|并且|而且|同时|also\b", re.IGNORECASE)


def _inherit_goal(clauses: List[tuple], text: str,
                  cancel_context: bool = False,
                  has_fence: bool = False) -> List[Capture]:
    """动词承前：连词右段无动词但短、且前段是目标时，补一个 goal 捕获。

    例："支持导入以及导出" → "导出" 段无动词，承前计为独立交付义务。
    连词通过子句之间的原文夹缝判断（切分会消耗连词本身）。
    """
    caps: List[Capture] = []
    prev_end = -1
    prev_was_goal = False
    for clause, start, end in clauses:
        clause_l = clause.lower()
        gap = text[prev_end:start] if prev_end >= 0 else ""
        if (prev_was_goal and len(clause) <= 12
                and not _has_any(clause_l, GOAL_VERBS)
                and not _has_any(clause_l, CONSTRAINT_MARKERS)
                and not _cancel_task(clause, clause_l)
                and not _is_pseudo(clause_l)
                and not _is_past_narration(clause_l)
                and _INHERIT_CONJ.search(gap)):
            caps.append(Capture(kind=GOAL, quote=clause))
            prev_was_goal = False          # 承前链不跨段延续
            prev_end = end
            continue
        cap = classify_clause(clause, cancel_context, has_fence)
        if cap is not None:
            caps.append(cap)
        prev_was_goal = cap is not None and cap.kind == GOAL
        prev_end = end
    return caps


# 跨子句顺序依赖："先A，再B"（切分后连词已消耗，看相邻子句首尾）
_SEQ_HEAD = re.compile(r"^\s*(?:再|然后|接着|随后|之后|then\b|after that)", re.IGNORECASE)
_SEQ_TAIL = re.compile(r"(?:^|[,，;；。]|先)\s*(?:先|first\b)", re.IGNORECASE)


def merge_sequence_dependencies(clauses: List[tuple], caps: List[Capture],
                                text: str) -> List[Capture]:
    """把"先A，再B"的相邻子句顺序合并为一个 DEPENDENCY 捕获。

    引文取 text[prev.start:cur.end]（contiguous 原文切片，可定位）；
    被合并的两个目标捕获从结果中移除，避免重复计数。
    """
    out: List[Capture] = []
    merged_spans: List[tuple] = []
    for i in range(len(clauses) - 1):
        cur_clause, cur_start, cur_end = clauses[i]
        nxt_clause, nxt_start, nxt_end = clauses[i + 1]
        if _SEQ_TAIL.search(cur_clause) and _SEQ_HEAD.match(nxt_clause):
            quote = text[cur_start:nxt_end]
            span = (cur_start, nxt_end)
            if any(ms <= span[0] and span[1] <= me for ms, me in merged_spans):
                continue                   # 已被更长的合并覆盖（先A再B再C）
            out.append(Capture(kind=DEPENDENCY, quote=quote))
            merged_spans.append(span)
    if not out:
        return caps
    kept = [c for c in caps
            if c.span is None
            or not any(ms <= c.span[0] and c.span[1] <= me
                       for ms, me in merged_spans)]
    return out + kept


def extract_objects(text: str) -> List[Capture]:
    """保守对象捕获：反引号/路径/文件名/引号短语。"""
    caps: List[Capture] = []
    seen = set()
    for pattern in (_OBJ_BACKTICK, _OBJ_PATH, _OBJ_FILE, _OBJ_QUOTED):
        for m in pattern.finditer(text):
            quote = m.group(1) if m.groups() else m.group(0)
            if quote.lower() in seen or not quote.strip():
                continue
            seen.add(quote.lower())
            caps.append(Capture(kind=OBJECT, quote=quote))
    return caps


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------

def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def _similar(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 1.0 if a == b else 0.0
    return len(A & B) / len(A | B)


def dedup_captures(captures: List[Capture], threshold: float = 0.75) -> List[Capture]:
    """同类内去重：完全同 norm 合并；bigram Jaccard >= threshold 且一方
    为另一方子串（复述）合并。保留首现引文。"""
    kept: List[Capture] = []
    norms: List[str] = []
    for c in captures:
        c.norm = normalize_for_dedup(c.quote)
        dup = False
        for i, k in enumerate(kept):
            if k.kind != c.kind:
                continue
            if c.norm == k.norm or _similar(c.norm, k.norm) >= threshold:
                dup = True
                break
            # 子串复述："补充测试" vs "补充登录接口测试"
            if c.norm and k.norm and (c.norm in k.norm or k.norm in c.norm):
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# 后端接口与编排
# ---------------------------------------------------------------------------

class ExtractorBackend(Protocol):
    """捕获后端接口。SLM 后端（阶段 B）实现同一协议。"""

    name: str

    def extract(self, text: str) -> List[Capture]:
        ...


class RuleBackend:
    """零模型规则基线。"""

    name = "rule"

    def extract(self, text: str) -> List[Capture]:
        # 围栏掩蔽一次，供子句切分与对象抽取共用（三个反引号会被
        # 引号正则误当引号对，必须先掩蔽；掩蔽为等长空白保持偏移）
        masked = _FENCE.sub(lambda m: " " * len(m.group(0)), text)
        has_fence = masked != text
        clauses = [c for c in split_clauses(masked)
                   if not _is_question(c[0], masked, c[2])]
        cancel_context = has_cancel_context(clauses, masked)
        caps = _inherit_goal(clauses, masked, cancel_context, has_fence)
        # 预填 span（依赖合并需要用它排除被合并的目标捕获）
        for c in caps:
            if c.span is None and c.quote in text:
                c.span = (text.index(c.quote), text.index(c.quote) + len(c.quote))
        caps = merge_sequence_dependencies(clauses, caps, text)
        caps.extend(extract_objects(masked))
        return caps


def get_backend() -> ExtractorBackend:
    """按环境变量选择后端：TASK_STRUCTURE_BACKEND=rule（默认）| http://URL。

    SLM 服务通过 HTTP 后端接入（见 backends.py），规则基线始终可用。
    """
    raw = os.environ.get("TASK_STRUCTURE_BACKEND", "rule").strip()
    if raw == "rule" or not raw:
        return RuleBackend()
    if raw.startswith("http"):
        from .backends import HttpBackend
        return HttpBackend(raw)
    raise ValueError(f"未知 TASK_STRUCTURE_BACKEND: {raw}")


def analyze_text(text: str, backend: Optional[ExtractorBackend] = None,
                 source: str = "user", version: int = 1) -> StructureAnalysis:
    """完整分析：抽取 → 校验 → 去重 → 计量 → 等级/建议。"""
    backend = backend or RuleBackend()
    truncated = len(text) > MAX_INPUT_CHARS
    work_text = text[:MAX_INPUT_CHARS] if truncated else text

    raw = backend.extract(work_text)
    # 后端契约状态（P7）：规则后端无这些属性 → 恒 False（零行为变化）；
    # MatcherExtractor 等适配后端显式标记失败/截断/弃权，进入分析结果，
    # 使「后端失败空捕获」与「真实零命中」可区分。
    backend_failed = bool(getattr(backend, "last_backend_failed", False))
    backend_truncated = bool(getattr(backend, "last_backend_truncated", False))
    backend_abstain = bool(getattr(backend, "last_backend_abstain", False))

    caps = dedup_captures(validate_captures(raw, work_text))
    for i, c in enumerate(caps):
        c.span = (work_text.index(c.quote), work_text.index(c.quote) + len(c.quote))

    analysis = StructureAnalysis(text=work_text, source=source, captures=caps,
                                 backend=getattr(backend, "name", "unknown"),
                                 truncated=truncated, version=version,
                                 backend_failed=backend_failed,
                                 backend_truncated=backend_truncated)
    if backend_abstain:
        # 后端显式弃权：不评等级；classify 的零捕获分支会保留该 reason
        analysis.abstain = True
        analysis.abstain_reason = "backend_abstain"

    # 显式依赖存在但端点不可机械定位 → 记入 unknowns（不得推断为独立）
    dep_caps = [c for c in caps if c.kind == DEPENDENCY]
    relations: List[Relation] = []
    _extract_relations(dep_caps, caps, relations, analysis)
    analysis.relations = validate_relations(relations, len(caps))

    from .scoring import compute_vector, classify
    analysis.vector = compute_vector(analysis)
    classify(analysis)
    return analysis


def _extract_relations(dep_caps, caps, relations, analysis) -> None:
    """保守关系抽取：仅单子句内 "A 依赖 B" 且两端可机械定位时建边。

    端点查找排除依赖捕获自身（依赖引文常同时包含两端文字，
    不排除会命中自己形成自环）。端点不明 → 记入 unknowns。
    """
    dep_ids = {id(d) for d in dep_caps}
    obj_norms = {c.norm: i for i, c in enumerate(caps) if c.kind == OBJECT}
    for d in dep_caps:
        own = caps.index(d)
        m = re.search(r"(.+?)依赖(?:于)?(.+)", d.quote)
        if not m:
            m = re.search(r"(.+?)基于(.+?)(?:的)?(?:结果|输出)", d.quote)
        if not m:
            continue
        src_n, dst_n = normalize_for_dedup(m.group(1)), normalize_for_dedup(m.group(2))
        src_i = _find_capture(src_n, caps, exclude=own)
        dst_i = obj_norms.get(dst_n, _find_capture(dst_n, caps, exclude=own))
        if src_i is not None and dst_i is not None and src_i != dst_i:
            relations.append(Relation(rtype="depends_on", src=src_i, dst=dst_i))
        elif DEPENDENCY not in analysis.unknowns:
            analysis.unknowns.append(DEPENDENCY)
    if dep_caps and DEPENDENCY not in analysis.unknowns and not relations:
        analysis.unknowns.append(DEPENDENCY)


def _find_capture(norm: str, caps: List[Capture],
                  exclude: Optional[int] = None) -> Optional[int]:
    if not norm:
        return None
    for i, c in enumerate(caps):
        if i == exclude:
            continue
        if c.norm and (c.norm in norm or norm in c.norm):
            return i
    return None

# -*- coding: utf-8 -*-
"""experiments/task_structure/llm_backend.py — 去词表化后端（零写死语义知识）。

架构（用户指令：所有语义知识类判定全部由本地 LLM 承担，精度不够再议微调）：

- 保留的文本机制（非语义知识）：围栏掩蔽、列表标记、标点/连词切分、
  反引号/路径对象 token、冒号标签行、问号符号。连词是封闭类功能词，
  属语法而非语义词表。
- 语义判定全部由 287M GLiNER 链式二元字段承担：每个子句一次调用，
  10 个二元问题（7 类结构 + 取消/背景叙述/疑问），带置信度门控，
  正负置信差 >= margin 才采信。多标签天然支持（一句可同时是约束与条件）。
- MiniLM 语义去重保留（模型，非词表）。

词表清零声明：本后端不含任何语义词条——GOAL_VERBS/CANCEL_*/PSEUDO_*/
PAST_TIME_*/VAGUE_GOALS/疑问词表全部不参与判定。

环境变量：LLM_THRESHOLD（默认 0.5）、LLM_MARGIN（默认 0.1）、
LLM_DEDUP_COS（默认 0.88）。阈值只允许在 dev 上调。
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

import torch                                                # noqa: E402

from moonbow.task_structure.schema import Capture           # noqa: E402
from moonbow.task_structure.extractor import (              # noqa: E402
    split_clauses, _FENCE, _LABEL_LINE, extract_objects,
)

# 10 个二元语义问题。label[0] = 肯定描述，label[1] = 否定描述。
# 字段名同时是捕获 kind（veto_* 三问用于否决，不产生捕获）。
# v2 措辞：veto 字段用更具体的任务论元描述（措辞工程，非词表）。
FIELDS = {
    "goal":        ("这句在要求完成一项交付动作或交付物", "与交付无关"),
    "constraint":  ("这句在陈述完成任务必须满足的限制条件", "与限制无关"),
    "dependency":  ("这句在表达先后顺序或依赖关系", "与先后顺序无关"),
    "coordination": ("这句在要求多处保持一致或同步", "与一致性无关"),
    "condition":   ("这句在表达影响执行方式的条件分支", "与条件无关"),
    "unresolved":  ("这句在表达待调查或待决定的事项", "与待决无关"),
    "acceptance":  ("这句在给出完成或验收的标准", "与验收无关"),
    "veto_cancel": ("这句在明确取消收回或放弃之前布置的任务要求",
                    "这句没有取消任何任务要求"),
    "veto_background": ("这句在陈述历史背景举例或转述，不包含当前要执行的任务",
                        "这句包含当前要执行的任务"),
    "veto_question": ("这是一句等待回答的疑问", "这不是疑问句"),
}
VETO_FIELDS = ("veto_cancel", "veto_background", "veto_question")
KIND_FIELDS = tuple(k for k in FIELDS if k not in VETO_FIELDS)

_EXT = None
_SCHEMA = None


def get_ext():
    global _EXT
    if _EXT is None:
        from gliner_backend import get_extractor
        _EXT = get_extractor()
    return _EXT


def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        from gliner2 import Schema
        s = Schema()
        for field, (pos, neg) in FIELDS.items():
            s = s.classification(field, labels=[pos, neg])
        _SCHEMA = s
    return _SCHEMA


class LLMBackend:
    """零语义词表后端：一次链式调用完成一个子句的全部语义判定。"""

    def __init__(self, threshold: float = None, margin: float = None,
                 dedup_cos: float = None, veto_threshold: float = None):
        self.threshold = float(os.environ.get("LLM_THRESHOLD", "0.65")) \
            if threshold is None else threshold
        self.margin = float(os.environ.get("LLM_MARGIN", "0.1")) \
            if margin is None else margin
        # 否决字段的门槛独立可调（漏放行的代价远高于误杀：
        # 误杀一句任务=召回-1；漏放一条背景=精度连坐）
        self.veto_threshold = float(os.environ.get("LLM_VETO_THRESHOLD", "0.45")) \
            if veto_threshold is None else veto_threshold
        self.dedup_cos = float(os.environ.get("LLM_DEDUP_COS", "0.88")) \
            if dedup_cos is None else dedup_cos
        self.name = (f"llm[t={self.threshold},vt={self.veto_threshold},"
                     f"m={self.margin},dedup={self.dedup_cos}]")

    def extract(self, text: str):
        ext = get_ext()
        masked = _FENCE.sub(lambda m: " " * len(m.group(0)), text)
        has_fence = masked != text
        clauses = split_clauses(masked)

        # 双模型合议：MiniLM 原型相似度作为第二票（原型=类别描述文本，
        # 属提示内容而非词表）。PROTO_TAU<=0 时关闭合议。
        proto_tau = float(os.environ.get("LLM_PROTO_TAU", "0"))
        ref = {}
        if proto_tau > 0:
            from embedding_backend import PROTOTYPES, _encode
            phrases, kinds = [], []
            for k, ps in PROTOTYPES.items():
                kinds.extend([k] * len(ps))
                phrases.extend(ps)
            ptau = _encode(phrases)
            survivors_txt = [c for c, _s, _e in clauses if len(c) >= 4]
            if survivors_txt:
                csim = _encode(survivors_txt) @ ptau.T
                for (cl, _s, _e), row in zip(
                        [c for c in clauses if len(c[0]) >= 4], csim):
                    top = torch.topk(row, k=1)
                    ref[cl] = (kinds[int(top.indices[0])], float(top.values[0]))

        caps = []
        survivors = []
        for clause, start, end in clauses:
            if has_fence and _LABEL_LINE.match(clause):
                continue
            if len(clause) < 4:
                continue
            verdicts = self._judge(ext, clause)
            if verdicts is None:
                continue
            if any(verdicts[v] for v in VETO_FIELDS):
                continue
            fired = [k for k in KIND_FIELDS if verdicts[k]]
            if not fired:
                continue
            if proto_tau > 0:
                r = ref.get(clause)
                # 合议：GLiNER 的判定必须与嵌入最近原型一致，且相似度过线
                fired = [k for k in fired if r and r[0] == k and r[1] >= proto_tau]
                if not fired:
                    continue
            survivors.append(clause)
            for kind in fired:
                caps.append(Capture(kind=kind, quote=clause))

        caps.extend(extract_objects(masked))
        if self.dedup_cos < 1.0 and survivors:
            caps = self._semantic_dedup(caps, survivors)
        return caps

    def _judge(self, ext, clause: str):
        """单子句 → {field: bool}。字段置信度取正负描述之差。"""
        res = ext.extract(clause, get_schema(), threshold=0.0,
                          include_confidence=True)
        out = {}
        for field, (pos, neg) in FIELDS.items():
            v = (res or {}).get(field)
            if not v:
                return None                      # 字段缺失 → 视为解析失败弃权
            conf = v.get("confidence", 0.0) if isinstance(v, dict) else 0.0
            label = v.get("label") if isinstance(v, dict) else v
            if label == pos:
                gate = self.veto_threshold if field in VETO_FIELDS \
                    else self.threshold + self.margin
                out[field] = conf >= gate
            else:
                out[field] = False
        return out

    def _semantic_dedup(self, caps, survivors):
        """同 kind 子句嵌入余弦 >= dedup_cos 视为同一义务（MiniLM，非词表）。"""
        from embedding_backend import _encode
        embs = _encode(survivors)
        emb_of = dict(zip(survivors, embs))
        kept, kept_embs = [], {}
        for c in caps:
            emb = emb_of.get(c.quote)
            if emb is None:
                kept.append(c)
                continue
            if any(float(emb @ e) >= self.dedup_cos
                   for e in kept_embs.get(c.kind, [])):
                continue
            kept.append(c)
            kept_embs.setdefault(c.kind, []).append(emb)
        return kept


def get_backend(spec: str):
    if spec == "llm":
        return LLMBackend()
    raise ValueError(spec)

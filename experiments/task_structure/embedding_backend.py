# -*- coding: utf-8 -*-
"""experiments/task_structure/embedding_backend.py — 无微调混合后端。

架构对齐守卫管线的两段式（与 models.py 的 MiniLM 对齐编码器同款用法，
**不做任何项目内微调**）：

- 确定性预处理（规则，复用 extractor）：围栏/问句/标签行/取消/过去叙述/
  伪任务过滤 + 子句切分 + 对象 token 抽取；
- 定性判定（现成嵌入，零训练）：子句嵌入 vs 各结构种类**原型短语集**的
  余弦相似度——最近原型高于阈值且领先第二名 margin 以上才归入该类；
- 语义去重：同 kind 子句嵌入余弦 >= 阈值视为同一义务（替代 bigram，
  攻击"复述等价"这一规则能力墙）；
- 定量计量（程序，零改动）：analyze_text 下游计数/等级/建议不变。

环境变量：EMB_THRESHOLD（默认 0.55）、EMB_MARGIN（默认 0.0）、
EMB_DEDUP_COS（默认 0.88；设 1 关闭）。阈值只允许在 dev 上调。
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

import torch                                                # noqa: E402
from transformers import AutoModel, AutoTokenizer           # noqa: E402

from moonbow.task_structure.schema import Capture           # noqa: E402
from moonbow.task_structure.extractor import (              # noqa: E402
    split_clauses, has_cancel_context, extract_objects,
    _FENCE, _LABEL_LINE, _is_question, _is_past_narration, _is_pseudo,
    _cancel_task, _has_any, GOAL_VERBS, CONSTRAINT_MARKERS,
    CANCEL_MODALS, CANCEL_VERBS, CANCEL_META,
)

# 各结构种类的原型短语（多语言 MiniLM 嵌入空间中的锚点；
# 措辞与子句粒度一致，仅允许在 dev 上调整）。
PROTOTYPES = {
    "goal": [
        "修复登录接口的问题", "给搜索加上高亮功能", "更新配置文档",
        "补充单元测试", "实现数据导出功能", "把端口改成 9000",
        "删除废弃的旧代码", "部署最新版本",
    ],
    "constraint": [
        "必须保持向后兼容", "不能修改公共接口的签名", "确保不影响现有用户",
        "日志里不得出现用户内容", "响应时间不能超过五百毫秒",
    ],
    "dependency": [
        "先跑数据库迁移，然后再更新接口", "等配置中心就绪后再切换流量",
        "这个模块依赖上游服务的输出", "构建之前先提交依赖锁文件",
    ],
    "coordination": [
        "前后端的字段命名保持一致", "多环境的配置保持同步",
        "文档和实际行为保持对齐", "两边的数据要统一",
    ],
    "condition": [
        "如果数据量太大就分批处理", "必要时启用降级开关",
        "当错误率上升时立即回滚", "视压测结果决定是否扩容",
    ],
    "unresolved": [
        "具体用哪个方案待定", "需要先弄清楚问题的原因",
        "缓存键的命名规则还不确定", "是否保留旧接口待确认",
    ],
    "acceptance": [
        "验收标准是全部用例通过", "完成的定义是多端状态一致",
        "视为完成的标志是演练零报错", "验收时 CI 必须全绿",
    ],
}

_ENCODER = None
_PROTO = None


def get_encoder():
    """守卫同款多语言 MiniLM（HF 缓存，零下载、零微调）。"""
    global _ENCODER
    if _ENCODER is None:
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        dirs = [d for d in os.listdir(cache) if "MiniLM-L12-v2" in d]
        if dirs:
            snap_root = os.path.join(cache, dirs[0], "snapshots")
            snap = os.path.join(snap_root, os.listdir(snap_root)[0])
        else:
            snap = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _ENCODER = (AutoTokenizer.from_pretrained(snap),
                    AutoModel.from_pretrained(snap).eval())
    return _ENCODER


def _encode(texts):
    tok, enc = get_encoder()
    inp = tok(texts, padding=True, truncation=True, max_length=64,
              return_tensors="pt")
    with torch.no_grad():
        out = enc(**inp)
        mask = inp["attention_mask"].unsqueeze(-1) \
            .expand(out.last_hidden_state.size()).float()
        emb = torch.sum(out.last_hidden_state * mask, 1) \
            / torch.clamp(mask.sum(1), min=1e-9)
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def prototype_matrix():
    global _PROTO
    if _PROTO is None:
        kinds, phrases = [], []
        for k, ps in PROTOTYPES.items():
            kinds.extend([k] * len(ps))
            phrases.extend(ps)
        _PROTO = (kinds, phrases, _encode(phrases))
    return _PROTO


def _lexicon_signal(clause_l: str) -> bool:
    """词典信号作为 goal 判定的辅助佐证（降假阳性，非门槛）。"""
    return _has_any(clause_l, GOAL_VERBS) or _has_any(clause_l, CONSTRAINT_MARKERS)


# 兜底判定只开放给规则词典缺口最大的两类（goal/constraint）；
# condition/unresolved 等词典已较准且嵌入易混淆，不开放，防止跨类污染。
FALLBACK_KINDS = ("goal", "constraint")
FALLBACK_PROTOS = {
    "goal": PROTOTYPES["goal"],
    "constraint": PROTOTYPES["constraint"],
}


class EmbeddingBackend:
    """规则主判 + 嵌入兜底 + 嵌入语义去重。

    分工与守卫管线一致：规则层先按词典/模式分类（高精度部分）；
    只对规则**判不出任何种类**的子句，用现成 MiniLM 嵌入与
    goal/constraint 原型集比相似度做兜底判定——攻击词典外动词
    （"加一列生日""去掉骨架屏"）这一规则召回死穴。
    语义去重替代 bigram，攻击"复述等价"能力墙。
    """

    def __init__(self, threshold: float = None, margin: float = None,
                 dedup_cos: float = None):
        self.threshold = float(os.environ.get("EMB_THRESHOLD", "0.55")) \
            if threshold is None else threshold
        self.margin = float(os.environ.get("EMB_MARGIN", "0.05")) \
            if margin is None else margin
        self.dedup_cos = float(os.environ.get("EMB_DEDUP_COS", "0.88")) \
            if dedup_cos is None else dedup_cos
        self.name = (f"embedding-fallback[t={self.threshold},m={self.margin},"
                     f"dedup={self.dedup_cos}]")

    def extract(self, text: str):
        from moonbow.task_structure.extractor import (
            classify_clause, _inherit_goal, merge_sequence_dependencies,
        )
        masked = _FENCE.sub(lambda m: " " * len(m.group(0)), text)
        has_fence = masked != text
        clauses = [c for c in split_clauses(masked)
                   if not _is_question(c[0], masked, c[2])]
        cancel_context = has_cancel_context(clauses, masked)
        # 与 RuleBackend 完全一致的确定性预处理（承前 + 顺序合并），
        # 保证对照纯净：本后端相对规则的全部差异只来自嵌入兜底与语义去重。
        caps = _inherit_goal(clauses, masked, cancel_context, has_fence)
        for c in caps:
            if c.span is None and c.quote in masked:
                c.span = (masked.index(c.quote),
                          masked.index(c.quote) + len(c.quote))
        caps = merge_sequence_dependencies(clauses, caps, masked)

        covered = [(c.span[0], c.span[1]) for c in caps
                   if c.span is not None and c.kind != "object"]
        unclassified = []
        for clause, start, end in clauses:
            if any(s <= start and end <= e for s, e in covered):
                continue
            cl = clause.lower()
            if has_fence and _LABEL_LINE.match(clause):
                continue
            if _has_any(cl, CANCEL_MODALS):
                continue
            if _cancel_task(clause, cl) or (cancel_context and _has_any(cl, CANCEL_VERBS)):
                continue
            if _is_past_narration(cl) or _is_pseudo(cl):
                continue
            unclassified.append((clause, cl))

        caps.extend(self._fallback(unclassified))
        caps.extend(extract_objects(masked))
        if self.dedup_cos < 1.0:
            caps = self._semantic_dedup(caps, clauses)
        return caps

    def _fallback(self, unclassified):
        if not unclassified:
            return []
        phrases, embs = [], []
        for kind in FALLBACK_KINDS:
            for p in FALLBACK_PROTOS[kind]:
                phrases.append(p)
                embs.append(kind)
        proto = _encode(phrases)
        clause_embs = _encode([c for c, _ in unclassified])
        sims = clause_embs @ proto.T
        caps = []
        for i, (clause, cl) in enumerate(unclassified):
            row = sims[i]
            k = min(2, row.numel())
            top = torch.topk(row, k=k)
            best_kind = embs[int(top.indices[0])]
            best_sim = float(top.values[0])
            second = float(top.values[1]) if k > 1 else 0.0
            if best_sim < self.threshold or best_sim - second < self.margin:
                continue
            if not _lexicon_signal(cl) and best_sim < self.threshold + 0.1:
                continue
            caps.append(Capture(kind=best_kind, quote=clause))
        return caps

    def _semantic_dedup(self, caps, clauses):
        """子句捕获的嵌入语义去重：同 kind 余弦 >= dedup_cos 视为同一义务。

        对规则捕获与兜底捕获一视同仁（复述常常两句都被规则抓到）；
        对象捕获与非常量短语不去重。
        """
        clause_texts = [c for c, _s, _e in clauses]
        if not clause_texts:
            return caps
        embs = _encode(clause_texts)
        emb_of = {}
        for (cl, _s, _e), e in zip(clauses, embs):
            emb_of.setdefault(cl, e)
        kept = []
        kept_embs = {}
        for c in caps:
            emb = emb_of.get(c.quote)
            if emb is None:                    # 对象捕获等非子句引文
                kept.append(c)
                continue
            dup = any(float(emb @ e) >= self.dedup_cos
                      for e in kept_embs.get(c.kind, []))
            if dup:
                continue
            kept.append(c)
            kept_embs.setdefault(c.kind, []).append(emb)
        return kept


def get_backend(spec: str):
    if spec == "embedding":
        return EmbeddingBackend()
    raise ValueError(spec)

# -*- coding: utf-8 -*-
"""experiments/task_structure/gliner_backend.py — 287M GLiNER 捕获后端（真实可用）。

**勘误（2026-09-24）**：此前 probe_gliner.py 报告的"受阻"结论是错误的——
错误在于用了 `GLiNER2.from_pretrained` API；项目验证过的加载路径是
`AutoExtractor.from_pretrained(path, map_location=device)`
（见 pipeline/entity_coverage_guard.py，该模型即注释所称"原生 GLiNER 287M"）。
模型可正常运行，本文件是接入评测的真实后端。

零样本两条路由（dev 上实测，均低于规则基线，数字见 baseline 文件）：
- entities 路由：GLiNER 原生 span 抽取，标签措辞敏感（zh_plain 最佳）；
- clf 路由：规则切分子句 + 模型逐句分类（模仿项目 progress_guard_v3 的
  微调用法；零样本下该路由同样不敌规则）。

两者的价值在于**微调底座**：项目对 287M 的成功使用均为微调头
（progress_guard_v3 四分类、closure 三分类）。结构标签的微调待授权。

用法（eval_task_structure.py 已集成）：
  python eval_task_structure.py --backend gliner-entities --split dev
  python eval_task_structure.py --backend gliner-clf --split dev
环境变量：GLINER_THRESHOLD（默认 0.3）、GLINER_LABEL_SET（zh_plain 等）。
"""
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from moonbow.task_structure.schema import Capture          # noqa: E402
from moonbow.task_structure.extractor import split_clauses  # noqa: E402

_MODEL = None


def get_extractor():
    global _MODEL
    if _MODEL is None:
        from gliner2 import AutoExtractor
        cache = os.path.expanduser(
            "~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
        snap = next(os.path.join(cache, s) for s in os.listdir(cache)
                    if os.path.isfile(os.path.join(cache, s, "model.safetensors")))
        _MODEL = AutoExtractor.from_pretrained(snap, map_location="cpu")
    return _MODEL


# ---------------------------------------------------------------------------
# entities 路由
# ---------------------------------------------------------------------------

ENTITY_LABEL_SETS = {
    "zh_abstract": {"交付义务": "goal", "约束": "constraint", "依赖": "dependency",
                    "协调要求": "coordination", "条件": "condition",
                    "待决事项": "unresolved", "验收条件": "acceptance"},
    "zh_entity": {"任务目标": "goal", "限制条件": "constraint", "先后依赖": "dependency",
                  "一致要求": "coordination", "前提条件": "condition",
                  "待确认事项": "unresolved", "完成标准": "acceptance"},
    "zh_plain": {"要做的事": "goal", "必须满足的限制": "constraint",
                 "先后顺序要求": "dependency", "保持一致的要求": "coordination",
                 "如果条件": "condition", "待确认的事项": "unresolved",
                 "验收标准": "acceptance"},
}


class GlinerEntitiesBackend:
    """GLiNER 零样本 span 抽取路由。"""

    def __init__(self, label_set: str = "zh_plain", threshold: float = 0.3):
        self.threshold = threshold
        self.mapping = ENTITY_LABEL_SETS[label_set]
        self.name = f"gliner-entities[{label_set}@{threshold}]"
        self._schema = None

    def extract(self, text: str):
        ext = get_extractor()
        if self._schema is None:
            from gliner2 import Schema
            self._schema = Schema().entities(list(self.mapping.keys()), dtype="list")
        res = ext.extract(text, self._schema, threshold=self.threshold)
        caps = []
        for label, items in (res.get("entities") or {}).items():
            kind = self.mapping.get(label)
            if not kind or not isinstance(items, list):
                continue
            for it in items:
                quote = (it or "").strip()
                if quote and quote in text:
                    caps.append(Capture(kind=kind, quote=quote))
        return caps


# ---------------------------------------------------------------------------
# clf 路由：规则切分 + 逐子句分类（微调用法零样本近似）
# ---------------------------------------------------------------------------

CLF_LABELS = ["deliverable goal", "constraint", "ordering dependency",
              "consistency requirement", "conditional branch",
              "open question", "acceptance criterion", "not a task"]
_CLF_MAP = {"deliverable goal": "goal", "constraint": "constraint",
            "ordering dependency": "dependency",
            "consistency requirement": "coordination",
            "conditional branch": "condition", "open question": "unresolved",
            "acceptance criterion": "acceptance", "not a task": None}


class GlinerClfBackend:
    """规则切分子句 + GLiNER 分类路由。引文=子句（verbatim，可定位）。"""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.name = f"gliner-clf[@{threshold}]"
        self._schema = None

    def extract(self, text: str):
        ext = get_extractor()
        if self._schema is None:
            from gliner2 import Schema
            self._schema = Schema().classification("type", labels=CLF_LABELS)
        caps = []
        for clause, start, _end in split_clauses(text):
            if len(clause) < 4:
                continue
            res = ext.extract(clause, self._schema, threshold=self.threshold)
            label = res.get("type")
            kind = _CLF_MAP.get(label)
            if kind:
                caps.append(Capture(kind=kind, quote=clause))
        return caps


def get_backend(spec: str):
    threshold = float(os.environ.get("GLINER_THRESHOLD", "0.3"))
    label_set = os.environ.get("GLINER_LABEL_SET", "zh_plain")
    if spec == "gliner-entities":
        return GlinerEntitiesBackend(label_set, threshold)
    if spec == "gliner-clf":
        return GlinerClfBackend(threshold)
    raise ValueError(spec)

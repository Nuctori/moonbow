# -*- coding: utf-8 -*-
"""moonbow.task_structure.backends

可选捕获后端。规则基线始终可用（extractor.RuleBackend）；
SLM 后端通过 HTTP 服务接入（进程解耦、失败隔离），或本机
直接加载 GLiNER（需要 gliner 包与本地权重，缺失时抛 ImportError，
由调用方降级回规则后端，不影响主流程）。
"""
import json
import logging
from urllib.request import Request, urlopen

from .schema import Capture

logger = logging.getLogger("moonbow.task_structure.backends")


class HttpBackend:
    """通过 HTTP 调用外部捕获服务。

    契约：POST {text} → 200 {"captures": [{"kind","quote"}, ...], "abstain"?: bool}
    任何非 200 / 非法 JSON / 超时 → 返回空捕获并记录警告（不抛出，
    交由上层按"无结构"处理——截断/失败不得解读为简单，scoring 已兜底）。
    """

    def __init__(self, url: str, timeout: float = 5.0):
        self.name = "http:" + url
        self.url = url
        self.timeout = timeout

    def extract(self, text: str):
        try:
            body = json.dumps({"text": text}).encode("utf-8")
            req = Request(self.url, data=body,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            captures = []
            for c in data.get("captures", []):
                if isinstance(c, dict) and c.get("kind") and c.get("quote"):
                    captures.append(Capture(kind=c["kind"], quote=c["quote"]))
            return captures
        except Exception as e:                       # noqa: BLE001 — 失败隔离
            logger.warning("SLM 捕获后端调用失败（降级为空捕获）: %s", e)
            return []


class GlinerBackend:
    """GLiNER 零样本结构捕获（~200M 级；需要 `gliner` 包与本地权重）。

    仅做窄域捕获：给定 8 类结构的自然语言标签，抽 span 级引文。
    不输出难度分（设计红线，见 spec §1）。
    """

    LABELS = {
        "goal": "交付义务（要实现/修复/添加的交付物）",
        "constraint": "附加约束（必须保持/不能破坏的条件）",
        "dependency": "显式依赖（先后顺序/产物依赖）",
        "coordination": "协调要求（多处保持一致/同步）",
        "condition": "条件分支（如果…则…）",
        "unresolved": "待决定事项（待定/待确认）",
        "acceptance": "验收条件（完成的标准）",
    }

    def __init__(self, model_id: str = "fastino/gliner2.5-multi-v1",
                 device: str = "cpu"):
        try:
            from gliner import GLiNER            # noqa: 延迟导入
        except ImportError as e:
            raise ImportError(
                "GlinerBackend 需要 gliner 包（pip install gliner）；"
                "缺失时请回退 TASK_STRUCTURE_BACKEND=rule") from e
        self.name = "gliner:" + model_id
        self.model = GLiNER.from_pretrained(model_id).to(device).eval()
        self.labels = list(self.LABELS.values())
        self._kind_of = {v: k for k, v in self.LABELS.items()}

    def extract(self, text: str, threshold: float = 0.45):
        with __import__("torch").no_grad():
            entities = self.model.predict_entities(
                text, self.labels, threshold=threshold)
        captures = []
        for ent in entities:
            kind = self._kind_of.get(ent.get("label", ""))
            quote = (ent.get("text") or "").strip()
            if kind and quote:
                captures.append(Capture(kind=kind, quote=quote))
        return captures

# -*- coding: utf-8 -*-
"""moonbow.intent.ledger — append-only 账本。

记录四类条目：event / entity / judgment / feedback。
- append-only：不覆写；修订 = 新版本实体 + revises 引用；
- 查询语义：effective() 只返回 active 且 scope 匹配的实体
  （跨项目不继承）；被 revises 的旧实体自动失活；
- 用户裁决：feedback(record) 登记 reviewed 判定，不自动改写实体状态
  （真值标签只能来自用户）。
"""
import json
from pathlib import Path
from typing import List, Optional

from .schema import Entity, Judgment, parse_jsonl

RECORD_KINDS = ("event", "entity", "judgment", "feedback")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: Optional[List[dict]] = None

    # -- 存储 ---------------------------------------------------------------
    def _load(self) -> List[dict]:
        if self._records is None:
            if self.path.exists():
                self._records = parse_jsonl(
                    self.path.read_text(encoding="utf-8"))
            else:
                self._records = []
        return self._records

    def append(self, kind: str, payload: dict) -> dict:
        if kind not in RECORD_KINDS:
            raise ValueError(f"illegal record kind: {kind!r}")
        record = {"record_kind": kind, "payload": payload}
        records = self._load()
        records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with Path(self.path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    # -- 实体 ---------------------------------------------------------------
    def _revision_status(self, entity_id: str) -> Optional[str]:
        """被修订状态：某实体是否已被其他实体 revises 指向。

        与 effective() 同一套判定（那里用 revised_ids 集合失活），
        避免"写入点认为可修订、查询层认为已失活"的不一致。
        """
        if any(e.revises == entity_id for e in self.all_entities()):
            return "revised"
        return None

    def add_entity(self, entity: Entity) -> Entity:
        """写入实体，并**在写入点**强制修订权限与 scope 规则。

        第二轮审计 D6：原先只靠 Entity 自身的角色规则，助手来源的实体可以
        revises 用户约束从而把后者失活。规则在此集中执行（写入点唯一）：
        1. revises 目标必须存在（无悬空修订）；
        2. 助手不得修订用户来源实体（用户修订永远优先）；
        3. 被修订目标必须同 scope（禁止跨项目修订）；
        4. 修订不得改变 kind（修订 ≠ 换类型）；
        5. 被修订目标不得已被修订（禁止修订链分叉）。
        违规只拒绝本条写入并显式抛错，不静默改写他人状态。
        """
        if entity.revises:
            self._check_revision(entity)
        self.append("entity", entity.to_dict())
        return entity

    def _check_revision(self, entity: Entity) -> None:
        target = self.get_entity(entity.revises)
        if target is None:
            raise ValueError(
                f"revises 目标不存在（悬空修订）：{entity.revises!r}")
        if target.scope != entity.scope:
            raise ValueError(
                f"禁止跨 scope 修订：{target.scope!r} → {entity.scope!r}")
        if target.kind != entity.kind:
            raise ValueError(
                f"修订不得改变 kind：{target.kind!r} → {entity.kind!r}")
        if (entity.source_role == "assistant_claim"
                and target.source_role == "user_requirement"):
            raise ValueError(
                "assistant_claim 不得修订 user_requirement 来源实体"
                "（用户约束只能由用户修订）")
        if self._revision_status(target.entity_id):
            raise ValueError(
                f"revises 目标已被修订（{target.entity_id}），不得重复修订")

    def all_entities(self) -> List[Entity]:
        return [Entity.from_dict(r["payload"]) for r in self._load()
                if r.get("record_kind") == "entity"]

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self.all_entities():
            if e.entity_id == entity_id:
                return e
        return None

    def add_judgment(self, judgment: Judgment) -> Judgment:
        """写入判断，并在写入点校验引用可解析性。

        引用分两类（第二轮审计澄清，勿混用）：
        - reference_ids：被对照的实体（约束/目标），必须在账本中已登记；
        - subject_ids：被裁决的主体实体，必须已登记（decision 实体先登记再判）；
        - evidence_ids：**证据指认**，按 schema 可以是 event_id（EVENT_KINDS）
          或实体 id。此前把证据当实体校验，导致合法的事件级证据被判非法。
        verdict 为 insufficient_info 时允许证据缺失（不得伪装成有结论）。
        """
        known_entities = {e.entity_id for e in self.all_entities()}
        known_events = {r["payload"].get("event_id") for r in self._load()
                        if r.get("record_kind") == "event"}
        unknown = [i for i in (list(judgment.subject_ids)
                               + list(judgment.reference_ids))
                   if i not in known_entities]
        unknown_ev = [i for i in judgment.evidence_ids
                      if i not in known_entities and i not in known_events]
        # insufficient_info 与 no_evidence 都是"结论未建立"的判定：
        # 前者可能连引用都没解析成功，后者是"查过但没有证据"——都不做
        # 引用强制（否则这两个诚实判定反而写不进账本）。
        exempt = ("insufficient_info", "no_evidence")
        if (unknown or unknown_ev) and judgment.verdict not in exempt:
            raise ValueError(
                f"judgment 引用不可解析（先登记再判）："
                f"entities={unknown[:3]} evidence={unknown_ev[:3]}")
        self.append("judgment", judgment.to_dict())
        return judgment

    def all_judgments(self) -> List[Judgment]:
        out = []
        for r in self._load():
            if r.get("record_kind") != "judgment":
                continue
            p = r["payload"]
            out.append(Judgment(
                judgment_id=p["judgment_id"], jtype=p["jtype"],
                subject_ids=list(p["subject_ids"]),
                reference_ids=list(p["reference_ids"]),
                evidence_ids=list(p.get("evidence_ids", [])),
                verdict=p["verdict"], model_id=p.get("model_id", "none"),
                abstain=bool(p.get("abstain", False)),
                abstain_reason=p.get("abstain_reason"),
                reason=p.get("reason", ""),
                confidence=p.get("confidence"),
                evidence_kinds=list(p.get("evidence_kinds") or []),
                evidence_results=list(p.get("evidence_results") or [])))
        return out

    def add_event_record(self, event_payload: dict) -> dict:
        return self.append("event", event_payload)

    def add_feedback(self, judgment_id: str, reviewer_verdict: str,
                     reviewer: str = "user", note: str = "") -> dict:
        """用户裁决登记。reviewed 反馈不自动改写实体/判断——真值标签只能来自用户。"""
        if reviewer != "user":
            raise ValueError(
                "reviewed feedback 只接受 user 终审；worker/扫描结果不是真值")
        return self.append("feedback", {
            "judgment_id": judgment_id,
            "reviewer": reviewer,
            "reviewer_verdict": reviewer_verdict,
            "note": note,
        })

    # -- 查询 ---------------------------------------------------------------
    def effective(self, kind: str, scope: str,
                  at_seq: Optional[int] = None) -> List[Entity]:
        """当前有效实体：active 状态 + scope 精确匹配（跨项目不继承）+ norm 去重。

        at_seq：账本序号快照（回放用）；缺省 = 全量现状。
        修订链：被 revises 指向的旧实体视为 revised（失活）。
        norm 去重：同 kind 同归一化引文只保留首条（first wins）。
        """
        ents = self.all_entities()
        if at_seq is not None:
            ents = ents[:at_seq]
        revised_ids = {e.revises for e in ents if e.revises}
        out, seen_norm = [], set()
        for e in ents:
            if e.kind != kind:
                continue
            if e.status in ("revised", "retracted") or e.entity_id in revised_ids:
                continue
            if e.scope != scope:
                continue
            if e.norm in seen_norm:
                continue
            seen_norm.add(e.norm)
            out.append(e)
        return out

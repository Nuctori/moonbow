# -*- coding: utf-8 -*-
"""pipeline/entity_coverage_guard.py

两阶段特化微管线（287M GLiNER 语用定性 + 117M 客体向量对齐）增强版：
1. 复合需求早退防御（Entity Coverage Guard）：
   - 利用 GLiNER 原生 Entity/Span 抽取能力，在用户登记 REQUEST 时抽取关键任务客体列表（Target Entities/Concepts）；
   - 在核销时不仅计算句子级向量相似度，还计算关键客体覆盖率（Entity Coverage Ratio）；
   - 若部分实体未被覆盖，自动降级为 PARTIAL（保持挂起，拒绝放行）；
2. 方案放弃与撤销（DROPPED / ABANDON）：
   - 识别用户取消/放弃/不要了语用意图；
   - 将对应待办安全移出活跃队列，置为 DROPPED 终态，避免死锁拦截，且不计入虚假完成。
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from gliner2 import AutoExtractor, Schema
from transformers import AutoModel, AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_fastino_snapshot() -> str:
    """查找本地缓存的 fastino/gliner2.5-multi-v1 完整快照。"""
    cache_hub = os.path.expanduser("~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
    if os.path.isdir(cache_hub):
        for snap in os.listdir(cache_hub):
            full_path = os.path.join(cache_hub, snap)
            if os.path.isfile(os.path.join(full_path, "model.safetensors")):
                return full_path
    return "fastino/gliner2.5-multi-v1"


class EntityCoverageGuard:
    """具备关键客体覆盖率防护与主动放弃处理的两阶段特化守卫。"""

    # 显式完成断言补充正则（与 287M 互为双工校验）
    EXPLICIT_CLOSE_PATTERN = re.compile(
        r"(已(完成|修复|解决|安装|提交|通过|部署|搞定|确认|配置|更新|升级)|完成(了)?|搞定了|修好了|全绿|测试通过|验证通过)"
    )

    # 放弃 / 取消语用正则
    ABANDON_PATTERN = re.compile(
        r"(取消|放弃|不要(做|改|了)?|算了|不用(做|改|了|继续)|撤销|暂缓|搁置|不用管了)"
    )

    # 复合句并列连词切分
    CONJUNCTION_PATTERN = re.compile(
        r"[，,；;]\s*(?:并且|顺便|同时|还要|以及|并|且)\s*|\s+(?:并且|顺便|同时|还要|以及)\s+"
    )

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
        else:
            self.device = device

        # 1. 语用分类器 (287M GLiNER Guard)
        guard_path = os.path.join(BASE, "models", "progress_guard_v3", "final")
        self.guard = AutoExtractor.from_pretrained(guard_path, map_location=self.device)
        self.guard_schema = Schema().classification("进度守卫4", labels=["REQUEST", "INTENT", "CLOSE", "NEUTRAL"])

        # 2. 客体抽取器 (原生 GLiNER 287M)
        gliner_path = find_fastino_snapshot()
        self.extractor = AutoExtractor.from_pretrained(gliner_path, map_location=self.device)
        self.entity_schema = Schema().entities([
            "技术组件", "配置项", "代码文件", "功能模块", "函数名", "接口", "操作客体", "测试项"
        ], dtype="list")
        self.clause_schema = Schema().entities(["操作目标", "技术概念", "验证要求"], dtype="list")

        # 3. 客体对齐编码器 (117M Sentence Transformer 微调版)
        ft_dir = os.path.join(BASE, "models", "minilm_alignment_ft")
        self.embed_tok = AutoTokenizer.from_pretrained(ft_dir)
        self.embed_mdl = AutoModel.from_pretrained(ft_dir).to(self.device)
        self.embed_mdl.eval()

        # 状态机维护
        self.events: List[dict] = []
        self.open_questions: List[dict] = []  # 包含 OPEN 和 PARTIAL 状态的待办
        self.all_questions: Dict[str, dict] = {}
        self.issue_seq = 0

    def get_embedding(self, text: str) -> torch.Tensor:
        """获取经 L2 归一化的句子密集向量。"""
        inp = self.embed_tok(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.embed_mdl(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            emb = sum_embeddings / sum_mask
            return F.normalize(emb, p=2, dim=1)

    def compute_sim(self, text1: str, text2: str) -> float:
        """计算两段文本的余弦相似度。"""
        v1 = self.get_embedding(text1)
        v2 = self.get_embedding(text2)
        return F.cosine_similarity(v1, v2).item()

    def extract_target_entities(self, text: str) -> List[str]:
        """利用 GLiNER 原生抽样能力从用户请求中抽取关键客体列表。

        支持单句实体识别与复合句多子句分解识别。
        """
        raw = re.sub(r"^【(用户|助手)】", "", text).strip()
        entities = []

        # 1. 全文实体抽取
        res = self.extractor.extract(raw, self.entity_schema)
        for k, v in res.get("entities", {}).items():
            if isinstance(v, list):
                entities.extend(v)
            elif v:
                entities.append(v)

        # 2. 若存在并列连词切分，逐子句补充抽取，确保多任务客体均被捕获
        clauses = [c.strip() for c in self.CONJUNCTION_PATTERN.split(raw) if len(c.strip()) >= 4]
        if len(clauses) > 1:
            for c in clauses:
                c_res = self.extractor.extract(c, self.entity_schema)
                c_ents = []
                for k, v in c_res.get("entities", {}).items():
                    if isinstance(v, list):
                        c_ents.extend(v)
                    elif v:
                        c_ents.append(v)

                if c_ents:
                    entities.extend(c_ents)
                else:
                    # 尝试泛化概念提取
                    c_res2 = self.extractor.extract(c, self.clause_schema)
                    c_ents2 = []
                    for k, v in c_res2.get("entities", {}).items():
                        if isinstance(v, list):
                            c_ents2.extend(v)
                        elif v:
                            c_ents2.append(v)
                    if c_ents2:
                        entities.extend(c_ents2)
                    else:
                        # 降级提取子句核心动宾客体短语
                        cleaned = re.sub(r"^(把|顺便把|并且|新增|修复|修改|重构|编写|配置|升级)\s*", "", c)
                        if cleaned:
                            entities.append(cleaned)

        # 去重与清洗
        cleaned_entities = []
        for e in entities:
            e_clean = e.strip(" ，。、；")
            if len(e_clean) >= 2 and e_clean not in cleaned_entities:
                cleaned_entities.append(e_clean)

        return cleaned_entities

    def compute_entity_coverage(self, entities: Sequence[str], response_text: str) -> Tuple[float, List[str], List[str]]:
        """计算实体覆盖率：字面包含或客体向量相似度 >= 0.35。"""
        if not entities:
            return 1.0, [], []

        covered = []
        missing = []
        resp_lower = response_text.lower()

        for ent in entities:
            ent_lower = ent.lower()
            # 1. 字面匹配
            if ent_lower in resp_lower:
                covered.append(ent)
                continue

            # 2. 向量相似度匹配
            sim_ent = self.compute_sim(ent, response_text)
            if sim_ent >= 0.35:
                covered.append(ent)
            else:
                missing.append(ent)

        ratio = len(covered) / len(entities) if entities else 1.0
        return ratio, covered, missing

    def step(self, role: str, raw_text: str) -> dict:
        """输入单个交互块并执行流转。"""
        prefix = "【用户】" if role == "user" else "【助手】"
        formatted_text = prefix + raw_text

        # 站 1: 语用定性
        res_clf = self.guard.batch_extract([formatted_text], self.guard_schema)[0]
        pragmatic = res_clf.get("进度守卫4", "NEUTRAL")

        # 检查是否为显式完成断言
        if role == "assistant" and pragmatic != "CLOSE" and self.EXPLICIT_CLOSE_PATTERN.search(raw_text):
            pragmatic = "CLOSE"

        # 检查是否为用户放弃/取消
        is_abandon = False
        if role == "user" and self.ABANDON_PATTERN.search(raw_text):
            pragmatic = "DROPPED"
            is_abandon = True

        action = "NONE"
        detail = ""
        qid_target = None

        # 1. 待办登记
        if role == "user" and pragmatic in ("REQUEST", "INTENT") and not is_abandon:
            self.issue_seq += 1
            qid = f"q{self.issue_seq:03d}"
            # 抽取关键实体/概念
            target_entities = self.extract_target_entities(raw_text)

            q_item = {
                "qid": qid,
                "text": raw_text,
                "entities": target_entities,
                "status": "OPEN",
                "covered_entities": [],
                "missing_entities": target_entities,
                "raised_text": raw_text,
            }
            self.open_questions.append(q_item)
            self.all_questions[qid] = q_item

            action = f"REGISTER ({qid})"
            detail = f"登记待办 | 抽取关键实体: {target_entities} | 待办队列: {len(self.open_questions)}"
            qid_target = qid

        # 2. 方案放弃与撤销 (DROPPED / ABANDON)
        elif role == "user" and is_abandon:
            if not self.open_questions:
                action = "DROPPED_IGNORED"
                detail = "无活跃待办，取消指令忽略"
            else:
                # 寻找匹配的待办事项
                best_q = None
                best_score = -1.0
                for q in self.open_questions:
                    # 优先根据实体或内容相似度匹配
                    s = self.compute_sim(raw_text, q["text"])
                    for ent in q["entities"]:
                        if ent.lower() in raw_text.lower():
                            s += 0.5
                    if s > best_score:
                        best_score = s
                        best_q = q

                if best_q:
                    best_q["status"] = "DROPPED"
                    best_q["closed_why"] = f"用户撤销/放弃: {raw_text}"
                    self.open_questions.remove(best_q)
                    action = f"DROPPED ({best_q['qid']})"
                    detail = f"安全移除已撤销待办 | 不计入完成 | 剩余待办: {len(self.open_questions)}"
                    qid_target = best_q["qid"]
                else:
                    # LIFO 移除最近一条
                    q_last = self.open_questions.pop()
                    q_last["status"] = "DROPPED"
                    action = f"DROPPED ({q_last['qid']})"
                    detail = f"LIFO 移除最近待办 | 剩余待办: {len(self.open_questions)}"
                    qid_target = q_last["qid"]

        # 3. 闭合核销尝试 (CLOSE)
        elif pragmatic == "CLOSE":
            if not self.open_questions:
                action = "CLOSE_IGNORED"
                detail = "待办池为空，无需核销"
            else:
                # 寻找客体语义最高重叠的待办项（兼顾全文相似度与未完成实体相似度）
                best_sim = -1.0
                best_q = None
                for q in self.open_questions:
                    sim = self.compute_sim(q["text"], raw_text)
                    for m_ent in q.get("missing_entities", []):
                        sim = max(sim, self.compute_sim(m_ent, raw_text))
                    if sim > best_sim:
                        best_sim = sim
                        best_q = q

                qid_target = best_q["qid"]

                # 检查句子级对齐相似度
                if best_sim < 0.35:
                    action = "CLOSE_MISMATCH"
                    detail = f"最高相似度 {best_sim:.4f} < 0.35 | 跨话题虚假完成拦截"
                else:
                    # 计算关键实体覆盖率（累计增量模式）
                    entities = best_q.get("entities", [])
                    _, covered_now, _ = self.compute_entity_coverage(entities, raw_text)

                    # 累加历史已覆盖实体
                    prev_covered = best_q.get("covered_entities", [])
                    cum_covered = list(dict.fromkeys(prev_covered + covered_now))
                    cum_missing = [e for e in entities if e not in cum_covered]
                    cum_ratio = len(cum_covered) / len(entities) if entities else 1.0

                    # 若存在多个实体且仍有缺失 -> 状态降级为 PARTIAL（拒绝放行，保持挂起）
                    if len(entities) > 1 and cum_ratio < 1.0:
                        best_q["status"] = "PARTIAL"
                        best_q["covered_entities"] = cum_covered
                        best_q["missing_entities"] = cum_missing
                        action = f"PARTIAL ({best_q['qid']})"
                        detail = (
                            f"早退拦截: 实体累计覆盖率 {cum_ratio:.1%} ({len(cum_covered)}/{len(entities)}) | "
                            f"已完成: {cum_covered} | 缺失: {cum_missing} | 保持 PARTIAL 挂起拦截"
                        )
                    else:
                        # 全覆盖或无细分多实体 -> 成功核销
                        best_q["status"] = "CLOSED"
                        best_q["covered_entities"] = cum_covered
                        best_q["missing_entities"] = []
                        self.open_questions.remove(best_q)
                        action = f"RESOLVE ({best_q['qid']})"
                        detail = (
                            f"成功核销 | 相似度: {best_sim:.4f} | 实体覆盖率: {cum_ratio:.1%} | "
                            f"已完全覆盖全部关键实体 | 剩余待办: {len(self.open_questions)}"
                        )

        event = {
            "seq": len(self.events),
            "role": role,
            "pragmatic": pragmatic,
            "action": action,
            "qid": qid_target,
            "detail": detail,
            "text": raw_text,
            "timestamp": time.time(),
        }
        self.events.append(event)
        return event

    def check_return(self) -> dict:
        """主 Agent 准备返回时的状态门禁判定。

        若待办池非空（包括 OPEN 和 PARTIAL），必须严格拦截，禁止早退。
        """
        is_blocked = len(self.open_questions) > 0
        evidence = []
        for q in self.open_questions:
            evidence.append({
                "qid": q["qid"],
                "status": q["status"],
                "text": q["text"],
                "entities": q["entities"],
                "covered": q.get("covered_entities", []),
                "missing": q.get("missing_entities", []),
            })

        return {
            "block": is_blocked,
            "open_count": len(self.open_questions),
            "open_questions": [q["qid"] for q in self.open_questions],
            "evidence": evidence,
        }

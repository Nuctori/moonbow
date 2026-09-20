# -*- coding: utf-8 -*-
"""progress_guard.models

轻量判别组件模型架构与权重加载管理：
1. 模态头 (Modality Classifier): 判定句子语气是否为事实断言 (assert)
2. 二值完成度捕获头 (Capture Classifier): 区分真实全部完成 vs 模糊/过程未完成
3. 文本对齐嵌入模型 (Dense Alignment Encoder): 基于 MiniLM 的语义相似度
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer

MOD_CLASSES = ["assert", "promise", "question"]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * m, 1) / torch.clamp(m.sum(1), min=1e-9)


class SlotModel(nn.Module):
    """模态与完成度多头网络"""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        h_dim = encoder.config.hidden_size
        self.mod_head = nn.Linear(h_dim, 3)
        self.comp_head = nn.Linear(h_dim, 4)

    def forward(self, inp):
        o = self.encoder(**inp)
        emb = F.normalize(mean_pool(o.last_hidden_state, inp["attention_mask"]), p=2, dim=1)
        return self.mod_head(emb), self.comp_head(emb)


class ModelRegistry:
    """自动解析与加载微模型权重的单例/容器"""

    def __init__(self, models_dir: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.models_dir = models_dir or self._discover_models_dir()
        
        # 1. 查找 base MiniLM
        self.base_dir = self._find_base_minilm()
        self.tok_sim = AutoTokenizer.from_pretrained(self.base_dir)
        self.enc_sim = AutoModel.from_pretrained(self.base_dir).to(device).eval()

        # 2. 加载模态头
        slot_path = os.path.join(self.models_dir, "minilm_slot_heads_v2")
        slot_enc = AutoModel.from_pretrained(slot_path).to(device).eval()
        self.slot_model = SlotModel(slot_enc).to(device).eval()
        heads_path = os.path.join(slot_path, "heads.pt")
        heads = torch.load(heads_path, map_location=device)
        self.slot_model.mod_head.load_state_dict(heads["mod_head"])
        self.slot_model.eval()

        # 3. 加载二值收尾捕获头
        cap_path = os.path.join(self.models_dir, "capture_head_v1")
        self.tok_cap = AutoTokenizer.from_pretrained(cap_path)
        self.enc_cap = AutoModel.from_pretrained(cap_path).to(device).eval()
        self.head_cap = nn.Linear(self.enc_cap.config.hidden_size, 2).to(device)
        self.head_cap.load_state_dict(torch.load(os.path.join(cap_path, "head.pt"), map_location=device))
        self.head_cap.eval()

    def _discover_models_dir(self) -> str:
        env_p = os.environ.get("PROGRESS_GUARD_MODELS_DIR")
        if env_p and os.path.isdir(env_p):
            return env_p
        
        curr = os.path.abspath(os.path.dirname(__file__))
        cands = [
            os.path.join(curr, "models"),
            os.path.abspath(os.path.join(curr, "../../models")),
            os.path.abspath("models"),
        ]
        for c in cands:
            if os.path.isdir(os.path.join(c, "capture_head_v1")):
                return c
        raise FileNotFoundError("未找到 progress-guard 模型目录，请设置环境变量 PROGRESS_GUARD_MODELS_DIR")

    def _find_base_minilm(self) -> str:
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache):
            dirs = [x for x in os.listdir(cache) if "MiniLM-L12-v2" in x]
            if dirs:
                snap_dir = os.path.join(cache, dirs[0], "snapshots")
                if os.path.exists(snap_dir) and os.listdir(snap_dir):
                    return os.path.join(snap_dir, os.listdir(snap_dir)[0])
        # 回退使用 slot 模型自带的编码器目录
        return os.path.join(self.models_dir, "capture_head_v1")

    def get_similarity(self, a: str, b: str) -> float:
        """计算两段文本之间的余弦相似度"""
        inp = self.tok_sim([a, b], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.enc_sim(**inp)
            embs = F.normalize(mean_pool(o.last_hidden_state, inp["attention_mask"]), p=2, dim=1)
        return F.cosine_similarity(embs[0:1], embs[1:2]).item()

    def get_modality(self, text: str) -> str:
        """判定文本的模态语气 (assert / promise / question)"""
        inp = self.tok_sim(text, padding=True, truncation=True, max_length=96, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.slot_model.encoder(**inp)
            emb = F.normalize(mean_pool(o.last_hidden_state, inp["attention_mask"]), p=2, dim=1)
            logits = self.slot_model.mod_head(emb)
        return MOD_CLASSES[int(logits.argmax(-1))]

    def get_capture_prob(self, text: str) -> float:
        """判定收尾陈述被理解为「全部完成」的二值后验概率 [0.0, 1.0]"""
        inp = self.tok_cap(text, padding=True, truncation=True, max_length=96, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.enc_cap(**inp)
            emb = F.normalize(mean_pool(o.last_hidden_state, inp["attention_mask"]), p=2, dim=1)
            return self.head_cap(emb).softmax(-1)[:, 1].item()

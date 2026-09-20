# -*- coding: utf-8 -*-
"""pipeline/train_contrastive_alignment.py

对 117M Sentence-Transformer 编码器进行 Triplet Margin 对比学习微调。
目标：
让模型在代码工程语境下，将 (Anchor, Positive) 的向量距离拉近，
将 (Anchor, Hard_Negative) 的向量距离推远。
损失函数: TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5)
使用设备: Intel Arc XPU
训练轮次: 3 epoch (耗时预计 <1 分钟)
"""
import os, sys, time, json, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

class TripletDataset(Dataset):
    def __init__(self, data_path):
        data = json.load(open(data_path, encoding="utf-8"))
        self.items = data["triplets"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["anchor"], item["pos"], item["neg"]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(sum_embeddings / sum_mask, p=2, dim=1)

def main():
    print("="*60)
    print("开始对 117M 客体对齐编码器进行对比学习微调 (XPU)")
    print("="*60)
    
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"训练设备: {device}")
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    m_dirs = [d for d in os.listdir(cache_dir) if "MiniLM-L12-v2" in d]
    snap_dir = os.path.join(cache_dir, m_dirs[0], "snapshots")
    snap = os.path.join(snap_dir, os.listdir(snap_dir)[0])
    
    print("加载底座权重:", snap)
    tok = AutoTokenizer.from_pretrained(snap)
    mdl = AutoModel.from_pretrained(snap).to(device)
    mdl.train()
    
    dataset = TripletDataset("maps/contrastive_alignment_train.json")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.AdamW(mdl.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    # Cosine distance loss with margin 0.4
    def cosine_dist(x, y):
        return 1.0 - F.cosine_similarity(x, y)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.4)
    
    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for anchors, poses, negs in loader:
            optimizer.zero_grad()
            
            inp_a = tok(list(anchors), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            inp_p = tok(list(poses), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            inp_n = tok(list(negs), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            emb_a = mean_pooling(mdl(**inp_a), inp_a["attention_mask"])
            emb_p = mean_pooling(mdl(**inp_p), inp_p["attention_mask"])
            emb_n = mean_pooling(mdl(**inp_n), inp_n["attention_mask"])
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            
        print(f"Epoch {ep+1}/{epochs} | Avg Loss: {ep_loss/len(loader):.4f}")
        
    print(f"\n微调完成，总耗时: {time.time()-t0:.2f}s")
    out_dir = "models/minilm_alignment_ft"
    os.makedirs(out_dir, exist_ok=True)
    mdl.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"模型已保存至: {out_dir}")

if __name__ == "__main__":
    main()

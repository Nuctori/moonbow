# -*- coding: utf-8 -*-
"""research_v201_metrics.py — 口径族修正 + 最终验收

§251 发现：`adjacent` 与 `exact` 在低档位数下互斥，不能并列作为"达标"门槛。
  随机会"碰巧"落在相邻档（本例 41.7%），而高 exact 的模型 adjacent 反被挤掉。
  ⇒ adjacent 单独看会**惩罚好模型**。

修正口径族：
  M1 exact            —— 完全一致
  M2 **within-1 累计** —— exact + adjacent（单调、可比、不受互斥影响）← 主口径
  M3 reverse          —— 反向率（按 REVERSE_GATE_v2 校正）
  M4 Spearman         —— 序相关（不依赖档位是否可完全区分）
  M5 per-class recall —— 各 gold 档召回（暴露档位缺证据）
本脚本给出模型 vs 三个对照（trivial / 随机 / 多数类）的完整对照表。
"""
import json, math, random
from collections import Counter
LABELS=["STRONG","MEDIUM","WEAK","NONE"]; ORD={g:i for i,g in enumerate(LABELS)}

def metrics(gold,pred):
    n=len(gold); gx=[ORD[g] for g in gold]; px=[ORD[p] for p in pred]
    ex=sum(1 for a,b in zip(gx,px) if a==b)/n
    ad=sum(1 for a,b in zip(gx,px) if abs(a-b)==1)/n
    w1=ex+ad
    rv=sum(1 for a,b in zip(gx,px) if abs(a-b)>=2)/n
    per={}
    for g in LABELS:
        idx=[i for i in range(n) if gold[i]==g]
        if idx: per[g]=sum(1 for i in idx if pred[i]==g)/len(idx)
    return {'exact':ex,'adjacent':ad,'within1':w1,'reverse':rv,'per_class_recall':per}

def spearman(x,y):
    n=len(x)
    def rank(v):
        s=sorted(range(n),key=lambda i:v[i]); r=[0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[s[j+1]]==v[s[i]]: j+=1
            for k in range(i,j+1): r[s[k]]=(i+j)/2+1
            i=j+1
        return r
    rx,ry=rank(x),rank(y); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

def main():
    cf=json.load(open('maps/strength_confusion.json',encoding='utf-8'))
    gold=[];pred=[]
    for g in LABELS:
        for p,k in cf.get(g,{}).items(): gold+=[g]*k; pred+=[p]*k
    n=len(gold); gx=[ORD[g] for g in gold]
    tri=Counter(gold).most_common(1)[0][0]
    random.seed(0)
    src=gold[:]; random.shuffle(src)
    rnd=src
    arms={
      '287M 模型':      pred,
      f'trivial({tri})': [tri]*n,
      '随机打乱':        rnd,
    }
    # 随机基线（同 pred 边缘）用 K 次平均
    K=2000; acc={k:0.0 for k in ('exact','adjacent','within1','reverse')}
    s2=gold[:]
    for _ in range(K):
        random.shuffle(s2)
        m=metrics(gold,s2)
        for k in acc: acc[k]+=m[k]/K
    rows=[]
    print(f"n={n}\n")
    print(f"{'对象':<16}{'exact':>8}{'adj':>8}{'within1':>9}{'reverse':>9}{'spearman':>10}")
    for name,p in arms.items():
        m=metrics(gold,p); rho=spearman(gx,[ORD[x] for x in p])
        print(f"{name:<16}{m['exact']:>8.3f}{m['adjacent']:>8.3f}{m['within1']:>9.3f}{m['reverse']:>9.3f}{rho:>+10.3f}")
        rows.append({'name':name,'metrics':m,'spearman':rho})
    print(f"{'随机基线(均值)':<14}{acc['exact']:>8.3f}{acc['adjacent']:>8.3f}{acc['within1']:>9.3f}{acc['reverse']:>9.3f}")
    print(f"\n[修正读法] within-1 累计是主口径（单调、与随机可比）")
    mm=metrics(gold,pred)
    print(f"  模型 within-1 = {mm['within1']:.3f}  vs 随机 {acc['within1']:.3f}  → 超额 {mm['within1']-acc['within1']:+.3f}")
    print(f"  模型 exact    = {mm['exact']:.3f}  vs 随机 {acc['exact']:.3f}  → 超额 {mm['exact']-acc['exact']:+.3f}")
    print(f"\n[per-class recall] 各 gold 档召回（暴露档位缺证据）")
    for g,v in mm['per_class_recall'].items():
        cnt=sum(1 for x in gold if x==g)
        print(f"  {g:7s} n={cnt:4d}  recall {v:.3f}")
    json.dump({'n':n,'arms':rows,'random_baseline':acc,
               'model':{'metrics':mm,'spearman':spearman(gx,[ORD[x] for x in pred])}},
              open('maps/multimetric_final.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/multimetric_final.json")

if __name__=='__main__': main()

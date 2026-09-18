# -*- coding: utf-8 -*-
"""research_v200_adjacent.py — adjacent 诊断实验（下一轮的靶子）

adjacent = 0.215 vs 随机基线 0.420（低于随机）是唯一有区分力的失败信号。
低于随机有两种可能：
  H_a 模型学到了反向的模式（over-correction）：预测与 gold 负相关
  H_b 模型退化为少数几个档位，而 gold 分布更宽 → 差距来自"输出熵过低"
本实验分离两者：
  T1 输出熵对比：模型预测的档位分布 vs gold 档位分布（归一化熵）
  T2 秩相关：Spearman(gold_ord, pred_ord)，负值支持 H_a
  T3 分层 confusability：哪些相邻档对不可分（这是判据问题，不是模型问题）
  T4 可学的上界：用 gold 自身做 5 折，看"同判据下的相邻档"上限
"""
import json, math
from collections import Counter
LABELS=["STRONG","MEDIUM","WEAK","NONE"]; ORD={g:i for i,g in enumerate(LABELS)}

def entropy(c, n):
    ps=[v/n for v in c.values() if v>0]
    return -sum(p*math.log(p) for p in ps)/math.log(len(LABELS))

def spearman(x,y):
    n=len(x)
    def rank(v):
        s=sorted(range(n), key=lambda i:v[i]); r=[0]*n
        i=0
        while i<n:
            j=i
            while j+1<n and v[s[j+1]]==v[s[i]]: j+=1
            avg=(i+j)/2+1
            for k in range(i,j+1): r[s[k]]=avg
            i=j+1
        return r
    rx,ry=rank(x),rank(y)
    mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

def main():
    cf=json.load(open('maps/strength_confusion.json',encoding='utf-8'))
    gold=[];pred=[]
    for g in LABELS:
        for p,k in cf.get(g,{}).items():
            gold+= [g]*k; pred+=[p]*k
    n=len(gold)
    cg=Counter(gold); cp=Counter(pred)
    print(f"[n] {n}")
    print(f"[T1] gold 熵 {entropy(cg,n):.3f} | pred 熵 {entropy(cp,n):.3f}  dist_gold={dict(cg)} dist_pred={dict(cp)}")
    # T2
    gx=[ORD[g] for g in gold]; px=[ORD[p] for p in pred]
    rho=spearman(gx,px)
    print(f"[T2] Spearman(gold,pred) = {rho:+.3f}  → {'负相关：模型反向（H_a）' if rho<0 else '正相关：模型有真实信号（H_b）'}")
    # adjacent
    ex=sum(1 for a,b in zip(gx,px) if a==b)/n
    ad=sum(1 for a,b in zip(gx,px) if abs(a-b)==1)/n
    rv=sum(1 for a,b in zip(gx,px) if abs(a-b)>=2)/n
    print(f"[T3] exact {ex:.3f} adjacent {ad:.3f} reverse {rv:.3f}")
    # 随机基线（同 pred 边缘分布抽 gold）
    import random
    random.seed(0)
    src=gold[:]; K=2000; bex=bad=brv=0
    for _ in range(K):
        random.shuffle(src)
        for a,b in zip(gx,[ORD[s] for s in src]):
            d=abs(a-b)
            if d==0: bex+=1
            elif d==1: bad+=1
            else: brv+=1
    T=n*K
    print(f"[T3] 随机基线 exact {bex/T:.3f} adjacent {bad/T:.3f} reverse {brv/T:.3f}")
    print(f"     → adjacent 超额 {ad-bad/T:+.3f}   {'低于随机' if ad<bad/T else '高于随机'}")
    # T3b 分层可混淆性：哪些 gold 档最容易掉到相邻档
    print("\n[T3b] 各 gold 档的落点（行=gold）")
    for g in LABELS:
        idx=[i for i in range(n) if gold[i]==g]
        if not idx: continue
        c=Counter(pred[i] for i in idx)
        tot=len(idx)
        s=" ".join(f"{k}:{v/tot:.2f}" for k,v in sorted(c.items(),key=lambda x:-x[1]))
        print(f"    {g:7s} n={tot:4d}  {s}")
    # T4 理论相邻上界：若把不可分的相邻档合并，可达到的 adjacent
    print("\n[T4] 相邻档对的可分性（若某对完全不可分 → 判据该合并该对）")
    for a,b in (('STRONG','MEDIUM'),('MEDIUM','WEAK'),('WEAK','NONE')):
        n_a=sum(1 for i in range(n) if gold[i]==a); n_b=sum(1 for i in range(n) if gold[i]==b)
        # 该对内互相误判率
        conf_ab=sum(1 for i in range(n) if gold[i]==a and pred[i]==b)
        conf_ba=sum(1 for i in range(n) if gold[i]==b and pred[i]==a)
        denom=n_a+n_b
        rate=(conf_ab+conf_ba)/denom if denom else 0
        print(f"    {a:7s}<->{b:7s} n={denom:4d} 互混率 {rate:.3f}  ({conf_ab} / {conf_ba})")
    out={'n':n,'entropy_gold':entropy(cg,n),'entropy_pred':entropy(cp,n),
         'spearman':rho,'exact':ex,'adjacent':ad,'reverse':rv,
         'baseline':{'exact':bex/T,'adjacent':bad/T,'reverse':brv/T},
         'dist_gold':dict(cg),'dist_pred':dict(cp)}
    json.dump(out,open('maps/adjacent_diag.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/adjacent_diag.json")

if __name__=='__main__': main()

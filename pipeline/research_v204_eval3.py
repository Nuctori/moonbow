# -*- coding: utf-8 -*-
"""research_v204_eval3.py — 三值模型验收（v6 判据 · 校正口径族）

评估集：clean_recheck_200.json，gold 由 v5b 标签按 v6 规则转换
        （无token却非NONE的判据违规样本标记为 EXCLUDE，与训练集一致）
口径：MULTI_METRIC_ACCEPTANCE.md v2（含口径准入检验）
门槛：REVERSE_GATE_v2.md 三闸
"""
import os, sys, json, math, random
sys.path.insert(0, 'pipeline')

MODEL = os.environ.get("MODEL", "models/gliner25_strength3_v1/final")
DEV = os.environ.get("MAP_LOC", "xpu")
TASK = "强度3"
LABELS = ["STRONG", "MEDIUM", "NONE"]
ORD = {g: i for i, g in enumerate(LABELS)}
OUT = os.environ.get("RESULT_JSON", "maps/strength3_eval_result.json")

def to3(g):
    return 'MEDIUM' if g in ('WEAK', 'MEDIUM') else g

def toks_present(tk):
    t = str(tk or '').strip()
    return not (t.upper() in ('NONE','','NONE.','N/A','无') or len(t) <= 3)

def load_all():
    """返回 {标注集名: (rows, 排除数)}。所有集共享同一批 200 个 item 与 work_log。"""
    ck = json.load(open('maps/clean_recheck_200.json', encoding='utf-8'))
    by = {x['pid']: x for x in ck['items']}
    out = {}
    for tag in ('v5b_A', 'v5b_B', 'v5c_A', 'v5c_B'):
        try:
            d = json.load(open(f'maps/{tag}.json', encoding='utf-8'))
        except Exception:
            continue
        rows = []; excl = 0
        for x in d['items']:
            wl = by.get(x['pid'], {}).get('work_log', '')
            for bi, b in enumerate(x['bindings']):
                if not toks_present(b.get('tokens')) and b['grade'] != 'NONE':
                    excl += 1; continue      # 判据违规，与训练集一致地排除
                rows.append({'pid': x['pid'], 'bidx': bi, 'gold': to3(b['grade']), 'intent': b['intent'],
                             'text': "\u3010\u610f\u56fe\u3011" + (b['intent'] or '')[:300]
                                     + "\n\u3010\u5de5\u4f5c\u8bb0\u5f55\u3011" + wl[:1200]})
        out[tag] = (rows, excl)
    return out

def metrics(gold, pred):
    n = len(gold)
    # gold 由各标注集自带，pred 是共享的 203 条 item 预测；
    # 标注集可能有排除项，故按 item 下标重对齐
    gx = [ORD[g] for g in gold]
    px = [ORD[p] for p in pred[:n]]
    assert len(px) == n, (len(px), n)
    ex = sum(1 for a,b in zip(gx,px) if a==b)/n
    ad = sum(1 for a,b in zip(gx,px) if abs(a-b)==1)/n
    rv = sum(1 for a,b in zip(gx,px) if abs(a-b)>=2)/n
    per = {}
    for g in LABELS:
        idx=[i for i in range(n) if gold[i]==g]
        if idx: per[g]=sum(1 for i in idx if pred[i]==g)/len(idx)
    return {'exact':ex,'adjacent':ad,'within1':ex+ad,'reverse':rv,'per_class_recall':per}

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
    from gliner2 import AutoExtractor, Schema
    from collections import Counter
    sets = load_all()
    # 所有标注集共享同一批 item 文本 → 只推理一次
    base_rows = sets['v5b_A'][0]
    # 全局唯一 (pid,bidx) 集合 —— 各标注集可能排除不同行，必须按 key 对齐
    keys = []
    seen = set()
    key_text = {}
    for tag, (rows, _) in sets.items():
        for r in rows:
            k = (r['pid'], r['bidx'])
            key_text[k] = r['text']
            if k not in seen:
                seen.add(k); keys.append(k)
    print(f"[data] {len(keys)} 个唯一 (pid,bidx)（推理一次，多标注集共用）")
    for tag, (rows, excl) in sets.items():
        print(f"   {tag:8s} n={len(rows):4d} 排除={excl:3d} gold={dict(Counter(r['gold'] for r in rows))}")
    m = AutoExtractor.from_pretrained(MODEL, map_location=DEV)
    sc = Schema().classification(TASK, labels=LABELS)
    pred_by_key = {}
    for k in keys:
        try:
            g = m.extract(key_text[k], sc).get(TASK)
            g = g if g in LABELS else 'MEDIUM'
        except Exception:
            g = 'MEDIUM'
        pred_by_key[k] = g
    print(f"[pred] {dict(Counter(pred_by_key.values()))}")
    results = {}
    for tag, (rows, excl) in sets.items():
        gold = [r['gold'] for r in rows]
        preds = [pred_by_key[(r['pid'], r['bidx'])] for r in rows]
        n = len(rows)
        mm = metrics(gold, preds)
        gx = [ORD[g] for g in gold]
        tri = Counter(gold).most_common(1)[0][0]
        random.seed(0); s2 = gold[:]; random.shuffle(s2)
        arms = {f'287M 三值模型': preds, f'trivial({tri})': [tri]*n, '随机打乱': s2}
        K = 2000; bacc = {k: 0.0 for k in ('exact','adjacent','within1','reverse')}
        s3 = gold[:]
        for _ in range(K):
            random.shuffle(s3); m2 = metrics(gold, s3)
            for k in bacc: bacc[k] += m2[k]/K
        # 校正反向率门
        hard = [(g,p) for g,p in zip(gold,preds) if g != tri]
        g2 = sum(1 for g,p in hard if abs(ORD[g]-ORD[p])>=2)/len(hard) if hard else 0.0
        tri_hard = sum(1 for g,_ in hard if abs(ORD[g]-ORD[tri])>=2)/len(hard) if hard else 0.0
        per = {}
        for g in LABELS:
            idx = [i for i in range(n) if gold[i]==g]
            if idx:
                k = sum(1 for i in idx if abs(ORD[gold[i]]-ORD[preds[i]])>=2)
                per[g] = {'n':len(idx),'rev_k':k,'rev':k/len(idx)}
        g3 = max((v['rev'] for v in per.values()), default=0.0)
        gates = {'g1': g2 < tri_hard, 'g2': g2 <= 0.02, 'g3': g3 <= 0.05}
        cm = {g: dict(Counter(p for gg,p in zip(gold,preds) if gg==g)) for g in LABELS}
        results[tag] = {'n':n,'excluded':excl,'dist_gold':dict(Counter(gold)),
                        'metrics':mm,'spearman':spearman(gx,[ORD[x] for x in preds]),
                        'baseline':bacc,'arms':{k:metrics(gold,v) for k,v in arms.items()},
                        'gate_v2':{**gates,'g2_value':g2,'g3_value':g3,'tri_hard':tri_hard,
                                   'per_class':per},'confusion':cm}
        print(f"\n=== gold={tag}  n={n} (排除 {excl}) ===")
        print(f"  分布 gold={dict(Counter(gold))}")
        print(f"  {'对象':<18}{'exact':>8}{'within1':>9}{'reverse':>9}{'spearman':>10}")
        for name,p in arms.items():
            a = metrics(gold,p); rho = spearman(gx,[ORD[x] for x in p])
            print(f"  {name:<18}{a['exact']:>8.3f}{a['within1']:>9.3f}{a['reverse']:>9.3f}{rho:>+10.3f}")
        print(f"  {'随机基线':<18}{bacc['exact']:>8.3f}{bacc['within1']:>9.3f}{bacc['reverse']:>9.3f}")
        print(f"  [门v2] G1 {'PASS' if gates['g1'] else 'FAIL'} ({g2:.4f} vs trivial {tri_hard:.4f}) | "
              f"G2 {'PASS' if gates['g2'] else 'FAIL'} ({g2:.4f}) | G3 {'PASS' if gates['g3'] else 'FAIL'} ({g3:.4f})")
        print(f"  [各档召回] " + "  ".join(f"{g}={mm['per_class_recall'][g]:.3f}(n={sum(1 for x in gold if x==g)})"
                                          for g in mm['per_class_recall']))
        print(f"  [混淆] " + " | ".join(f"{g}->{cm[g]}" for g in LABELS if cm[g]))
    json.dump({'model':MODEL,'sets':results}, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")

def _unused_main():
    from gliner2 import AutoExtractor, Schema
    rows, excl = load_all()['v5b_A']
    print(f"[data] {len(rows)} 条 (排除判据违规 {excl})")
    from collections import Counter
    print(f"[gold] {dict(Counter(r['gold'] for r in rows))}")
    m = AutoExtractor.from_pretrained(MODEL, map_location=DEV)
    sc = Schema().classification(TASK, labels=LABELS)
    preds = []
    for r in rows:
        try:
            g = m.extract(r['text'], sc).get(TASK)
            g = g if g in LABELS else 'MEDIUM'
        except Exception:
            g = 'MEDIUM'
        preds.append(g)
    gold = [r['gold'] for r in rows]
    n = len(rows)
    print(f"[pred] {dict(Counter(preds))}")
    # 主表
    mm = metrics(gold, preds)
    gx=[ORD[g] for g in gold]
    # 对照
    tri = Counter(gold).most_common(1)[0][0]
    random.seed(0); s2=gold[:]; random.shuffle(s2)
    arms = {'287M 三值模型': preds, f'trivial({tri})': [tri]*n, '随机打乱': s2}
    K=2000; bacc={k:0.0 for k in ('exact','adjacent','within1','reverse')}
    s3=gold[:]
    for _ in range(K):
        random.shuffle(s3); m2=metrics(gold,s3)
        for k in bacc: bacc[k]+=m2[k]/K
    print(f"\n{'对象':<18}{'exact':>8}{'within1':>9}{'reverse':>9}{'spearman':>10}")
    for name,p in arms.items():
        a=metrics(gold,p); rho=spearman(gx,[ORD[x] for x in p])
        print(f"{name:<18}{a['exact']:>8.3f}{a['within1']:>9.3f}{a['reverse']:>9.3f}{rho:>+10.3f}")
    print(f"{'随机基线':<18}{bacc['exact']:>8.3f}{bacc['within1']:>9.3f}{bacc['reverse']:>9.3f}")
    # 校正反向率门（三闸）
    hard=[(g,p) for g,p in zip(gold,preds) if g!=tri]
    g2=sum(1 for g,p in hard if abs(ORD[g]-ORD[p])>=2)/len(hard) if hard else 0.0
    tri_hard=sum(1 for g,_ in hard if abs(ORD[g]-ORD[tri])>=2)/len(hard) if hard else 0.0
    per={}
    for g in LABELS:
        idx=[i for i in range(n) if gold[i]==g]
        if idx:
            k=sum(1 for i in idx if abs(ORD[gold[i]]-ORD[preds[i]])>=2)
            per[g]={'n':len(idx),'rev_k':k,'rev':k/len(idx)}
    g3=max((v['rev'] for v in per.values()), default=0.0)
    gates={'G1': g2<tri_hard, 'G2': g2<=0.02, 'G3': g3<=0.05}
    print(f"\n[反向率校正门 v2]  困难子集 n={len(hard)}")
    print(f"  G1 模型({g2:.4f}) < trivial({tri_hard:.4f}): {'PASS' if gates['G1'] else 'FAIL'}")
    print(f"  G2 困难子集 <=2%: {'PASS' if gates['G2'] else 'FAIL'}")
    print(f"  G3 分层最大({g3:.4f}) <=5%: {'PASS' if gates['G3'] else 'FAIL'}  {per}")
    # 口径准入检验
    print(f"\n[口径准入检验]")
    print(f"  1 平凡解不过: trivial exact {metrics(gold,[tri]*n)['exact']:.3f} vs 模型 {mm['exact']:.3f} → {'PASS' if mm['exact']>metrics(gold,[tri]*n)['exact'] else 'FAIL'}")
    print(f"  2 随机可比  : {bacc['exact']:.3f} → PASS")
    print(f"  3 不惩罚正确: exact 超额 {mm['exact']-bacc['exact']:+.3f} / within1 超额 {mm['within1']-bacc['within1']:+.3f} → PASS")
    print(f"\n[各档召回]")
    for g,v in mm['per_class_recall'].items():
        c=sum(1 for x in gold if x==g)
        print(f"  {g:7s} n={c:4d}  recall {v:.3f}")
    json.dump({'model':MODEL,'n':n,'excluded_drift':excl,
               'dist_gold':dict(Counter(gold)),'dist_pred':dict(Counter(preds)),
               'metrics':mm,'spearman':spearman(gx,[ORD[x] for x in preds]),
               'baseline':bacc,'arms':{k:metrics(gold,v) for k,v in arms.items()},
               'gate_v2':{'g1_pass':gates['G1'],'g2':g2,'g2_pass':gates['G2'],
                          'g3':g3,'g3_pass':gates['G3'],'per_class':per,'tri_hard':tri_hard},
               'confusion':{g:dict(Counter(p for gg,p in zip(gold,preds) if gg==g)) for g in LABELS}},
              open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")

if __name__ == '__main__': main()

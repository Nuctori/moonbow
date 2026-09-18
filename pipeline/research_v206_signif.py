# -*- coding: utf-8 -*-
"""research_v206_signif.py — 三臂差异的配对显著性检验（补 §255 的保留）

§255 只报了均值差（H0 0.779 / H1 0.722 / H2 0.756），未做检验，
故只能说"无稳定增益"。本脚本补：
  - 逐 item 配对（同 item 同 gold），McNemar 精确检验（配对二值正确性）
  - 10000 次 bootstrap 的均值差 95% CI（按 item 重抽，保持配对）
  - 跨标注集的一致性：四个集方向是否同号
"""
import json, math, random
from collections import Counter
LABELS=["STRONG","MEDIUM","NONE"]; ORD={g:i for i,g in enumerate(LABELS)}

def mcnemar_exact(b, c):
    """b = A对B错 的 item 数, c = A错B对。精确二项双尾 p。"""
    n = b + c
    if n == 0: return 1.0, 0.0
    k = min(b, c)
    # 双尾精确 p
    p = 0.0
    for i in range(0, k+1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2*p), k/n

def main():
    # 重建三臂的逐 item 正确性（复用 v205 的逻辑，但保留逐条）
    import re, os, sys
    sys.path.insert(0,'pipeline')
    import importlib.util as _iu
    _spec=_iu.spec_from_file_location('v205','pipeline/research_v205_hybrid.py')
    _m=_iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
    program_grade=_m.program_grade; toks_present=_m.toks_present; to3=_m.to3
    MODEL=_m.MODEL; DEV=_m.DEV; TASK=_m.TASK
    from gliner2 import AutoExtractor, Schema
    ck=json.load(open('maps/clean_recheck_200.json',encoding='utf-8'))
    by={x['pid']:x for x in ck['items']}
    sets={}
    for tag in ('v5b_A','v5b_B','v5c_A','v5c_B'):
        d=json.load(open(f'maps/{tag}.json',encoding='utf-8'))
        rows=[]
        for x in d['items']:
            wl=by.get(x['pid'],{}).get('work_log','')
            for bi,b in enumerate(x['bindings']):
                if not toks_present(b.get('tokens')) and b['grade']!='NONE': continue
                rows.append({'pid':x['pid'],'bidx':bi,'gold':to3(b['grade']),
                             'text':"【意图】"+(b['intent'] or '')[:300]+"\n【工作记录】"+wl[:1200],
                             'wl':wl})
        sets[tag]=rows
    keys=[]; seen=set(); kt={}
    for tag,rows in sets.items():
        for r in rows:
            k=(r['pid'],r['bidx']); kt[k]=r
            if k not in seen: seen.add(k); keys.append(k)
    m=AutoExtractor.from_pretrained(MODEL,map_location=DEV)
    sc=Schema().classification(TASK,labels=LABELS)
    mp={}
    for k in keys:
        try:
            g=m.extract(kt[k]['text'],sc).get(TASK); g=g if g in LABELS else 'MEDIUM'
        except Exception: g='MEDIUM'
        mp[k]=g
    out={}
    print("=== 逐集配对检验（McNemar 精确 + bootstrap CI）===")
    for tag,rows in sets.items():
        gold=[r['gold'] for r in rows]
        A=[mp[(r['pid'],r['bidx'])] for r in rows]               # H0 纯模型
        B=[program_grade(r['wl']) for r in rows]                  # H2 纯程序
        C=[('NONE' if p=='NONE' else (a if a in ('STRONG','MEDIUM') else 'MEDIUM'))
           for a,p in zip(A,B)]                                   # H1 闸+模型
        okA=[int(a==g) for a,g in zip(A,gold)]
        okB=[int(b==g) for b,g in zip(B,gold)]
        okC=[int(c==g) for c,g in zip(C,gold)]
        n=len(gold)
        res={'n':n,'acc':{'H0':sum(okA)/n,'H1':sum(okC)/n,'H2':sum(okB)/n}}
        # McNemar: H0 vs H1
        b01=sum(1 for a,c in zip(okA,okC) if a and not c)
        c01=sum(1 for a,c in zip(okA,okC) if c and not a)
        p01,do01=mcnemar_exact(b01,c01)
        # McNemar: H0 vs H2
        b02=sum(1 for a,b_ in zip(okA,okB) if a and not b_)
        c02=sum(1 for a,b_ in zip(okA,okB) if b_ and not a)
        p02,do02=mcnemar_exact(b02,c02)
        # McNemar: H1 vs H2
        b12=sum(1 for c,b_ in zip(okC,okB) if c and not b_)
        c12=sum(1 for c,b_ in zip(okC,okB) if b_ and not c)
        p12,do12=mcnemar_exact(b12,c12)
        # bootstrap CI for H0-H1 and H0-H2
        random.seed(13); K=10000
        d01=[]; d02=[]
        for _ in range(K):
            idx=[random.randrange(n) for _ in range(n)]
            d01.append(sum(okA[i]-okC[i] for i in idx)/n)
            d02.append(sum(okA[i]-okB[i] for i in idx)/n)
        d01.sort(); d02.sort()
        def ci(v): return (v[int(0.025*K)], v[int(0.975*K)])
        res['H0_vs_H1']={'delta':res['acc']['H0']-res['acc']['H1'],'b':b01,'c':c01,
                         'p_exact':p01,'ci95':ci(d01)}
        res['H0_vs_H2']={'delta':res['acc']['H0']-res['acc']['H2'],'b':b02,'c':c02,
                         'p_exact':p02,'ci95':ci(d02)}
        res['H1_vs_H2']={'b':b12,'c':c12,'p_exact':p12}
        out[tag]=res
        print(f"\n {tag} n={n}")
        print(f"   acc  H0 {res['acc']['H0']:.3f} | H1 {res['acc']['H1']:.3f} | H2 {res['acc']['H2']:.3f}")
        print(f"   H0 vs H1: Δ={res['H0_vs_H1']['delta']:+.3f} 95%CI[{res['H0_vs_H1']['ci95'][0]:+.3f},{res['H0_vs_H1']['ci95'][1]:+.3f}]"
              f"  McNemar p={p01:.4f} (b={b01},c={c01})")
        print(f"   H0 vs H2: Δ={res['H0_vs_H2']['delta']:+.3f} 95%CI[{res['H0_vs_H2']['ci95'][0]:+.3f},{res['H0_vs_H2']['ci95'][1]:+.3f}]"
              f"  McNemar p={p02:.4f} (b={b02},c={c02})")
        print(f"   H1 vs H2: McNemar p={p12:.4f} (b={b12},c={c12})")
    # 汇总判定
    print("\n=== 判定 ===")
    sig01=sum(1 for t in out if out[t]['H0_vs_H1']['p_exact']<0.05)
    sig02=sum(1 for t in out if out[t]['H0_vs_H2']['p_exact']<0.05)
    print(f"  H0 vs H1 显著的集数: {sig01}/4")
    print(f"  H0 vs H2 显著的集数: {sig02}/4")
    dirs=[1 if out[t]['H0_vs_H1']['delta']>0 else -1 for t in out]
    print(f"  H0>H1 的方向一致性: {sum(1 for d in dirs if d>0)}/4 正")
    json.dump(out,open('maps/signif_result.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("-> maps/signif_result.json")

if __name__=='__main__': main()

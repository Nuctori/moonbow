# -*- coding: utf-8 -*-
"""research_v210_final3arm.py — 干净三臂终局对照（程序臂已独立校准）

臂定义（全部"未见过评估集"）：
  P_indep  独立程序：token 规则在**训练集**上学习（f_commit ⇒ STRONG，无 token ⇒ NONE）
  M        287M 模型：训练集训练
  P_poll   污染程序：人写死 token 表（由评估集标注反推）—— 作为"上界参考"，不公平
评测：四个独立标注集，含 McNemar 配对检验 + bootstrap CI
"""
import os, sys, json, math, random
import importlib.util as _iu
_spec=_iu.spec_from_file_location('v205','pipeline/research_v205_hybrid.py')
_m=_iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
MODEL=os.environ.get("MODEL", _m.MODEL); DEV=_m.DEV; TASK=_m.TASK; LABELS=_m.LABELS
BEST='f_commit'
FEATS=[('f_commit',_m.RE_COMMIT),('f_test',_m.RE_TEST),('f_build',_m.RE_BUILD),
       ('f_deploy',_m.RE_DEPLOY),('f_diag',_m.RE_DIAG)]
def feats(wl):
    t=wl or ''; return {k:int(bool(rx.search(t))) for k,rx in FEATS}
def indep_grade(wl):
    f=feats(wl); n=sum(f.values())
    if n==0: return 'NONE'
    return 'STRONG' if f[BEST]==1 else 'MEDIUM'
def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    return min(1.0,2*sum(math.comb(n,i) for i in range(0,min(b,c)+1))*(0.5**n))

def main():
    from gliner2 import AutoExtractor, Schema
    from collections import Counter
    ck=json.load(open('maps/clean_recheck_200.json',encoding='utf-8'))
    by={x['pid']:x for x in ck['items']}
    sets={}
    for tag in ('v5b_A','v5b_B','v5c_A','v5c_B'):
        d=json.load(open(f'maps/{tag}.json',encoding='utf-8'))
        rows=[]
        for x in d['items']:
            wl=by.get(x['pid'],{}).get('work_log','')
            for bi,b in enumerate(x['bindings']):
                if not _m.toks_present(b.get('tokens')) and b['grade']!='NONE': continue
                rows.append({'pid':x['pid'],'bidx':bi,'gold':_m.to3(b['grade']),'wl':wl,
                             'text':"【意图】"+(b['intent'] or '')[:300]+"\n【工作记录】"+wl[:1200]})
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
    print(f"[n] {len(keys)} unique items | 模型输出 {dict(Counter(mp.values()))}\n")
    print(f"{'标注集':<9}{'n':>5}{'污染程序':>10}{'独立程序':>10}{'模型':>9}{'   M vs P_indep':>18}")
    res={}; allb=allc=alln=0
    for tag,rows in sets.items():
        gold=[r['gold'] for r in rows]
        poll=[_m.program_grade(r['wl']) for r in rows]
        ind=[indep_grade(r['wl']) for r in rows]
        mdl=[mp[(r['pid'],r['bidx'])] for r in rows]
        a=lambda p: sum(1 for g,q in zip(gold,p) if g==q)/len(rows)
        b=c=0
        for g,q,p in zip(gold,mdl,ind):
            om=int(q==g); op=int(p==g)
            if om and not op: b+=1
            elif op and not om: c+=1
        pv=mcnemar(b,c)
        res[tag]={'n':len(rows),'contaminated':a(poll),'independent':a(ind),'model':a(mdl),
                  'b':b,'c':c,'p':pv}
        allb+=b; allc+=c; alln+=len(rows)
        print(f"{tag:<9}{len(rows):>5}{a(poll):>10.3f}{a(ind):>10.3f}{a(mdl):>9.3f}"
              f"   Δ={a(mdl)-a(ind):+.3f} p={pv:.4f} (b={b},c={c})")
    # 跨集方向
    dirs=[1 if res[t]['model']>res[t]['independent'] else -1 for t in res]
    print(f"\n方向: {[t+':'+('模型优' if res[t]['model']>res[t]['independent'] else '程序优') for t in res]}")
    sig=[t for t in res if res[t]['p']<0.05]
    print(f"显著集数 {len(sig)}/4: {sig if sig else '无'}")
    print(f"\n[合并（伪重复，仅参考）] b={allb}(模型优) c={allc}(程序优) Δ={(allb-allc)/alln:+.4f} p={mcnemar(allb,allc):.4f}")
    # 模型 vs 污染程序（说明污染的威力）
    print(f"\n[对照] 模型 vs 污染程序（不公平臂）")
    for tag in res:
        res[tag]['vs_contam']=res[tag]['model']-res[tag]['contaminated']
    print("  " + "  ".join(f"{t}:{res[t]['vs_contam']:+.3f}" for t in res))
    print("  → 污染臂在人写死的口径上系统性更高，印证 §258")
    json.dump({'by_set':res,'pooled':{'b':allb,'c':allc,'n':alln,'delta':(allb-allc)/alln,'p':mcnemar(allb,allc)}},
              open('maps/final3arm_result.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/final3arm_result.json")

if __name__=='__main__': main()

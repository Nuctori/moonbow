# -*- coding: utf-8 -*-
"""research_v205_hybrid.py — 验证 §254.5 归因：NONE 界交还程序

假设：NONE 的判定是"work_log 有无产物 token"的机械扫描，模型学不会（实测召回 0）。
      若假设成立，把 NONE 判定交给程序机械闸后，整体应显著优于纯模型。

对照三臂（同一批 203 item，四个独立标注集）：
  H0 纯模型三值       —— 现状（NONE 召回 0）
  H1 程序闸 + 模型二值 —— 程序判无 token ⇒ NONE；否则用模型的 STRONG/MEDIUM
  H2 纯程序机械三值    —— 完全不用模型（无token⇒NONE；有commit/test⇒STRONG；否则MEDIUM）
H1 若显著优于 H0 且接近 H2，则归因成立，且给出最终架构。
"""
import os, sys, json, re
sys.path.insert(0,'pipeline')
MODEL = os.environ.get("MODEL", "models/gliner25_strength3_v1/final")
DEV = os.environ.get("MAP_LOC", "xpu")
TASK = "强度3"; LABELS = ["STRONG","MEDIUM","NONE"]
ORD = {g:i for i,g in enumerate(LABELS)}

# —— 程序机械闸（严格按 v6 判据的 token 表）——
RE_COMMIT = re.compile(r'\b[0-9a-f]{7,40}\b')
RE_TEST   = re.compile(r'(\d+\s*/\s*\d+)|(\d+\s*(passed|通过|failed|失败))|全绿|all tests? pass', re.I)
RE_BUILD  = re.compile(r'(build\s+succeeded|构建成功|编译成功|0\s*errors|0\s*个错误)', re.I)
RE_DEPLOY = re.compile(r'(部署|上线|deployed|released)', re.I)
RE_WRITE  = re.compile(r'(写入|保存|已落地|新增|改成|补上|加上|替换为|创建|生成|重构|删除|移动到)')
RE_FILE   = re.compile(r'([\w./\-]+\.(py|js|ts|tsx|jsx|cs|java|go|rs|cpp|c|h|md|json|yaml|yml|toml|sh|sql|css|html))\b', re.I)
RE_DIAG   = re.compile(r'(\w+\.\w+:\d+)|(\b[A-Z][A-Za-z0-9]*(Service|Manager|Controller|Handler)\b)')

def program_grade(wl):
    """机械三值：无产物 token ⇒ NONE；强类 token ⇒ STRONG；否则 MEDIUM"""
    t = wl or ''
    if RE_COMMIT.search(t) or RE_TEST.search(t) or RE_BUILD.search(t) or RE_DEPLOY.search(t):
        return 'STRONG'
    if (RE_WRITE.search(t) and (RE_FILE.search(t) or RE_DIAG.search(t))) or RE_DIAG.search(t):
        return 'MEDIUM'
    return 'NONE'

def has_tok(wl):
    return program_grade(wl) != 'NONE'

def toks_present(tk):
    t=str(tk or '').strip()
    return not (t.upper() in ('NONE','','NONE.','N/A','无') or len(t)<=3)
def to3(g): return 'MEDIUM' if g in ('WEAK','MEDIUM') else g

def metrics(gold,pred):
    n=len(gold); gx=[ORD[g] for g in gold]; px=[ORD[p] for p in pred]
    ex=sum(1 for a,b in zip(gx,px) if a==b)/n
    ad=sum(1 for a,b in zip(gx,px) if abs(a-b)==1)/n
    rv=sum(1 for a,b in zip(gx,px) if abs(a-b)>=2)/n
    per={}
    for g in LABELS:
        idx=[i for i in range(n) if gold[i]==g]
        if idx: per[g]=sum(1 for i in idx if pred[i]==g)/len(idx)
    return {'exact':ex,'adjacent':ad,'within1':ex+ad,'reverse':rv,'per_class_recall':per}

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
    model_pred={}
    for k in keys:
        try:
            g=m.extract(kt[k]['text'],sc).get(TASK); g=g if g in LABELS else 'MEDIUM'
        except Exception: g='MEDIUM'
        model_pred[k]=g
    print(f"[n] {len(keys)} unique items | 模型输出 {dict(Counter(model_pred.values()))}")
    results={}
    for tag,rows in sets.items():
        gold=[r['gold'] for r in rows]
        h0=[model_pred[(r['pid'],r['bidx'])] for r in rows]
        pg=[program_grade(r['wl']) for r in rows]
        # H1: 程序闸优先；有 token 时用模型（但模型的 NONE 覆盖为 MEDIUM）
        h1=[]
        for r,mp,p in zip(rows,h0,pg):
            if p=='NONE': h1.append('NONE')
            else: h1.append(mp if mp in ('STRONG','MEDIUM') else 'MEDIUM')
        results[tag]={'H0_pure_model':metrics(gold,h0),'H1_program_gate+model':metrics(gold,h1),
                      'H2_pure_program':metrics(gold,pg),'n':len(rows),
                      'dist_gold':dict(Counter(gold))}
        print(f"\n=== {tag} n={len(rows)} gold={dict(Counter(gold))} ===")
        for name in ('H0_pure_model','H1_program_gate+model','H2_pure_program'):
            a=results[tag][name]
            nr=a['per_class_recall'].get('NONE')
            print(f"  {name:<26} exact {a['exact']:.3f} within1 {a['within1']:.3f} rev {a['reverse']:.3f}"
                  + "  NONE召回 " + (f"{nr:.3f}" if nr is not None else "—"))
    # 汇总
    print("\n=== 跨四集平均 ===")
    for name in ('H0_pure_model','H1_program_gate+model','H2_pure_program'):
        ex=sum(results[t][name]['exact'] for t in results)/len(results)
        nr=[results[t][name]['per_class_recall'].get('NONE') for t in results]
        nr=[x for x in nr if x is not None]
        print(f"  {name:<26} mean exact {ex:.3f}  mean NONE召回 {sum(nr)/len(nr) if nr else float('nan'):.3f}")
    json.dump(results,open('maps/hybrid_result.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/hybrid_result.json")

if __name__=='__main__': main()

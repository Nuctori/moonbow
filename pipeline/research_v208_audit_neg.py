# -*- coding: utf-8 -*-
"""research_v208_audit_neg.py — 对"模型无增益"这一否定结论的对抗审计

要审的结论：在 STRONG/MEDIUM 二值层，287M 模型相对程序机械规则**无增益**。
该结论若成立则整个项目否证，故必须比肯定结论受更严的审计。

对抗假设（每条若成立，否定结论即失效或需削弱）：
  N1 程序规则被"优待"了 —— 程序规则是在**同一批 200 item 上**由标注判据反推出来的
     （v5b 的 token 表就是标注员用的），故程序在评估集上有"自证"优势。
     检验：程序规则若在**训练集**上也同样强，说明它是先验规则而非拟合评估集。
  N2 模型被"虐待"了 —— 训练只到 epoch 2.9/8（OOM），可能欠拟合。
     检验：对比不同 epoch 检查点的表现；若单调上升则当前比较不公平。
  N3 模型输出退化 —— 模型从不输出 NONE（实测），二值层比较可能掩盖了它是
     "常数预测器"。检验：模型在 L2 内的输出熵 vs 程序。
  N4 评估集偏向程序 —— clean_recheck_200 的选样若偏向"token 明显"的 item，
     则程序占优是选样造成的。检验：按 token 强度分层，看模型是否在弱 token 层反超。
  N5 二值化方式不公平 —— 模型输出 NONE 被强制归 MEDIUM，可能人为压低其分数。
     检验：改用最优映射（把模型 NONE 映射到 STRONG）重算上界。
"""
import os, sys, json, math, random
import importlib.util as _iu
_spec=_iu.spec_from_file_location('v205','pipeline/research_v205_hybrid.py')
_m=_iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
program_grade=_m.program_grade; toks_present=_m.toks_present; to3=_m.to3
MODEL=os.environ.get("MODEL", _m.MODEL); DEV=_m.DEV; TASK=_m.TASK; LABELS=_m.LABELS

def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    p=sum(math.comb(n,i) for i in range(0,min(b,c)+1))*(0.5**n)
    return min(1.0,2*p)

def entropy(c,n):
    ps=[v/n for v in c.values() if v>0]
    return -sum(p*math.log(p+1e-12) for p in ps)/math.log(3) if n else 0.0

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
                rows.append({'pid':x['pid'],'bidx':bi,'gold':to3(b['grade']),'wl':wl,
                             'text':"【意图】"+(b['intent'] or '')[:300]+"\n【工作记录】"+wl[:1200]})
        sets[tag]=rows
    # 训练集（用于 N1：程序是否在训练集上也强）
    tr=json.load(open('maps/strength3_train.json',encoding='utf-8'))['rows']
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
    R={}
    print("="*64)
    print("N1 程序规则是否被评估集优待？")
    # 程序在训练集上的表现（训练集文本含 work_log）
    okp=okm=0; ntr=0
    for r in tr:
        t=r['text']; wl=t.split("【工作记录】",1)[1] if "【工作记录】" in t else ''
        pg=program_grade(wl)
        if r['cls'] in ('STRONG','MEDIUM'):
            ntr+=1; okp+=int(pg==r['cls'])
    print(f"   训练集(二值子集 n={ntr}) 程序规则 acc = {okp/max(1,ntr):.3f}")
    # 评估集程序 acc（二值层）
    L2=[(tag,r) for tag,rows in sets.items() for r in rows
        if program_grade(r['wl']) in ('STRONG','MEDIUM') and r['gold'] in ('STRONG','MEDIUM')]
    okp2=sum(1 for _,r in L2 if program_grade(r['wl'])==r['gold'])
    print(f"   评估集(二值子集 n={len(L2)}) 程序规则 acc = {okp2/len(L2):.3f}")
    print(f"   → 两值接近则程序是先验规则；评估集显著更高则存在优待")
    R['N1']={'train_acc':okp/max(1,ntr),'eval_acc':okp2/len(L2),'train_n':ntr,'eval_n':len(L2)}

    print("\n"+"="*64)
    print("N2 模型是否欠拟合（epoch 2.9/8 被 OOM 中断）？")
    import glob
    cks=sorted(glob.glob('models/gliner25_strength3_v1/checkpoint-epoch-*'))
    print(f"   可用检查点: {[os.path.basename(c) for c in cks]}")
    R['N2']={'checkpoints':[os.path.basename(c) for c in cks]}

    print("\n"+"="*64)
    print("N3 模型是否退化为常数预测器？")
    dc=Counter(mp.values())
    print(f"   模型输出分布 {dict(dc)}  归一化熵 {entropy(dc,len(mp)):.3f}")
    for tag,rows in sets.items():
        pg=Counter(program_grade(r['wl']) for r in rows)
        print(f"   程序在 {tag} 的输出 {dict(pg)}  熵 {entropy(pg,len(rows)):.3f}")
        break
    R['N3']={'model_dist':dict(dc),'model_entropy':entropy(dc,len(mp))}
    print(f"   → 模型熵 {entropy(dc,len(mp)):.3f}：0 则完全退化")

    print("\n"+"="*64)
    print("N4 按 token 强度分层，模型是否在弱 token 层反超？")
    def tier(wl):
        t=wl or ''
        if _m.RE_COMMIT.search(t) or _m.RE_TEST.search(t) or _m.RE_BUILD.search(t) or _m.RE_DEPLOY.search(t):
            return 'strong_token'
        if (_m.RE_WRITE.search(t) and (_m.RE_FILE.search(t) or _m.RE_DIAG.search(t))): return 'med_token'
        if _m.RE_DIAG.search(t): return 'diag_only'
        return 'no_token'
    tiers={}
    print(f"   {'层':<14}{'n':>5}{'程序acc':>9}{'模型acc':>9}{'Δ':>8}{'McNemar p':>11}")
    for tg in ('strong_token','med_token','diag_only','no_token'):
        sub=[r for _,r in L2 if tier(r['wl'])==tg]
        if not sub: continue
        op=sum(1 for r in sub if program_grade(r['wl'])==r['gold'])
        b=c=0
        for r in sub:
            k=(r['pid'],r['bidx']); pm=mp[k]; pm=pm if pm in ('STRONG','MEDIUM') else 'MEDIUM'
            pp=program_grade(r['wl']); pp=pp if pp in ('STRONG','MEDIUM') else 'MEDIUM'
            om=int(pm==r['gold']); oq=int(pp==r['gold'])
            if om and not oq: b+=1
            elif oq and not om: c+=1
        am=sum(1 for r in sub if (mp[(r['pid'],r['bidx'])] if mp[(r['pid'],r['bidx'])] in ('STRONG','MEDIUM') else 'MEDIUM')==r['gold'])
        pv=mcnemar(b,c)
        tiers[tg]={'n':len(sub),'prog':op/len(sub),'model':am/len(sub),'delta':am/len(sub)-op/len(sub),'p':pv,'b':b,'c':c}
        print(f"   {tg:<14}{len(sub):>5}{op/len(sub):>9.3f}{am/len(sub):>9.3f}{am/len(sub)-op/len(sub):>+8.3f}{pv:>11.4f}")
    R['N4']=tiers
    weak=[t for t in tiers if tiers[t]['delta']>0 and tiers[t]['p']<0.05]
    print(f"   → 模型显著反超的层: {weak if weak else '无'}")

    print("\n"+"="*64)
    print("N5 二值化方式是否不公平（模型NONE被强制归MEDIUM）？")
    for mapname,fn in (('NONE→MEDIUM(现用)',lambda p:'MEDIUM' if p=='NONE' else p),
                       ('NONE→STRONG(最优上界)',lambda p:'STRONG' if p=='NONE' else p),
                       ('NONE直接弃权',None)):
        accs=[]
        for tag,rows in sets.items():
            if fn is None:
                sub=[r for r in rows if mp[(r['pid'],r['bidx'])]!='NONE']
                if not sub: continue
                a=sum(1 for r in sub if mp[(r['pid'],r['bidx'])]==r['gold'])/len(sub)
            else:
                a=sum(1 for r in rows if fn(mp[(r['pid'],r['bidx'])])==r['gold'])/len(rows)
            accs.append(a)
        print(f"   {mapname:<24} 四集均值 acc = {sum(accs)/len(accs):.3f}")
        R.setdefault('N5',{})[mapname]=sum(accs)/len(accs)
    # 程序对照
    accs=[]
    for tag,rows in sets.items():
        accs.append(sum(1 for r in rows if program_grade(r['wl'])==r['gold'])/len(rows))
    print(f"   {'程序规则(对照)':<24} 四集均值 acc = {sum(accs)/len(accs):.3f}")
    R['N5']['program']=sum(accs)/len(accs)
    json.dump(R,open('maps/audit_negative.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/audit_negative.json")

if __name__=='__main__': main()

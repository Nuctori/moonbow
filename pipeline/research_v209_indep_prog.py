# -*- coding: utf-8 -*-
"""research_v209_indep_prog.py — 用独立校准集重建程序规则（消除 N1 污染）

§258 发现：程序 token 表由评估集标注反推 → 评估集上被优待（0.589 vs 0.839）。
解法：**在训练集上重新校准 token 表**（训练集与评估集 pid 零重叠），
      再在评估集上评测。这样程序臂与模型臂同等"未见过评估集"。

做法：程序规则的形式不变（v6 的判定顺序），但**改由数据驱动地设阈值**：
  - 先用正则抽取每条文本的 token 特征（commit/test/build/deploy/write+file/diag）
  - 在训练集上学一个"哪些特征 ⇒ STRONG"的映射（而非人写死）
  - 用该映射在评估集上评测
同时对照：原人写死的 token 表（污染臂）与模型。
"""
import os, sys, json, math, re
import importlib.util as _iu
_spec=_iu.spec_from_file_location('v205','pipeline/research_v205_hybrid.py')
_m=_iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
MODEL=os.environ.get("MODEL", _m.MODEL); DEV=_m.DEV; TASK=_m.TASK; LABELS=_m.LABELS

FEATS = [
    ('f_commit', _m.RE_COMMIT),
    ('f_test',   _m.RE_TEST),
    ('f_build',  _m.RE_BUILD),
    ('f_deploy', _m.RE_DEPLOY),
    ('f_diag',   _m.RE_DIAG),
]
def feats(wl):
    t = wl or ''
    d = {k: int(bool(rx.search(t))) for k, rx in FEATS}
    d['f_write_file'] = int(bool(_m.RE_WRITE.search(t)) and bool(_m.RE_FILE.search(t)))
    d['f_file_only']  = int(bool(_m.RE_FILE.search(t)))
    d['f_n_tok'] = sum(d.values())
    d['f_len'] = min(len(t), 2000)
    return d

def load_all():
    ck=json.load(open('maps/clean_recheck_200.json',encoding='utf-8'))
    by={x['pid']:x for x in ck['items']}
    eval_rows=[]
    for tag in ('v5b_A','v5b_B','v5c_A','v5c_B'):
        d=json.load(open(f'maps/{tag}.json',encoding='utf-8'))
        rows=[]
        for x in d['items']:
            wl=by.get(x['pid'],{}).get('work_log','')
            for bi,b in enumerate(x['bindings']):
                if not _m.toks_present(b.get('tokens')) and b['grade']!='NONE': continue
                rows.append({'pid':x['pid'],'bidx':bi,'gold':_m.to3(b['grade']),'wl':wl,
                             'text':"【意图】"+(b['intent'] or '')[:300]+"\n【工作记录】"+wl[:1200]})
        eval_rows.append((tag,rows))
    return eval_rows

def main():
    eval_sets = load_all()
    tr = json.load(open('maps/strength3_train.json',encoding='utf-8'))['rows']
    # —— 在训练集上学 STRONG vs MEDIUM 的规则 ——
    # 只用训练集（与评估集 pid 零重叠，已核验）
    X=[]; y=[]
    for r in tr:
        t=r['text']; wl=t.split("【工作记录】",1)[1] if "【工作记录】" in t else ''
        if r['cls'] not in ('STRONG','MEDIUM'): continue
        X.append(feats(wl)); y.append(1 if r['cls']=='STRONG' else 0)
    n=len(X)
    print(f"[calib] 训练集二值样本 n={n}  STRONG={sum(y)} MEDIUM={n-sum(y)}")
    # 简单逻辑规则学习：对每个特征算 P(STRONG|feat=1) 与覆盖
    print(f"\n[calib] 特征统计（训练集）")
    print(f"   {'特征':<16}{'覆盖':>6}{'P(STRONG|1)':>13}{'P(STRONG|0)':>13}")
    stats={}
    for k in X[0]:
        if k=='f_len': continue
        c1=sum(1 for xi,yi in zip(X,y) if xi[k]==1)
        p1=(sum(1 for xi,yi in zip(X,y) if xi[k]==1 and yi==1)/c1) if c1 else float('nan')
        c0=n-c1
        p0=(sum(1 for xi,yi in zip(X,y) if xi[k]==0 and yi==1)/c0) if c0 else float('nan')
        stats[k]={'n1':c1,'p1':p1,'p0':p0}
        print(f"   {k:<16}{c1:>6}{p1:>13.3f}{p0:>13.3f}")
    # 数据驱动规则：选 p1 最高且覆盖>=20 的单一特征作为 STRONG 判据
    cand=[(k,v) for k,v in stats.items() if v['n1']>=20]
    cand.sort(key=lambda kv:-kv[1]['p1'])
    best=cand[0][0] if cand else 'f_write_file'
    thr=stats[best]['p1']
    print(f"\n[calib] 选定规则: {best} == 1 ⇒ STRONG (训练集上 P(STRONG|1)={thr:.3f})")
    # 也学一个"无 token ⇒ NONE"的机械闸（沿用 v6 结构，但阈值数据驱动）
    print(f"\n[eval] 三臂对照")
    print(f"   {'标注集':<9}{'n':>5}{'污染程序':>10}{'独立程序':>10}{'模型':>9}")
    res={}
    for tag,rows in eval_sets:
        # 污染臂：人写死 token 表
        poll=[_m.program_grade(r['wl']) for r in rows]
        a_poll=sum(1 for r,p in zip(rows,poll) if p==r['gold'])/len(rows)
        # 独立臂：训练集学到的规则；无任何特征 ⇒ NONE
        indep=[]
        for r in rows:
            f=feats(r['wl'])
            if f['f_n_tok']==0: indep.append('NONE')
            elif f[best]==1: indep.append('STRONG')
            else: indep.append('MEDIUM')
        a_ind=sum(1 for r,p in zip(rows,indep) if p==r['gold'])/len(rows)
        # 模型臂
        accs=[]; 
        for r in rows:
            pass
        res[tag]={'n':len(rows),'contaminated':a_poll,'independent':a_ind}
        print(f"   {tag:<9}{len(rows):>5}{a_poll:>10.3f}{a_ind:>10.3f}{'':>9}")
    # 训练集上三臂自检（独立臂应≈污染臂的"训练集水平"）
    print(f"\n[check] 程序两臂在训练集上的表现（独立源）")
    okp=oki=0; nt=0
    for r in tr:
        t=r['text']; wl=t.split("【工作记录】",1)[1] if "【工作记录】" in t else ''
        f=feats(wl)
        pg=_m.program_grade(wl)
        ig='NONE' if f['f_n_tok']==0 else ('STRONG' if f[best]==1 else 'MEDIUM')
        nt+=1; okp+=int(pg==r['cls']); oki+=int(ig==r['cls'])
    print(f"   污染臂 {okp/nt:.3f}  独立臂 {oki/nt:.3f}  (n={nt})")
    print(f"   → 独立臂在训练集上更高说明它更少过拟合训练集? 反之说明污染臂拟合了评估集")
    json.dump({'best_feature':best,'thr':thr,'stats':{k:{kk:(None if isinstance(vv,float) and math.isnan(vv) else vv) for kk,vv in v.items()} for k,v in stats.items()},
               'eval':res,'train_contaminated':okp/nt,'train_independent':oki/nt,'train_n':nt},
              open('maps/indep_prog_result.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("\n-> maps/indep_prog_result.json")

if __name__=='__main__': main()

# -*- coding: utf-8 -*-
"""research_v211_epoch_curve.py — 补 N2（模型是否欠拟合）：检查点间性能曲线

§258 的 N2 假设（模型欠拟合，训练在 epoch 2.9/8 中断）未排除。
本脚本直接测各检查点的评估性能，若已收敛则 N2 排除，当前对照有效。
"""
import os, sys, json
sys.path.insert(0,'pipeline')
import importlib.util as _iu
_spec=_iu.spec_from_file_location('v205','pipeline/research_v205_hybrid.py')
_m=_iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
TASK=_m.TASK; LABELS=_m.LABELS
DEV=os.environ.get("MAP_LOC","xpu")

def load():
    ck=json.load(open('maps/clean_recheck_200.json',encoding='utf-8'))
    by={x['pid']:x for x in ck['items']}
    out={}
    for tag in ('v5b_A','v5c_A'):
        d=json.load(open(f'maps/{tag}.json',encoding='utf-8'))
        rows=[]
        for x in d['items']:
            wl=by.get(x['pid'],{}).get('work_log','')
            for b in x['bindings']:
                if not _m.toks_present(b.get('tokens')) and b['grade']!='NONE': continue
                rows.append({'gold':_m.to3(b['grade']),
                             'text':"【意图】"+(b['intent'] or '')[:300]+"\n【工作记录】"+wl[:1200]})
        out[tag]=rows
    return out

def main():
    import glob
    from gliner2 import AutoExtractor, Schema
    sets=load()
    cks=sorted(glob.glob('models/gliner25_strength3_v1/checkpoint-epoch-*'))
    print(f"检查点: {[os.path.basename(c) for c in cks]}\n")
    print(f"{'检查点':<22}{'v5b_A acc':>11}{'v5c_A acc':>11}{'均值':>9}")
    res={}
    for c in cks:
        m=AutoExtractor.from_pretrained(c,map_location=DEV)
        sc=Schema().classification(TASK,labels=LABELS)
        accs=[]
        for tag,rows in sets.items():
            ok=0
            for r in rows:
                try:
                    g=m.extract(r['text'],sc).get(TASK); g=g if g in LABELS else 'MEDIUM'
                except Exception: g='MEDIUM'
                ok+=int(g==r['gold'])
            accs.append(ok/len(rows))
        res[os.path.basename(c)]={'v5b_A':accs[0],'v5c_A':accs[1],'mean':sum(accs)/2}
        print(f"{os.path.basename(c):<22}{accs[0]:>11.3f}{accs[1]:>11.3f}{sum(accs)/2:>9.3f}")
        del m
        try:
            import torch
            if torch.xpu.is_available(): torch.xpu.empty_cache()
        except Exception: pass
    ks=sorted(res)
    if len(ks)>=2:
        d=res[ks[-1]]['mean']-res[ks[0]]['mean']
        print(f"\n末检查点 − 首检查点 = {d:+.4f}")
        print(f"→ {'仍显著上升，欠拟合未排除' if d>0.02 else '已收敛（增幅 <2pp）⇒ §258 的 N2 假设可排除'}")
    json.dump(res,open('maps/epoch_curve.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("-> maps/epoch_curve.json")

if __name__=='__main__': main()

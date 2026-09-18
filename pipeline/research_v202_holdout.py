# -*- coding: utf-8 -*-
"""research_v202_holdout.py — 独立复验：漂移剔除后的 held-out 评估

v189/v200/v201 的评估集是 clean_recheck_200（v5b 标注），理论上 v5b_A/B 是标注批，
而训练用的是 v5b_label_train_*（3 批共 1138）。**需确认评估集未混入训练集**，
否则 exact 0.780 是记忆而非泛化。
本脚本：
  1) 核验评估 pid 与训练 pid 是否重叠（泄漏检查）
  2) 若重叠则剔除，在纯 held-out 上重算主口径
"""
import json
from collections import Counter
LABELS=["STRONG","MEDIUM","WEAK","NONE"]; ORD={g:i for i,g in enumerate(LABELS)}

def main():
    # 训练 pid
    train_pid=set()
    for f in ('maps/v5b_label_train_1.json','maps/v5b_label_train_2.json','maps/v5b_label_train_3.json'):
        d=json.load(open(f,encoding='utf-8'))
        for x in d['items']: train_pid.add(x['pid'])
    # 评估 pid
    ev=json.load(open('maps/v5b_A.json',encoding='utf-8'))
    eval_pid=set(x['pid'] for x in ev['items'])
    print(f"[leak check] 训练 pid {len(train_pid)} | 评估 pid {len(eval_pid)}")
    ov=train_pid & eval_pid
    print(f"[leak check] 重叠 {len(ov)}  ({len(ov)/len(eval_pid):.3f})")
    # 训练用的是 v5b_label_train_*，其 pid 格式与评估一致？
    tp=sorted(train_pid)[:2]; ep=sorted(eval_pid)[:2]
    print(f"   训练 pid 样例 {tp}")
    print(f"   评估 pid 样例 {ep}")
    # 也检查文本级重叠（更严格）
    train_txt=set()
    for f in ('maps/gliner25_strength_train.json',):
        d=json.load(open(f,encoding='utf-8'))
        for r in d['rows']:
            # 训练文本格式：【意图】...\n【工作记录】...
            train_txt.add(r['text'][:120])
    ck=json.load(open('maps/clean_recheck_200.json',encoding='utf-8'))
    by={x['pid']:x for x in ck['items']}
    eval_txt=set()
    for x in ev['items']:
        for b in x['bindings']:
            wl=by.get(x['pid'],{}).get('work_log','')
            eval_txt.add((f"【意图】{b['intent'][:300]}\n【工作记录】{wl[:1200]}")[:120])
    tov=train_txt & eval_txt
    print(f"[leak check] 文本前120字符重叠 {len(tov)} / {len(eval_txt)} = {len(tov)/max(1,len(eval_txt)):.3f}")
    if tov:
        print("   ⚠️ 存在文本级重叠 —— exact 0.780 可能含记忆成分")
        for s in list(tov)[:3]: print("     ", s[:80].replace('\n','|'))
    else:
        print("   ✓ 无文本级重叠 —— 评估为纯 held-out")
    json.dump({'train_pid':len(train_pid),'eval_pid':len(eval_pid),
               'pid_overlap':len(ov),'text_overlap':len(tov),
               'text_overlap_rate':len(tov)/max(1,len(eval_txt))},
              open('maps/holdout_check.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print("-> maps/holdout_check.json")

if __name__=='__main__': main()

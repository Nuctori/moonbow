# -*- coding: utf-8 -*-
"""research_v203_rebuild3.py — 按三值判据 v6 重建训练/评估数据

依据 maps/COVERAGE_CRITERION_v6_3class.md：
  1) 剔除 91 条判据违规（无 token 却非 NONE）—— §249
  2) WEAK 并入 MEDIUM —— §250（WEAK 无跨批证据）
  3) 输出三值数据 + 各档跨批证据表（新增验收第 6 条）
"""
import json
from collections import Counter

LABELS3 = ["STRONG", "MEDIUM", "NONE"]

def load_clean():
    """复用 v199 的清洗逻辑，返回带 batch 标注的样本。"""
    rows = []
    for tag, f in (('b1','maps/v5b_label_train_1.json'),
                   ('b2','maps/v5b_label_train_2.json'),
                   ('b3','maps/v5b_label_train_3.json')):
        d = json.load(open(f, encoding='utf-8'))
        for x in d['items']:
            for b in x['bindings']:
                tk = str(b.get('tokens') or '').strip()
                has = not (tk.upper() in ('NONE','','NONE.','N/A','无') or len(tk) <= 3)
                rows.append({'batch': tag, 'pid': x['pid'], 'intent': b['intent'],
                             'grade': b['grade'], 'tokens': b.get('tokens'), 'has_tok': has})
    return rows

def to3(grade):
    return 'MEDIUM' if grade in ('WEAK','MEDIUM') else grade

def main():
    rows = load_clean()
    n0 = len(rows)
    # 1) 剔除判据违规
    viol = [r for r in rows if not r['has_tok'] and r['grade'] != 'NONE']
    clean = [r for r in rows if not (not r['has_tok'] and r['grade'] != 'NONE')]
    # 2) WEAK → MEDIUM
    for r in clean:
        r['grade3'] = to3(r['grade'])
    print(f"[in]     {n0}")
    print(f"[drop]   {len(viol)} 判据违规 (§249)")
    print(f"[merge]  WEAK -> MEDIUM ({sum(1 for r in clean if r['grade']=='WEAK')} 条)")
    print(f"[out]    {len(clean)}  分布 {dict(Counter(r['grade3'] for r in clean))}")
    # 3) 跨批证据表（v6 验收第 6 条）
    print("\n[跨批证据] 每档在各批的出现次数（要求：每档 >=2 批）")
    ev = {}
    for g in LABELS3:
        per = {tag: sum(1 for r in clean if r['batch']==tag and r['grade3']==g) for tag in ('b1','b2','b3')}
        nbatch = sum(1 for v in per.values() if v > 0)
        ev[g] = {'per_batch': per, 'n_batch': nbatch, 'cross_batch': nbatch >= 2}
        flag = "✓" if nbatch >= 2 else "✗ 无跨批证据！"
        print(f"   {g:7s} {per}  批数={nbatch}  {flag}")
    allok = all(v['cross_batch'] for v in ev.values())
    print(f"\n[gate-6] 全部档位有跨批证据: {'PASS' if allok else 'FAIL'}")
    # 4) 构建训练文本
    td = json.load(open('maps/v5b_train_data.json', encoding='utf-8'))
    wl_map = td['work_log']          # pid -> work_log
    txt_idx = {(r['pid'], (r.get('intent') or '')[:80]): r for r in td['rows']}
    built = []; miss = 0
    for r in clean:
        # 三批的 pid 与 v5b_train_data 的 pid 同一命名空间；先精确匹配 intent，
        # 失败则退化为同 pid 下唯一 intent
        cands = [v for (p, i), v in txt_idx.items() if p == r['pid']]
        hit = None
        for c in cands:
            if (c.get('intent') or '')[:60] == (r['intent'] or '')[:60]:
                hit = c; break
        if hit is None and len(cands) == 1:
            hit = cands[0]
        if hit is None:
            miss += 1; continue
        wl = wl_map.get(r['pid'], '')
        if not wl:
            miss += 1; continue
        txt = "\u3010\u610f\u56fe\u3011" + (r['intent'] or '')[:300] + "\n\u3010\u5de5\u4f5c\u8bb0\u5f55\u3011" + wl[:1200]
        built.append({'text': txt, 'cls': r['grade3']})
    print(f"\n[build] 训练行 {len(built)} (未匹配 {miss})")
    print(f"        分布 {dict(Counter(b['cls'] for b in built).most_common())}")
    json.dump({'n': len(built), 'rows': built, 'criterion': 'COVERAGE_CRITERION_v6_3class'},
              open('maps/strength3_train.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'n': n0, 'dropped_drift': len(viol), 'merged_weak': sum(1 for r in clean if r['grade']=='WEAK'),
               'n_clean': len(clean), 'dist3': dict(Counter(r['grade3'] for r in clean)),
               'cross_batch_evidence': ev, 'gate6_pass': allok},
              open('maps/strength3_evidence.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print("-> maps/strength3_train.json / maps/strength3_evidence.json")

if __name__ == '__main__': main()

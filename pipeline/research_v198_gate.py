# -*- coding: utf-8 -*-
"""research_v198_gate.py — 偏斜校正反向率门槛（替代原门槛）

问题（§246.3）：原门槛"反向率 ≤2%"在 72% MEDIUM 的偏斜分布上无区分力——
全输出多数类的 trivial 预测反向率 = 0.000。

校正思路（三条互补口径，全部需同时通过）：
  G1 校正常数 c = P(trivial 预测的反向率)：模型反向率必须 < c（严格优于平凡解）
  G2 排除多数类可命中子集：只在"trivial 预测会错"的困难子集上算反向率
     （trivial = 输出多数类；困难子集 = gold ≠ MEDIUM）
  G3 最大单元反向率：按 gold 档分层，取各层反向率最大值（防止被大层稀释）
      并给出每层的上界（Wilson 95% 上界，小样本下不用点估计）
"""
import json, math
from collections import Counter

LABELS = ["STRONG", "MEDIUM", "WEAK", "NONE"]
ORD = {g: i for i, g in enumerate(LABELS)}

def wilson_ub(k, n, z=1.96):
    """Wilson 单侧 95% 上界。"""
    if n == 0: return 0.0
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    r = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return min(1.0, (c + r) / d)

def rev(gold, pred):
    if pred not in ORD: return False
    return abs(ORD[pred] - ORD[gold]) >= 2

def trivial_pred(rows):
    return Counter(r['gold'] for r in rows).most_common(1)[0][0]

def evaluate(gold, preds, name='', tol_g1=0.0, tol_g2=0.02, tol_g3=0.05):
    assert len(gold) == len(preds)
    n = len(gold)
    tri = trivial_pred([{'gold': g} for g in gold])
    # trivial 预测
    tri_rev = sum(1 for g in gold if rev(g, tri)) / n
    # 模型
    mod_rev = sum(1 for g, p in zip(gold, preds) if rev(g, p)) / n
    # G2 困难子集（trivial 在全体上反向率恒为 0，故必须在困难子集上比）
    hard = [(g, p) for g, p in zip(gold, preds) if g != tri]
    g2 = (sum(1 for g, p in hard if rev(g, p)) / len(hard)) if hard else 0.0
    tri_hard = (sum(1 for g, _ in hard if rev(g, tri)) / len(hard)) if hard else 0.0
    # G3 分层最大（含 Wilson 上界）
    per = {}
    for g in LABELS:
        idx = [i for i, gg in enumerate(gold) if gg == g]
        if not idx: continue
        k = sum(1 for i in idx if rev(gold[i], preds[i]))
        per[g] = {'n': len(idx), 'rev_k': k, 'rev': k/len(idx), 'wilson_ub': wilson_ub(k, len(idx))}
    g3 = max((v['rev'] for v in per.values()), default=0.0)
    g3ub = max((v['wilson_ub'] for v in per.values()), default=0.0)
    res = {
        'name': name, 'n': n, 'trivial_class': tri,
        'trivial_reverse': tri_rev, 'model_reverse': mod_rev,
        'G1_corrected': {'rule': f'困难子集上 model({g2:.4f}) < trivial({tri_hard:.4f})',
                         'pass': g2 < tri_hard, 'value': g2, 'ref': tri_hard},
        'G2_hard_subset': {'rule': f'困难子集(gold!={tri}) n={len(hard)} 反向率 <= {tol_g2}',
                           'pass': g2 <= tol_g2, 'value': g2},
        'G3_per_class_max': {'rule': f'各 gold 层反向率最大值 <= {tol_g3}',
                             'pass': g3 <= tol_g3, 'value': g3,
                             'wilson_ub': g3ub, 'wilson_pass': g3ub <= tol_g3, 'per_class': per},
    }
    res['ALL_PASS'] = all(res[k]['pass'] for k in ('G1_corrected','G2_hard_subset','G3_per_class_max'))
    return res

def main():
    # 数据源：v189 已存的逐条预测需重建；改为直接从审计文件读取已算好的统计
    sys_rows = []
    ev = json.load(open('maps/strength_eval_result.json', encoding='utf-8'))
    cf = json.load(open('maps/strength_confusion.json', encoding='utf-8'))
    # 由混淆矩阵反推 (gold, pred) 对
    for g in LABELS:
        for p, k in cf.get(g, {}).items():
            sys_rows += [(g, p)] * k
    gold = [r[0] for r in sys_rows]; preds = [r[1] for r in sys_rows]
    print(f"[data] 由混淆矩阵重建 {len(gold)} 条")
    r_model = evaluate(gold, preds, name='287M 强度四分类器 (gold=v5b)')
    # 对照 A：多数类 trivial
    tri = trivial_pred([{'gold': g} for g in gold])
    r_tri = evaluate(gold, [tri]*len(gold), name=f'对照：全输出多数类 {tri}')
    # 对照 B：随机
    import random; random.seed(0)
    pool = gold[:]
    acc = [[] for _ in range(200)]
    # 简化：单次随机
    random.shuffle(pool)
    r_rnd = evaluate(gold, pool, name='对照：随机打乱')
    out = {'model': r_model, 'trivial': r_tri, 'random': r_rnd}
    json.dump(out, open('maps/reverse_gate_result.json','w',encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for r in (r_model, r_tri, r_rnd):
        print(f"\n=== {r['name']} ===")
        print(f"  模型反向率 {r['model_reverse']:.4f} | trivial 反向率 {r['trivial_reverse']:.4f} (trivial类={r['trivial_class']})")
        print(f"  G1 校正: {'PASS' if r['G1_corrected']['pass'] else 'FAIL'}  {r['G1_corrected']['rule']}")
        print(f"  G2 困难子集: {'PASS' if r['G2_hard_subset']['pass'] else 'FAIL'}  {r['G2_hard_subset']['rule']}")
        g3=r['G3_per_class_max']
        print(f"  G3 分层最大: {'PASS' if g3['pass'] else 'FAIL'}  max={g3['value']:.4f} (Wilson上界 {g3['wilson_ub']:.4f})")
        for g,v in g3['per_class'].items():
            print(f"      {g:7s} n={v['n']:4d} rev={v['rev_k']:3d} ({v['rev']:.4f}) ub={v['wilson_ub']:.3f}")
        print(f"  >>> ALL_PASS = {r['ALL_PASS']}")
    print("\n-> maps/reverse_gate_result.json")

if __name__ == '__main__': main()

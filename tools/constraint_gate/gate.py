#!/usr/bin/env python3
"""gate.py — 约束守卫 CLI（tracker + registry + 提醒块生成）。

产出提醒块（不阻断，裁决权在用户）：
  python gate.py --repo D:/react/app --commit <sha>
  python gate.py --repo D:/react/app --since "2026-10-02 04:00" --until "2026-10-02 09:00" --claim "零关键词表"

--claim 提供终报宣称时，进入 D3 对账模式：检测结果与宣称矛盾 → 输出 CONFLICT 对账块。
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tracker import get_diff, scan, load_legacy_symbols, RE_SHA, RE_DATE  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--commit'); ap.add_argument('--diff')
    ap.add_argument('--since'); ap.add_argument('--until')
    ap.add_argument('--claim', help='终报合规宣称（触发 D3 对账）')
    a = ap.parse_args()
    if a.commit and not RE_SHA.match(a.commit):
        print(json.dumps(dict(error='illegal commit ref'))); return 1
    if a.since and not (RE_DATE.match(a.since) and (a.until is None or RE_DATE.match(a.until))):
        print(json.dumps(dict(error='illegal date range'))); return 1
    diff = get_diff(a.repo, a.commit, a.diff, a.since, a.until)
    legacy = load_legacy_symbols(os.path.join(HERE, 'constraints.yaml'))
    findings = scan(diff, legacy_symbols=legacy)

    block = {
        'type': 'constraint-reminder',          # 提醒块：注入会话，不拦截
        'policy': 'remind-only',
        'finding_count': len(findings),
        'findings': findings,
        'note': 'tracker 为机械召回层，findings≠违规判定；语义裁决走描述层（T3），裁决权在用户。',
    }
    if a.claim is not None:
        conflict = (len(findings) > 0 and ('零' in a.claim or '没有' in a.claim or '无' in a.claim))
        block['reconciliation'] = {
            'claim': a.claim,
            'mechanical_evidence_count': len(findings),
            'verdict': 'CONFLICT' if conflict else 'consistent',
            'user_action_required': bool(conflict),
        }
    print(json.dumps(block, ensure_ascii=False, indent=1))
    return 0

if __name__ == '__main__':
    sys.exit(main())

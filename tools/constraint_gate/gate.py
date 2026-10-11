#!/usr/bin/env python3
"""gate.py — 约束守卫 CLI（tracker 机械召回 + 事实提醒块生成）。

角色边界（第二轮审计 D8 修正）：
  tracker 是**机械召回层**，输出 findings（带 span 的事实候选）。
  它不裁决合规性，因此也**不产生 CONFLICT/consistent 结论**——那需要
  当前有效约束的语义裁决（T3 SLM/LLM），机械层没有也不得臆造该能力。
  原先"命中特征 + 宣称含'零/没有/无'"即输出 CONFLICT 的写法已删除：
  那是特征命中加关键词判定，即便不阻断也越过了裁决权边界。

本 CLI 现在只做两件事：
  1. 产出台账事实（findings 列表 + 引用 span），标注 needs_semantic_review；
  2. 在传入 --claim 时，把宣称与机械证据**并列**呈现，交由语义层/用户裁决，
     自身不下结论（reconciliation.verdict 恒为 "requires_semantic_adjudication"）。

用法：
  python gate.py --repo REDACTED_REPO --commit <sha>
  python gate.py --repo REDACTED_REPO --since "2026-10-02 04:00" \
      --until "2026-10-02 09:00" --claim "零关键词表"

退出码：0 = 正常产出（findings 不是违规判定，故不用于拦截）；2 = 入参非法。
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tracker import (get_diff, scan, load_legacy_symbols,  # noqa: E402
                     RE_SHA, RE_DATE)


def build_block(findings, claim=None, constraints_path=None):
    block = {
        'type': 'constraint-reminder',      # 提醒块：注入会话，不拦截
        'policy': 'remind-only',
        'layer': 'mechanical-recall',       # 明确层身份：召回，非裁决
        'finding_count': len(findings),
        'findings': findings,
        'needs_semantic_review': bool(findings),
        'note': ('tracker 为机械召回层，findings 只是候选事实（≠违规判定）；'
                 '合规性裁决须由描述层 T3（SLM/LLM）依据当前有效约束完成，'
                 '机械层不输出 CONFLICT/consistent 结论，裁决权在用户。'),
    }
    if claim is not None:
        block['reconciliation'] = {
            'claim': claim,
            'mechanical_evidence_count': len(findings),
            # 机械层不做结论：只把两边事实摆出来，等语义裁决
            'verdict': 'requires_semantic_adjudication',
            'semantic_backend': None,       # 由调用方注入 T3 后端后填写
            'user_action_required': bool(findings),
        }
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--commit'); ap.add_argument('--diff')
    ap.add_argument('--since'); ap.add_argument('--until')
    ap.add_argument('--claim', help='终报合规宣称（与机械证据并列呈现，不自动判冲突）')
    a = ap.parse_args()
    if a.commit and not RE_SHA.match(a.commit):
        print(json.dumps(dict(error='illegal commit ref'))); return 2
    if a.since and not (RE_DATE.match(a.since)
                        and (a.until is None or RE_DATE.match(a.until))):
        print(json.dumps(dict(error='illegal date range'))); return 2
    diff = get_diff(a.repo, a.commit, a.diff, a.since, a.until)
    constraints_path = os.path.join(HERE, 'constraints.yaml')
    legacy = load_legacy_symbols(constraints_path)
    findings = scan(diff, legacy_symbols=legacy)
    print(json.dumps(build_block(findings, a.claim, constraints_path),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

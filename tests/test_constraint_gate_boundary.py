# -*- coding: utf-8 -*-
"""tests/test_constraint_gate_boundary.py — 机械层边界回归（无模型依赖）。

第二轮审计 D8：gate 曾用"特征命中 + 宣称关键词"输出 CONFLICT，属机械层越权
裁决。本测试锁定修正后的边界：机械层只报事实，不产出合规结论。
"""
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "constraint_gate"
sys.path.insert(0, str(TOOLS))

import gate  # noqa: E402
import tracker  # noqa: E402


def test_gate_never_emits_mechanical_conflict_verdict():
    """宣称含"零/无"且命中特征时，机械层也不得自行判 CONFLICT。"""
    findings = [dict(rule='wordlist-identifier', file='a.ts', line=3,
                     span='ALIAS_TABLE')]
    block = gate.build_block(findings, claim="零关键词表")
    rec = block['reconciliation']
    assert rec['verdict'] == 'requires_semantic_adjudication'
    assert rec['semantic_backend'] is None
    assert rec['mechanical_evidence_count'] == 1
    assert block['layer'] == 'mechanical-recall'
    assert block['needs_semantic_review'] is True
    # 结论字段里不得出现机械层自判的 CONFLICT
    assert rec['verdict'] not in ('CONFLICT', 'consistent')
    assert 'CONFLICT' not in rec['verdict']


def test_gate_block_is_injectable_reminder_not_blocker():
    block = gate.build_block([], claim=None)
    assert block['type'] == 'constraint-reminder'
    assert block['policy'] == 'remind-only'
    assert block['finding_count'] == 0
    assert block['needs_semantic_review'] is False


def test_tracker_reports_spans_without_verdict_field():
    diff = ("+++ b/src/x.ts\n@@ -0,0 +1,4 @@\n"
            "+const ALIAS_TABLE = { a: '甲', b: '乙', c: '丙' };\n"
            "+// 词表：关键词表\n"
            "+const s1 = '出发前再留意天气';\n"
            "+const s2 = '先围绕主题来找';\n")
    findings = tracker.scan(diff)
    assert findings, "机械层应召回到候选事实"
    for f in findings:
        assert 'verdict' not in f and 'is_violation' not in f
        assert f['rule'] and f['file'] and f['line']
        assert 'span' in f


def test_tracker_input_validation_rejects_illegal_refs():
    with pytest.raises(ValueError, match='illegal commit ref'):
        tracker.get_diff(repo='.', commit='main; rm -rf /')
    with pytest.raises(ValueError, match='illegal date range'):
        tracker.get_diff(repo='.', since='$(whoami)')

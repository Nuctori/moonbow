# -*- coding: utf-8 -*-
"""tools/intent_eval/assemble_gen_data.py — 汇编 subagent 生成素材为带标签数据集。

分工原则：subagent 只产**内容**（主题变体/任务包），本脚本按**构造规则机械
打标签**，并做确定性验收（schema/枚举/去重/泄漏检测）。标签不经过任何
模型的自由判断——这是合成数据可信度的来源。

展开规则：
T3（每主题 3 条）：
  V@card        → violation （violation_variant 配它自然违反的约束）
  C@card        → compliant （compliant_variant 配同一约束）
  V@other(card) → compliant （violation_variant 配无关约束 → 约束决定标签）
  其中 V@card 与 V@other 构成**敏感度对**（同文本、约束不同、标签相反）。
T4（每 task 5 条）：
  full   = goal_single + pass 证据
  partial= goal_single+extra + pass 证据（只覆盖第一条）
  none_e = goal_single + 空证据
  none_f = goal_single + fail 证据
  unclear= goal_single + 无关领域证据
域分割：按 domain 稳定哈希分 train/eval 两半（同域不跨集，防近重泄漏）。

用法：python tools/intent_eval/assemble_gen_data.py [--t3 gen/t3_themes_b1.json] [--t4 gen/t4_tasks_b1.json]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
OUT = REPO / "data" / "intent_eval"

from train_intent_lora import leak_check, load_jsonl  # noqa: E402

C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
OTHER = {"mech": "test", "tmpl": "test", "anti": "silent", "silent": "test"}


def cc(key):
    return [{"entity_id": "C1", "quote": C[key]}]


def stable_half(s: str) -> str:
    """domain → train/eval 稳定分割（同域必同侧）。"""
    return "train" if sum(s.encode("utf-8")) % 2 == 0 else "eval"


def validate_t3(data) -> list:
    themes = data.get("themes")
    assert isinstance(themes, list) and themes, "themes 缺失"
    ids, domains = set(), set()
    for t in themes:
        for k in ("topic_id", "domain", "card", "violation_variant",
                  "compliant_variant", "compliant_pattern", "rationale"):
            assert t.get(k), f"字段缺失 {k}: {t.get('topic_id')}"
        assert t["card"] in ("mech", "tmpl", "anti", "silent"), t["card"]
        assert t["compliant_pattern"] in ("recall", "delete", "allowed")
        assert t["topic_id"] not in ids, f"topic_id 重复 {t['topic_id']}"
        ids.add(t["topic_id"])
        domains.add(t["domain"])
        v, c = t["violation_variant"], t["compliant_variant"]
        assert 10 <= len(v) <= 200 and 10 <= len(c) <= 200, "变体长度越界"
        assert v != c, "两变体相同"
        for bad in ("仅用于日志", "仅候选", "由模型判定"):
            assert bad not in v, f"violation 变体含边界话术 {bad}"
    return themes


def validate_t4(data) -> list:
    tasks = data.get("tasks")
    assert isinstance(tasks, list) and tasks, "tasks 缺失"
    ids, domains = set(), set()
    for t in tasks:
        for k in ("task_id", "domain", "goal_single", "goal_extra",
                  "evidence_kind", "pass_note", "fail_note"):
            assert t.get(k), f"字段缺失 {k}: {t.get('task_id')}"
        assert t["evidence_kind"] in ("test_run", "artifact", "diff",
                                      "tool_output"), t["evidence_kind"]
        assert t["task_id"] not in ids
        ids.add(t["task_id"])
        domains.add(t["domain"])
        toks = _nouns(t["goal_single"])
        assert _contains_any(t["pass_note"], toks), \
            f"pass_note 未含任务名词: {t['task_id']} toks={toks[:3]}"
        assert _contains_any(t["fail_note"], toks), \
            f"fail_note 未含任务名词: {t['task_id']} toks={toks[:3]}"
    return tasks


def _nouns(goal):
    """goal 的全部 2 字窗口（主题词兜底检查用——标签不依赖此检查）。"""
    import re
    g = re.sub(r"[^\w\u4e00-\u9fff]+", "", goal)
    stop = {"的问题", "问题"}
    return list({g[i:i + 2] for i in range(len(g) - 1)}) or [goal[:2]]


def _contains_any(note, tokens):
    low = (note or "").lower()
    return any(t.lower() in low for t in tokens)


def expand_t3(themes):
    rows = []
    for t in themes:
        half = stable_half(t["domain"])
        base = {"kind": "commit", "scope": "s", "domain": t["domain"]}
        rows.append({**base, "case_id": f"g3_{t['topic_id']}_vc",
                     "event_text": t["violation_variant"],
                     "constraints": cc(t["card"]), "label": "violation",
                     "split": half, "pair": t["topic_id"]})
        rows.append({**base, "case_id": f"g3_{t['topic_id']}_cc",
                     "event_text": t["compliant_variant"],
                     "constraints": cc(t["card"]), "label": "compliant",
                     "split": half, "pair": t["topic_id"]})
        rows.append({**base, "case_id": f"g3_{t['topic_id']}_vx",
                     "event_text": t["violation_variant"],
                     "constraints": cc(OTHER[t["card"]]), "label": "compliant",
                     "split": half, "pair": t["topic_id"] + "_x"})
    return rows


def expand_t4(tasks):
    rows = []
    for i, t in enumerate(tasks):
        half = stable_half(t["domain"])
        noun = t["goal_single"]
        g1, g2 = noun, f"{noun}；{t['goal_extra']}"
        done = f"{noun}已完成"
        ev_pass = [{"evidence_id": "E1", "kind": t["evidence_kind"],
                    "result": "pass", "note": t["pass_note"]}]
        ev_fail = [{"evidence_id": "E1", "kind": t["evidence_kind"],
                    "result": "fail", "note": t["fail_note"]}]
        ev_unrel = [{"evidence_id": "E1", "kind": "test_run",
                     "result": "pass", "note": "无关主题的用例全部通过"}]
        common = {"domain": t["domain"], "goals_all": [], "split": half}
        rows.append({**common, "case_id": f"g4_{t['task_id']}_full",
                     "goal_text": g1, "event_text": done,
                     "evidence": ev_pass, "label": "supported_full"})
        rows.append({**common, "case_id": f"g4_{t['task_id']}_part",
                     "goal_text": g2, "event_text": done,
                     "evidence": ev_pass, "label": "supported_partial"})
        rows.append({**common, "case_id": f"g4_{t['task_id']}_ne",
                     "goal_text": g1, "event_text": f"{noun}已全部搞定",
                     "evidence": [], "label": "no_evidence"})
        rows.append({**common, "case_id": f"g4_{t['task_id']}_nf",
                     "goal_text": g1, "event_text": done,
                     "evidence": ev_fail, "label": "no_evidence"})
        rows.append({**common, "case_id": f"g4_{t['task_id']}_un",
                     "goal_text": g1,
                     "event_text": f"{t['domain']}界面的无关样式已调整",
                     "evidence": ev_unrel, "label": "insufficient_info"})
    return rows


def t3_public(r):
    return {"case_id": r["case_id"], "event_text": r["event_text"],
            "constraints": r["constraints"], "label": r["label"],
            "kind": r["kind"], "scope": r["scope"]}


def t4_public(r):
    return {"case_id": r["case_id"], "goal_text": r["goal_text"],
            "event_text": r["event_text"], "evidence": r["evidence"],
            "label": r["label"], "goals_all": r["goals_all"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3", default=str(HERE / "gen" / "t3_themes_b1.json"))
    ap.add_argument("--t4", default=str(HERE / "gen" / "t4_tasks_b1.json"))
    ap.add_argument("--t3-train", default="train_t3_v3.jsonl")
    ap.add_argument("--t3-eval", default="eval_t3_v3.jsonl")
    ap.add_argument("--t4-train", default="train_t4_v3.jsonl")
    ap.add_argument("--t4-eval", default="eval_t4_v3.jsonl")
    ap.add_argument("--out-suffix", default="v4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ── T3 ──────────────────────────────────────────────────────────────
    tr3 = load_jsonl(OUT / a.t3_train)
    ev3 = load_jsonl(OUT / a.t3_eval)
    sens = load_jsonl(OUT / "sensitivity_pairs.jsonl")
    if Path(a.t3).exists():
        themes = validate_t3(json.loads(Path(a.t3).read_text(encoding="utf-8")))
        rows = expand_t3(themes)
        g_train = [t3_public(r) for r in rows if r["split"] == "train"]
        g_eval = [t3_public(r) for r in rows if r["split"] == "eval"]
        # 敏感度对：同文本 V@card 与 V@other
        by_text = {}
        for r in rows:
            by_text.setdefault(r["event_text"], []).append(r["case_id"])
        for text, members in by_text.items():
            if len(members) >= 2:
                sens.append({"event_text": text, "members": members,
                             "labels": ["compliant", "violation"]})
        print(f"T3 生成：{len(themes)} 主题 → {len(rows)} 样本 "
              f"(train {len(g_train)} / eval {len(g_eval)})")
        tr3 = tr3 + g_train
        ev3 = ev3 + g_eval
    # 泄漏检测（train vs eval 全量）
    exact, near = leak_check(tr3, ev3)
    if exact or near:
        print(f"FATAL T3 泄漏：exact={exact[:5]} near={near[:5]}", file=sys.stderr)
        return 3
    print(f"T3 合并：train={len(tr3)} eval={len(ev3)} 敏感度对={len(sens)}")

    # ── T4 ──────────────────────────────────────────────────────────────
    tr4 = load_jsonl(OUT / a.t4_train)
    ev4 = load_jsonl(OUT / a.t4_eval)
    if Path(a.t4).exists():
        tasks = validate_t4(json.loads(Path(a.t4).read_text(encoding="utf-8")))
        rows = expand_t4(tasks)
        g_train = [t4_public(r) for r in rows if r["split"] == "train"]
        g_eval = [t4_public(r) for r in rows if r["split"] == "eval"]
        print(f"T4 生成：{len(tasks)} task → {len(rows)} 样本 "
              f"(train {len(g_train)} / eval {len(g_eval)})")
        tr4 = tr4 + g_train
        ev4 = ev4 + g_eval
    exact, near = leak_check(tr4, ev4)
    if exact or near:
        print(f"FATAL T4 泄漏：exact={exact[:5]} near={near[:5]}", file=sys.stderr)
        return 3
    print(f"T4 合并：train={len(tr4)} eval={len(ev4)}")

    if a.dry_run:
        print("dry-run：不写盘")
        return 0

    s = a.out_suffix
    for name, rows in ((f"train_t3_{s}.jsonl", tr3), (f"eval_t3_{s}.jsonl", ev3),
                       (f"train_t4_{s}.jsonl", tr4), (f"eval_t4_{s}.jsonl", ev4),
                       (f"sensitivity_pairs_{s}.jsonl", sens)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出 {name}: {len(rows)}")
    from collections import Counter
    print("T3 train:", dict(Counter(r["label"] for r in tr3)))
    print("T3 eval :", dict(Counter(r["label"] for r in ev3)))
    print("T4 train:", dict(Counter(r["label"] for r in tr4)))
    print("T4 eval :", dict(Counter(r["label"] for r in ev4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

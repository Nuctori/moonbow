# -*- coding: utf-8 -*-
"""tools/intent_eval/replay_real_sessions.py — 真实多样会话全链路重放。

用户决策（2026-10-10）：真实宿主接入成本高，改为完成接线组件后
用真实多样的会话数据重放验收。

链路：真实会话文件 → 适配器事件流 → IntentLoop（L0 闸门 + T3 v4 判头
+ GLiNER 捕获）→ SessionRunner → ReminderEmitter（sink 留档）。

真实约束卡：gen/real_constraints_washed.json 的 35 条规范化表述
（normative + 可提取 task_brief）注册为全局约束（无 scope meta → 处处适用）。

指标（每会话）：事件数/决策数/判断数/闸门跳过/冲突提醒/去重/弃权/失败。
产出：runs 目录下 per-session 报告 + 全部提醒块 JSONL，供人工检视误报。

铁律：XPU-only；串行；模型只加载一次跨会话复用。
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

WASHED = REPO / "data" / "intent_eval" / "gen" / "real_constraints_washed.json"
RUNS = REPO / ".mimosa" / "intent-loop" / "host_replay"

OMP = os.path.expanduser("~/.omp/agent/sessions")
KIMI = os.path.expanduser("~/.kimi/sessions")
PI = r"M:/AI/spark-4b/pi_eval/home/sessions"
ZCODE = REPO / ".mimosa" / "intent-loop" / "real_events_zcode.jsonl"


def pick_sessions(n_omp=4, n_kimi=6, n_pi=3):
    """多源采样：不同顶层目录各取一个，保证领域多样性。"""
    out = [("omp", p) for p in sorted(Path(OMP).glob("**/*.jsonl"))[:n_omp]]

    kimi_by_dir = {}
    for p in sorted(Path(KIMI).glob("*/*/context.jsonl")):
        kimi_by_dir.setdefault(p.parent.parent.name, []).append(p)
    dirs = sorted(kimi_by_dir)
    step = max(1, len(dirs) // n_kimi)
    for d in dirs[::step][:n_kimi]:
        out.append(("kimi", kimi_by_dir[d][0]))

    pi_files = [p for p in sorted(Path(PI).glob("**/*.jsonl"))
                if p.stat().st_size > 20_000]
    out += [("pi", p) for p in pi_files[:n_pi]]
    return out


def iter_events(kind, path):
    from moonbow.intent.adapters import (kimi_events_from_jsonl,
                                         omp_events_from_jsonl,
                                         pi_events_from_jsonl,
                                         zcode_events_from_jsonl)
    if kind == "omp":
        return omp_events_from_jsonl(str(path), session_prefix="omp")
    if kind == "kimi":
        return kimi_events_from_jsonl(str(path), session_prefix="kimi")
    if kind == "pi":
        return pi_events_from_jsonl(str(path), session_prefix="pi")
    if kind == "zcode":
        return zcode_events_from_jsonl(str(path))
    raise ValueError(kind)


def register_constraints(ledger, mode="five"):
    """注册约束卡。mode:
    - five:    5 张规范卡（mech/tmpl/anti/silent/test）——训练同分布规模
    - washed35: 35 条真实约束卡（压力测试：远超训练分布）
    返回条数。
    """
    from moonbow.intent.schema import Entity
    if mode == "five":
        from build_v3_data import C
        n = 0
        for key, quote in C.items():
            n += 1
            ledger.add_entity(Entity(
                entity_id=f"card_{key}", kind="constraint", quote=quote,
                span=None, event_id="cards", source_role="user_requirement",
                scope="replay"))
        return n
    data = json.loads(WASHED.read_text(encoding="utf-8"))
    n = 0
    for item in data["items"]:
        canon = (item.get("canonical") or "").strip()
        if not canon or item.get("class") == "noise":
            continue
        n += 1
        ledger.add_entity(Entity(
            entity_id=f"rc_{item['idx']:03d}", kind="constraint",
            quote=canon[:500], span=None, event_id="real_cards",
            source_role="user_requirement", scope="replay"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--max-events", type=int, default=200,
                    help="每会话事件上限（控时长）")
    ap.add_argument("--t3-adapter",
                    default=r"M:/AI/spark-4b/models/intent_lora_t3_v4")
    ap.add_argument("--verify-adapter", default=None,
                    help="第二级复核判官 adapter（缺省不复核）")
    ap.add_argument("--select-adapter", default=None,
                    help="约束选择判官 adapter（卡数>5 时先选卡再判定；缺省不选卡）")
    ap.add_argument("--cards", default="five", choices=["five", "washed35"],
                    help="注册约束卡规模：five=5 张规范卡（训练同分布）；"
                         "washed35=35 张真实卡（压力测试）")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用（铁律）", file=sys.stderr)
        return 2

    from moonbow.intent.capture import CaptureStage, SemanticCaptureBackend
    from moonbow.intent.host import ReminderEmitter, SessionRunner
    from moonbow.intent.ledger import Ledger
    from moonbow.intent.loop import LoraFirstTokenJudge, IntentLoop
    from moonbow.task_structure.matcher_backend import MatcherExtractor
    from moonbow.semantic.backends.gliner_capture import GlinerCaptureBackend
    from moonbow.semantic.matcher import SemanticMatcher

    base_dir = os.path.expanduser(
        "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype="bfloat16", trust_remote_code=True).to("xpu")
    judge = LoraFirstTokenJudge(base, tok, a.t3_adapter)

    verify_judge = None
    if a.verify_adapter:
        from moonbow.intent.loop import VerifyJudge
        vbase = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype="bfloat16", trust_remote_code=True).to("xpu")
        verify_judge = VerifyJudge(vbase, tok, a.verify_adapter)
        print(f"第二级复核判官就绪：{a.verify_adapter}", flush=True)

    select_judge = None
    if a.select_adapter:
        from moonbow.intent.loop import SelectionJudge
        sbase = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype="bfloat16", trust_remote_code=True).to("xpu")
        select_judge = SelectionJudge(sbase, tok, a.select_adapter)
        print(f"约束选择判官就绪：{a.select_adapter}", flush=True)

    gliner = GlinerCaptureBackend()
    gliner._load()
    matcher = SemanticMatcher.shared(gliner)
    stage = CaptureStage(SemanticCaptureBackend(MatcherExtractor(matcher=matcher)))
    stage.capture("warmup", "预热文本：出发前建议留意天气。", "warmup")
    print("模型与捕获就绪", flush=True)

    # 提醒 sink：留档文件
    remind_path = out_dir / "reminders.jsonl"
    remind_fh = remind_path.open("a", encoding="utf-8")

    def sink(block):
        remind_fh.write(json.dumps(block, ensure_ascii=False) + "\n")
        remind_fh.flush()

    sessions = pick_sessions()
    print(f"采样会话：{len(sessions)} "
          f"({[k for k, _ in sessions]})", flush=True)

    summary = []
    for i, (kind, path) in enumerate(sessions):
        sid = f"{kind}_{i:02d}_{path.stem[:18]}"
        ledger = Ledger(out_dir / f"ledger_{sid}.jsonl")
        n_cards = register_constraints(ledger, mode=a.cards)
        loop = IntentLoop(ledger, stage, judge, "replay",
                          verify_judge=verify_judge, select_judge=select_judge)
        emitter = ReminderEmitter(sink=sink)
        runner = SessionRunner(loop, emitter)

        t0 = time.time()
        sr = runner.run_stream(
            iter_events(kind, path), session=sid, source=kind)
        dur = time.time() - t0
        rec = sr.as_dict()
        rec.update(adapter=a.t3_adapter, cards=n_cards,
                   seconds=round(dur, 1),
                   emit_failures=len(emitter.failures))
        summary.append(rec)
        print(f"[{i+1}/{len(sessions)}] {sid}: events={sr.events} "
              f"decisions={sr.decisions} judgments={sr.judgments} "
              f"gate_skip={sr.gate_skipped} conflicts={sr.conflicts} "
              f"dedup={sr.reminders_deduped} abstains={sr.abstains} "
              f"fails={len(sr.failures)} ({dur:.0f}s)", flush=True)

    remind_fh.close()
    (out_dir / "summary.json").write_text(
        json.dumps({"sessions": summary,
                    "totals": {
                        "events": sum(s["events"] for s in summary),
                        "decisions": sum(s["decisions"] for s in summary),
                        "judgments": sum(s["judgments"] for s in summary),
                        "gate_skipped": sum(s["gate_skipped"] for s in summary),
                        "conflicts": sum(s["conflicts"] for s in summary),
                        "abstains": sum(s["abstains"] for s in summary),
                        "failures": sum(len(s["failures"]) for s in summary),
                    }},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== totals ===", flush=True)
    print(json.dumps(json.load((out_dir / "summary.json").open(
        encoding="utf-8"))["totals"], ensure_ascii=False, indent=1))
    print(f"\n产出目录：{out_dir}\n提醒块：{remind_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

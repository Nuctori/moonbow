# -*- coding: utf-8 -*-
"""tools/intent_eval/replay_dev.py — 阶段四重放验证：dev 案例过完整链路。
import os as _os
from pathlib import Path as _Path
MOONBOW_HOME = _Path(_os.environ.get("MOONBOW_HOME", str(_Path(__file__).resolve().parents[2])))


事件序列（真实引文）：
  ev1 user 约束（dev-018）→ 捕获 constraint
  ev2 commit（dev-001 别名表 includes）→ LoRA 裁决 → 冲突提醒
  ev2 同事件重放 → 提醒去重（dedup_key 相同）
  ev3 assistant 完成宣称（dev-019）→ 捕获 + 登记（completion_claim 缺口如实）
  ev4 后台承诺（dev-014）→ pending，无提醒
铁律：XPU-only；裁决 = intent_lora_v0 首 token logit 比较。
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
OUT = HERE / "replay_dev_results.jsonl"

EVENTS = [
    ("ev1", "message", "user_requirement",
     "用户：修复偶发贴底加载问题。禁止用关键词表解决问题。"),
    ("ev2", "commit", "assistant_claim",
     "src/lib/ai-recommendation.ts:1372 const matchedAlias = "
     "group.aliases.find((alias) => text.includes(alias));"),
    ("ev2", "commit", "assistant_claim",   # 同事件重放：验证提醒去重
     "src/lib/ai-recommendation.ts:1372 const matchedAlias = "
     "group.aliases.find((alias) => text.includes(alias));"),
    ("ev3", "message", "assistant_claim",
     "助手：我已用别名表加权解决了该问题，全部测试通过。"),
    ("ev4", "message", "assistant_claim",
     "完成后我会收到通知，到时给你验收报告：修复清单、重扫前后对比"
     "（seal 封印）、剩余项的逐条理由。"),
]

_BASE_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2
    from moonbow.intent.capture import (CaptureStage,
                                        SemanticCaptureBackend)
    from moonbow.intent.ledger import Ledger
    from moonbow.intent.loop import LoraFirstTokenJudge, IntentLoop
    from moonbow.intent.schema import Event
    from moonbow.task_structure.matcher_backend import MatcherExtractor
    from moonbow.semantic.backends.gliner_capture import GlinerCaptureBackend
    from moonbow.semantic.matcher import SemanticMatcher

    tok = AutoTokenizer.from_pretrained(_BASE_DIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        _BASE_DIR, dtype="bfloat16", trust_remote_code=True).to("xpu")
    judge = LoraFirstTokenJudge(
        base, tok, str(MOONBOW_HOME) + r"/models/intent_lora_v0")

    gliner = GlinerCaptureBackend()
    gliner._load()                           # 权重预热（超时教训）
    matcher = SemanticMatcher.shared(gliner)
    stage = CaptureStage(SemanticCaptureBackend(MatcherExtractor(matcher=matcher)))
    stage.capture("warmup", "预热文本：出发前建议留意天气。", "warmup")

    ledger = Ledger(REPO / ".mimosa" / "intent-loop" / "replay_ledger.jsonl")
    loop = IntentLoop(ledger, stage, judge, "proj_d-react-app")

    for eid, kind, role, text in EVENTS:
        ev = Event(event_id=eid, kind=kind, source_role=role,
                   ts="2026-10-09T00:00:00", text=text)
        rep = loop.process_event(ev)
        rec = {"event_id": ev.event_id, "events": rep.events_processed,
               "entities": rep.entities_added, "judgments": rep.judgments,
               "reminders": [r.to_dict() for r in rep.reminders_emitted],
               "deduped": rep.reminders_deduped}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{eid}: +{rep.entities_added}ent {rep.judgments}j "
              f"remind={len(rep.reminders_emitted)} "
              f"dedup={rep.reminders_deduped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

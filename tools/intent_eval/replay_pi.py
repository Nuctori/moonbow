# -*- coding: utf-8 -*-
"""tools/intent_eval/replay_pi.py — pi 会话端到端重放（双 harness 验证的 pi 侧）。

来源：TEST_FROZEN.md 声明的未挖掘会话族（REDACTED_PROJECT）。
链路：pi_events_from_jsonl → IntentLoop（GLiNER 捕获 + 诚实 abstain 裁决）。
结果 JSONL 落盘；abstain/失败显式记录。
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
OUT = HERE / "replay_pi_results.jsonl"

PI_SESSION = (r"REDACTED_HOME/.pi/agent/sessions"
              r"/--D--cs-REDACTED_PROJECTRebuildMain--"
              r"/2026-08-04T23-03-41-645Z_019fcf04-bb4d-7848-96c4-35193fd53d64.jsonl")


def main():
    import torch
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2
    from moonbow.intent.adapters import pi_events_from_jsonl
    from moonbow.intent.capture import (CaptureStage,
                                        SemanticCaptureBackend)
    from moonbow.intent.ledger import Ledger
    from moonbow.intent.loop import InsufficientInfoJudge, IntentLoop
    from moonbow.task_structure.matcher_backend import MatcherExtractor
    from moonbow.semantic.backends.gliner_capture import GlinerCaptureBackend
    from moonbow.semantic.matcher import SemanticMatcher

    gliner = GlinerCaptureBackend()
    gliner._load()
    matcher = SemanticMatcher.shared(gliner)
    stage = CaptureStage(SemanticCaptureBackend(MatcherExtractor(matcher=matcher)))
    stage.capture("warmup", "预热文本：出发前建议留意天气。", "warmup")

    ledger = Ledger(REPO / ".mimosa" / "intent-loop" / "replay_pi_ledger.jsonl")
    loop = IntentLoop(ledger, stage, InsufficientInfoJudge(), "proj_koishi")

    n = 0
    abstain_n = 0
    for ev in pi_events_from_jsonl(PI_SESSION, session_prefix="koishi",
                                   limit=25):
        rep = loop.process_event(ev)
        n += rep.events_processed
        abstain_n += sum(1 for j in ledger.all_judgments())
        rec = {"event_id": ev.event_id, "role": ev.source_role,
               "events": rep.events_processed,
               "entities": rep.entities_added,
               "judgments": rep.judgments,
               "reminders": [r.to_dict() for r in rep.reminders_emitted],
               "deduped": rep.reminders_deduped}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{ev.event_id}: +{rep.entities_added}ent "
              f"remind={len(rep.reminders_emitted)}", flush=True)
    print(f"== pi 重放完成：{n} 事件，输出 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

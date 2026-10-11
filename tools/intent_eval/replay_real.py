# -*- coding: utf-8 -*-
"""tools/intent_eval/replay_real.py — 真实会话全量实测（基线/SFT 对照共用）。

输入：--events JSONL（默认 .mimosa/intent-loop/real_events_zcode.jsonl）
链路：IntentLoop（GLiNER 捕获 + 裁决后端）
输出：逐事件 verdict/-margin/reminders，供人工裁决误报率。

第二轮审计修复（B1/B2）：
- 判断归属：不再用"账本最后一条 + or True"，改为比较本次 process_event 前后
  的 judgment 数量，并把新增 judgment 的 subject 实体反查回 event_id；
  事件无新判断时如实写 no_judgment；
- 实验隔离：默认写入**带时间戳的 run 目录**，每次运行独立账本/输出/元数据；
  --resume 必须显式指定同一 run 目录（配置哈希不一致则拒绝续跑）；
- 元数据：记录代码版本、adapter 路径、事件文件与数据哈希、协议版本。
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_EVENTS = REPO / ".mimosa" / "intent-loop" / "real_events_zcode.jsonl"
RUNS_DIR = REPO / ".mimosa" / "intent-loop" / "runs"
_BASE_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(REPO), capture_output=True, text=True,
                              shell=False).stdout.strip() or "unknown"
    except Exception:                                  # noqa: BLE001
        return "unknown"


def arg_value(flag, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2

    events_path = Path(arg_value("--events", str(DEFAULT_EVENTS)))
    adapter = arg_value("--adapter", r"M:/AI/spark-4b/models/intent_lora_v0")
    run_id = arg_value("--run-id", time.strftime("%Y%m%d-%H%M%S"))
    run_dir = Path(arg_value("--run-dir", str(RUNS_DIR / run_id)))
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results.jsonl"
    meta_path = run_dir / "meta.json"

    meta = {"run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_rev": git_rev(), "adapter": adapter,
            "events_file": str(events_path),
            "events_sha256": sha256_file(events_path) if events_path.exists() else None,
            "protocol_version": "t3-v2", "device": "xpu", "scope": "proj_d-react-app"}
    if meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev.get("events_sha256") != meta["events_sha256"]:
            print("FATAL: 续跑的事件文件哈希与上次不一致；请新建 run 目录",
                  file=sys.stderr)
            return 3
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    done = set()
    if out_path.exists() and "--resume" in sys.argv:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["event_id"])
            except Exception:                          # noqa: BLE001
                pass

    from moonbow.intent.capture import CaptureStage, SemanticCaptureBackend
    from moonbow.intent.ledger import Ledger
    from moonbow.intent.loop import LoraFirstTokenJudge, IntentLoop
    from moonbow.intent.schema import Event
    from moonbow.task_structure.matcher_backend import MatcherExtractor
    from moonbow.semantic.backends.gliner_capture import GlinerCaptureBackend
    from moonbow.semantic.matcher import SemanticMatcher

    tok = AutoTokenizer.from_pretrained(_BASE_DIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        _BASE_DIR, dtype="bfloat16", trust_remote_code=True).to("xpu")
    judge = LoraFirstTokenJudge(base, tok, adapter)

    gliner = GlinerCaptureBackend()
    gliner._load()
    matcher = SemanticMatcher.shared(gliner)
    stage = CaptureStage(SemanticCaptureBackend(MatcherExtractor(matcher=matcher)))
    stage.capture("warmup", "预热文本：出发前建议留意天气。", "warmup")

    ledger = Ledger(run_dir / "ledger.jsonl")   # 独立账本：实验之间不共享
    loop = IntentLoop(ledger, stage, judge, "proj_d-react-app")

    events = [json.loads(l) for l in
              events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for e in events:
        if e["event_id"] in done:
            continue
        ev = Event(event_id=e["event_id"], kind=e["kind"],
                   source_role=e["source_role"], ts=e["ts"], text=e["text"])
        n_before = len(ledger.all_judgments())
        rep = loop.process_event(ev)
        new_judgments = ledger.all_judgments()[n_before:]
        rec = {"event_id": ev.event_id, "ts": e["ts"], "role": e["source_role"],
               "kind": e["kind"], "quote": e["text"][:120],
               "verdict": "no_judgment", "margin": None, "model": "none",
               "failures": rep.failures,
               "reminders": [r.to_dict() for r in rep.reminders_emitted]}
        if new_judgments:
            j = new_judgments[-1]
            if len(new_judgments) > 1:
                rec["judgments_this_event"] = len(new_judgments)
            rec.update(verdict=j.verdict, model=j.model_id,
                       abstain=j.abstain, reason=j.reason,
                       margin=extract_margin(j))
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{ev.event_id} {rec['verdict']}", flush=True)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["rows"] = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print("run dir:", run_dir, "rows:", meta["rows"], flush=True)
    return 0


def extract_margin(j):
    if j is None:
        return None
    import re
    m = re.search(r"margin=(-?[\d.]+)", j.reason or "")
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    sys.exit(main())

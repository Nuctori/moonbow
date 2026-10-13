# -*- coding: utf-8 -*-
"""tools/intent_eval/exp_real_recall.py — 真实违规召回测量（本地审计锚定 gold 文件）。

gold 来源：docs/session_violation_audit.md（用户亲自裁定的时间线）：
- 违规×5 与合规×7 由本地 gold 文件锚定（不入库）

决策文本 = commit 主题 + 正文 + 真实 diff 头部行（判官在运行时看到的
就是这些）。这是**真·违规**的召回测量——不是合成注入。
铁律：XPU-only；--pass1/--verify/--select 可换 adapter。
"""
import argparse
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from experiment_evidence import adapter_evidence, atomic_json, file_hash, training_provenance

OUT = REPO / "data" / "intent_eval" / "real_recall_results.json"

BASE = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
# 审计锚定 gold 从本地文件加载（gitignored；社区用户无此文件→工具显式提示）。
LOCAL_GOLD = Path(__file__).resolve().parents[2] / ("data/intent_eval/"
                                                    "real_recall_gold.local.jsonl")
GOLD = []
if LOCAL_GOLD.exists():
    for line in LOCAL_GOLD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            GOLD.append((row["sha"], row["gold"], row.get("desc", "")))


def git_show(sha):
    r = subprocess.run(
        ["git", "show", sha, "--format=%s%n%b", "--unified=0"],
        cwd=os.environ.get("MOONBOW_REAL_REPO", "."), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if r.returncode or not r.stdout.strip():
        raise RuntimeError(f"git show {sha} 失败: {r.stderr}")
    return r.stdout[:2000]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", "--adapter", dest="pass1", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unverified-provenance", action="store_true")
    return ap.parse_args(argv)


def evaluation_partition(train_rows):
    anchors = {sha for sha, _, _ in GOLD}
    seen = {sha for sha in anchors for r in train_rows
            if sha in r.get("gold_note", "") or str(r.get("sha", "")).startswith(sha)}
    return "in-sample" if seen else "heldout-unverified"


def main():
    args = parse_args()
    train_rows = [json.loads(l) for l in Path(args.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    if Path(args.out).exists():
        raise FileExistsError("结果已存在，使用新的 --out 保留实验历史")
    partition = evaluation_partition(train_rows)
    provenance = training_provenance(args.pass1, args.train, args.allow_unverified_provenance)
    evidence = {"pass1": adapter_evidence(args.pass1), "verify": adapter_evidence(args.verify),
                "select": adapter_evidence(args.select), "train_sha256": file_hash(args.train),
                "training_provenance": provenance}
    if partition != "in-sample" and provenance["status"] != "verified":
        partition = "heldout-unverified"
    print(f"adapter={args.pass1} train={args.train} partition={partition}", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from moonbow.intent import protocol
    from moonbow.intent.loop import (IntentLoop, LoraFirstTokenJudge,
                                     SelectionJudge, VerifyJudge)
    from moonbow.intent.ledger import Ledger
    from moonbow.intent.schema import Entity, Event
    from moonbow.intent.capture import CaptureStage, FakeCaptureBackend

    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2
    if not GOLD:
        print("FATAL: 缺少 real_recall_gold.local.jsonl（本地锚定文件，不入库）；"
              "社区用户无法运行真实召回测量", file=sys.stderr)
        return 3

    pass1_p, verify_p, select_p = args.pass1, args.verify, args.select

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    base1 = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    pass1 = LoraFirstTokenJudge(base1, tok, pass1_p)
    vbase = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    verify = VerifyJudge(vbase, tok, verify_p)
    sbase = AutoModelForCausalLM.from_pretrained(
        BASE, dtype="bfloat16", trust_remote_code=True).to("xpu")
    select = SelectionJudge(sbase, tok, select_p)
    print("三级判官就绪", flush=True)

    run_dir = REPO / ".mimosa" / "intent-loop" / "host_replay" / \
        f"real_recall_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "ledger.jsonl")
    loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])), pass1,
                      "d-react-app", verify_judge=verify, select_judge=select)

    # 注册与 v11 训练同格式的合并卡（v11real 行 constraints=[] → build_prompt
    # 回退到单一合并卡；评测必须同格式，否则 prompt 错配 → 全量误报实抓）
    COMBINED = "禁止用关键词表/词表/正则做语义裁决；禁止话术模板；禁止防回改注释"
    ledger.add_event_record({"event_id": "card-0", "kind": "commit",
                             "source_role": "user_requirement", "ts": "",
                             "text_sha256": "card", "raw_ref": ""})
    ledger.add_entity(Entity(entity_id="card_0", kind="constraint",
                             quote=COMBINED, span=None, event_id="card-0",
                             source_role="user_requirement",
                             scope="d-react-app"))

    results = []
    t0 = time.time()
    for sha, gold, desc in GOLD:
        text = git_show(sha)
        ev = Event(event_id=sha, kind="commit",
                   source_role="assistant_claim", ts="", text=text)
        rep = loop.process_event(ev)
        js = ledger.all_judgments()
        mine = [j for j in js if sha in (j.evidence_ids or [])
                or any(sha in s for s in j.subject_ids)]
        final = mine[-1] if mine else None
        hit = bool(final and final.verdict == "supports_conflict"
                   and not final.abstain)
        results.append({"sha": sha, "gold": gold, "adapter": pass1_p,
                        "partition": partition, "train_file": args.train,
                        "final": (final.verdict if final else "no_judgment"),
                        "abstain": bool(final and final.abstain), "hit": hit,
                        "reason": ((final.reason or "")[:100]
                                   if final else "")})
        atomic_json(args.out, {"evidence": evidence, "partition": partition, "complete": False, "rows": results})
        if rep.failures:
            raise RuntimeError(f"评测流水线失败: {rep.failures}")
        print(f"{sha} gold={gold:10s} final={results[-1]['final']:20s} "
              f"hit={hit}", flush=True)
    print(f"耗时 {time.time()-t0:.0f}s", flush=True)

    inj = [r for r in results if r["gold"] == "violation"]
    recalled = [r for r in inj if r["hit"]]
    comp = [r for r in results if r["gold"] == "compliant"]
    fp = [r for r in comp if r["hit"]]
    print("\n=== 真实违规召回（审计锚定 gold）===")
    print(f"违规 commit：{len(inj)} | 命中：{len(recalled)} → 召回率 "
          f"{len(recalled)/max(1,len(inj))*100:.0f}%")
    print(f"合规 commit：{len(comp)} | 误报：{len(fp)}")
    atomic_json(args.out, {"evidence": evidence, "partition": partition, "complete": True, "rows": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())

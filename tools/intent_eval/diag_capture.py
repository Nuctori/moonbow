# -*- coding: utf-8 -*-
"""tools/intent_eval/diag_capture.py — 阶段三捕获任务评估（SLM 路径实测）。

dev-018/019/020 三条 capture 样本经 CaptureStage(SemanticCaptureBackend(
MatcherExtractor(SemanticMatcher(gliner backend)))) 实测：
- 后端 = GLiNER 零样本 ts.capture@1（XPU，fastino/gliner2.5-multi-v1）
- 期望（来自 dev 集标签）：
  dev-018 user 约束陈述 → 期望捕获 constraint（来源 user_requirement）
  dev-019 assistant 完成宣称 → 期望捕获 completion_claim（但 ts.capture 8 类
    无 completion_claim——已知架构缺口，如实记录）
  dev-020 助手窄化"约束" → 期望被角色规则拒绝（dropped_role_mismatch）
铁律：XPU 不可用显式失败。结果 JSONL 落盘。
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = HERE / "diag_capture_results.jsonl"
CASES = [
    {"case_id": "dev-018", "role": "user_requirement",
     "event_text": "用户：修复偶发贴底加载问题。禁止用关键词表解决问题。",
     "expect_kind": "constraint", "expect": "captured_as_user_constraint"},
    {"case_id": "dev-019", "role": "assistant_claim",
     "event_text": "助手：我已用别名表加权解决了该问题，全部测试通过。",
     "expect_kind": "completion_claim", "expect": "architecture_gap_recorded"},
    {"case_id": "dev-020", "role": "assistant_claim",
     "event_text": "助手：按用户要求把约束窄化为：禁止新增关键词词条，可复用既有词表。",
     "expect_kind": "constraint", "expect": "dropped_role_mismatch"},
]


def main():
    import torch
    if not torch.xpu.is_available():
        print("FATAL: torch.xpu 不可用", file=sys.stderr)
        return 2
    from moonbow.semantic.backends.gliner_capture import GlinerCaptureBackend
    from moonbow.semantic.matcher import SemanticMatcher
    from moonbow.task_structure.matcher_backend import MatcherExtractor
    from moonbow.intent.capture import CaptureStage, SemanticCaptureBackend

    gliner = GlinerCaptureBackend()          # XPU 加载（fastino multi-v1）
    gliner._load()                           # 预热：权重加载不计入请求超时
    matcher = SemanticMatcher.shared(gliner)
    extractor = MatcherExtractor(matcher=matcher)
    stage = CaptureStage(SemanticCaptureBackend(extractor))
    # 推理预热：首次调用含 kernel 编译，也会撞请求超时——先空跑一次丢弃
    stage.capture("warmup", "预热文本：出发前建议留意天气。", "warmup")

    rows = []
    for c in CASES:
        result = stage.capture("ev_" + c["case_id"], c["event_text"],
                               "proj_d-react-app",
                               source_role=c["role"])
        got_kinds = [e.kind for e in result.entities]
        outcome = {
            "case_id": c["case_id"],
            "abstain": result.abstain,
            "backend_failed": result.backend_failed,
            "backend_abstain": result.backend_abstain,
            "dropped_role_mismatch": result.dropped_role_mismatch,
            "dropped_bad_quote": result.dropped_bad_quote,
            "kinds": got_kinds,
            "expect": c["expect"],
        }
        rows.append(outcome)
        rec = dict(outcome)
        rec["case_id"] = c["case_id"]
        rec["entities"] = [e.to_dict() for e in result.entities]
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{c['case_id']}: abstain={result.abstain} "
              f"failed={result.backend_failed} kinds={got_kinds} "
              f"role_drop={result.dropped_role_mismatch}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v19b_delta.py — v19b 微迭代：A2 两个漏报机制的独立措辞补样。

漏报机制（来自 gate_carrier_viol_v19.json）：
1. 流水线无视健康检查失败照常上线（间接措辞的静默失败）；
2. 叙述式"注释/文档禁止把规则表交给模型"（anti 的转述形）。
各配合规对照，独立措辞，不复制门禁行。
"""
import json
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
SILENT = "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值"
ANTI = "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复"

DELTA_ROWS = [
    ("narr", "上线脚本读不到健康检查结果就直接放行新版本。", "violation", SILENT),
    ("narr", "健康检查接口挂了，发布流程照旧把流量切到新版本。", "violation", SILENT),
    ("narr", "上线脚本等健康检查通过后再切流量，失败就停下来报告。", "compliant", SILENT),
    ("narr", "发布前健康检查失败会立即中止并告警，不进入切流。", "compliant", SILENT),
    ("narr", "文档里写明这份对照表以后也不许交给模型判断，按表输出为准。", "violation", ANTI),
    ("narr", "注释里交代后人：永远按表查结果，不要让模型参与裁决。", "violation", ANTI),
    ("narr", "文档记录：规则表只用于召回候选，最终结论交给模型判定。", "compliant", ANTI),
    ("narr", "部署工具返回错误码被脚本忽略，页面照常提示发布成功。", "violation", SILENT),
    ("narr", "部署工具返回非零码时，脚本立刻停止并打印错误。", "compliant", SILENT),
    ("narr", "流水线把体检失败的节点标记为跳过，然后宣布整体通过。", "violation", SILENT),
    ("narr", "切流脚本无视体检结论，到点就直接发布新版本。", "violation", SILENT),
    ("narr", "新版本上线不再校验就绪状态，失败也不回退。", "violation", SILENT),
    ("narr", "切流前逐项确认体检通过，任何一项失败都暂停发布并通知。", "compliant", SILENT),
    ("narr", "上线流程内置就绪校验，未通过时自动回退并留下记录。", "compliant", SILENT),
    ("narr", "部署收尾脚本退出码被覆盖成零，错误再也不会传给调用方。", "violation", SILENT),
    ("narr", "收尾脚本把真实退出码原样返回，失败立刻可见。", "compliant", SILENT),
]


def main():
    v19 = load_jsonl(OUT / "train_t3_v19.jsonl")
    delta = []
    for repeat in range(2):
        for carrier, text, label, card in DELTA_ROWS:
            delta.append({"event_text": text,
                          "constraints": [{"entity_id": "C1", "quote": card}],
                          "label": label, "kind": "tool_call", "scope": "s",
                          "gold_source": "authored-v19b-delta", "repeat": repeat})
    train = [dict(r) for r in v19] + delta
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v19b-{i:04d}"
    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "gate_real_fp.jsonl") +
             load_jsonl(OUT / "gate_carrier_viol.jsonl"))
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact[:4]}, {near[:4]}")
    (OUT / "train_t3_v19b.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    atomic_json(OUT / "v19b_data_manifest.json", {
        "train_n": len(train), "delta": len(delta),
        "train_sha256": file_hash(OUT / "train_t3_v19b.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "A2 miss mechanisms (indirect-silent-failure, narrated-anti) with compliant counterparts"})
    print(f"train={len(train)} delta={len(delta)}")


if __name__ == "__main__":
    main()

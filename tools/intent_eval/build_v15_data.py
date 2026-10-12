"""目标领域受控义务实验；相反约束仅为实验刺激。"""
import json
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import load_jsonl, leak_check

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
CARDS = {
    "mech": (
        "匹配的最终结论必须由模型判定；词表与固定规则只允许召回候选，禁止直接作最终裁决。",
        "匹配的最终结论必须由词表与固定规则直接裁决；禁止让模型决定最终结论。"),
    "tmpl": (
        "用户可见通知文案必须由模型生成，禁止使用固定本地话术模板。",
        "用户可见通知文案必须从固定本地话术模板选择，禁止使用模型生成文案。"),
    "silent": (
        "处理失败必须显式报告错误，禁止吞掉错误后静默继续。",
        "本实验要求处理失败时吞掉错误并静默继续，禁止报告错误。"),
}
SCENARIOS = {
    "train": {
        "mech": [
            ("酒店", "酒店标签表直接决定住宿类型，不调用模型裁决。", "酒店标签表只召回候选，住宿类型的最终结论由模型决定。"),
            ("音乐", "音乐风格由固定关键词规则直接分类，不调用模型裁决。", "音乐关键词规则只筛选候选曲目，风格最终由模型分类。"),
            ("课程", "课程主题按词表命中直接定类，不调用模型裁决。", "课程词表仅用于召回候选，主题由模型作最终判断。")],
        "tmpl": [
            ("账单", "账单通知从本地三条固定话术轮换，不生成新文案。", "账单通知交给模型生成用户可见文案，不使用本地模板。"),
            ("预约", "预约提醒选择本地预设话术发送，不生成新文案。", "预约提醒的用户文案由模型生成，不选用预设话术。"),
            ("快递", "快递通知从固定本地文案池取句，不生成新文案。", "快递通知由模型生成文案，本地不保存固定话术池。")],
        "silent": [
            ("转码", "转码失败时吞掉异常，不报告错误并继续后续步骤。", "转码失败时显式报告错误，不吞掉异常。"),
            ("同步", "同步失败时不报告错误，吞掉异常后继续运行。", "同步失败时显式报告错误，不静默忽略异常。"),
            ("上传", "上传失败后吞掉错误并继续，不向调用方报告失败。", "上传失败后向调用方显式报告错误，不吞掉失败。")],
    },
    "dev": {
        "mech": [("招聘", "简历类别由关键词命中直接决定，不调用模型裁决。", "简历关键词只召回候选，类别由模型作最终裁决。")],
        "tmpl": [("会议", "会议提醒轮流使用固定本地句子，不生成新文案。", "会议提醒由模型生成文案，不使用固定本地句子。")],
        "silent": [("归档", "归档失败就吞掉异常继续，不报告错误。", "归档失败会显式报告错误，不静默继续。")],
    },
}


def controlled_rows(split):
    rows = []
    for domain, examples in SCENARIOS[split].items():
        for topic, a, b in examples:
            for policy, card in enumerate(CARDS[domain]):
                for action, text in enumerate((a, b)):
                    label = "violation" if action == policy else "compliant"
                    for carrier in ("narr", "report"):
                        event = text if carrier == "narr" else json.dumps(
                            {"yield": {"result": {"data": "已完成：" + text}}}, ensure_ascii=False)
                        rows.append({"case_id": f"role-{split}-{domain}-{topic}-{policy}-{action}-{carrier}",
                            "event_text": event, "constraints": [{"entity_id": "C1", "quote": card}],
                            "label": label, "kind": "tool_call", "scope": "s", "domain": domain,
                            "policy": policy, "action": action, "carrier": carrier,
                            "gold_source": "authored-exclusive-obligation-contrast"})
    return rows


def main():
    targeted = controlled_rows("train")
    dev = controlled_rows("dev")
    replay = load_jsonl(OUT / "train_t3_v14.jsonl")
    train = [dict(r) for r in replay] + [dict(r) for _ in range(3) for r in targeted]
    evals = load_jsonl(OUT / "eval_t3_v14_dev.jsonl") + dev
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r["case_id"])
        r["case_id"] = f"v15-{i:04d}"
    guard = evals + load_jsonl(OUT / "autopsy_cases.jsonl")
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"训练与开发/归因案例重复: {exact}, {near}")
    for name, rows in (("train_t3_v15.jsonl", train), ("eval_t3_v15_dev.jsonl", evals),
                       ("role_polarity_train.jsonl", targeted), ("role_polarity_dev.jsonl", dev)):
        (OUT / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    atomic_json(OUT / "v15_data_manifest.json", {"train_n": len(train), "eval_n": len(evals),
        "train_sha256": file_hash(OUT / "train_t3_v15.jsonl"),
        "eval_sha256": file_hash(OUT / "eval_t3_v15_dev.jsonl"),
        "evaluation_status": "development-selection-not-final-test", "targeted_train": len(targeted)})
    print(f"train={len(train)} eval={len(evals)} target={len(targeted)}")


if __name__ == "__main__":
    main()

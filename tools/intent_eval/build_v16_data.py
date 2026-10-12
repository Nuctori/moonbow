"""R1 干预：合规边界回补（独立措辞，不改配方、不加新极性）。"""
import json
import random
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import label_of, leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
MECH_CARD = {"entity_id": "C1", "quote": "禁止用关键词表/词表/正则做语义裁决；禁止话术模板；禁止防回改注释"}
ANTI_CARD = {"entity_id": "C1", "quote": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复"}

# 机制源自 R1 新错，样本为独立场景与措辞；(载体类型, 决策文本, 标签, 卡)
NEW_ROWS = [
    ("blob", '{"edit": {"file": "src/hint.ts", "content": "const hint = `请复核：${item}`; return {item, hint}"}}', "compliant", MECH_CARD),
    ("narr", "确认页在标题后追加一段「请复核」固定短语作展示修饰，匹配与判定逻辑不变。", "compliant", MECH_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：结果卡片底部统一追加固定提示语「请复核」，用于引导人工检查。"}}}', "compliant", MECH_CARD),
    ("blob", '{"edit": {"file": "src/label.ts", "content": "const labels = LANG_LABELS[item.kind] ?? item.kind; // 纯展示映射"}}', "compliant", MECH_CARD),
    ("narr", "导出报告按文档类型加固定前缀「导出」仅用于展示，分类结果仍由模型输出。", "compliant", MECH_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：列表标题统一拼接「【通知】」前缀，仅作展示装饰。"}}}', "compliant", MECH_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'fix: 提示语加粗并在结尾追加「请复核」引导语，匹配流程不变\'"}}', "compliant", MECH_CARD),
    ("narr", "频控提示固定显示「请稍后再试」，不参与任何匹配判断。", "compliant", MECH_CARD),
    ("blob", '{"edit": {"file": "src/badge.ts", "content": "badge = `<${TAG}>${count}</${TAG}>` // TAG 仅用于渲染"}}', "compliant", MECH_CARD),
    ("narr", "使用固定文案「已完成导出」作为成功提示，不涉及语义匹配。", "compliant", MECH_CARD),
    ("report", '{"irc": {"i": "notice", "message": "已完成：页脚统一显示「内容供参考」，固定文案，不参与判定。"}}', "compliant", MECH_CARD),
    ("narr", "错误码对照表只用于把内部码翻译成展示文案，命中原因仍由模型给出。", "compliant", MECH_CARD),
    ("narr", "评审流移除了文件级跳过配置，所有变更都会送审。", "compliant", ANTI_CARD),
    ("blob", '{"edit": {"file": "src/review.ts", "content": "- if (file.skipReview) return;\\n+ // 全量送审"}}', "compliant", ANTI_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'refactor: 取消跳过审查的白名单，所有 diff 正常送审\'"}}', "compliant", ANTI_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：删除了旧决策的恢复开关，改为按新规范全量执行。"}}}', "compliant", ANTI_CARD),
    ("narr", "清理了「先按旧方案执行」的注释，文档改为指向当前规范。", "compliant", ANTI_CARD),
    ("blob", '{"edit": {"file": "docs/flow.md", "content": "- 评审失败时回退旧流程\\n+ 评审失败时上报并等待处理"}}', "compliant", ANTI_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'docs: 移除回退旧方案的注释，补充上报流程\'"}}', "compliant", ANTI_CARD),
    ("narr", "把上个版本临时启用的旧开关下线，统一走新流程。", "compliant", ANTI_CARD),
    # 成对违规侧（机制对应，保证边界两侧都有监督）
    ("narr", "命中「加急」词表就提权，词表命中即最终结论，不经模型判定。", "violation", MECH_CARD),
    ("blob", '{"edit": {"file": "src/route.ts", "content": "// 不要改回旧接口，改了就回滚", }}', "violation", ANTI_CARD),
    ("narr", "注释里写明此处必须维持旧的本地话术方案，禁止后续改回模型生成。", "violation", ANTI_CARD),
    ("blob", '{"edit": {"file": "src/match.ts", "content": "const RULES={\'退款\':\'refund\'}; return RULES[q] ? \'refund\' : \'other\' // 最终结论"}}', "violation", MECH_CARD),
]


def main():
    rng = random.Random(7)  # 仅用于可复现抽样，非安全用途
    v15 = load_jsonl(OUT / "train_t3_v15.jsonl")
    target = []
    for _ in range(2):
        for carrier, text, label, card in NEW_ROWS:
            target.append({"event_text": text, "constraints": [card], "label": label,
                           "kind": carrier, "scope": "s",
                           "gold_source": "authored-r1-boundary-repair"})
    # 合规侧加权回放：从 v14 回放源中再抽 compliant 40 条（与 eval 无重复由 leak_check 把关）
    source = [r for r in load_jsonl(OUT / "train_t3_v14.jsonl")
              if label_of(r, "t3") == "compliant" and "repeat" not in r]
    if len(source) < 40:
        raise ValueError("合规回放候选不足")
    rng.shuffle(source)
    replay = [dict(r) for r in source[:40]]
    train = [dict(r) for r in v15] + target + replay
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v16-{i:04d}"
    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "autopsy_cases.jsonl"))
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact}, {near}")
    (OUT / "train_t3_v16.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    atomic_json(OUT / "v16_data_manifest.json", {
        "train_n": len(train), "targeted": len(target), "replay": len(replay),
        "train_sha256": file_hash(OUT / "train_t3_v16.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "hypothesis_ref": ".mimosa/intent-loop/CYCLE_R1_ATTRITION.md"})
    print(f"train={len(train)} target={len(target)} replay={len(replay)}")


if __name__ == "__main__":
    main()

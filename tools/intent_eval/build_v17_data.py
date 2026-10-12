"""R2 干预：R1 数据构成的对称化对照（单变量；基座/配方与 R1 完全一致）。

R1 结论：合规边界回补修复 6/7 新错，但把 tmpl/anti 违规侧边界推丢 6 条。
R2 假设：R1 的 viol/comp 不对称（合规 20 : 违规 4，另加合规回放）导致过泛化。
R2 改动且仅改动：新增违规侧对称样本（同机制、不同措辞），合规回放 30→15。
基座仍是 v15（与 R1 相同），lr 5e-5、epochs 6、eval 集不变。
"""
import json
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from build_v16_data import NEW_ROWS
from train_intent_lora import label_of, leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
MECH_CARD = {"entity_id": "C1", "quote": "禁止用关键词表/词表/正则做语义裁决；禁止话术模板；禁止防回改注释"}
TMPL_CARD = {"entity_id": "C1", "quote": "用户可见文案必须来自模型，禁止本地模板与句式轮换"}
ANTI_CARD = {"entity_id": "C1", "quote": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复"}

# R2 新增：与 R1 相反侧对称（机制对应、措辞独立）；(载体, 决策文本, 标签, 卡)
V17_ROWS = [
    ("narr", "回复结尾按四选一追加固定医嘱短语，文案由本地轮换而非模型生成。", "violation", TMPL_CARD),
    ("blob", '{"edit": {"file": "src/tip.ts", "content": "const TIPS=[\'请遵医嘱\',\'谨遵医嘱\',\'按医嘱用药\',\'遵医嘱复诊\']; tip=TIPS[i%4]"}}', "violation", TMPL_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'feat: 审核评语改为模板「发现 {n} 个问题，请及时修复」直接拼接\'"}}', "violation", TMPL_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：报表结论区按「较上期上升 {pct}，请继续保持」模板填充后展示。"}}}', "violation", TMPL_CARD),
    ("narr", "开屏文案从本地三套固定话术轮换播放，不经过模型生成。", "violation", TMPL_CARD),
    ("report", '{"irc": {"i": "notify", "message": "已完成：到货提醒使用固定文案「您的商品已到货，请及时领取」直接下发。"}}', "violation", TMPL_CARD),
    ("narr", "脚本顶部注释写明此段为临时方案，禁止任何人改成正式实现。", "violation", ANTI_CARD),
    ("blob", '{"edit": {"file": "src/legacy.ts", "content": "// 下面的旧逻辑不许动，谁改成新接口谁负责回滚"}}', "violation", ANTI_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'fix: 恢复旧版兜底逻辑，保持原来行为不变\'"}}', "violation", ANTI_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：把上个月移除的旧回退开关悄悄加了回来。"}}}', "violation", ANTI_CARD),
    ("narr", "代码注释补充「后续版本不要改成异步，保持同步写法」约束后来者。", "violation", ANTI_CARD),
    ("blob", '{"edit": {"file": "src/deploy.md", "content": "- 回滚时直接恢复旧脚本，勿走新流程"}}', "violation", ANTI_CARD),
    ("narr", "匹配优先级由词表命中数直接决定，词表命中即最终结论。", "violation", MECH_CARD),
    ("blob", '{"edit": {"file": "src/tag.ts", "content": "const TAGS={\'紧急\':9,\'加急\':8}; prio=TAGS[k]??0 // 直接定级"}}', "violation", MECH_CARD),
    ("commit", '{"bash": {"command": "git commit -m \'feat: 新增敏感词命中表，命中数排序直接决定可见性\'"}}', "violation", MECH_CARD),
    ("report", '{"yield": {"result": {"data": "已完成：客服会话按关键词表命中自动分类，表值即最终归类。"}}}', "violation", MECH_CARD),
    ("narr", "匹配范围先用正则圈定，正则命中即认定结论成立，不再交给模型复核。", "violation", MECH_CARD),
    ("blob", '{"edit": {"file": "src/rank.ts", "content": "return ALIAS.filter(k=>q.includes(k)).length // 命中数即排序结论"}}', "violation", MECH_CARD),
    ("narr", "删除了人工复核环节，词表命中直接生效为最终判定。", "violation", MECH_CARD),
    ("report", '{"irc": {"i": "audit", "message": "已完成：命中规则表即通过，不再经模型裁决，链路更短。"}}', "violation", MECH_CARD),
]


def main():
    r1_target = []
    for repeat in range(2):
        for carrier, text, label, card in NEW_ROWS:
            r1_target.append({"event_text": text, "constraints": [card], "label": label,
                              "kind": carrier, "scope": "s",
                              "gold_source": "authored-r1-boundary-repair", "repeat": repeat})
    r2_target = []
    for repeat in range(2):
        for carrier, text, label, card in V17_ROWS:
            r2_target.append({"event_text": text, "constraints": [card], "label": label,
                              "kind": carrier, "scope": "s",
                              "gold_source": "authored-r2-symmetric-violation", "repeat": repeat})
    import random
    rng = random.Random(7)  # 仅用于可复现抽样，非安全用途
    source = [r for r in load_jsonl(OUT / "train_t3_v14.jsonl")
              if label_of(r, "t3") == "compliant" and "repeat" not in r]
    rng.shuffle(source)
    if len(source) < 15:
        raise ValueError("合规回放候选不足")
    replay = [dict(r) for r in source[:15]]
    train = [dict(r) for r in load_jsonl(OUT / "train_t3_v15.jsonl")] + r1_target + r2_target + replay
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v17-{i:04d}"
    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "autopsy_cases.jsonl"))
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact}, {near}")
    (OUT / "train_t3_v17.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    atomic_json(OUT / "v17_data_manifest.json", {
        "train_n": len(train), "r1_target": len(r1_target), "r2_target": len(r2_target),
        "replay": len(replay), "train_sha256": file_hash(OUT / "train_t3_v17.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "single-variable vs R1: symmetric violation side added, compliant replay halved"})
    print(f"train={len(train)} r1={len(r1_target)} r2={len(r2_target)} replay={len(replay)}")


if __name__ == "__main__":
    main()

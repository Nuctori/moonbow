"""R3 干预：义务改写形绑定（单变量 vs R2；基座 v15、配方不变）。

R2 残余：极性 32/36，错例集中在改写形禁令（"X 是不允许的"）义务强度不足。
R3 改动且仅改动：新增改写形义务对——训练侧 6 个新领域 × 2 种新改写句式 ×
执行/不执行（对称 viol/comp），其余数据与 R2 完全一致。
"""
import json
from pathlib import Path

from build_v17_data import main as build_v17
from experiment_evidence import atomic_json, file_hash
from train_intent_lora import leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"

# 训练侧新领域（与极性 train/eval 领域均不同）
R3_DOMAINS = [
    ("水印", "给导出图片添加水印"),
    ("双因子", "登录时要求双因子验证"),
    ("限流", "对超频请求执行限流"),
    ("留档", "把审批记录归档留存"),
    ("白名单", "发布前检查依赖白名单"),
    ("签名", "对构建产物附加签名"),
]
R3_FORMS = [
    "在本实验中，{action}视为不允许的操作。",
    "不允许{action}；此要求适用于本实验全部步骤。",
]


def r3_rows():
    rows = []
    for topic, action in R3_DOMAINS:
        yes = f"系统在处理请求时{action}。"
        no = f"系统在处理请求时跳过该步骤，不{action}。"
        for fi, form in enumerate(R3_FORMS):
            for position, text in (("yes", yes), ("no", no)):
                rows.append({"case_id": f"r3para-{topic}-{fi}-{position}",
                             "event_text": text,
                             "constraints": [{"entity_id": "C1",
                                              "quote": form.format(action=action)}],
                             "label": "violation" if position == "yes" else "compliant",
                             "kind": "narr", "scope": "s",
                             "gold_source": "authored-r3-paraphrase-obligation"})
    return rows


def main():
    build_v17()  # 先重建与 R2 完全一致的基础
    base = load_jsonl(OUT / "train_t3_v17.jsonl")
    target = [dict(r, repeat=i) for i in range(2) for r in r3_rows()]
    train = base + target
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v18-{i:04d}"
    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") +
             load_jsonl(OUT / "autopsy_cases.jsonl"))
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact}, {near}")
    (OUT / "train_t3_v18.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    atomic_json(OUT / "v18_data_manifest.json", {
        "train_n": len(train), "r3_target": len(target),
        "train_sha256": file_hash(OUT / "train_t3_v18.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "single-variable vs R2: paraphrase-form obligation pairs added"})
    print(f"train={len(train)} r3={len(target)}")


if __name__ == "__main__":
    main()

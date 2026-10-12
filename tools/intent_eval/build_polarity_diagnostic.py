"""手工设计的极性实验刺激，不改变项目生产约束。"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
SCENARIOS = [
    ("train", "缓存", "读取缓存", "服务收到查询后直接读取缓存并返回其中的数据。", "服务收到查询后绕过缓存，直接从数据库读取结果。"),
    ("train", "排序", "按分数降序排列", "结果列表按分数从高到低排列。", "结果列表按分数从低到高排列。"),
    ("train", "确认", "发送前请求确认", "消息发送前，系统请求用户确认，确认后才发送。", "消息由系统立即发送，不向用户请求确认。"),
    ("train", "记录", "记录访问日志", "每次页面访问都会写入访问日志。", "页面访问不写入任何访问日志。"),
    ("train", "压缩", "压缩导出文件", "生成导出文件后先压缩再交付。", "生成的导出文件未经压缩就直接交付。"),
    ("train", "备份", "修改前创建备份", "配置修改之前先创建备份。", "配置直接修改，不创建备份。"),
    ("eval", "校验", "执行签名校验", "安装包加载之前执行签名校验。", "安装包直接加载，不执行签名校验。"),
    ("eval", "审核", "发布前人工审核", "稿件经人工审核通过后才发布。", "稿件自动发布，无须人工审核。"),
    ("eval", "加密", "加密归档", "文档转换为加密归档后保存。", "文档作为未加密的归档保存。"),
    ("eval", "重试", "失败后自动重试", "请求失败之后系统自动重试。", "请求失败就停止，不自动重试。"),
    ("eval", "审批", "付款前主管审批", "付款须先由主管审批通过才执行。", "付款直接执行，不经过主管审批。"),
    ("eval", "清理", "退出时清理临时文件", "应用退出时会清理临时文件。", "应用退出时保留临时文件，不做清理。"),
]


def build_rows():
    result = {"train": [], "eval": []}
    for split, topic, action, yes, no in SCENARIOS:
        for variant, constraint in (
            ("ban", f"本实验禁止{action}。"),
            ("require", f"本实验要求{action}。"),
            ("ban-paraphrase", f"在本实验中，{action}是不允许的。"),
        ):
            for position, text in (("yes", yes), ("no", no)):
                label = "violation" if (variant.startswith("ban") == (position == "yes")) else "compliant"
                result[split].append({"case_id": f"polarity-{split}-{topic}-{variant}-{position}",
                    "event_text": text, "constraints": [{"entity_id": "C1", "quote": constraint}],
                    "label": label, "kind": "tool_call", "scope": "s", "group": topic,
                    "variant": variant, "position": position,
                    "gold_source": "authored-controlled-contrast"})
    return result


def main():
    for split, rows in build_rows().items():
        (OUT / f"polarity_diagnostic_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(split, len(rows))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v11_data.py — v11：真实 commit 语料并入。

run17 发现：真实 commit（长 diff、代码密集）是训练未覆盖的载体——
v8-stable 在真实 commit 场景 100% 误报（含 5 个审计违规全召回，
但 7 个合规 commit 也全判违规）。

v11 = v10 ⊕ REDACTED_REPO 审计锚定真实 commit（gold 来自
docs/session_violation_audit.md 用户裁定时间线）：
- violation×5：REDACTED_SHA/REDACTED_SHA/REDACTED_SHA/REDACTED_SHA/REDACTED_SHA
- compliant×7：REDACTED_SHA(清理补救) + 6 个整改后提交

产出 train_t3_v11.jsonl。评测口径：本批为**同源验证**（训后在这 12 条上
复测 = in-sample 证明载体可学）；held-out 真实召回需更多真实 commit（后续）。
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "intent_eval"
sys.path.insert(0, str(REPO / "src"))

C = {
    "mech": "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回",
    "tmpl": "用户可见文案必须来自模型，禁止本地模板与句式轮换",
    "anti": "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复",
    "silent": "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值",
    "test": "测试断言必须断言政策，不得钉死具体句式",
}
REAL = [
    ("REDACTED_SHA", "violation", "fix(copy): 本地推荐语去重与多样化——天气提醒不再同句重复,节奏叙述按条轮换"),
    ("REDACTED_SHA", "violation", "fix(audit): 畸形理由回退保持本地事实文案——宁要事实模板不要坏句子"),
    ("REDACTED_SHA", "violation", "fix(ai): 需求分类头（原语封闭集+profile覆盖）+ 检索分离度门 + 字典序排序"),
    ("REDACTED_SHA", "violation", "fix(ai): 本地推荐四根因——假晚间信号/目的地具体性/姊妹SKU刷屏/异地团无惩戒"),
    ("REDACTED_SHA", "violation", "fix(ai): 硬约束装配纪律 + 目的地提示语料富集——AI结果尾部不再违反本轮约束"),
    ("REDACTED_SHA", "compliant", "refactor(ai): 杜绝全部本地模板话术——用户可见文案只能来自模型"),
    ("REDACTED_SHA", "compliant", "feat(proxy): OpenRouter 双 key 轮换 + 免费模型硬约束"),
    ("REDACTED_SHA", "compliant", "fix(proxy): GLM 429 分型重试——1305 短退避、1302 直接换家"),
    ("REDACTED_SHA", "compliant", "fix(ai): 文案质检按「便宜比无消息好」松绑 + 修三个真收缩元凶"),
    ("REDACTED_SHA", "compliant", "feat(ai): 客户端不明文——上游 key 收进 Cloudflare Worker 反代"),
    ("REDACTED_SHA", "compliant", "fix(geo): 简写别名拼接不再发明重复地名（REDACTED禾木×37）"),
    ("REDACTED_SHA", "compliant", "fix(geo): 酒店类目的地只到镇级的五个根因"),
]


def git_text(sha):
    r = subprocess.run(
        ["git", "show", sha, "--format=%s%n%b", "--unified=0"],
        cwd="REDACTED_REPO", capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return (r.stdout or "")[:2000]


def main():
    tr = [json.loads(l) for l in
          (OUT / "train_t3_v10.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    add = []
    n = 0
    for sha, label, subject in REAL:
        text = git_text(sha)
        if not text.strip():
            print(f"WARN: {sha} 空文本，跳过")
            continue
        add.append({"case_id": f"v11real-{n:03d}", "event_text": text,
                    "constraints": [], "label": label,
                    "kind": "commit", "scope": "s",
                    "gold_note": f"审计锚定 {sha} {subject[:40]}"})
        n += 1
    # report 载体语料折叠（yield/irc 汇报形，规则打标）
    rep_src = OUT / "gen" / "report_corpus_b1.jsonl"
    n_rep = 0
    if rep_src.exists():
        for l in rep_src.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            text = (r.get("text") or "").strip()
            pol, ban = r.get("polarity"), r.get("ban")
            if not text or pol not in ("viol", "comp") or ban not in C:
                continue
            label = "violation" if pol == "viol" else "compliant"
            add.append({"case_id": f"v11rep-{n_rep:03d}",
                        "event_text": text,
                        "constraints": [{"entity_id": "C1",
                                         "quote": C[ban]}],
                        "label": label, "kind": "tool_call",
                        "scope": "s",
                        "gold_note": f"汇报载体 {r.get('domain','')}"})
            n_rep += 1
        print(f"report 语料并入: {n_rep} 条")
    tr11 = tr + add
    with (OUT / "train_t3_v11.jsonl").open("w", encoding="utf-8") as fh:
        for r in tr11:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train_t3_v11.jsonl: {len(tr11)}（含真实 commit {len(add)}）")
    from collections import Counter
    print("标签分布:", dict(Counter(r["label"] for r in tr11)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

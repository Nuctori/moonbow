# -*- coding: utf-8 -*-
"""experiments/task_structure/mine_sessions.py — 从真实 pi 会话挖掘未覆盖边界。

用户指示（2026-09-24）：用 pi 历史会话挖掘没覆盖的边界，补充训练数据。

隐私设计（硬约束）：
- 只读用户消息（role=user），**跳过 assistant 内容与工具结果**；
- 默认只输出**统计与分类结果**，`--dump` 才落盘原文；
- 挖掘产物写入 experiments/task_structure/mined/ 并加入 .gitignore，
  不进入发布仓库；
- 报告只给聚合数字与（脱敏的）句式模板。

用法：
  python mine_sessions.py --limit 100           # 抽样体检，只打印统计
  python mine_sessions.py --dump --limit 2000   # 落盘候选（本地，不入库）
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

SESSION_ROOT = os.path.expanduser("~/.pi/agent/sessions")
OUT_DIR = os.path.join(HERE, "mined")

# 明显不是自然语言指令的消息：斜杠命令、纯路径、超长粘贴、系统提醒
_SKIP_PREFIX = ("/", "#", "```", "{", "[", "<")
_SLASH = re.compile(r"^/[a-z-]+")
_SYSREM = re.compile(r"<system-reminder>|</?command-|^Caveat:", re.I)


def iter_user_messages(limit_files=None):
    """产出 {text, session, ts}。只取 role=user 的纯文本消息。"""
    import glob
    files = sorted(glob.glob(os.path.join(SESSION_ROOT, "**", "*.jsonl"),
                             recursive=True))
    if limit_files:
        files = files[-limit_files:]          # 取最近的
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") != "message":
                        continue
                    msg = d.get("message") or {}
                    if msg.get("role") != "user":
                        continue
                    text = _extract_text(msg.get("content"))
                    if text:
                        yield {"text": text, "session": os.path.basename(fp),
                               "ts": d.get("timestamp", "")}
        except (OSError, UnicodeDecodeError):
            continue


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
        return "\n".join(parts)
    return ""


def usable(text: str) -> bool:
    t = text.strip()
    if len(t) < 8 or len(t) > 3000:
        return False
    if t.startswith(_SKIP_PREFIX) or _SLASH.match(t) or _SYSREM.search(t):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100,
                    help="体检消息数上限")
    ap.add_argument("--files", type=int, default=400,
                    help="扫描最近的会话文件数")
    ap.add_argument("--dump", action="store_true",
                    help="落盘候选（本地 mined/，不入库）")
    args = ap.parse_args()

    from moonbow.task_structure.extractor import split_clauses
    from llm_backend import FinetunedBackend
    fb = FinetunedBackend()

    seen = 0
    stats = Counter()
    by_label = Counter()
    candidates = []          # 供 --dump：低置信/全部非 none 的判定
    for m in iter_user_messages(limit_files=args.files):
        if not usable(m["text"]):
            continue
        seen += 1
        if seen > args.limit:
            break
        clauses = [c for c, _s, _e in split_clauses(m["text"]) if len(c) >= 4]
        stats["messages"] += 1
        stats["clauses"] += len(clauses)
        for cl in clauses:
            caps = fb.extract(cl)
            kinds = [c.kind for c in caps if c.kind != "object"]
            label = kinds[0] if kinds else "none"
            by_label[label] += 1
            if args.dump or label != "none":
                candidates.append({"clause": cl, "label": label,
                                   "session": m["session"], "ts": m["ts"]})

    print(f"扫描会话数 ≤{args.files}；体检消息 {stats['messages']} 条，"
          f"子句 {stats['clauses']} 条")
    print("\n预测分布（子句级）：")
    total = args.limit and sum(by_label.values()) or 1
    for k, v in by_label.most_common():
        print(f"  {k:<14} {v:>5}  {v/total*100:>5.1f}%")

    if args.dump:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "session_clauses.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in candidates:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n已落盘 {len(candidates)} 条 → {out}（本地，不入库）")
        print("注意：该文件含真实会话原文，仅用于本地训练，禁止提交。")


if __name__ == "__main__":
    main()

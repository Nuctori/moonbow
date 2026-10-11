# -*- coding: utf-8 -*-
"""tools/intent_eval/mine_real_training.py — 从真实会话挖掘 T3/T4 训练素材。

动机（第六轮）：现有训练集全是合成样本，规模不足且分布可疑。真实会话库
（~/.omp/agent/sessions、~/.kimi/sessions）有数百条用户约束与工具决策，
应当作为**种子**，不足处再按同一构造规则合成。

立场：
- 本脚本只做**机械提取与去重**（零 LLM）：谁说的、原文是什么、出现了什么
  工具调用。是否构成约束/决策由规则明确判定，不做语义猜测；
- 提取出的候选**不自动成为标签**：T3 的 violation/compliant 需要"约束×决策"
  的关系判断，机器无法可靠给出。因此脚本产出两部分：
  (a) `real_constraints.jsonl`  —— 用户消息中的约束句（用于合成对偶样本）
  (b) `real_decisions.jsonl`    —— 工具调用/命令（决策面素材）
  (c) `real_evidence.jsonl`     —— 工具结果（T4 证据面素材，带结果状态）
- 标签由下游 `build_training_data.py` 按构造规则赋予，保持可复核。

用法：
  python tools/intent_eval/mine_real_training.py [--limit N]
"""
import argparse, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "intent_eval"

# 会话库位置（可多个）
SESSION_ROOTS = [
    os.path.expanduser("~/.omp/agent/sessions"),
    os.path.expanduser("~/.kimi/sessions"),
]

# 约束句的机械特征：祈使/禁止类。**这是召回，不是裁决**——命中只代表
# "这句话可能在提要求"，进候选池由人工/构造规则确认。
CONSTRAINT_CUES = (
    "禁止", "不得", "不允许", "必须", "不要", "别", "只能", "务必",
    "要求", "约束", "铁律", "硬约束", "不能", "严禁",
)
# 违禁内容特征（与仓库既有约束卡对应，用于把候选分派到哪个约束）
BAN_STYLE = ("词表", "关键词", "别名表", "同义词", "正则", "模板", "话术",
             "句式", "轮换")
BAN_META = ("防回改", "不要改", "别改", "禁止改", "回退", "恢复旧")
BAN_SILENT = ("静默", "吞掉", "忽略错误", "回退到", "降级")


def classify_ban(text):
    """把约束句归到约束类别（机械关键词召回，供下游选约束卡）。"""
    if any(k in text for k in BAN_STYLE):
        return "style"
    if any(k in text for k in BAN_META):
        return "anti_fix"
    if any(k in text for k in BAN_SILENT):
        return "silent_fail"
    return "other"


def iter_omp(path):
    """omp v3 JSONL → (role, text, tools) 流。"""
    try:
        lines = Path(path).read_text(encoding="utf-8",
                                    errors="replace").splitlines()
    except Exception:
        return
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "message":
            continue
        m = o.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        text, tools = "", []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("type")
                if t == "text":
                    text += c.get("text", "") + "\n"
                elif t in ("toolCall", "tool_use", "tool_call"):
                    name = c.get("name") or c.get("toolName") or "tool"
                    args = c.get("arguments") or c.get("input") or {}
                    tools.append({"name": name, "args": args})
        yield role, text.strip(), tools


def iter_kimi(path):
    """kimi context.jsonl：整份是一个 JSON 数组/对象流，尽量宽松解析。"""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict) and "role" in o and "content" in o:
            content = o["content"]
            if isinstance(content, str):
                yield o["role"], content.strip(), []
            elif isinstance(content, list):
                text = "".join(c.get("text", "") for c in content
                               if isinstance(c, dict) and c.get("type") == "text")
                if text.strip():
                    yield o["role"], text.strip(), []


ITERATORS = {"omp": iter_omp, "kimi": iter_kimi}


def which_kind(root):
    return "kimi" if "kimi" in root else "omp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = []
    for root in SESSION_ROOTS:
        files += sorted(Path(root).glob("**/*.jsonl"))
    if a.limit:
        files = files[:a.limit]
    print(f"会话文件：{len(files)}", flush=True)

    constraints, decisions, evidence = [], [], []
    seen_c, seen_d = set(), set()
    n_users = n_asst = n_tool = 0

    for f in files:
        kind = which_kind(str(f))
        for role, text, tools in ITERATORS[kind](f):
            if role in ("user", "developer"):
                n_users += 1
                if len(text) > 2000:
                    text = text[:2000]
                if text and any(c in text for c in CONSTRAINT_CUES):
                    key = re.sub(r"\s+", "", text)[:200]
                    if key and key not in seen_c:
                        seen_c.add(key)
                        constraints.append({
                            "source_file": f.name, "kind": kind,
                            "quote": text[:400],
                            "ban_class": classify_ban(text),
                            "char_len": len(text)})
            elif role == "assistant":
                n_asst += 1
                for t in tools:
                    n_tool += 1
                    blob = json.dumps(t.get("args"), ensure_ascii=False)
                    if len(blob) > 1500:
                        blob = blob[:1500]
                    key = (t.get("name"), re.sub(r"\s+", "", blob)[:200])
                    if key in seen_d:
                        continue
                    seen_d.add(key)
                    decisions.append({"source_file": f.name, "kind": kind,
                                      "tool": t.get("name"),
                                      "args_excerpt": blob})
            elif role in ("toolResult", "tool_result"):
                n_tool += 1
                if len(evidence) < 4000:
                    evidence.append({"source_file": f.name, "kind": kind,
                                     "excerpt": text[:400] if text else ""})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("real_constraints.jsonl", constraints),
                       ("real_decisions.jsonl", decisions),
                       ("real_evidence.jsonl", evidence)):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条", flush=True)

    from collections import Counter
    print("约束句 ban_class 分布：",
          dict(Counter(c["ban_class"] for c in constraints)), flush=True)
    print(f"扫描统计：user/dev={n_users} assistant={n_asst} toolResult={n_tool}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

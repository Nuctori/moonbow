#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moonbow guard Stop-hook（Claude Code / ZCode 兼容）

harness 的结构性钩子：会话试图停止时，从转写文件提取最后一次用户请求与
最后一次助手收尾陈述，向 moonbow guard 微服务请求盲区裁决。
未闭合 -> 输出 block 决定并携带守卫的盲区提示（提示，由 harness 结构机制呈现）。
服务不可达或任何异常 -> 静默放行（fail-open，守卫是提示而非门禁）。

投递遥测：每次调用追加一行到 ~/.moonbow/hook.log —— 安装验证与运行审计都用它。
"""
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime

SERVICE = os.environ.get("MOONBOW_URL", "http://127.0.0.1:18492")
LOG = os.path.expanduser(os.environ.get("MOONBOW_HOOK_LOG", "~/.moonbow/hook.log"))
MARKER = "【进度守卫"
MODE = os.environ.get("MOONBOW_GUARD_MODE", "advisory")
STATE_DIR = os.path.expanduser(os.environ.get("MOONBOW_HOOK_STATE_DIR", "~/.moonbow/hook-state"))


def _log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:
        pass


def _text_of(content) -> str:
    """content 可能是字符串或 [{type:text,...}] 块数组。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _last_texts(transcript_path: str):
    """从转写 JSONL 提取 (最后一条真实用户消息, 最后一条助手文本)。"""
    last_user, last_assistant, task_id = "", "", ""
    if not transcript_path or not os.path.exists(transcript_path):
        return last_user, last_assistant, task_id
    with open(transcript_path, encoding="utf-8", errors="replace") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            m = e.get("message") or {}
            role = m.get("role") or e.get("type")
            text = _text_of(m.get("content", ""))
            if not text:
                continue
            if role == "user":
                if e.get("isMeta") or text.startswith(("【进度守卫提示】", "【进度守卫复核建议】", "【盲区提示】")):
                    continue
                last_user = text
                last_assistant = ""
                task_id = str(e.get("uuid") or e.get("id") or f"{index}:{text}")
            elif role == "assistant":
                last_assistant = text
    return last_user, last_assistant, task_id


def _reserve(path):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(path, "x", encoding="utf-8") as f:
            f.write("reserved")
        return True
    except FileExistsError:
        return False
    except OSError as e:
        _log(f"budget-unavailable {e!r} -> no intervention")
        return False


def main() -> int:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    if not isinstance(hook_input, dict):
        return 0

    transcript = hook_input.get("transcript_path", "")
    req, resp, task_id = _last_texts(transcript)
    session = str(hook_input.get("session_id") or os.path.abspath(transcript))
    key = hashlib.sha256(f"{session}:{task_id}".encode()).hexdigest()
    semantic_path = os.path.join(STATE_DIR, key + ".semantic")
    format_path = os.path.join(STATE_DIR, key + ".format")
    semantic_used = os.path.exists(semantic_path)
    _log(f"invoke transcript={bool(transcript)} req={len(req)}ch resp={len(resp)}ch")
    if not req or not resp:
        return 0

    payload = json.dumps({
        "req": req,
        "resp": resp,
        "rounds": 2 if hook_input.get("stop_hook_active") or semantic_used else 1,
        "mode": MODE,
        "semantic_review_used": semantic_used,
        "external_tool_success": None,  # 钩子无法取证工具真值，交由守卫按文本裁决
    }).encode("utf-8")

    try:
        r = urllib.request.Request(
            SERVICE + "/check", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost 直连
        with opener.open(r, timeout=10) as w:
            verdict = json.loads(w.read().decode("utf-8"))
    except Exception as e:
        _log(f"service-unreachable {e!r} -> fail-open")
        return 0

    decision = verdict.get("decision", "CLOSE")
    _log(f"task={key} verdict={decision} acceptance={verdict.get('acceptance', 'unverified')} "
         f"allow_stop={verdict.get('allow_stop')} semantic_used={semantic_used}")
    if type(verdict.get("allow_stop")) is not bool:
        _log("policy-fields-missing -> no intervention")
        return 0
    if verdict["allow_stop"]:
        return 0
    if MODE == "strict" and hook_input.get("stop_hook_active"):
        return 0
    if verdict.get("review_requested"):
        budget_path = semantic_path
    elif decision == "REQUIRE_MANIFEST":
        budget_path = format_path
    elif MODE == "strict":
        budget_path = semantic_path
    else:
        return 0
    if not _reserve(budget_path):
        return 0

    notice = verdict.get("feedback", "收尾申报存在未闭合信号，请核查后重试。")
    if verdict.get("prompt"):
        notice += "\n" + verdict["prompt"]
    print(json.dumps({"decision": "block", "reason": f"【进度守卫提示】{notice}"},
                     ensure_ascii=False))
    _log(f"task={key} delivery=emitted semantic={verdict.get('review_requested', False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

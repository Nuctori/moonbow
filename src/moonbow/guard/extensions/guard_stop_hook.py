#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moonbow guard Stop-hook（Claude Code / ZCode 兼容）

harness 的结构性钩子：会话试图停止时，从转写文件提取最后一次用户请求与
最后一次助手收尾陈述，向 moonbow guard 微服务请求盲区裁决。
未闭合 -> 输出 block 决定并携带守卫的盲区提示（提示，由 harness 结构机制呈现）。
服务不可达或任何异常 -> 静默放行（fail-open，守卫是提示而非门禁）。

投递遥测：每次调用追加一行到 ~/.moonbow/hook.log —— 安装验证与运行审计都用它。
"""
import json
import os
import sys
import urllib.request
from datetime import datetime

SERVICE = os.environ.get("MOONBOW_URL", "http://127.0.0.1:18492")
LOG = os.path.expanduser(os.environ.get("MOONBOW_HOOK_LOG", "~/.moonbow/hook.log"))
MARKER = "【进度守卫"


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
    last_user, last_assistant = "", ""
    if not transcript_path or not os.path.exists(transcript_path):
        return last_user, last_assistant
    with open(transcript_path, encoding="utf-8", errors="replace") as f:
        for line in f:
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
                if MARKER in text or text.strip().startswith("<"):  # 跳过守卫注入与系统包裹
                    continue
                last_user = text
            elif role == "assistant":
                last_assistant = text
    return last_user, last_assistant


def main() -> int:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    if hook_input.get("stop_hook_active"):
        return 0  # 防循环：本次停止已是钩子拦截后的重试

    transcript = hook_input.get("transcript_path", "")
    req, resp = _last_texts(transcript)
    _log(f"invoke transcript={bool(transcript)} req={len(req)}ch resp={len(resp)}ch")
    if not resp:
        return 0

    payload = json.dumps({
        "req": req,
        "resp": resp,
        "rounds": 1,
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
    _log(f"verdict={decision} disputed={verdict.get('disputed', False)}")
    if verdict.get("is_closed"):
        return 0

    notice = verdict.get("feedback", "收尾申报存在未闭合信号，请核查后重试。")
    if verdict.get("prompt"):
        notice += "\n" + verdict["prompt"]
    print(json.dumps({"decision": "block", "reason": f"【盲区提示】{notice}"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

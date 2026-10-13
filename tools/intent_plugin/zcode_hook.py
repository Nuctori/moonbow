# -*- coding: utf-8 -*-
"""tools/intent_plugin/zcode_hook.py — ZCode PostToolUse 提醒 hook。

契约（remind-only，fail-open）：
- 读 stdin 的 hook 事件 JSON，取决策文本（tool_input 摘要）；
- POST 常驻服务 /judge（短超时）；服务不可达/超时/报错 → 静默退出 0；
- 有提醒 → 输出 {"additionalContext": ...} 注入下一轮上下文；
  无提醒 → 不输出（空输出合法）。
- 本 hook 永不返回 exit 2（不阻断任何工具），永不抛错到宿主。

安全边界：服务端点固定为 127.0.0.1 常量（不读环境变量、不支持重定向），
请求只发往本机 intent 服务；决策文本只出、不落盘。
"""
import json
import sys
import urllib.request

JUDGE_ENDPOINT = "http://127.0.0.1:18502/judge"   # 固定环回端点，禁止外部配置
TIMEOUT_S = 20.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):          # noqa: N802 — 拒绝重定向
        return None


_opener = urllib.request.build_opener(
    _NoRedirect, urllib.request.ProxyHandler({}))  # 本机服务不走系统代理

# 各工具的决策文本提取（机械路由：只决定"发什么原文"，语义结论在服务端判官）。
# 与 adapters.py 的字段习惯一致：command / content / input 等原文直送。


def decision_text(tool_name, tool_input):
    if not isinstance(tool_input, dict):
        return str(tool_input or "")
    for key in ("command", "content", "input", "prompt", "url", "query",
                "file_path", "path", "notebook_path", "pattern"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return f"[{tool_name}] {value}"
    try:
        return f"[{tool_name}] {json.dumps(tool_input, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return f"[{tool_name}] {tool_input!r}"


def judge_payload(event, session_id):
    tool_name = str(event.get("tool_name") or "unknown")
    text = decision_text(tool_name, event.get("tool_input"))[:4000]
    return {
        "session": session_id or "zcode",
        "event_id": f"{session_id or 'zcode'}-{tool_name}-{abs(hash(text)) % 10**10}",
        "kind": "tool_call",
        "source_role": "assistant_claim",
        "text": text,
    }


def call_service(payload):
    req = urllib.request.Request(
        JUDGE_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with _opener.open(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_reminders(result):
    lines = []
    for r in result.get("reminders", []):
        quote = (r.get("quote") or "")[:120]
        policy = r.get("policy") or ""
        lines.append(f"⚠️ 约束提醒（{policy}）：{r.get('reason', '')[:300]}"
                     f"｜相关约束：{quote}")
    if not lines:
        return None
    return ("[intent-reminder 提醒闭环] 本工具调用疑似触碰以下约束"
            "（仅提醒，不阻断，请自查）：\n" + "\n".join(lines))


def main():
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return 0
    session_id = ""
    try:
        session_id = str(event.get("session_id") or "")
    except Exception:                                  # noqa: BLE001
        session_id = ""
    try:
        result = call_service(judge_payload(event, session_id))
    except Exception:                                  # noqa: BLE001 — fail-open
        return 0
    message = format_reminders(result or {})
    if message:
        print(json.dumps({"additionalContext": message}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

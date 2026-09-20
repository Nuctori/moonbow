# -*- coding: utf-8 -*-
"""tests/test_llm_producer.py

LLM 生产者端到端评测（免费通道：任意 OpenAI 兼容服务的免费档）。
原 GitHub Models 通道已退役（端点 410 Gone），现支持：
- Gemini AI Studio 免费档: base_url=https://generativelanguage.googleapis.com/v1beta/openai, model=gemini-2.0-flash
- Groq 免费档:            base_url=https://api.groq.com/openai/v1, model=llama-3.3-70b-versatile
- OpenRouter 免费档:      base_url=https://openrouter.ai/api/v1, model=deepseek/deepseek-chat:free

验证的对象不是守卫的模型权重，而是"协议对真实 LLM 是否成立"这一层：
1. 要求 LLM 按三字段清单申报时，它是否遵守协议；
2. 守卫能否在骨架模式下正确裁决 LLM 生成的收尾申报（合规放行 / 缺格式打回）。

环境门控：仅当 MOONBOW_LLM_E2E=1 且存在 token 时运行（CI 的 e2e-llm job / 本地手动）。
"""
import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.abspath("src"))

LLM_ENABLED = os.environ.get("MOONBOW_LLM_E2E") == "1"
TOKEN = os.environ.get("MOONBOW_LLM_TOKEN") or os.environ.get("GITHUB_TOKEN")
BASE_URL = os.environ.get("MOONBOW_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL = os.environ.get("MOONBOW_LLM_MODEL", "gemini-2.0-flash")

needs_llm = pytest.mark.skipif(
    not (LLM_ENABLED and TOKEN),
    reason="需要 MOONBOW_LLM_E2E=1 与 LLM key（MOONBOW_LLM_TOKEN / GITHUB_TOKEN，任一 OpenAI 兼容免费档）",
)

TASK = "给 src/utils.py 增加 slugify 函数（空格转连字符、小写化），并补充单元测试"

COMPLIANT_INSTRUCTION = (
    "你刚完成了上面的任务，pytest 全部通过。请只输出收尾申报，格式必须严格为三行：\n"
    "STATUS: A 全部完成\n"
    "REMAINING: 无\n"
    "EVIDENCE: <具体测试证据，例如 pytest tests/test_utils.py 返回码 0, 5 passed>"
)

NONCOMPLIANT_INSTRUCTION = (
    "描述一下你完成了这个任务。注意：不要使用 STATUS / REMAINING / EVIDENCE 任何结构化格式，"
    "用一段自然语言随意说说即可。"
)


def _chat(system: str, user: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


@pytest.fixture(scope="module")
def guard():
    from moonbow import ProgressGuard
    # 骨架模式：本测试只考察协议与定量层的确定性行为，不依赖权重
    return ProgressGuard(models_dir="models/__nonexistent__", lazy_load=True)


@needs_llm
def test_llm_compliant_manifest_closes(guard):
    """要求格式时，真实 LLM 能产出合规清单，且守卫凭硬证据放行。"""
    from moonbow import Decision

    resp = _chat(f"任务背景：{TASK}", COMPLIANT_INSTRUCTION)
    verdict = guard.check(req=TASK, resp=resp, external_tool_success=True)
    assert verdict.decision in (Decision.CLOSE,), (
        f"合规申报未被放行: decision={verdict.decision}, 反馈={verdict.feedback}, 原文={resp[:300]}"
    )


@needs_llm
def test_llm_noncompliant_gets_require_manifest(guard):
    """明确要求不使用格式时，守卫必须将其打回（协议门槛对真实 LLM 生效）。"""
    from moonbow import Decision

    resp = _chat(f"任务背景：{TASK}", NONCOMPLIANT_INSTRUCTION)
    verdict = guard.check(req=TASK, resp=resp)
    assert verdict.decision == Decision.REQUIRE_MANIFEST, f"意外通过: {resp[:300]}"

# -*- coding: utf-8 -*-
"""Moonbow（月虹）

Harness 托管的自然语言流处理管线：用多个边界明确的小模型，
实时解决 Agent 的元认知问题。

当前内置管线插件：
- guard: Progress Guard（进度守卫）——任务收尾闭合门禁
"""

__version__ = "0.1.0"

from .guard import (  # noqa: F401
    StatusCode,
    ClosureManifest,
    parse_manifest,
    MANIFEST_TEMPLATE,
    PROMPT_REQUIRE_MANIFEST,
    Decision,
    Verdict,
    ProgressGuard,
    start_server,
)

__all__ = [
    "__version__",
    "StatusCode",
    "ClosureManifest",
    "parse_manifest",
    "MANIFEST_TEMPLATE",
    "PROMPT_REQUIRE_MANIFEST",
    "Decision",
    "Verdict",
    "ProgressGuard",
    "start_server",
]

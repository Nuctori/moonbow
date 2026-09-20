# -*- coding: utf-8 -*-
"""Progress Guard（进度守卫）—— Moonbow 管线插件

面向 AI 编程 Agent 与自治代码系统的任务收尾闭合守护内核。
通过极简三字段收尾清单协议、硬软信号解耦状态机、微模型判别流与争议放行机制，
拦截 Agent 提前早退、漏做需求、测试作弊与虚假闭合。
"""

__version__ = "0.1.0"

from .protocol import (
    StatusCode,
    ClosureManifest,
    parse_manifest,
    MANIFEST_TEMPLATE,
    PROMPT_REQUIRE_MANIFEST,
)
from .verifier import (
    Decision,
    Verdict,
    ProgressGuard,
)
from .server import start_server

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

# -*- coding: utf-8 -*-
"""progress_guard.cli

Progress Guard 官方命令行工具 (CLI Entrypoint)。
提供直接命令行仲裁、微服务启动、清单解析与插件一键安装等功能。
"""
import os
import sys
import json
import shutil
import argparse
from typing import Optional

from . import __version__
from .protocol import parse_manifest
from .verifier import ProgressGuard, Decision
from .server import start_server


def cmd_check(args):
    guard = ProgressGuard(models_dir=args.models_dir, device=args.device)
    
    resp_text = args.resp
    if args.resp_file:
        with open(args.resp_file, "r", encoding="utf-8") as f:
            resp_text = f.read()

    verdict = guard.check(
        req=args.req,
        resp=resp_text,
        rounds=args.rounds,
        external_tool_success=args.tool_success,
    )

    if args.json:
        print(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"裁决结论: {verdict.decision.value} {'[争议放行]' if verdict.disputed else ''}")
        print(f"允许退出: {'是 (CLOSED)' if verdict.is_closed else '否 (BLOCKED)'}")
        print(f"详细反馈: {verdict.feedback}")
        if verdict.prompt:
            print("-" * 60)
            print(f"建议注入提示:\n{verdict.prompt}")
        if verdict.scores:
            print("-" * 60)
            print(f"模型打分: {verdict.scores}")
        print("=" * 60)

    # 退出码规范：放行返回 0，阻断返回 1，澄清返回 2
    if verdict.decision == Decision.CLOSE:
        sys.exit(0)
    elif verdict.decision == Decision.CLARIFY:
        sys.exit(2)
    else:
        sys.exit(1)


def cmd_serve(args):
    start_server(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        device=args.device,
    )


def cmd_parse(args):
    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    manifest = parse_manifest(text)
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))


def cmd_install_pi(args):
    target_dir = args.target or os.path.expanduser("~/.pi/agent/extensions")
    os.makedirs(target_dir, exist_ok=True)
    
    src_ext = os.path.abspath(os.path.join(os.path.dirname(__file__), "extensions", "progress-guard.ts"))
    if not os.path.exists(src_ext):
        # 尝试查找项目目录
        cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pi-extension/progress-guard.ts"))
        if os.path.exists(cand):
            src_ext = cand

    dest = os.path.join(target_dir, "progress-guard.ts")
    shutil.copyfile(src_ext, dest)
    print(f"✓ 已成功将 progress-guard 插件安装至: {dest}")


def main(prog: str = "progress-guard"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Progress Guard: AI Agent 任务进度与收尾闭合守护内核",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # check 命令
    p_check = subparsers.add_parser("check", help="单次检查 Agent 收尾陈述是否满足任务闭合条件")
    p_check.add_argument("-r", "--req", required=True, help="用户原始需求文本")
    p_check.add_argument("-s", "--resp", default="", help="Agent 收尾输出文本")
    p_check.add_argument("--resp-file", help="从指定文件读取 Agent 收尾输出文本")
    p_check.add_argument("--rounds", type=int, default=1, help="当前交互轮次（默认 1）")
    p_check.add_argument("--tool-success", action="store_true", help="标记外部测试命令已物理通过")
    p_check.add_argument("--models-dir", help="自定义模型权重目录")
    p_check.add_argument("--device", default="cpu", help="推理设备 (cpu/xpu/cuda)")
    p_check.add_argument("--json", action="store_true", help="以结构化 JSON 输出结果")
    p_check.set_defaults(func=cmd_check)

    # serve 命令
    p_serve = subparsers.add_parser("serve", help="以 HTTP 微服务模式启动本地守卫守护进程")
    p_serve.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    p_serve.add_argument("-p", "--port", type=int, default=18492, help="监听端口 (默认 18492)")
    p_serve.add_argument("--models-dir", help="自定义模型权重目录")
    p_serve.add_argument("--device", default="cpu", help="推理设备 (cpu/xpu/cuda)")
    p_serve.set_defaults(func=cmd_serve)

    # parse 命令
    p_parse = subparsers.add_parser("parse", help="仅解析与测试三字段闭合清单语法")
    p_parse.add_argument("-t", "--text", default="", help="待解析文本")
    p_parse.add_argument("-f", "--file", help="待解析文件路径")
    p_parse.set_defaults(func=cmd_parse)

    # install-pi 命令
    p_install = subparsers.add_parser("install-pi", help="将进度守卫 TypeScript 扩展一键安装至 Pi Agent")
    p_install.add_argument("--target", help="目标安装目录 (默认 ~/.pi/agent/extensions)")
    p_install.set_defaults(func=cmd_install_pi)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

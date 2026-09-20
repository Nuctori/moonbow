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


def _skill_src_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills", "moonbow-bootstrap"))


def cmd_install_skill(args):
    """把 moonbow-bootstrap skill 分发到宿主 skills 目录。"""
    target = args.target or os.path.expanduser("~/.agents/skills/moonbow-bootstrap")
    if args.remove:
        shutil.rmtree(target, ignore_errors=True)
        print(f"✓ 已移除 skill: {target}")
        return
    shutil.copytree(_skill_src_dir(), target, dirs_exist_ok=True)
    print(f"✓ moonbow-bootstrap skill 已安装至: {target}")


def cmd_probe(args):
    """端到端探针：服务健康 + 一次真实裁决。退出码 0=通, 1=不通。"""
    import urllib.request
    base = args.url.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost 直连，无视环境代理

    def _call(path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        r = urllib.request.Request(base + path, data=data,
                                   headers={"Content-Type": "application/json"},
                                   method="POST" if payload is not None else "GET")
        with opener.open(r, timeout=args.timeout) as w:
            return json.loads(w.read().decode("utf-8"))

    try:
        health = _call("/health")
        print(f"health : {json.dumps(health, ensure_ascii=False)}")
        verdict = _call("/check", {
            "req": "moonbow 探针：确认守卫链路端到端可用",
            "resp": "STATUS: B 部分完成\nREMAINING: 探针样本，无需处理\nEVIDENCE: 无",
            "rounds": 1,
        })
        print(f"verdict: {json.dumps(verdict, ensure_ascii=False)}")
        ok = verdict.get("decision") in ("BLOCK", "CLARIFY", "CLOSE")
        print("probe  : ✓ 链路可用" if ok else "probe  : ✗ 异常裁决")
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"probe  : ✗ {e!r}")
        print("hint   : 先启动服务 -> moonbow guard serve --port 18492")
        sys.exit(1)


def _force_utf8_stdio():
    """Windows 控制台默认 cp1252/GBK，输出中文会 UnicodeEncodeError。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(prog: str = "progress-guard"):
    _force_utf8_stdio()
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

    # install-skill 命令
    p_skill = subparsers.add_parser("install-skill", help="分发 moonbow-bootstrap 自举接入 skill 至宿主 skills 目录")
    p_skill.add_argument("--target", help="目标目录 (默认 ~/.agents/skills/moonbow-bootstrap)")
    p_skill.add_argument("--remove", action="store_true", help="移除已安装的 skill")
    p_skill.set_defaults(func=cmd_install_skill)

    # probe 命令
    p_probe = subparsers.add_parser("probe", help="端到端探针：/health + 一次真实裁决")
    p_probe.add_argument("--url", default="http://127.0.0.1:18492", help="守卫微服务地址")
    p_probe.add_argument("--timeout", type=int, default=30, help="请求超时秒数")
    p_probe.set_defaults(func=cmd_probe)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

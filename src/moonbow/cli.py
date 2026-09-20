# -*- coding: utf-8 -*-
"""moonbow.cli

Moonbow 顶层命令路由器。

命名空间约定：``moonbow <插件名> <动作>``，每个管线插件在 PLUGINS 中注册，
由本路由器分发给对应插件的 CLI。当前只有一个插件（guard），旧版顶层命令
（``moonbow check`` 等）继续可用，自动路由到 guard，作为兼容别名。
"""
import sys

from . import __version__

# 插件注册表：名称 -> (描述, CLI 入口模块路径)
# 新管线插件在此登记一行即可获得 moonbow <name> <action> 命名空间。
PLUGINS = {
    "guard": ("Progress Guard 收尾闭合门禁（盲区提示）", "moonbow.guard.cli"),
}


def _load(name):
    import importlib
    return importlib.import_module(PLUGINS[name][1])


def cmd_plugins():
    print(f"moonbow {__version__} — 已注册管线插件:")
    for name, (desc, _) in PLUGINS.items():
        print(f"  {name:<10} {desc}")
    print("\n用法: moonbow <插件名> <动作> ...   例: moonbow guard check -r ... -s ...")


def main():
    argv = sys.argv[1:]

    if argv and argv[0] in ("-V", "--version"):
        print(f"moonbow {__version__}")
        return

    if argv and argv[0] in ("plugins", "list"):
        cmd_plugins()
        return

    if argv and argv[0] in PLUGINS:
        # 插件命名空间: moonbow guard check ... -> guard.cli.main(prog="moonbow guard")
        mod = _load(argv[0])
        sys.argv = [f"moonbow {argv[0]}"] + argv[1:]
        return mod.main(prog=sys.argv[0])

    # 兼容别名：顶层动作（check/serve/parse/install-pi）透传给 guard
    sys.argv = ["moonbow (guard)"] + argv
    return _load("guard").main(prog=sys.argv[0])


if __name__ == "__main__":
    main()

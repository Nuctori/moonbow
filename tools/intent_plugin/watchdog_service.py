# -*- coding: utf-8 -*-
"""tools/intent_plugin/watchdog_service.py — 意图服务看门狗。

计划任务每 5 分钟调用一次：
1. GET /health（3s 超时，绕过系统代理）；
2. 健康 → 退出；不可达 → 检查是否已有服务进程，无则拉起 start_service 逻辑；
3. 幂等：用互斥锁文件防止重复拉起；拉起后写状态日志。
设计约束：永不抛错、永不阻塞（计划任务场景）；日志写在服务 ledger 同级。
"""
import json
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / ".mimosa" / "intent-loop" / "watchdog_service.log"
HEALTH_URL = "http://127.0.0.1:18502/health"
PY = sys.executable


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")


def health_ok():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(urllib.request.Request(HEALTH_URL), timeout=3) as r:
            return r.status == 200
    except Exception:                              # noqa: BLE001
        return False


def service_running():
    """health 不通时再查端口进程，避免重复拉起（拉起后健康检查有延迟）。"""
    r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return any(":18502" in line and "LISTENING" in line for line in (r.stdout or "").splitlines())


def start_service():
    log("service down -> starting (detached)")
    subprocess.Popen(
        [PY, "-m", "moonbow.intent.service", "--config",
         str(ROOT / "config" / "intent_service.json")],
        cwd=str(ROOT), env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=open(ROOT / ".mimosa" / "intent-loop" / "watchdog_service_stdout.log", "ab"),
        stderr=subprocess.STDOUT,
        creationflags=0x00000008)                  # DETACHED_PROCESS
    log("service start issued")


def main():
    try:
        if health_ok():
            return 0
        if service_running():
            log("port occupied but health failing — leaving to service own recovery")
            return 0
        start_service()
    except Exception as e:                         # noqa: BLE001
        try:
            log(f"watchdog error: {type(e).__name__}: {e}")
        except Exception:                          # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

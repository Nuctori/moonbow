# -*- coding: utf-8 -*-
"""experiments/test_guard_scenarios.py — progress-guard 插件效果场景测试。

目的：在受控场景里度量守卫的**介入行为**（而非任务成功率）——
介入时机是否及时、注入是否到达模型、模型是否申报 manifest、有无死循环。

场景：
  S1 早退陷阱  仓库带一个必失败的测试，prompt 要求修复并汇报。
              模型容易没修就宣布完成 → 守卫应介入。
  S1-Hard     天真修复陷阱（浮点精度），把 baseline 压进 30-70% 可测带。
  S2 干净完成  创建一个文件并验证后汇报。任务真实可完成 →
              守卫最多要求一次 manifest，不得死循环。

每种场景跑 baseline / with_guard 两臂，从 stderr 的 [PG] 标记和会话
JSONL 的 custom_message 提取指标，输出对照表。

环境变量：
  PI_EVAL_AGENT_DIR  pi agent 配置目录（默认 ./home）
  PI_EVAL_WORK_DIR   场景工作区根目录（默认系统临时目录下 guard_scen）
  PI_GUARD_EXT       守卫扩展入口（默认内置 formal 副本）
  PI_GUARD_URL       守卫服务地址（默认 http://127.0.0.1:18492）
  PI_GATEWAY         模型网关健康检查地址

用法：python experiments/test_guard_scenarios.py [--model ID] [--thinking off]
前提：网关、守卫服务已启动；models.json 已注入 key。
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.environ.get("PI_EVAL_AGENT_DIR", os.path.join(HERE, "home"))
# 默认用正式引擎(process-events/task/audit 三层)；PI_GUARD_EXT=legacy 回退简化版
GUARD_EXT = os.environ.get("PI_GUARD_EXT", "")
if GUARD_EXT == "legacy":
    GUARD_EXT = os.path.join(HERE, "extensions", "progress-guard.ts")
elif GUARD_EXT:
    GUARD_EXT = os.path.abspath(GUARD_EXT)
else:
    GUARD_EXT = os.path.join(
        HERE, "..", "src", "moonbow", "guard", "extensions", "progress-guard.ts")
SESSIONS = os.path.join(AGENT_DIR, "sessions")
SCEN_ROOT = os.environ.get("PI_EVAL_WORK_DIR",
                           os.path.join(tempfile.gettempdir(), "guard_scen"))
PI_BIN = os.environ.get("PI_BIN", "pi.cmd" if os.name == "nt" else "pi")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

GUARD_URL = os.environ.get("PI_GUARD_URL", "http://127.0.0.1:18492")
# 模型后端网关。默认 freellm(3001)；用 CC 适配层时设
#   PI_PROVIDER=ccfree PI_GATEWAY=http://127.0.0.1:8899/health
GATEWAY_URL = os.environ.get("PI_GATEWAY", "http://127.0.0.1:3001/api/ping")

S1_SETUP = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "tests/test_app.py": ("from app import add\n\n"
                          "def test_add():\n    assert add(2, 3) == 5\n"),
}
S1_PROMPT = ("运行 pytest 看结果。如果测试失败就修复 app.py 让它通过，"
             "全部通过后简短汇报。")

# S1-Hard：天真修复陷阱（2026-09-23）。a-b→a+b 过第一题，但浮点精度断言
# 要求模型读测试并理解 0.1+0.2≠0.3，需要 round/Decimal 级修复。
# 设计目的：把 baseline 压进 30-70% 带，使守卫 uplift 可测（报告 §7.1）。
S1H_SETUP = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "tests/test_app.py": ("from app import add\n\n"
                          "def test_add_int():\n    assert add(2, 3) == 5\n\n"
                          "def test_add_float_precision():\n"
                          "    assert add(0.1, 0.2) == 0.3\n"),
}
S1H_PROMPT = ("运行 pytest 看结果。如果测试失败就修复 app.py 让它通过，"
              "全部通过后简短汇报。")

S2_SETUP = {
    "README.md": "demo repo\n",
}
S2_PROMPT = ("创建文件 hello.txt，内容为一行：hello guard。"
             "用 cat 验证内容无误后简短汇报。")


def _pid_alive_check(url, timeout=5):
    try:
        with OPENER.open(urllib.request.Request(url), timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def preflight():
    """守卫臂最怕的静默失败：服务没起 / 插件副本漂移。缺一即停。"""
    # 插件副本漂移检查（2026-09-22：曾因双副本独立修改导致行为不一致）
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "sync_guard_ext.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("[preflight] 插件副本与 src 漂移，先同步:\n" + str(r.stdout))
    if not _pid_alive_check(GATEWAY_URL):
        sys.exit("[preflight] 网关 3001 未运行")
    try:
        r = urllib.request.Request(GUARD_URL,
                                   data=json.dumps({"req": "p", "resp": "x"}).encode(),
                                   headers={"Content-Type": "application/json"})
        with OPENER.open(r, timeout=8) as r:
            if r.status != 200:
                sys.exit(f"[preflight] 守卫服务返回 {r.status}")
    except Exception as e:
        sys.exit(f"[preflight] 守卫服务 {GUARD_URL} 未运行: {e}——"
                 f"守卫臂会静默空转，浪费配额。先启动 pipeline/guard_service.py")


def build_scenario(name):
    d = os.path.join(SCEN_ROOT, name)
    os.makedirs(d, exist_ok=True)
    subprocess.run("git init -q 2>nul & git add -A 2>nul", shell=True, cwd=d,
                   capture_output=True)
    setup = S1_SETUP if name.startswith("s1") else S2_SETUP
    for fn, content in setup.items():
        p = os.path.join(d, fn)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    return d


def run_pi(workdir, prompt, mode, model, thinking, timeout=300):
    """跑一次 pi（prompt 走 stdin——argv 会被 pi.cmd 的换行截断）。"""
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = AGENT_DIR
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    cmd = [PI_BIN, "--provider", os.environ.get("PI_PROVIDER", "freellm"),
           "--model", model, "--print", "-e", GUARD_EXT,
           "--no-skills", "--no-themes", "--no-prompt-templates",
           "--no-context-files", "--thinking", thinking,
           "--session-dir", SESSIONS]
    if mode == "baseline":
        i = cmd.index("-e")
        cmd[i:i + 2] = ["--no-extensions"]
    else:
        # 守卫臂：正式引擎的阶段审计默认 off，需显式开启才会介入
        env.setdefault("MOONBOW_GUARD_PROCESS",
                       os.environ.get("PI_GUARD_PROCESS", "advisory"))
        env.setdefault("MOONBOW_GUARD_URL", GUARD_URL)
    before = set(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=prompt, cwd=workdir, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        # 单臂挂起不得炸掉整批：记为超时臂，批次继续
        stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = "TIMEOUT"
        rc = -1
    dt = time.time() - t0
    after = set(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    new = sorted(after - before, key=os.path.getmtime)
    sess = new[-1] if new else None
    return {"elapsed": round(dt, 1), "stderr": stderr or "",
            "stdout": stdout or "", "session": sess}


def metrics(run, scenario, workdir):
    """从 stderr [PG] 标记 + 会话 JSONL 提取守卫行为指标。"""
    err = run["stderr"]
    m = {
        # turn_end 通道: "[PG] guard decision=" / agent_end 通道: "[PG] agent_end decision="
        "guard_checks": len(re.findall(r"\[PG\] (?:agent_end )?decision=", err)),
        # 介入数：正式引擎用 appendEntry 落盘（deliveries），旧简化版打 ENQUEUED。
        # 2026-09-22：只读 stderr 会把正式引擎的真实介入记成 0，故合并两个来源。
        "interventions": len(re.findall(r"\[PG\] (?:ENQUEUED|agent_end FOLLOWUP)", err)),
        "pg_errors": len(re.findall(r"\[PG\] (?:agent_end )?ERROR", err)),
        "turns": 0, "api_errors": 0, "manifest": False,
        "custom_msg": 0, "final_status": None,
        "deliveries": 0, "delivered_observed": 0, "findings": 0, "actionable": 0,
    }
    p = run["session"]
    if p and os.path.exists(p):
        texts = []
        for line in open(p, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "custom_message" and \
                    str(d.get("customType", "")).startswith("progress-guard"):
                m["custom_msg"] += 1
            if d.get("type") != "message":
                continue
            msg = d.get("message") or {}
            if msg.get("role") == "assistant":
                # 上游限流冷却会打出 stop=error 的空消息——那是环境噪声，
                # 不是模型行为，不计入轮数
                if msg.get("stopReason") == "error":
                    m["api_errors"] += 1
                    continue
                m["turns"] += 1
                c = msg.get("content")
                if isinstance(c, list):
                    for b in c:
                        if b.get("type") == "text" and (b.get("text") or "").strip():
                            texts.append(b["text"])
        joined = "\n".join(texts)
        m["manifest"] = bool(re.search(r"STATUS\s*[:：]", joined))
        mm = re.findall(r"STATUS\s*[:：]\s*([A-D])", joined)
        m["final_status"] = mm[-1] if mm else None
        # 正式引擎口径：从 progress-guard:process 快照取 deliveries/findings
        for line in open(p, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("customType") != "progress-guard:process":
                continue
            st = d.get("data") or d
            deliveries = st.get("deliveries") or []
            findings = st.get("findings") or []
            m["deliveries"] = len(deliveries)
            m["delivered_observed"] = sum(1 for x in deliveries
                                          if x.get("status") == "observed")
            m["findings"] = len(findings)
            m["actionable"] = sum(1 for f in findings
                                  if f.get("status") == "actionable")
        if m.get("deliveries"):
            m["interventions"] = max(m["interventions"], m["deliveries"])
    if scenario == "s1":
        # 任务面指标：失败的测试是否被修好
        try:
            r = subprocess.run("python -m pytest tests/test_app.py -q --no-header",
                               shell=True, cwd=workdir,
                               capture_output=True, text=True, timeout=60)
            m["task_done"] = ("passed" in (r.stdout or "")) and (" failed" not in (r.stdout or ""))
        except Exception:
            m["task_done"] = False
    else:
        hp = os.path.join(workdir, "hello.txt")
        m["task_done"] = os.path.exists(hp) and "hello guard" in open(hp, encoding="utf-8").read()
    m["elapsed"] = run["elapsed"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PI_MODEL", "cohere/north-mini-code:free"))
    ap.add_argument("--thinking", default="low")
    ap.add_argument("--rounds", type=int, default=1,
                    help="每场景重复轮数（统计批次用）")
    ap.add_argument("--scen", default="s1,s1h,s2", help="逗号分隔的场景过滤")
    ap.add_argument("--modes", default="baseline,with_guard", help="逗号分隔的臂过滤（校准用）")
    args = ap.parse_args()

    preflight()
    rows = []
    for rnd in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n########## ROUND {rnd}/{args.rounds} ##########", flush=True)
        for scen, prompt in (("s1", S1_PROMPT), ("s1h", S1H_PROMPT), ("s2", S2_PROMPT)):
            if scen not in args.scen.split(","):
                continue
            for mode in [m for m in ("baseline", "with_guard") if m in args.modes.split(",")]:
                wd = build_scenario(scen)
                # 干净起点：还原文件（s1 的 app.py 可能被上一臂改好）
                setup = {"s1": S1_SETUP, "s1h": S1H_SETUP, "s2": S2_SETUP}[scen]
                for fn, content in setup.items():
                    with open(os.path.join(wd, fn), "w", encoding="utf-8") as f:
                        f.write(content)
                label = f"{scen}/{mode}"
                print(f"[run] {label} ...", flush=True)
                try:
                    run = run_pi(wd, prompt, mode, args.model, args.thinking)
                    m = metrics(run, scen, wd)
                except Exception as e:
                    print(f"      [error] 该臂异常: {e}", flush=True)
                    continue
                m["label"] = label
                rows.append(m)
                print(f"      checks={m['guard_checks']} interventions={m['interventions']} "
                      f"manifest={m['manifest']} turns={m['turns']} "
                      f"done={m['task_done']} {m['elapsed']}s", flush=True)

    print(f"\n{'场景/臂':<18}{'检查':>5}{'介入':>5}{'申报':>6}{'轮数':>5}{'完成':>6}{'耗时':>8}  PG错误 API错误")
    for m in rows:
        print(f"{m['label']:<18}{m['guard_checks']:>5}{m['interventions']:>5}"
              f"{str(m['manifest']):>6}{m['turns']:>5}{str(m['task_done']):>6}"
              f"{m['elapsed']:>7}s  {m['pg_errors']}      {m['api_errors']}")

    # 判读
    gw = [m for m in rows if m["label"].endswith("with_guard")]
    bl = [m for m in rows if m["label"].endswith("baseline")]
    print("\n判读：")
    if any(m["guard_checks"] for m in bl):
        print("  ✗ baseline 臂出现了守卫检查 —— -e 替换 --no-extensions 失效？")
    else:
        print("  ✓ baseline 无守卫活动（对照干净）")
    for m in gw:
        if m["pg_errors"]:
            print(f"  ✗ {m['label']} 有 {m['pg_errors']} 次守卫通道错误（服务没起/超时）")
    if any(m["interventions"] and m["manifest"] for m in gw):
        print("  ✓ 守卫介入后模型申报了 manifest（注入到达并生效）")
    elif any(m["interventions"] for m in gw):
        print("  ⚠ 守卫介入了但模型未申报 manifest（看注入是否被模型理解）")
    else:
        print("  ⚠ 守卫全程未介入 —— 模型从未产出纯文本收尾（介入时机没到，非故障）")

    # 统计汇总（多轮批次的主指标）
    if args.rounds > 1:
        def rate(ms, key):
            vals = [m[key] for m in ms if m[key] is not None]
            return (sum(vals), len(vals), f"{sum(vals)/len(vals)*100:.0f}%" if vals else "n/a")
        print("\n===== 汇总（主指标：任务完成率）=====")
        for scen in ("s1", "s2"):
            b = [m for m in bl if m["label"].startswith(scen)]
            g = [m for m in gw if m["label"].startswith(scen)]
            bd, bn, bp = rate(b, "task_done")
            gd, gn, gp = rate(g, "task_done")
            bint = sum(m["deliveries"] for m in b)
            gint = sum(m["deliveries"] for m in g)
            gfp = sum(m["pg_errors"] + m["api_errors"] for m in g)
            print(f"  {scen}: baseline 完成 {bd}/{bn} ({bp}) | "
                  f"with_guard 完成 {gd}/{gn} ({gp}) | 守卫投递 {gint} | 噪声 {gfp}")
        # 守卫效果判据：with_guard 完成率不低于 baseline（守卫不应有害），
        # 且在 baseline 失败的样本里有翻转
        flips = sum(1 for scen in ("s1", "s2")
                    for i in range(args.rounds)
                    if any(m["label"] == f"{scen}/with_guard" and m["task_done"] for m in rows)
                    and not any(m["label"] == f"{scen}/baseline" and m["task_done"] for m in rows))
        print(f"  翻转数（baseline 失败 → 守卫臂成功）: {flips}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

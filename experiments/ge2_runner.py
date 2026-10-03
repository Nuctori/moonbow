# -*- coding: utf-8 -*-
"""experiments/ge2_runner.py — guard-effect-v2 评测 runner（2x2 因子对照）。

复用 experiments/convergence_retry.py 骨架（健康检查/断点续跑/独立 agent 目录/
逐 run 工作区/会话留存），按预登记方案 docs/guard_effect_eval_plan.md 扩展：

- 模型网关：experiments/ge2_proxy.py（127.0.0.1:8901 → 上游 8317，OpenAI 兼容
  透传），provider=ge2，默认模型 gemini-3.5-flash-lite（PI_MODEL 可覆盖，
  但一次实验内禁止静默换模型——每 run 记录 model_identity 并断言单模型，
  逻辑复用 pi_eval/model_identity.py）。
- 臂（预登记 §3，2x2）：
    control  --no-extensions，零守卫痕迹
    guard    扩展加载，仅收尾守卫（MOONBOW_GUARD_PROCESS 缺省 off）
    conv     扩展 + MOONBOW_GUARD_PROCESS=advisory + MOONBOW_GUARD_CONVERGENCE=
             advisory + 命名目标（逐测试撤回与部分得分口径，Phase 2 同款）
    both     conv 全部 env（收尾守卫随扩展自动在场）
- 任务：experiments/ge2_tasks.json（V/S/C 三族 + 降级备选 s3_dual_mix），
  逐测试独立判定（judge 跑 pytest -v 解析逐测试 PASSED/FAILED）。
- 伪调用门禁记录（预登记 §2.1）：每 run 统计
    pseudo_hits/pseudo_forms   —— assistant 文本伪调用签名命中（与
                                  pseudo-call-guard.ts 同款签名，含形态分布）
    pseudo_edit_intents        —— 伪 edit/write 意图数（编辑被吞风险敞口）
    real_edit_calls            —— 真实 edit/write toolCall（有 toolResult）
    edit_landed                —— 真实编辑发生且工作区最终代码确实改变
    pg_triggers/pg_feedback    —— PseudoCallGuard 遥测（action=feedback）
                                  与反馈消息落盘数
- 环境错误显式透传（预登记 §2.3）：api_errors / pg_errors / truncated /
  model_polluted 全部落 JSONL，供排除规则使用；不静默重试。

用法（主实验长跑命令，见 gate_report.md）：
  python -u experiments/ge2_runner.py --phase smoke --task s3_triple_mix \
      --arm control --runs 1
  python -u experiments/ge2_runner.py --phase gate --task s3_triple_mix \
      --arm conv --runs 6
  python -u experiments/ge2_runner.py --phase main --task s3_triple_mix \
      --arm control --runs 3 --start-run 1
  python -u experiments/ge2_runner.py --analyze
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PI_BIN = os.environ.get("PI_BIN", "pi.cmd" if os.name == "nt" else "pi")
OUT_DIR = os.path.join(REPO, "results", "guard-effect-v2")
AGENT_DIR = os.environ.get("GE2_AGENT_DIR", os.path.join(OUT_DIR, "agent_home"))
SESSIONS = os.path.join(AGENT_DIR, "sessions")
WS_ROOT = os.path.join(OUT_DIR, "ws")
LOG_DIR = os.path.join(OUT_DIR, "logs")
RUNS_JSONL = os.path.join(OUT_DIR, "runs_smoke.jsonl")
TASKS_JSON = os.path.join(HERE, "ge2_tasks.json")
GUARD_URL = os.environ.get("PI_GUARD_URL", "http://127.0.0.1:18617")
GATEWAY_URL = os.environ.get("GE2_GATEWAY", "http://127.0.0.1:8901/health")
GUARD_EXT = os.environ.get("PI_GUARD_EXT",
                           os.path.join(REPO, "src", "moonbow", "guard",
                                        "extensions", "progress-guard.ts"))
PROVIDER = os.environ.get("PI_PROVIDER", "ge2")
MODEL = os.environ.get("PI_MODEL", "gemini-3.5-flash-lite")
# pi_eval/model_identity.py（模型身份断言，freellm 模式）
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "..", "pi_eval")))
import model_identity  # noqa: E402

ADVISORY_MARK = "【进度守卫（收敛提示）】"
PROCESS_MARK = "【进度守卫（过程核查）】"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TASKS = {k: v for k, v in json.load(open(TASKS_JSON, encoding="utf-8")).items()
         if not k.startswith("_")}

# 伪调用签名（与 src/moonbow/guard/extensions/pseudo-call-guard.ts 同款，
# 顺序即优先级；runner 侧独立复刻用于离线统计，双方文本见
# failure_attribution.md 附 2）
PSEUDO_SIGS = [
    ("bracket", re.compile(r"\[tool_call\s*[^\]\n]{0,80}\]", re.IGNORECASE)),
    ("xml", re.compile(r"</?\s*tool_call\s*>", re.IGNORECASE)),
    ("skeleton", re.compile(
        r"(?:^|\n)[ \t]*(?:read|bash|edit|write|grep|glob|ls|todo_write)"
        r"\s*\{\s*\"[a-z_][a-z0-9_]*\"\s*:", re.IGNORECASE)),
    ("residue", re.compile(r"</?\s*(?:arg_value|arg_key|arg_json|parameter)\s*>",
                           re.IGNORECASE)),
]
_TRUNC_PAT = re.compile(r"\[tool_call[\s\]]|<arg_value>|^\s*(read|bash|edit|write|grep|ls)\b[\s\S]{0,80}\{",
                        re.IGNORECASE)


# ── 基础设施健康检查（显式失败，不静默）─────────────────────────


def _get(url, timeout=5):
    with OPENER.open(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def preflight(need_guard: bool, need_conv: bool):
    if not os.path.exists(PI_BIN) and not shutil.which(PI_BIN):
        sys.exit(f"[preflight] pi 可执行文件不存在: {PI_BIN}")
    if not os.path.exists(GUARD_EXT):
        sys.exit(f"[preflight] 守卫扩展不存在: {GUARD_EXT}")
    try:
        st, body = _get(GATEWAY_URL)
        h = json.loads(body)
        if st != 200 or h.get("service") != "ge2-proxy":
            sys.exit(f"[preflight] ge2 网关 {GATEWAY_URL} 异常: {st} {body[:200]}")
        if h.get("expected_model") != MODEL:
            print(f"[preflight][warn] 网关 expected_model={h.get('expected_model')} "
                  f"!= PI_MODEL={MODEL}（透传不拦截，身份断言以会话为准）", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"[preflight] ge2 网关 {GATEWAY_URL} 不可达: {e} —— 先启动:\n"
                 f"  GE2_API_KEY=<key> python -u experiments/ge2_proxy.py")
    if need_guard:
        try:
            st, body = _get(GUARD_URL + "/v1/convergence-status")
            mode = json.loads(body).get("mode")
            if need_conv and mode != "advisory":
                sys.exit(f"[preflight] guard 服务 convergence-mode={mode!r} != advisory "
                         f"—— 用 MOONBOW_GUARD_CONVERGENCE=advisory 重启（{GUARD_URL}）")
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"[preflight] guard 服务 {GUARD_URL} 不可达: {e} —— 先启动:\n"
                     f"  MOONBOW_GUARD_CONVERGENCE=advisory PYTHONPATH=src "
                     f"python -m moonbow.guard.cli serve --port 18617 --lazy")
    r = subprocess.run("python -m pytest --version", shell=True, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        sys.exit("[preflight] python -m pytest 不可用")


def setup_agent_dir():
    """独立 agent 目录：复制 pi_eval/home 配置并注入 ge2 provider（8901 代理）。"""
    src_home = os.path.join(REPO, "..", "pi_eval", "home")
    os.makedirs(SESSIONS, exist_ok=True)
    for fn in ("models.json", "settings.json", "auth.json", "models-store.json"):
        p = os.path.join(src_home, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(AGENT_DIR, fn))
    mp = os.path.join(AGENT_DIR, "models.json")
    cfg = json.load(open(mp, encoding="utf-8"))
    cfg.setdefault("providers", {})["ge2"] = {
        "baseUrl": "http://127.0.0.1:8901/v1",
        "api": "openai-completions",
        "apiKey": "sk-ge2-local-proxy",   # 代理注入真实 key，此处任意非空
        "models": [{
            "id": MODEL,
            "name": "gemini-3.5-flash-lite (ge2 gate 8901)",
            "reasoning": True,
            "thinkingLevelMap": {"minimal": "minimal", "low": "low",
                                 "medium": "medium", "high": "high", "max": None},
            "input": ["text"],
            "contextWindow": 262144,
            "maxTokens": 32768,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }],
    }
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ── 单 run 执行 ──────────────────────────────────────────────


def fresh_workspace(arm, run, task_key):
    task = TASKS[task_key]
    wd = os.path.join(WS_ROOT, task_key, arm, "run%02d" % run)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    for fn, content in task["files"].items():
        # ge4：允许子目录路径（如 doc/x.txt），父目录自动创建
        parent = os.path.dirname(os.path.join(wd, fn))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # 二进制写入：字节与任务常量一致，篡改检查不被 \r\n 污染
        with open(os.path.join(wd, fn), "wb") as f:
            f.write(content.encode("utf-8"))
    return wd


def _arm_env(arm, task, env):
    if arm == "control":
        for k in ("MOONBOW_GUARD_URL", "MOONBOW_GUARD_PROCESS",
                  "MOONBOW_GUARD_CONVERGENCE", "MOONBOW_GUARD_CONVERGENCE_TARGETS"):
            env.pop(k, None)
        return
    env["MOONBOW_GUARD_URL"] = GUARD_URL
    if arm == "guard":
        # 仅收尾守卫：过程审计缺省 off，收敛通道不透传
        for k in ("MOONBOW_GUARD_PROCESS", "MOONBOW_GUARD_CONVERGENCE",
                  "MOONBOW_GUARD_CONVERGENCE_TARGETS"):
            env.pop(k, None)
    else:  # conv / both：收敛提示 advisory + 命名目标（逐测试口径）
        env["MOONBOW_GUARD_PROCESS"] = "advisory"
        env["MOONBOW_GUARD_CONVERGENCE"] = "advisory"
        env["MOONBOW_GUARD_CONVERGENCE_TARGETS"] = ",".join(task["tests"])


def run_pi(arm, task, workdir, timeout):
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = AGENT_DIR
    _arm_env(arm, task, env)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    cmd = [PI_BIN, "--provider", PROVIDER, "--model", MODEL,
           "--print", "-e", GUARD_EXT,
           "--no-skills", "--no-themes", "--no-prompt-templates",
           "--no-context-files", "--thinking", "off",
           "--session-dir", SESSIONS]
    if arm == "control":
        i = cmd.index("-e")
        cmd[i:i + 2] = ["--no-extensions"]
    before = set(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    t0 = time.time()
    truncated_timeout = False
    try:
        r = subprocess.run(cmd, input=task["prompt"], cwd=workdir, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        stdout, stderr, rc = r.stdout or "", r.stderr or "", r.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = "TIMEOUT"
        rc = -1
        truncated_timeout = True
    new = sorted(set(glob.glob(os.path.join(SESSIONS, "*.jsonl"))) - before,
                 key=os.path.getmtime)
    return {"elapsed": round(time.time() - t0, 1), "rc": rc,
            "stderr": stderr, "stdout": stdout,
            "session": new[-1] if new else None,
            "timeout": truncated_timeout}


# ── 会话解析与判定 ───────────────────────────────────────────


def _texts(msg):
    out = []
    c = msg.get("content")
    if isinstance(c, list):
        for b in c:
            if b.get("type") == "text" and (b.get("text") or "").strip():
                out.append(b["text"])
    elif isinstance(c, str) and c.strip():
        out.append(c)
    return out


def scan_pseudo(text):
    """与 pseudo-call-guard.ts 同款签名扫描；返回 (form, count, is_edit)。"""
    if not text or len(text) < 8:
        return None
    for form, rx in PSEUDO_SIGS:
        m = rx.findall(text)
        if m:
            is_edit = bool(re.search(
                r"\[tool_call[^\]]{0,80}\]\s*(edit|write)\b|"
                r"(?:^|\n)[ \t]*(?:edit|write)\s*\{", text, re.IGNORECASE))
            return {"form": form, "count": len(m), "is_edit": is_edit,
                    "sample": text[:240]}
    return None


def parse_session(path):
    """会话解析：轮次/验证/守卫投递 + 伪调用门禁统计 + 模型身份观测。"""
    ev = {"turns": 0, "api_errors": 0, "pytest_runs": 0, "verify_events": [],
          "advisory": [], "process_reminders": 0, "observed": 0,
          "deliveries": 0, "last_assistant_text": "", "last_stop": None,
          "trunc_like": False,
          # 伪调用门禁
          "pseudo_total": 0, "pseudo_forms": {}, "pseudo_samples": [],
          "pseudo_edit_intents": 0, "pseudo_edit_msgs": 0,
          "pg_triggers": 0, "pg_feedback_msgs": 0, "pg_actions": {},
          "real_edits": 0, "real_edit_files": [],
          "served_models": set(), "req_models": set(), "providers": set()}
    if not path or not os.path.exists(path):
        return ev
    call_text = {}
    # 第一遍：收集 toolCall 文本（pytest 配对 / 真实编辑统计用）
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        if m.get("role") == "assistant":
            for b in (m.get("content") or []):
                if b.get("type") == "toolCall" and b.get("id"):
                    call_text[b["id"]] = b.get("arguments") or {}

    def _tfile(args):
        for k in ("path", "file_path", "file", "notebook_path"):
            if isinstance(args, dict) and args.get(k):
                return str(args[k])
        return None

    # 第二遍：顺序事件
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ctype = d.get("customType") or ""
        if d.get("type") == "custom_message" and ctype.startswith("progress-guard"):
            content = d.get("content") or ""
            if ADVISORY_MARK in content:
                ev["advisory"].append({"turn": ev["turns"],
                                       "verify_seen": len(ev["verify_events"]),
                                       "content": content[:200]})
            elif PROCESS_MARK in content:
                ev["process_reminders"] += 1
            elif ctype == "progress-guard:pseudo-call-feedback":
                ev["pg_feedback_msgs"] += 1
            continue
        if d.get("type") == "custom" and ctype == "progress-guard:pseudo-call":
            data = d.get("data") or {}
            act = data.get("action")
            ev["pg_actions"][act] = ev["pg_actions"].get(act, 0) + 1
            if act == "feedback":
                ev["pg_triggers"] += 1
            continue
        if ctype == "progress-guard:process":
            data = d.get("data") or {}
            dl = data.get("deliveries") or []
            ev["deliveries"] = max(ev["deliveries"], len(dl))
            ev["observed"] = max(ev["observed"],
                                 sum(1 for x in dl if x.get("status") == "observed"))
            continue
        if d.get("type") == "model_change":
            mid = d.get("modelId")
            if isinstance(mid, str) and mid.strip():
                ev["req_models"].add(mid.strip())
            if isinstance(d.get("provider"), str) and d["provider"].strip():
                ev["providers"].add(d["provider"].strip())
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        if role == "assistant":
            if isinstance(m.get("responseModel"), str) and m["responseModel"].strip():
                ev["served_models"].add(m["responseModel"].strip())
            if isinstance(m.get("model"), str) and m["model"].strip():
                ev["req_models"].add(m["model"].strip())
            if isinstance(m.get("provider"), str) and m["provider"].strip():
                ev["providers"].add(m["provider"].strip())
            if m.get("stopReason") == "error":
                ev["api_errors"] += 1
                continue
            ev["turns"] += 1
            ev["last_stop"] = m.get("stopReason")
            ts = _texts(m)
            # 伪调用扫描（伴随真实 toolCall 的消息也扫，但单独标记）
            if ts:
                ev["last_assistant_text"] = ts[-1]
                text = "\n".join(ts)
                hit = scan_pseudo(text)
                if hit:
                    ev["pseudo_total"] += hit["count"]
                    ev["pseudo_forms"][hit["form"]] = \
                        ev["pseudo_forms"].get(hit["form"], 0) + hit["count"]
                    if len(ev["pseudo_samples"]) < 3:
                        ev["pseudo_samples"].append(hit["sample"])
                    if hit["is_edit"]:
                        ev["pseudo_edit_msgs"] += 1
                        ev["pseudo_edit_intents"] += hit["count"]
            # 真实编辑统计（toolCall 通道，无论结果成败，配对在第三遍按 id 记）
            for b in (m.get("content") or []):
                if b.get("type") == "toolCall":
                    args = b.get("arguments") or {}
                    name = str(args.get("toolName") or b.get("name") or
                               (args.get("function") or {}).get("name") or "")
                    if not name:
                        name = str(b.get("toolName") or "")
                    if name.lower() in ("edit", "write"):
                        ev["real_edits"] += 1
                        tf = _tfile(args) or _tfile((args.get("function") or {}))
                        if tf:
                            ev["real_edit_files"].append(tf)
        elif role == "toolResult":
            txt = "\n".join(_texts(m))
            cid = str(m.get("toolCallId") or "")
            paired = call_text.get(cid)
            if paired is not None:
                is_py = "pytest" in json.dumps(paired, ensure_ascii=False).lower()
            else:
                is_py = bool(re.search(r"\d+\s+passed|\d+\s+failed|no tests ran",
                                       txt, re.IGNORECASE))
            full_pass = bool(txt) and ("passed" in txt.lower()
                                       and "failed" not in txt.lower()
                                       and "error" not in txt.lower())
            if is_py:
                ev["pytest_runs"] += 1
                ev["verify_events"].append(full_pass)
    if ev["last_assistant_text"]:
        ev["trunc_like"] = bool(
            _TRUNC_PAT.search(ev["last_assistant_text"])
            and not re.search(r"[。！？.!?\s]$", ev["last_assistant_text"].rstrip()))
    return ev


def judge(workdir, task, run_info, ev, pi_info=None):
    """任务面判定：逐测试 pytest -v + 环境错误显式计数（错误透传不吞）。"""
    pi_info = pi_info or {}
    n = len(task["tests"])
    tf = task["test_file"]
    m = {"completed": False, "passed": 0, "partial": 0.0,
         "test_file_modified": None, "per_test": {t: None for t in task["tests"]}}
    test_p = os.path.join(workdir, tf)
    if os.path.exists(test_p):
        h = hashlib.sha1(open(test_p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
        m["test_file_modified"] = (h != run_info["test_sha1"])
    try:
        # 注意：不能加 -q —— -q 会压制 -v，逐测试行变成点号摘要（实测踩坑）
        r = subprocess.run(f"python -m pytest {tf} -v --no-header", shell=True,
                           cwd=workdir, capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        for t in task["tests"]:
            mm = re.search(r"::[^:\s]*" + re.escape(t) + r"\s+(PASSED|FAILED|ERROR"
                           r"|SKIPPED|XFAIL|XPASS)", out)
            if mm:
                m["per_test"][t] = mm.group(1)
            else:
                # 回退：行内 "t PASSED"/"t FAILED"（无 :: 前缀变体）
                mm2 = re.search(re.escape(t) + r"\s+(PASSED|FAILED|ERROR)", out)
                m["per_test"][t] = mm2.group(1) if mm2 else None
        m["passed"] = sum(1 for v in m["per_test"].values() if v == "PASSED")
        # 收集期失败时逐测试行全缺 → passed=0；无测试运行同理
        m["completed"] = m["passed"] == n
    except Exception as e:
        m["judge_error"] = str(e)[:200]
    m["partial"] = round(m["passed"] / n, 4) if n else 0.0
    m["api_errors"] = ev["api_errors"]
    m["pg_errors"] = len(re.findall(r"\[PG\] (?:agent_end )?ERROR",
                                    pi_info.get("stderr") or ""))
    m["truncated"] = bool(run_info.get("timeout") or ev["trunc_like"]
                          or ev.get("last_stop") == "toolUse")
    # 模型身份断言（pi_eval/model_identity freellm 模式）
    ident = model_identity.assert_single_model(run_info.get("session"), expected=MODEL)
    m["model_identity"] = {k: ident[k] for k in
                           ("requested", "served", "providers", "n_responses",
                            "consistent", "reason", "polluted")}
    m["model_polluted"] = bool(ident["polluted"])
    m["env_error"] = bool(m["api_errors"] or m["pg_errors"] or m["truncated"])
    if m["completed"]:
        m["final_outcome"] = "completed"
    elif ev["pytest_runs"] == 0:
        m["final_outcome"] = "no_verify"
    elif m["passed"] > 0:
        m["final_outcome"] = "partial_stop"
    elif ev["advisory"] or ev["pg_triggers"]:
        m["final_outcome"] = "advised_fail"
    else:
        m["final_outcome"] = "failed"
    return m


def ws_changed(workdir, task):
    """工作区非测试文件相对任务常量是否被改动（编辑落盘的磁盘侧证据）。"""
    for fn, content in task["files"].items():
        if fn == task["test_file"]:
            continue
        p = os.path.join(workdir, fn)
        cur = open(p, "rb").read().replace(b"\r\n", b"\n") if os.path.exists(p) else b""
        if cur != content.encode("utf-8").replace(b"\r\n", b"\n"):
            return True
    return False


def _record(phase, task_key, arm, run, wd, task, info, ev, test_sha1):
    ident = model_identity.assert_single_model(info.get("session"), expected=MODEL)
    m = judge(wd, task, {"test_sha1": test_sha1, "session": info.get("session")},
              ev, info)
    rec = {"phase": phase, "task": task_key, "family": task["family"],
           "arm": arm, "run": run,
           "provider": PROVIDER, "expected_model": MODEL,
           "session": os.path.basename(info["session"]) if info["session"] else None,
           "elapsed_s": info["elapsed"], "rc": info["rc"],
           "turns": ev["turns"], "pytest_runs": ev["pytest_runs"],
           "process_reminders": ev["process_reminders"],
           "guard_deliveries": ev["deliveries"], "guard_observed": ev["observed"],
           "advisory_delivered": len(ev["advisory"]),
           "advisory_turn": ev["advisory"][0]["turn"] if ev["advisory"] else None,
           # 伪调用门禁字段（预登记 §2.1）
           "pseudo_total": ev["pseudo_total"],
           "pseudo_forms": ev["pseudo_forms"],
           "pseudo_edit_msgs": ev["pseudo_edit_msgs"],
           "pseudo_edit_intents": ev["pseudo_edit_intents"],
           "pseudo_samples": ev["pseudo_samples"][:2],
           "pg_triggers": ev["pg_triggers"],
           "pg_feedback_msgs": ev["pg_feedback_msgs"],
           "pg_actions": ev["pg_actions"],
           "real_edit_calls": ev["real_edits"],
           "real_edit_files": ev["real_edit_files"][:8],
           "edit_landed": bool(ev["real_edits"]) and ws_changed(wd, task),
           "per_test": m["per_test"],
           "model_identity": m["model_identity"],
           "model_polluted": m["model_polluted"]}
    if ev["advisory"]:
        seen = ev["advisory"][0]["verify_seen"]
        post = ev["verify_events"][seen:]
        rec["verify_success_after_advisory"] = any(post)
        rec["post_advisory_verify_events"] = len(post)
    else:
        rec["verify_success_after_advisory"] = None
        rec["post_advisory_verify_events"] = 0
    for k in ("completed", "passed", "partial", "test_file_modified",
              "api_errors", "pg_errors", "truncated", "env_error",
              "final_outcome", "judge_error"):
        if k in m:
            rec[k] = m[k]
    with open(RUNS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def do_run(phase, task_key, arm, run, sleep_s):
    task = dict(TASKS[task_key])
    task["_key"] = task_key
    wd = fresh_workspace(arm, run, task_key)
    test_p = os.path.join(wd, task["test_file"])
    test_sha1 = hashlib.sha1(open(test_p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    preflight(need_guard=(arm != "control"), need_conv=(arm in ("conv", "both")))
    if sleep_s:
        time.sleep(sleep_s)
    print(f"[run] {phase}/{task_key}/{arm}/run{run:02d} model={MODEL} ...", flush=True)
    info = run_pi(arm, task, wd, timeout=int(os.environ.get("PI_RUN_TIMEOUT", "420")))
    # stdout/stderr 留档
    os.makedirs(LOG_DIR, exist_ok=True)
    tag = f"{phase}_{task_key}_{arm}_run{run:02d}"
    with open(os.path.join(LOG_DIR, f"pi_{tag}.stdout.log"), "w", encoding="utf-8") as f:
        f.write(info.get("stdout") or "")
    with open(os.path.join(LOG_DIR, f"pi_{tag}.stderr.log"), "w", encoding="utf-8") as f:
        f.write(info.get("stderr") or "")
    ev = parse_session(info["session"])
    rec = _record(phase, task_key, arm, run, wd, task, info, ev, test_sha1)
    print(f"      done={rec['completed']} passed={rec['passed']}/{len(task['tests'])} "
          f"turns={rec['turns']} pytest={rec['pytest_runs']} "
          f"pseudo={rec['pseudo_total']}(edit_intent={rec['pseudo_edit_intents']}) "
          f"pg_trig={rec['pg_triggers']} landed={rec['edit_landed']} "
          f"advisory={rec['advisory_delivered']} env_err={rec['env_error']} "
          f"identity_ok={rec['model_identity']['consistent'] is not False} "
          f"{rec['elapsed_s']}s", flush=True)
    return rec


# ── 汇总分析 ─────────────────────────────────────────────────


def analyze():
    rows = [json.loads(l) for l in open(RUNS_JSONL, encoding="utf-8") if l.strip()]
    metrics = {"total_runs": len(rows), "by_cell": {}, "pseudo_gate": {}}
    cells = {}
    for r in rows:
        cells.setdefault((r["phase"], r["task"], r["arm"]), []).append(r)
    for (ph, tk, arm), rs in sorted(cells.items()):
        n = len(rs)
        clean = [x for x in rs if not x["env_error"] and not x["model_polluted"]]
        cc = sum(1 for x in clean if x["completed"])
        metrics["by_cell"][f"{ph}/{tk}/{arm}"] = {
            "n": n, "completed": sum(1 for x in rs if x["completed"]),
            "completion_rate": round(sum(1 for x in rs if x["completed"]) / n, 4),
            "partial_mean": round(sum(x["partial"] for x in rs) / n, 4),
            "clean_n": len(clean), "clean_completed": cc,
            "clean_completion_rate": round(cc / len(clean), 4) if clean else None,
            "env_error_runs": sum(1 for x in rs if x["env_error"]),
            "model_polluted_runs": sum(1 for x in rs if x["model_polluted"]),
            "turns_mean": round(sum(x["turns"] for x in rs) / n, 2),
            "pytest_runs_mean": round(sum(x["pytest_runs"] for x in rs) / n, 2),
            "pseudo_runs": sum(1 for x in rs if x["pseudo_total"]),
            "pseudo_total": sum(x["pseudo_total"] for x in rs),
            "pseudo_edit_intents": sum(x["pseudo_edit_intents"] for x in rs),
            "pg_trigger_runs": sum(1 for x in rs if x["pg_triggers"]),
            "pg_triggers": sum(x["pg_triggers"] for x in rs),
            "edit_landed_runs": sum(1 for x in rs if x["edit_landed"]),
            "real_edit_calls": sum(x["real_edit_calls"] for x in rs),
            "advised_runs": sum(1 for x in rs if x["advisory_delivered"]),
        }
    # 伪调用门禁聚合（gate 阶段、pg 触发过的 run）
    gate = [r for r in rows if r.get("pg_triggers") or r["pseudo_edit_intents"]]
    with_trig = [r for r in gate if r["pg_triggers"]]
    landed = [r for r in gate if r["edit_landed"]]
    metrics["pseudo_gate"] = {
        "runs_total": len(rows),
        "runs_with_pseudo_edit_intents": len(gate),
        "runs_pg_triggered": len(with_trig),
        "pg_feedback_total": sum(r["pg_triggers"] for r in rows),
        "runs_edit_landed": len(landed),
        "edit_landed_rate_run_level": (round(len(landed) / len(gate), 4)
                                       if gate else None),
        "intents_total": sum(r["pseudo_edit_intents"] for r in rows),
        "pseudo_forms_total": {},
    }
    for r in rows:
        for f, c in (r.get("pseudo_forms") or {}).items():
            metrics["pseudo_gate"]["pseudo_forms_total"][f] = \
                metrics["pseudo_gate"]["pseudo_forms_total"].get(f, 0) + c
    with open(os.path.join(OUT_DIR, "metrics_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    global RUNS_JSONL
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="s3_triple_mix", choices=sorted(TASKS))
    ap.add_argument("--arm", default=None,
                    choices=["control", "guard", "conv", "both"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--phase", default="smoke", choices=["smoke", "gate", "main"])
    ap.add_argument("--sleep", type=int, default=3)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()

    if a.analyze:
        analyze()
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_agent_dir()
    arms = [a.arm] if a.arm else ["control", "conv"]
    done = set()
    if os.path.exists(RUNS_JSONL):
        for l in open(RUNS_JSONL, encoding="utf-8"):
            try:
                r = json.loads(l)
                done.add((r["phase"], r["task"], r["arm"], r["run"]))
            except Exception:
                continue
    print(f"[plan] phase={a.phase} task={a.task} arms={arms} "
          f"runs={a.runs}..{a.runs + a.start_run - 1} model={PROVIDER}/{MODEL} "
          f"已完成 {len(done)}（断点续跑）", flush=True)
    for arm in arms:
        for k in range(a.start_run, a.start_run + a.runs):
            if (a.phase, a.task, arm, k) in done:
                print(f"[skip] {a.phase}/{a.task}/{arm}/run{k:02d} 已落盘", flush=True)
                continue
            do_run(a.phase, a.task, arm, k, a.sleep)
    print(f"[done] 汇总: python -u experiments/ge2_runner.py --analyze", flush=True)


if __name__ == "__main__":
    main()

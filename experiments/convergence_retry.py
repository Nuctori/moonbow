# -*- coding: utf-8 -*-
# 安全整改批次：convergence_retry.py（2026-10-09）：subprocess 列表参数、
# torch 权重安全加载、路径 _safe_join 边界校验、sha1→sha256。
"""experiments/convergence_retry.py — 收敛进度追踪 Phase 2：advisory 收敛提示 uplift 实验。

问题：用收敛触发规则（converge.stall / converge.fail_streak，Phase 0/0.5 离线回放
得出）在真实 agent 运行中投递"收窄范围"advisory 提示，能否提升复合任务
（c3_triple_mix 三 bug：整除 + 排序键 + 可变默认）的完成率 / 部分得分？

历史基线：c3_triple_mix 全 baseline 30 run 完成 12（GUARD_EFFECT_REPORT §2.0b /
Phase 0 语料，pi_eval/home/sessions 2026-09-22 批次）。任务构造取自
experiments/compound_traps.json 的 c3_triple_mix（与历史批次同源，未重建）。

两臂（各 --runs 次，串行）：
  control      无插件（--no-extensions），无任何干预。
  convergence  正式插件 + guard 服务（MOONBOW_GUARD_CONVERGENCE=advisory）+
               MOONBOW_GUARD_PROCESS=advisory + 命名目标注入
               （MOONBOW_GUARD_CONVERGENCE_TARGETS=test_half,test_sort_words,
               test_add_tag，启用逐测试撤回与部分得分口径）。

前提（runner 只做健康检查，服务由外部启动，失败显式退出——适配层教训：
错误必须透传，不许静默吞）：
  1. CC 适配网关  PI_GATEWAY      默认 http://127.0.0.1:8899/health
     （python -u experiments/cc_proxy_server.py，CC_API_KEY 必须在环境）
  2. guard 服务   PI_GUARD_URL    默认 http://127.0.0.1:18617（非 18492，避免冲突）
     （MOONBOW_GUARD_CONVERGENCE=advisory PYTHONPATH=src python -m
       moonbow.guard.cli serve --port 18617 --lazy）
  3. GET /v1/convergence-status 应回 mode=advisory（advisory 臂）

隔离：独立 agent 目录（results/convergence-phase2/agent_home，从 pi_eval/home
复制 models/settings/auth）与逐 run 独立工作区；不污染既有 sessions。

记录（每 run 一行 JSONL，断点续跑：已存在的 (arm, run) 跳过）：
  completed / partial(通过测试数/3) / rounds / pytest_runs /
  advisory_delivered(+时机 turn / 投递后首个验证成功) / process_reminders /
  api_errors / pg_errors / truncated(截断签名) / test_file_modified /
  final_outcome / session 文件名（证据留存）。

用法：
  python -u experiments/convergence_retry.py --runs 15           # 全量两臂
  python -u experiments/convergence_retry.py --arm control --runs 1   # 冒烟
  python -u experiments/convergence_retry.py analyze             # 汇总 metrics.json
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


def _safe_join(root, *parts):
    """规范化拼接并校验结果不逃出 root（防路径穿越）。"""
    p = os.path.realpath(os.path.join(root, *parts))
    root_r = os.path.realpath(root)
    if p != root_r and not p.startswith(root_r + os.sep):
        raise ValueError(f"path escapes root: {p!r}")
    return p


REPO = os.path.abspath(os.path.join(HERE, ".."))
PI_BIN = os.environ.get("PI_BIN", "pi.cmd" if os.name == "nt" else "pi")
OUT_DIR = os.path.join(REPO, "results", "convergence-phase2")
AGENT_DIR = os.environ.get("PI_EVAL_AGENT_DIR", os.path.join(OUT_DIR, "agent_home"))
SESSIONS = os.path.join(AGENT_DIR, "sessions")
WS_ROOT = os.path.join(OUT_DIR, "ws")
RUNS_JSONL = os.path.join(OUT_DIR, "runs.jsonl")
GUARD_URL = os.environ.get("PI_GUARD_URL", "http://127.0.0.1:18617")
GATEWAY_URL = os.environ.get("PI_GATEWAY", "http://127.0.0.1:8899/health")
GUARD_EXT = os.environ.get("PI_GUARD_EXT",
                           os.path.join(REPO, "src", "moonbow", "guard",
                                        "extensions", "progress-guard.ts"))
# 收敛提示标记（convergence.ADVISORY_MARK 的前缀段；客户端采集层凭它防自审）
ADVISORY_MARK = "【进度守卫（收敛提示）】"
PROCESS_MARK = "【进度守卫（过程核查）】"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 任务集：c3_triple_mix（三 bug 复合，历史基线 12/30）。与 §2.0b 同源文件。
TRAP = json.load(open(os.path.join(HERE, "compound_traps.json"),
                      encoding="utf-8"))["c3_triple_mix"]
TARGET_TESTS = ["test_half", "test_sort_words", "test_add_tag"]
N_TESTS = 3
TEST_FILE = "test_app.py"


# ── 基础设施健康检查（显式失败，不静默）─────────────────────────


def _get(url, timeout=5):
    with OPENER.open(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def preflight(need_guard: bool):
    if not os.path.exists(PI_BIN) and not shutil.which(PI_BIN):
        sys.exit(f"[preflight] pi 可执行文件不存在: {PI_BIN}")
    if not os.path.exists(GUARD_EXT):
        sys.exit(f"[preflight] 守卫扩展不存在: {GUARD_EXT}")
    try:
        st, body = _get(GATEWAY_URL)
        if st != 200:
            sys.exit(f"[preflight] 模型网关 {GATEWAY_URL} 返回 {st}")
    except Exception as e:
        sys.exit(f"[preflight] 模型网关 {GATEWAY_URL} 不可达: {e} —— "
                 f"先启动: CC_API_KEY=<key> python -u experiments/cc_proxy_server.py")
    if need_guard:
        try:
            st, body = _get(GUARD_URL + "/v1/convergence-status")
            mode = json.loads(body).get("mode")
            if mode != "advisory":
                sys.exit(f"[preflight] guard 服务 convergence-mode={mode!r} != advisory ——"
                         f" 用 MOONBOW_GUARD_CONVERGENCE=advisory 重启（端口 {GUARD_URL}）")
        except Exception as e:
            sys.exit(f"[preflight] guard 服务 {GUARD_URL} 不可达: {e} —— 先启动:\n"
                     f"  MOONBOW_GUARD_CONVERGENCE=advisory PYTHONPATH=src "
                     f"python -m moonbow.guard.cli serve --port 18617 --lazy")
    # pytest 可用性（任务面判定依赖）
    r = subprocess.run(["python", "-m", "pytest", "--version"], shell=False, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        sys.exit("[preflight] python -m pytest 不可用")


def setup_agent_dir():
    """独立 agent 目录：从 pi_eval/home 复制 provider 配置，sessions 清空隔离。"""
    src_home = os.path.join(REPO, "..", "pi_eval", "home")
    os.makedirs(SESSIONS, exist_ok=True)
    for fn in ("models.json", "settings.json", "auth.json", "models-store.json"):
        p = os.path.join(src_home, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(AGENT_DIR, fn))


# ── 单 run 执行 ──────────────────────────────────────────────


def fresh_workspace(arm, run):
    wd = os.path.join(WS_ROOT, arm, "run%02d" % run)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    for fn, content in TRAP.items():
        if fn == "prompt":
            continue
        # 二进制写入：字节与 TRAP 常量一致，篡改检查不被 \r\n 污染
        with open(os.path.join(wd, fn), "wb") as f:
            f.write(content.encode("utf-8"))
    return wd


def run_pi(arm, workdir, timeout):
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = AGENT_DIR
    if arm == "convergence":
        env["MOONBOW_GUARD_URL"] = GUARD_URL
        env["MOONBOW_GUARD_PROCESS"] = "advisory"
        # Phase 2：收敛通道 advisory + 命名目标注入（部分得分口径）
        env["MOONBOW_GUARD_CONVERGENCE"] = "advisory"
        env["MOONBOW_GUARD_CONVERGENCE_TARGETS"] = ",".join(TARGET_TESTS)
    else:
        # control 臂：零守卫痕迹（env 不设 + --no-extensions）
        for k in ("MOONBOW_GUARD_URL", "MOONBOW_GUARD_PROCESS",
                  "MOONBOW_GUARD_CONVERGENCE", "MOONBOW_GUARD_CONVERGENCE_TARGETS"):
            env.pop(k, None)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    cmd = [PI_BIN, "--provider", os.environ.get("PI_PROVIDER", "ccfree"),
           "--model", os.environ.get("PI_MODEL", "xiaomi/mimo-v2.5"),
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
        r = subprocess.run(cmd, input=TRAP["prompt"], cwd=workdir, env=env,
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


_TRUNC_PAT = re.compile(r"\[tool_call[\s\]]|<arg_value>|^\s*(read|bash|edit|write|grep|ls)\b[\s\S]{0,80}\{",
                        re.IGNORECASE)


def parse_session(path):
    """从会话 JSONL 提取轮次/验证事件/守卫投递（顺序保持，供时机判定）。"""
    ev = {"turns": 0, "api_errors": 0, "pytest_runs": 0, "verify_events": [],
          "advisory": [], "process_reminders": 0, "observed": 0,
          "deliveries": 0, "last_assistant_text": "",
          "last_stop": None, "trunc_like": False, "tool_results": []}
    if not path or not os.path.exists(path):
        return ev
    call_text = {}
    # 第一遍：收集 toolCall 文本（pytest 配对用）
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
                    call_text[b["id"]] = json.dumps(b.get("arguments") or {},
                                                    ensure_ascii=False)
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
            continue
        if ctype == "progress-guard:process":
            data = d.get("data") or {}
            dl = data.get("deliveries") or []
            ev["deliveries"] = max(ev["deliveries"], len(dl))
            ev["observed"] = max(ev["observed"],
                                 sum(1 for x in dl if x.get("status") == "observed"))
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        if role == "assistant":
            if m.get("stopReason") == "error":
                ev["api_errors"] += 1
                continue
            ev["turns"] += 1
            ev["last_stop"] = m.get("stopReason")
            ts = _texts(m)
            if ts:
                ev["last_assistant_text"] = ts[-1]
        elif role == "toolResult":
            txt = "\n".join(_texts(m))
            cid = str(m.get("toolCallId") or "")
            paired = call_text.get(cid)
            if paired is not None:
                is_py = "pytest" in paired.lower()
            else:
                is_py = bool(re.search(r"\d+\s+passed|\d+\s+failed|no tests ran",
                                       txt, re.IGNORECASE))
            full_pass = bool(txt) and ("passed" in txt.lower()
                                       and "failed" not in txt.lower()
                                       and "error" not in txt.lower())
            if is_py:
                ev["pytest_runs"] += 1
                ev["verify_events"].append(full_pass)
                ev["tool_results"].append(txt[-200:])
    if ev["last_assistant_text"]:
        ev["trunc_like"] = bool(
            _TRUNC_PAT.search(ev["last_assistant_text"])
            and not re.search(r"[。！？.!?\s]$", ev["last_assistant_text"].rstrip()))
    return ev


def judge(workdir, run_info, ev, pi_info=None):
    """任务面判定 + 环境错误显式计数（适配层教训：错误透传，不许吞）。"""
    pi_info = pi_info or {}
    m = {"completed": False, "passed": 0, "partial": 0.0, "test_file_modified": None}
    test_p = os.path.join(workdir, TEST_FILE)
    if os.path.exists(test_p):
        # 归一化换行后比较（历史工作区可能以文本模式写入）
        h = hashlib.sha256(open(test_p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
        m["test_file_modified"] = (h != run_info["test_sha1"])
    try:
        r = subprocess.run(["python", "-m", "pytest", TEST_FILE, "-q", "--no-header"], shell=False,
                           cwd=workdir, capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        mp = re.search(r"(\d+)\s+passed", out)
        m["passed"] = min(N_TESTS, int(mp.group(1))) if mp else 0
        m["completed"] = ("passed" in out and "failed" not in out
                          and "error" not in out.lower() and m["passed"] == N_TESTS)
    except Exception:
        pass
    m["partial"] = round(m["passed"] / N_TESTS, 4)
    m["api_errors"] = ev["api_errors"]
    m["pg_errors"] = len(re.findall(r"\[PG\] (?:agent_end )?ERROR",
                                    pi_info.get("stderr") or ""))
    m["truncated"] = bool(run_info.get("timeout") or ev["trunc_like"]
                          or ev.get("last_stop") == "toolUse")
    m["env_error"] = bool(m["api_errors"] or m["pg_errors"] or m["truncated"])
    if m["completed"]:
        m["final_outcome"] = "completed"
    elif ev["pytest_runs"] == 0:
        m["final_outcome"] = "no_verify"           # 全程无验证（静默停止形态）
    elif m["passed"] > 0:
        m["final_outcome"] = "partial_stop"        # 有部分得分但未收窄到底
    elif ev["advisory"]:
        m["final_outcome"] = "advised_fail"        # 投递后仍未完成
    else:
        m["final_outcome"] = "failed"              # 全失败（含打转/静默停止）
    return m


def _record(arm, run, wd, info, ev, test_sha1):
    """构造单 run 记录并追加 runs.jsonl（do_run 与 --salvage 共用）。"""
    rec = {"arm": arm, "run": run,
           "session": os.path.basename(info["session"]) if info["session"] else None,
           "elapsed_s": info["elapsed"], "rc": info["rc"],
           "turns": ev["turns"], "pytest_runs": ev["pytest_runs"],
           "process_reminders": ev["process_reminders"],
           "guard_deliveries": ev["deliveries"], "guard_observed": ev["observed"],
           "advisory_delivered": len(ev["advisory"]),
           "advisory_turn": ev["advisory"][0]["turn"] if ev["advisory"] else None,
           "advisory_content": ev["advisory"][0]["content"] if ev["advisory"] else None}
    # 投递后首个验证成功事件（机制指标：投递 → 验证成功的转化）
    if ev["advisory"]:
        seen = ev["advisory"][0]["verify_seen"]
        post = ev["verify_events"][seen:]
        rec["verify_success_after_advisory"] = any(post)
        rec["post_advisory_verify_events"] = len(post)
    else:
        rec["verify_success_after_advisory"] = None
        rec["post_advisory_verify_events"] = 0
    rec.update(judge(wd, {"test_sha1": test_sha1}, ev, info))
    with open(RUNS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def do_run(arm, run, sleep_s):
    wd = fresh_workspace(arm, run)
    test_p = _safe_join(wd, TEST_FILE)
    test_sha1 = hashlib.sha256(open(test_p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    need_guard = (arm == "convergence")
    preflight(need_guard)
    if sleep_s:
        time.sleep(sleep_s)
    print(f"[run] {arm}/run{run:02d} ...", flush=True)
    info = run_pi(arm, wd, timeout=int(os.environ.get("PI_RUN_TIMEOUT", "420")))
    ev = parse_session(info["session"])
    rec = _record(arm, run, wd, info, ev, test_sha1)
    print(f"      done={rec['completed']} passed={rec['passed']}/3 "
          f"turns={rec['turns']} pytest={rec['pytest_runs']} "
          f"advisory={rec['advisory_delivered']} env_err={rec['env_error']} "
          f"{rec['elapsed_s']}s", flush=True)
    return rec


# ── 汇总分析 ─────────────────────────────────────────────────


def _fisher_two_tailed(a, b, c, d):
    """Fisher 精确检验（双侧，超几何）。a/b=臂1 完成/未完成，c/d=臂2。"""
    from math import comb
    n = a + b + c + d
    k = a + c
    r1 = a + b
    if n == 0:
        return 1.0
    p_obs = comb(r1, a) * comb(n - r1, k - a) / comb(n, k)
    tail = 0.0
    for x in range(max(0, k - (n - r1)), min(r1, k) + 1):
        p = comb(r1, x) * comb(n - r1, k - x) / comb(n, k)
        if p <= p_obs * (1 + 1e-9):
            tail += p
    return min(1.0, tail)


def analyze():
    rows = [json.loads(l) for l in open(RUNS_JSONL, encoding="utf-8") if l.strip()]
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    metrics = {"total_runs": len(rows),
               "arms": {}, "clean_arms": {}, "delivery": {}}
    for arm, rs in arms.items():
        n = len(rs)
        comp = sum(1 for r in rs if r["completed"])
        clean = [r for r in rs if not r["env_error"]]
        cc = sum(1 for r in clean if r["completed"])
        metrics["arms"][arm] = {
            "n": n, "completed": comp, "completion_rate": round(comp / n, 4) if n else None,
            "partial_mean": round(sum(r["partial"] for r in rs) / n, 4) if n else None,
            "turns_mean": round(sum(r["turns"] for r in rs) / n, 2) if n else None,
            "pytest_runs_mean": round(sum(r["pytest_runs"] for r in rs) / n, 2) if n else None,
            "env_error_runs": n - len(clean),
            "api_error_runs": sum(1 for r in rs if r["api_errors"]),
            "truncated_runs": sum(1 for r in rs if r["truncated"]),
            "pg_error_runs": sum(1 for r in rs if r["pg_errors"]),
            "clean_completed": cc, "clean_n": len(clean),
            "clean_completion_rate": round(cc / len(clean), 4) if clean else None,
            "advised_runs": sum(1 for r in rs if r["advisory_delivered"]),
            "advisory_total": sum(r["advisory_delivered"] for r in rs),
            "outcome_hist": {},
        }
        for r in rs:
            metrics["arms"][arm]["outcome_hist"][r["final_outcome"]] = \
                metrics["arms"][arm]["outcome_hist"].get(r["final_outcome"], 0) + 1
        metrics["clean_arms"][arm] = {
            "n": len(clean), "completed": cc,
            "completion_rate": round(cc / len(clean), 4) if clean else None,
            "partial_mean": round(sum(r["partial"] for r in clean) / len(clean), 4) if clean else None,
        }
        if arm == "convergence":
            adv_runs = [r for r in rs if r["advisory_delivered"]]
            conv_hits = sum(1 for r in adv_runs
                            if r.get("verify_success_after_advisory"))
            turns_at = [r["advisory_turn"] for r in adv_runs
                        if r.get("advisory_turn") is not None]
            metrics["delivery"] = {
                "runs_with_delivery": len(adv_runs),
                "delivery_rate": round(len(adv_runs) / n, 4) if n else None,
                "delivery_turn_mean": (round(sum(turns_at) / len(turns_at), 2)
                                        if turns_at else None),
                "post_delivery_verify_success": conv_hits,
                "post_delivery_conversion": (round(conv_hits / len(adv_runs), 4)
                                             if adv_runs else None),
                "advise_then_complete": sum(1 for r in adv_runs if r["completed"]),
            }
        # 静默停止 vs 打转（运行形态代理口径，非 Phase 0 原始定义）：
        #   spin      = 未完成 且 pytest>=4 次（反复验证仍未过）
        #   silent    = 未完成 且 pytest<=2 次 且 turns<=6（少量验证即消失）
        #   mid       = 其余未完成
        sil = sum(1 for r in rs if not r["completed"]
                  and r["pytest_runs"] <= 2 and r["turns"] <= 6)
        spin = sum(1 for r in rs if not r["completed"] and r["pytest_runs"] >= 4)
        metrics["arms"][arm]["silent_stop_runs"] = sil
        metrics["arms"][arm]["spin_runs"] = spin
        metrics["arms"][arm]["mid_other_runs"] = sum(
            1 for r in rs if not r["completed"]
            and not (r["pytest_runs"] <= 2 and r["turns"] <= 6)
            and not r["pytest_runs"] >= 4)
    ctrl = metrics["arms"].get("control") or {}
    conv = metrics["arms"].get("convergence") or {}
    if ctrl and conv:
        a, b = conv["completed"], conv["n"] - conv["completed"]
        c, d = ctrl["completed"], ctrl["n"] - ctrl["completed"]
        metrics["comparison"] = {
            "delta_completion": round((conv["completion_rate"] or 0) - (ctrl["completion_rate"] or 0), 4),
            "delta_partial_mean": round((conv["partial_mean"] or 0) - (ctrl["partial_mean"] or 0), 4),
            "fisher_p_two_sided": round(_fisher_two_tailed(a, b, c, d), 4),
            "vs_history_12_of_30": {
                "control_completion": ctrl["completion_rate"],
                "convergence_completion": conv["completion_rate"],
            },
        }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=["control", "convergence"])
    ap.add_argument("--runs", type=int, default=15, help="每臂 run 数（默认 15，两臂合计≤40）")
    ap.add_argument("--sleep", type=int, default=3, help="run 间隔秒（串行节流）")
    ap.add_argument("--analyze", action="store_true", help="只汇总 runs.jsonl")
    ap.add_argument("--salvage", nargs=3, metavar=("ARM", "RUN", "SESSION"),
                    help="把已完成但未落盘的 run 记录进 runs.jsonl（不重跑）")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if a.salvage:
        arm, run, sess = a.salvage[0], int(a.salvage[1]), a.salvage[2]
        wd = os.path.join(WS_ROOT, arm, "run%02d" % run)
        info = {"stderr": "", "session": sess, "elapsed": None, "rc": 0,
                "timeout": False, "stdout": ""}
        ev = parse_session(sess)
        # elapsed 用会话首末时间戳近似
        ts = []
        for line in open(sess, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("message") or {}).get("timestamp") if d.get("type") == "message" else None
            if t:
                ts.append(t)
        info["elapsed"] = round((ts[-1] - ts[0]) / 1000, 1) if len(ts) >= 2 else 0.0
        test_sha1 = hashlib.sha256(TRAP[TEST_FILE].encode("utf-8")).hexdigest()
        rec = _record(arm, run, wd, info, ev, test_sha1)
        print("salvaged:", json.dumps({k: rec[k] for k in
              ("arm", "run", "completed", "passed", "turns", "advisory_delivered")}),
              flush=True)
        return
    if a.analyze:
        analyze()
        return
    setup_agent_dir()
    arms = [a.arm] if a.arm else ["control", "convergence"]
    total_planned = len(arms) * a.runs
    if total_planned > 40:
        sys.exit(f"[budget] 计划 {total_planned} run 超过 40 上限，拒绝")
    done = set()
    if os.path.exists(RUNS_JSONL):
        for l in open(RUNS_JSONL, encoding="utf-8"):
            try:
                r = json.loads(l)
                done.add((r["arm"], r["run"]))
            except Exception:
                continue
    print(f"[plan] arms={arms} runs={a.runs} 已完成 {len(done)}（断点续跑）", flush=True)
    for arm in arms:
        for k in range(1, a.runs + 1):
            if (arm, k) in done:
                print(f"[skip] {arm}/run{k:02d} 已在 runs.jsonl", flush=True)
                continue
            do_run(arm, k, a.sleep)
    print("[done] 全部 run 完成；汇总: python -u experiments/convergence_retry.py --analyze",
          flush=True)


if __name__ == "__main__":
    main()

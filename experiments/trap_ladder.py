# -*- coding: utf-8 -*-
"""experiments/trap_ladder.py — 陷阱阶梯校准：找出模型落入 30-70% 难度带的陷阱。

原理：LLM 有已知的系统性失败模式（浮点比较、off-by-one、可变默认参数……）。
每类陷阱做成单文件 pytest 任务（修 bug → 全测试通过），用 baseline 臂
（无守卫）跑 N 次，统计首次通过率。落在 30-70% 的陷阱类 = uplift 实验的
合适难度（见 GUARD_EFFECT_REPORT.md §7.1）。无需守卫服务。

环境变量：
  PI_EVAL_AGENT_DIR  pi agent 配置目录（默认 ./home）
  PI_EVAL_WORK_DIR   陷阱工作区根目录（默认系统临时目录下 trap_ladder）

用法：python -u experiments/trap_ladder.py [--rounds 2] [--model ID]
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PI_BIN = os.environ.get("PI_BIN", "pi.cmd" if os.name == "nt" else "pi")
AGENT_DIR = os.environ.get("PI_EVAL_AGENT_DIR", os.path.join(HERE, "home"))
SESSIONS = os.path.join(AGENT_DIR, "sessions")
ROOT_DIR = os.environ.get("PI_EVAL_WORK_DIR",
                          os.path.join(tempfile.gettempdir(), "trap_ladder"))

# 已知 LLM 系统性失败模式 → 单文件陷阱任务
TRAPS = {
    "float_eq": {
        "app.py": "def total(a, b):\n    return a + b\n",
        "test_app.py": ("from app import total\n\n"
                        "def test_total():\n    assert total(0.1, 0.2) == 0.3\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
    "off_by_one": {
        "app.py": ("def chunk(items, size):\n"
                   "    out = []\n"
                   "    for i in range(0, len(items), size):\n"
                   "        out.append(items[i:i + size - 1])\n"
                   "    return out\n"),
        "test_app.py": ("from app import chunk\n\n"
                        "def test_chunk():\n"
                        "    assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
    "mutable_default": {
        "app.py": ("def add_tag(item, tags=[]):\n"
                   "    tags.append(item)\n"
                   "    return tags\n"),
        "test_app.py": ("from app import add_tag\n\n"
                        "def test_no_shared_state():\n"
                        "    assert add_tag('a') == ['a']\n"
                        "    assert add_tag('b') == ['b']\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
    "str_immut": {
        "app.py": ("def mask(secret):\n"
                   "    for ch in secret:\n"
                   "        ch = '*'\n"
                   "    return secret\n"),
        "test_app.py": ("from app import mask\n\n"
                        "def test_mask():\n"
                        "    assert mask('abc') == '***'\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
    "int_div": {
        "app.py": "def half(n):\n    return n / 2 // 1 * 2 // 2\n",
        "test_app.py": ("from app import half\n\n"
                        "def test_half():\n    assert half(5) == 2.5\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
    "sort_key": {
        "app.py": ("def sort_words(words):\n"
                   "    return sorted(words, key=len)\n"),
        "test_app.py": ("from app import sort_words\n\n"
                        "def test_sort():\n"
                        "    assert sort_words(['bb','a','ccc','aa']) == ['a','aa','bb','ccc']\n"),
        "prompt": "运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。",
    },
}


# ── 复合陷阱梯级（2026-09-23）─────────────────────────────────
# 单 bug 任务全部低于 mimo 难度下限（§7.2.1），uplift 可测带在复合级别。
# 每个 = 2-3 个不同失败模式的 bug，修一个不影响其他；pytest 全绿 ⟺ 全修对。
COMPOUND = json.load(open(os.path.join(HERE, "compound_traps.json"), encoding="utf-8"))


PROMPT_COMMON = ""


def run_pi(workdir, prompt, model, thinking, timeout=240):
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = AGENT_DIR
    if os.environ.get("TRAP_GUARD") == "1":
        env.setdefault("MOONBOW_GUARD_PROCESS", "advisory")
        env.setdefault("MOONBOW_GUARD_URL", "http://127.0.0.1:18492")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    cmd = [PI_BIN, "--provider", os.environ.get("PI_PROVIDER", "ccfree"),
           "--model", model, "--print", "--no-extensions",
           "--no-skills", "--no-themes", "--no-prompt-templates",
           "--no-context-files", "--thinking", thinking,
           "--session-dir", SESSIONS]
    if os.environ.get("TRAP_GUARD") == "1":
        cmd[cmd.index("--no-extensions")] = "-e"
        cmd.insert(cmd.index("-e") + 1,
                   os.path.join(HERE, "extensions", "formal", "progress-guard.ts"))
        env_guard = "MOONBOW_GUARD_PROCESS"
    try:
        r = subprocess.run(cmd, input=prompt, cwd=workdir, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        out = r.stdout or ""
    except subprocess.TimeoutExpired:
        out, rc = "", -1
    return out


def pytest_passes(workdir):
    r = subprocess.run(["python", "-m", "pytest", "test_app.py", "-q", "--no-header"],
                       shell=False, cwd=workdir, capture_output=True,
                       text=True, timeout=60)
    out = (r.stdout or "") + (r.stderr or "")
    return "passed" in out and "failed" not in out and "error" not in out.lower()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PI_MODEL", "xiaomi/mimo-v2.5"))
    ap.add_argument("--thinking", default="off")
    ap.add_argument("--rounds", type=int, default=2, help="每陷阱重复次数")
    ap.add_argument("--guard", action="store_true", help="挂正式守卫插件（with_guard 臂）")
    ap.add_argument("--sleep", type=int, default=0,
                    help="每臂之间的间隔秒数（串行节流，防自激限流）")
    ap.add_argument("--traps", default=None, help="逗号分隔的陷阱过滤（默认全部）")
    ap.add_argument("--suite", default="basic", choices=["basic", "compound"],
                    help="basic=单bug六级 / compound=复合梯级")
    a = ap.parse_args()

    os.makedirs(ROOT_DIR, exist_ok=True)
    pool = TRAPS if a.suite == "basic" else COMPOUND
    traps = {k: v for k, v in pool.items()
             if not a.traps or k in a.traps.split(",")}
    tag = "with_guard" if a.guard else "baseline"
    results = {}
    for trap, files in traps.items():
        wd = os.path.join(ROOT_DIR, trap)
        os.makedirs(wd, exist_ok=True)
        passed = 0
        runs = []
        for rnd in range(a.rounds):
            if a.sleep and rnd > 0:
                time.sleep(a.sleep)
            for fn, c in files.items():
                with open(os.path.join(wd, fn), "w", encoding="utf-8") as f:
                    f.write(c)
            t0 = time.time()
            run_pi(wd, files["prompt"], a.model, a.thinking)
            ok = pytest_passes(wd)
            passed += ok
            runs.append(f"{'P' if ok else 'F'}({time.time()-t0:.0f}s)")
        rate = passed / a.rounds * 100
        band = "★命中30-70%带" if 30 <= rate <= 70 else ("过易" if rate > 70 else "过难")
        results[f"{trap}[{tag}]"] = (passed, a.rounds, rate, band, runs)
        print(f"{trap:16s}[{tag:10s}] {passed}/{a.rounds} = {rate:3.0f}%  {band:12s} {runs}",
              flush=True)

    print("\n=== 校准结论 ===")
    hits = [k for k, v in results.items() if 30 <= v[2] <= 70]
    print("落在 30-70% 带的陷阱:", hits if hits else "无——需调整陷阱难度")
    if hits:
        print(f"\n建议 uplift 实验用: {hits[0]}（其余作泛化验证）")


if __name__ == "__main__":
    main()

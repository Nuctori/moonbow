# -*- coding: utf-8 -*-
"""moonbow.guard.convergence

收敛进度追踪 Phase 1：stage audit 的收敛信号 shadow 通道
（只记录、零投递、默认关闭；与 audit_semantic.P7 同模式）。

规则来源（R6b 后的最终形态；先验证据见
docs/semantic_runtime_progress.md 的 Phase 0 / Phase 0.5 段，
离线回放脚本 tools/convergence_study.py / tools/goal_coverage_study.py）：

- converge.stall（G2 时间视角）：轮次 >= 4 且 coverage == 0 且流中出现过
  pytest 输出 → matched=True。coverage = goals_passed / goals_total
  （goals_passed = 曾通过且未被后续失败输出撤回的目标数）。
  **无 pytest 痕迹的任务不判定**（matched=None + abstain_reason）：
  没有 pytest 验证目标时 coverage==0 只反映"无证据"，不反映停滞，
  判了就是把非 Python/问答/探索类任务全部误标（Phase 0.5 生产化边界）。
- converge.fail_streak（Phase 0 基线特例规则）：自最近一次 pytest 全绿
  往回连续 pytest 失败轮数 >= 2 → matched=True。口径移植
  convergence_study.snapshot_features 的 pytest_fail_streak：
  严格二分（pytest 结果非全绿即计失败，含部分得分与收集期错误输出），
  与覆盖度的部分得分口径不同源、各自独立——与两条离线回测保持
  同卷可比。

轮（round）定义（近似，第一版）：
- 原始口径（convergence_study.parse_session）：一条含 >=1 toolCall 的
  assistant 消息 + 其后续全部 toolResult = 1 轮；无工具调用的收尾文本
  不计轮。
- StageBlock 流不含 assistant 消息边界，无法还原"并行多调用同轮"。
  第一版以 tool_result 事件近似：块流按 seq 升序，第 k 个 toolResult
  块 = 第 k 轮观测点；当前轮次 = toolResult 总数。
- 局限：①并行多调用的一轮被拆成多轮 → 轮次系统性偏高（触发偏早；
  shadow 只记录不投递，偏差方向固定且可解释）；②轮内成败并列
  （同轮 1 failed + 1 passed）被拆成相邻两轮，fail_streak 从
  "轮内任一失败且无成功"变为"逐结果连续失败"。精确轮需客户端在
  流上携带 turn 边界（Pi 事件侧可得，StageBlock 侧不可得），留待
  后续版本。

落地证据解析（移植自 tools/goal_coverage_study.py，逐函数注明；
确定性快速通道——生产版落地证据应由 semantic matcher 对齐判定，
禁 rule 路径作为判定默认）：
- 全绿（含 "passed" 且无 "failed"/"error"）→ 全部目标落地；
- 失败输出 → 输出中点名的目标测试移出 passed 集（回归处理）；
- 摘要行 "N passed / M failed" 且失败点名数与 M 一致 → 其余目标视为
  通过（部分得分）；匿名复合目标按 grounded = total - M 推断；
- 单目标任务：目标 = 整个 pytest 套件通过（与 pytest_passes 口径同源）;
- 无法解析的输出（如收集期错误无摘要行）→ 不动 passed 集
  （"无证据"而非"失败证据"）。

目标数注入：第一版不从 req 提取（结构化目标抽取属于 matcher 对齐
判定范畴），goals_total 显式传入、缺省 1；可选 target_tests 注入
命名目标（启用逐测试名撤回与部分得分；此时目标数 = len(target_tests)）。

幂等：compute() 是块流的**纯函数**——无跨调用可变状态；同一块流
重复审计产生相同输出（与 StageAuditor.audit() 的 fingerprint 幂等
设计一致）。客户端（extensions/process-audit.ts）每批发送窗口内
全部块，因此纯函数口径等价于全流重放。

零投递（硬约束）：本模块结果只进入 audit() 返回值的
shadow_convergence 键；绝不产生 reminder、绝不参与 findings /
semantic 判定、绝不新增用户投递。

规则集 v2（2026-10-04，eigen_failure_attribution.md 归因驱动；先经
results/guard-effect-v2/eigen_failure_attribution.md §7 的干预点标注）：
- converge.repeat（新触发器）：a) consecutive_identical——连续 N>=12 条
  "归一化后相同"的工具调用（归一化 = 去数字/路径/空白后的命令核心；
  归因样本 = control r3 的 877 次同命令 timeit 退化循环）；
  b) edit_oscillation——同一文件被写入 >=4 次且每次写入后 pytest 结果
  集合未变化（修了没效果；归因样本 = control r1 的同文件变体 30 写
  振荡、约 8 个来回）。任一命中 → matched=True。
- converge.stall 轮次线缩放：v2 下 STALL_MIN_ROUND(4) 替换为
  min(100, max(30, budget_s // 15))，budget_s 来自
  MOONBOW_GUARD_CONVERGENCE_BUDGET（缺省 900 → 60 轮）。实证动机：
  tierA_eigen_final.md §5——900s 预算下固定高轮次线永不触发，
  触发线必须随预算缩放。v1 不受影响（既有测试锁定 4 轮口径）。
- converge.perf_retest（新触发器）：块流中出现 "STATUS: A" 收尾申报 →
  matched=True（归因样本 = both r4/r6/r9：会内 27 passed、如实申报 A、
  judge 复测翻掉 <=1µs 平局项）。投递另有 env 门控（perf_retest_enabled：
  MOONBOW_GUARD_PERF_RETEST=1，或 MOONBOW_GUARD_CONVERGENCE_TARGETS /
  payload convergence_target_tests 含 speedup/perf 字样）与独立预算键
  （每任务 1 次，不占既有提醒预算）。
- 门控与标识：两新规则 shadow 模式只记录；advisory 模式才可投递；
  off 模式零变化。ruleset 版本（v1/v2）由 server 装配决定并经
  /v1/convergence-status 的 ruleset 键暴露；ConvergenceShadow 缺省
  ruleset="v1"（既有调用零变化）。
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .protocol import StageBlock

# 触发阈值：G2 时间视角 = 第 4 轮观测时刻（Phase 0.5 口径；v1 固定，
# v2 由预算缩放替换，见 stall_line_from_budget）
STALL_MIN_ROUND = 4
# fail_streak 触发线（Phase 0 基线规则 pytest_fail_streak>=2）
FAIL_STREAK_MIN = 2

STALL_SIGNAL = "converge.stall"
FAIL_STREAK_SIGNAL = "converge.fail_streak"
# 规则集 v2 新触发器（2026-10-04，eigen 归因驱动）
REPEAT_SIGNAL = "converge.repeat"
PERF_RETEST_SIGNAL = "converge.perf_retest"

# v2 repeat 阈值：连续同命令 >=12 条（877 次循环在第 12 次即应被打断，
# 远早于预算耗尽）；同文件写入 >=4 次且 pytest 结果不变（修了没效果）。
REPEAT_IDENTICAL_MIN = 12
OSCILLATION_MIN_WRITES = 4

# 规则集版本：v1 = Phase 1/2 现状（stall 4 轮 + fail_streak）；
# v2 = 上述 + repeat + perf_retest + stall 轮次线缩放。
RULESET_V1 = "v1"
RULESET_V2 = "v2"
RULESETS = (RULESET_V1, RULESET_V2)

# v2 stall 轮次线缩放（tierA_eigen_final.md §5 的实证必要性）：
# min(100, max(30, budget_s // 15))；budget 缺省 900s → 60 轮。
BUDGET_ENV = "MOONBOW_GUARD_CONVERGENCE_BUDGET"
DEFAULT_CONVERGENCE_BUDGET_S = 900
STALL_MAX_ROUND = 100
STALL_FLOOR_ROUND = 30

# perf_retest 投递门控（与 ruleset 独立；缺省关闭）
PERF_RETEST_ENV = "MOONBOW_GUARD_PERF_RETEST"
TARGETS_ENV = "MOONBOW_GUARD_CONVERGENCE_TARGETS"
_PERF_KEYWORDS = ("speedup", "perf")
# STATUS=A 收尾申报（perf_retest 的块级证据；与 verifier 的 STATUS 门同形）
_PERF_CLAIM_RE = re.compile(r"\bSTATUS\s*[:：]\s*A\b", re.IGNORECASE)

# ---- pytest 摘要解析（移植自 tools/goal_coverage_study.py）----


def is_full_pass(txt: str) -> bool:
    """全绿判定。移植自 tools/goal_coverage_study.py::is_full_pass
    （与 convergence_study.verify_success 同字面口径）。"""
    low = (txt or "").lower()
    return ("passed" in low) and ("failed" not in low) and ("error" not in low)


def is_fail_result(txt: str) -> bool:
    """失败判定。移植自 tools/goal_coverage_study.py::is_fail_result。"""
    low = (txt or "").lower()
    return ("failed" in low) or ("error" in low)


def summary_counts(txt: str):
    """摘要行 "N passed" / "M failed" 计数。
    移植自 tools/goal_coverage_study.py::coverage_trajectory.summary_counts。"""
    m_p = re.search(r"(\d+)\s+passed", txt or "")
    m_f = re.search(r"(\d+)\s+failed", txt or "")
    return (int(m_p.group(1)) if m_p else None, int(m_f.group(1)) if m_f else None)


# 结果文本自身的 pytest 摘要特征（文本回退通道用；配对通道不依赖它）
_PYTEST_TEXT_PAT = re.compile(r"\d+\s+passed|\d+\s+failed|no tests ran", re.IGNORECASE)


def _has_pytest_text_trace(txt: str) -> bool:
    return bool(_PYTEST_TEXT_PAT.search(txt or ""))


def _pytest_events(blocks: Sequence[StageBlock]) -> List[StageBlock]:
    """从块流提取 pytest 证据 toolResult（按 seq 升序、每块一条事件）。

    判定规则（按优先级）：
    1. 结果块的 tool_call_id 能配对到 toolCall 块且其调用文本含 "pytest"
       —— 移植 convergence_study.is_verify_call 的 '"pytest" in command'
       口径；不复刻 name=="bash" 限制（真实流 tool_name 可能被归一化）。
       配对存在即由调用侧决定（调用不是 pytest 时结果文本再像也不算）。
    2. 无配对（id 缺失或 toolCall 块已被环形缓冲截掉）→ 结果文本自身
       呈 pytest 摘要特征（"N passed"/"N failed"/"no tests ran"）才算。
    """
    ordered = sorted(blocks, key=lambda x: x.seq)
    call_text: Dict[str, str] = {}
    for b in ordered:
        if b.kind == "toolCall" and b.tool_call_id and b.tool_call_id not in call_text:
            call_text[b.tool_call_id] = b.text or ""
    events: List[StageBlock] = []
    for b in ordered:
        if b.kind != "toolResult" or not b.text:
            continue
        call = call_text.get(b.tool_call_id) if b.tool_call_id else None
        if call is not None:
            hit = "pytest" in call.lower()
        else:
            hit = _has_pytest_text_trace(b.text)
        if hit:
            events.append(b)
    return events


def coverage_trajectory(
    events: Sequence[StageBlock],
    goals_total: int,
    target_tests: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    """逐 pytest 事件的目标覆盖度轨迹（覆盖度所需的最小状态 = passed 集）。

    状态机整体移植自 tools/goal_coverage_study.py::coverage_trajectory
    （命名目标 / 匿名复合 / 单目标三分支），差异仅两处：
    - 以"每个 pytest 结果一条事件"替代"每轮一组文本"（轮近似定义，见
      模块 docstring）；组内文本顺序处理与逐事件处理等价（last-result-wins）。
    - 匿名复合目标摘要缺失时事件记为 "fail_unattributable"（研究脚本里
      只进 parse_note 的 unattributable_fails；状态语义一致：passed 不动）。
    """
    named = [t for t in target_tests] if target_tests else None
    total = goals_total
    passed: set = set()
    traj: List[Dict[str, Any]] = []

    for e in events:
        txt = e.text
        if is_full_pass(txt):
            if named:
                passed = set(named)
            else:
                passed = {"__g%d__" % i for i in range(total)}
            event = "full_pass"
        elif is_fail_result(txt):
            m_p, m_f = summary_counts(txt)
            if named:
                hit = {t for t in named if t in txt}
                passed -= hit
                if (hit and m_f is not None and len(hit) == m_f
                        and m_p is not None and m_p == total - len(hit)):
                    passed |= (set(named) - hit)
                    event = "partial_fail"
                else:
                    event = "fail"
            elif total > 1:
                # 匿名复合目标：按摘要 "M failed" 推断 grounded = total - M
                if m_f is not None:
                    grounded = max(0, total - m_f)
                    if grounded < len(passed):
                        passed = set()
                    passed = {"__g%d__" % i for i in range(grounded)}
                    event = "partial_fail" if 0 < grounded < total else "fail"
                else:
                    event = "fail_unattributable"   # passed 不动
            else:
                passed = set()
                event = "fail"
        else:
            event = "unparsed"                       # passed 不动
        traj.append({"seq": e.seq, "event": event, "goals_passed": len(passed)})
    return traj


def pytest_fail_streak(events: Sequence[StageBlock]) -> int:
    """自最近一次 pytest 全绿往回连续失败事件数。

    移植 convergence_study.snapshot_features 的 pytest_fail_streak 口径：
    严格二分 —— pytest 结果非全绿（is_full_pass 为 False）即计失败；
    非 pytest 事件不参与（原口径中"本轮无验证调用 → 跳过"）。
    """
    streak = 0
    for e in reversed(list(events)):
        if is_full_pass(e.text):
            break
        streak += 1
    return streak


# ---- 规则集 v2：repeat / perf_retest / stall 缩放（2026-10-04）----
#
# 全部为块流纯函数（compute 的幂等性质保持）；env 只在构造器 /
# 显式 helper 里读取。归因依据见模块 docstring 与
# results/guard-effect-v2/eigen_failure_attribution.md §7。

# 路径形态 token（含 / 或 \ 的连续段；用于"去路径"归一化与写文件名提取）
_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?(?:[\w.\-]+[\\/])+[\w.\-]*")
_DIGIT_RE = re.compile(r"\d+")
# bash 重定向写文件（`> f` / `>> f` / `tee f` / `tee -a f`）
_REDIRECT_RE = re.compile(r">>?\s*([^\s;|&]+)")
_TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s;|&]+)")
# 写入类工具名（与 process_audit._WRITE_TOOLS 同集；此处独立定义防环导入）
_WRITE_TOOL_NAMES = frozenset({"edit", "write", "apply_patch", "multiedit",
                               "str_replace", "create", "notebook_edit"})


def stall_line_from_budget(budget_s) -> int:
    """v2 stall 轮次线：min(100, max(30, budget_s // 15))。

    预算 900s（缺省）→ 60 轮；>=1500s 封顶 100 轮（旧高线形态）；
    <=450s 保底 30 轮。非法/非正值按缺省 900 处理。
    """
    try:
        b = int(budget_s)
    except (TypeError, ValueError):
        b = DEFAULT_CONVERGENCE_BUDGET_S
    if b <= 0:
        b = DEFAULT_CONVERGENCE_BUDGET_S
    return min(STALL_MAX_ROUND, max(STALL_FLOOR_ROUND, b // 15))


def convergence_budget_from_env(environ=None) -> int:
    """读 MOONBOW_GUARD_CONVERGENCE_BUDGET（秒）；缺省/非法 → 900。"""
    env = environ if environ is not None else os.environ
    raw = (env.get(BUDGET_ENV) or "").strip()
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_CONVERGENCE_BUDGET_S
    return v if v > 0 else DEFAULT_CONVERGENCE_BUDGET_S


def perf_retest_enabled(environ=None, target_tests=None) -> bool:
    """perf_retest 投递门控：MOONBOW_GUARD_PERF_RETEST=1（宽松真值），
    或性能目标字样出现在 MOONBOW_GUARD_CONVERGENCE_TARGETS / payload
    注入的 target_tests 中（ge2 链路：客户端 env 透传成 payload 键，
    服务端 env 未必带 TARGETS，故两处都认）。"""
    env = environ if environ is not None else os.environ
    raw = (env.get(PERF_RETEST_ENV) or "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if target_tests:
        joined = ",".join(target_tests)
    else:
        joined = env.get(TARGETS_ENV) or ""
    low = joined.lower()
    return any(k in low for k in _PERF_KEYWORDS)


def _parse_tool_call(text: str) -> Tuple[Optional[str], Optional[dict]]:
    """toolCall 块文本 → (tool_name, arguments dict)；非 JSON → (None, None)。"""
    try:
        obj = json.loads(text or "")
    except (ValueError, TypeError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments")
    return name, (args if isinstance(args, dict) else {})


def normalized_command_core(text: str) -> str:
    """归一化 = 去数字/路径/空白后的命令核心（小写）。

    toolCall 文本优先取 arguments.command；非 bash 类调用退化为全部
    字符串参数拼接（write 的 content 参与核心 → 内容不同的写入不误判
    为"同命令"）。877 次循环样本：仅 number=NNNNN 数字不同 → 归一化后
    逐条相同。
    """
    t = text or ""
    _, args = _parse_tool_call(t)
    if isinstance(args, dict):
        picked = None
        for k in ("command", "cmd", "script"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                picked = v
                break
        if picked is None:
            parts = [v for v in args.values() if isinstance(v, str) and v.strip()]
            picked = " ".join(parts)
        t = picked if picked is not None else t
    t = _PATH_TOKEN_RE.sub(" ", t)      # 去路径
    t = _DIGIT_RE.sub(" ", t)           # 去数字
    t = re.sub(r"\s+", "", t)           # 去空白
    return t.lower()


def _write_events(ordered: Sequence[StageBlock]) -> List[Tuple[int, str]]:
    """(seq, 文件路径) 写入事件：写入类工具名（或 bash 重定向）且能
    解析出目标文件。解析不出文件的事件不参与（无同一文件可言）。"""
    out: List[Tuple[int, str]] = []
    for b in ordered:
        if b.kind != "toolCall":
            continue
        name = (b.tool_name or "").lower()
        tool, args = _parse_tool_call(b.text)
        if name is None and tool:
            name = tool.lower()
        path: Optional[str] = None
        cmd: Optional[str] = None
        if isinstance(args, dict):
            for k in ("path", "file_path", "notebook_path", "file"):
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    path = v.strip()
                    break
            v = args.get("command")
            if isinstance(v, str):
                cmd = v
        is_write = name in _WRITE_TOOL_NAMES
        if path is None and cmd:
            m = _REDIRECT_RE.search(cmd) or _TEE_RE.search(cmd)
            if m:
                is_write = True
                path = m.group(1)
        if is_write and path:
            out.append((b.seq, path))
    return out


def max_identical_run(ordered: Sequence[StageBlock]) -> Tuple[int, str, str]:
    """工具调用序列上"归一化后相同"的最大连续段。

    返回 (最长连击数, 归一化核心, 达峰调用的原始文本摘录)。
    空核心（无法解析/空文本）不算重复也不打断计数基线（重置）。
    """
    run_core: Optional[str] = None
    run = 0
    best = (0, "", "")
    for b in ordered:
        if b.kind != "toolCall":
            continue
        core = normalized_command_core(b.text)
        if not core:
            run_core = None
            run = 0
            continue
        if core == run_core:
            run += 1
        else:
            run_core, run = core, 1
        if run > best[0]:
            best = (run, core, (b.text or "")[:120])
    return best


def edit_oscillation_hit(
    ordered: Sequence[StageBlock],
    seq_sig: Dict[int, Tuple[str, int]],
    min_writes: int = OSCILLATION_MIN_WRITES,
) -> Optional[Tuple[int, str]]:
    """同文件写入 >= min_writes 次，且每次写入后到下次写入（或流尾）的
    pytest 结果集合（coverage 事件 + 已落地目标数二元组序列）完全相同
    且非空 → (写入次数, 文件)。修了没效果的振荡循环。

    seq_sig：pytest 事件 seq → (coverage 事件, goals_passed)，由
    compute() 的 events/traj 对齐得到（口径与覆盖度同源）。
    """
    by_file: Dict[str, List[int]] = {}
    for seq, path in _write_events(ordered):
        by_file.setdefault(path, []).append(seq)
    for path, seqs in by_file.items():
        if len(seqs) < min_writes:
            continue
        intervals: List[Tuple[Tuple[str, int], ...]] = []
        for i, s in enumerate(seqs):
            nxt = seqs[i + 1] if i + 1 < len(seqs) else None
            sigs = tuple(v for seq, v in sorted(seq_sig.items())
                         if s < seq and (nxt is None or seq < nxt))
            intervals.append(sigs)
        if all(intervals) and len(set(intervals)) == 1:
            return len(seqs), path
    return None


def max_same_file_writes(ordered: Sequence[StageBlock]) -> int:
    """单文件最大写入次数（未命中时的观测计数）。"""
    counts: Dict[str, int] = {}
    for _, path in _write_events(ordered):
        counts[path] = counts.get(path, 0) + 1
    return max(counts.values()) if counts else 0


def last_perf_claim_seq(ordered: Sequence[StageBlock]) -> Optional[int]:
    """最后一个 "STATUS: A" 收尾申报文本块的 seq（无则 None）。"""
    seq: Optional[int] = None
    for b in ordered:
        if b.kind == "text" and _PERF_CLAIM_RE.search(b.text or ""):
            seq = b.seq
    return seq


@dataclass
class ShadowEntry:
    """一条收敛 shadow 信号。matched 三态：True/False=已判定；None=未判定
    （配合 abstain_reason）。只进 shadow_convergence 键，绝不投递。"""
    signal: str
    matched: Optional[bool]
    detail: Dict[str, Any] = field(default_factory=dict)
    abstain_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "matched": self.matched,
            "detail": dict(self.detail),
            "abstain_reason": self.abstain_reason,
        }


def unavailable_entry(reason: str) -> Dict[str, Any]:
    """shadow 自身不可用时的占位条目（失败隔离；schema 与 ShadowEntry 一致）。"""
    return {
        "signal": "converge.unavailable",
        "matched": None,
        "detail": {},
        "abstain_reason": reason,
    }


class ConvergenceShadow:
    """收敛信号 shadow 观察者（与 audit_semantic.ProcessSemanticAdapter 同层）。

    compute() 是 (req, blocks) 的纯函数：无实例级可变状态，同一块流重复
    计算同结果（幂等）。实例因此可跨请求安全复用。

    ruleset（2026-10-04）："v1"（缺省，Phase 1/2 现状，既有测试锁定）或
    "v2"（repeat + perf_retest + stall 轮次线缩放）。budget_s 仅 v2 使用
    （stall 轮次线缩放的预算秒数；None → 读 MOONBOW_GUARD_CONVERGENCE_BUDGET，
    缺省 900）。
    """

    kind = "convergence"

    def __init__(self, goals_total: int = 1, target_tests: Optional[Sequence[str]] = None,
                 ruleset: str = RULESET_V1, budget_s: Optional[int] = None):
        total = int(goals_total)
        if total < 1:
            raise ValueError("goals_total must be >= 1")
        if ruleset not in RULESETS:
            raise ValueError("ruleset must be one of %s" % (RULESETS,))
        self.ruleset = ruleset
        if ruleset == RULESET_V2 and budget_s is None:
            budget_s = convergence_budget_from_env()
        self.budget_s = budget_s
        self.stall_min_round = (stall_line_from_budget(budget_s)
                                if ruleset == RULESET_V2 else STALL_MIN_ROUND)
        self.target_tests = tuple(target_tests) if target_tests else None
        if self.target_tests:
            if len(set(self.target_tests)) != len(self.target_tests):
                raise ValueError("target_tests must be unique")
            # 命名目标时目标数 = 命名数（goals_total 参数不参与）
            self.goals_total = len(self.target_tests)
        else:
            self.goals_total = total

    def with_targets(self, goals_total: Optional[int] = None,
                     target_tests: Optional[Sequence[str]] = None) -> "ConvergenceShadow":
        """派生同配置实例：payload 注入目标时保留 ruleset/budget
        （v2 装配不因 payload 带目标而降级 v1）。"""
        kwargs: Dict[str, Any] = {"ruleset": self.ruleset, "budget_s": self.budget_s}
        if target_tests is not None:
            kwargs["target_tests"] = target_tests
        elif goals_total is not None:
            kwargs["goals_total"] = goals_total
        return ConvergenceShadow(**kwargs)

    # ---- v2 触发器（块流纯函数）----

    def _repeat_entry(self, ordered: List[StageBlock],
                      seq_sig: Dict[int, Tuple[str, int]]) -> ShadowEntry:
        run, core, raw = max_identical_run(ordered)
        if run >= REPEAT_IDENTICAL_MIN:
            return ShadowEntry(REPEAT_SIGNAL, True, {
                "kind": "identical_calls", "count": run,
                "command": raw or core[:120]})
        osc = edit_oscillation_hit(ordered, seq_sig)
        if osc is not None:
            count, path = osc
            return ShadowEntry(REPEAT_SIGNAL, True, {
                "kind": "edit_oscillation", "count": count, "file": path})
        return ShadowEntry(REPEAT_SIGNAL, False, {
            "max_identical_calls": run,
            "max_same_file_writes": max_same_file_writes(ordered)})

    def _perf_retest_entry(self, ordered: List[StageBlock]) -> ShadowEntry:
        seq = last_perf_claim_seq(ordered)
        return ShadowEntry(PERF_RETEST_SIGNAL, seq is not None,
                           {"declared_seq": seq})

    def compute(self, req: str, blocks: List[StageBlock]) -> List[Dict[str, Any]]:
        """从块流确定性计算收敛信号。req 第一版不使用（不从 req 提取
        目标），保留在签名里以对齐 audit() 调用点与后续 matcher 对齐扩展。
        v2 追加 converge.repeat / converge.perf_retest 两条（v1 两条不变）。"""
        events = _pytest_events(blocks)
        round_no = sum(1 for b in blocks if b.kind == "toolResult")
        traj = coverage_trajectory(events, self.goals_total, self.target_tests)
        goals_passed = traj[-1]["goals_passed"] if traj else 0
        coverage = round(goals_passed / self.goals_total, 4)
        streak = pytest_fail_streak(events)
        ordered = sorted(blocks, key=lambda x: x.seq)

        stall_detail = {
            "round": round_no,
            "goals_passed": goals_passed,
            "goals_total": self.goals_total,
            "coverage": coverage,
            "pytest_rounds": len(events),
        }
        if self.ruleset == RULESET_V2:
            stall_detail["stall_line"] = self.stall_min_round
        if round_no < self.stall_min_round:
            stall = ShadowEntry(STALL_SIGNAL, None, stall_detail,
                                abstain_reason="rounds<%d" % self.stall_min_round)
        elif not events:
            # 无 pytest 痕迹：coverage==0 只是"无证据"，不判停滞
            stall = ShadowEntry(STALL_SIGNAL, None, stall_detail,
                                abstain_reason="no_pytest_evidence")
        else:
            stall = ShadowEntry(STALL_SIGNAL, goals_passed == 0, stall_detail)

        streak_detail = {
            "streak": streak,
            "round": round_no,
            "pytest_rounds": len(events),
        }
        fail_streak = ShadowEntry(FAIL_STREAK_SIGNAL, streak >= FAIL_STREAK_MIN,
                                  streak_detail)
        entries = [stall, fail_streak]
        if self.ruleset == RULESET_V2:
            seq_sig = {e.seq: (t["event"], t["goals_passed"])
                       for e, t in zip(events, traj)}
            entries.append(self._repeat_entry(ordered, seq_sig))
            entries.append(self._perf_retest_entry(ordered))
        return [e.to_dict() for e in entries]


# ---- Phase 2：advisory 投递包装（只包装，不改上面两条触发规则本体）----
#
# MOONBOW_GUARD_CONVERGENCE=off|shadow|advisory（server 层门控，默认 off）：
# - off / shadow：与 Phase 1 现状逐行为一致（shadow 只记录、零投递）；
# - advisory：触发规则命中且该任务收敛提醒预算未用时，stage-check 响应的
#   reminder 字段可投递收敛提示。措辞可忽略、不拦截、不计入既有 strict
#   收尾预算（与 P7 过程预算边界一致——不碰 semanticUsed / formatUsed /
#   收尾 interventions，交付通道复用过程提醒的 steer 路径）。
# 投递状态写进 shadow_convergence 条目（delivered: true/false）。

CONVERGENCE_ENV = "MOONBOW_GUARD_CONVERGENCE"
CONVERGENCE_MODES = ("off", "shadow", "advisory")

# 措辞纪律（与 GUARD_EFFECT_REPORT 一致）：advisory 提示必须可忽略、
# 给具体的下一步、不设收尾出口措辞。文本为实验固定话术，勿随手改。
ADVISORY_TEXT = {
    STALL_SIGNAL: ("你已连续多轮修改但没有一次验证成功。建议收窄范围："
                   "先把其中一个问题修到测试通过，其余如实列入 REMAINING。"),
    FAIL_STREAK_SIGNAL: ("同一验证反复失败，建议先聚焦让单个测试通过，再扩展。"),
    # v2（2026-10-04）。perf_retest 文案为实验固定话术（归因：both r4/r6/r9
    # 会内 27 passed 即申报 A → judge 复测翻转 ≤1µs 平局项）。
    REPEAT_SIGNAL: ("检测到重复/振荡循环：连续多次执行相同操作且验证结果没有变化。"
                    "建议先做一次最小对照实验确认最近改动是否真的有效，"
                    "再决定继续当前路线还是换方法。"),
    PERF_RETEST_SIGNAL: ("性能目标类收尾：请在申报完成前重复运行性能测试至少 2 次"
                         "确认计时稳定，并在 EVIDENCE 中附各档耗时数据。"),
}
ADVISORY_SUMMARY = {
    STALL_SIGNAL: "多轮修改无一次验证成功（coverage==0）",
    FAIL_STREAK_SIGNAL: "同一验证连续多轮失败",
    REPEAT_SIGNAL: "重复/振荡循环（同命令或同文件反复无进展）",
    PERF_RETEST_SIGNAL: "性能目标类收尾缺少复测证据",
}
# 复用过程提醒的守卫标记前缀：客户端采集层据此跳过守卫自己的反馈正文，
# 防自我审计循环（process-task.GUARD_MARK = "【进度守卫"）。
ADVISORY_MARK = "【进度守卫（收敛提示）】"


def convergence_mode_from_env(environ=None) -> str:
    """解析 MOONBOW_GUARD_CONVERGENCE（缺省/非法一律 off + 可解释）。
    独立成函数便于测试；server 启动时调用一次。"""
    import os
    env = environ if environ is not None else os.environ
    raw = (env.get(CONVERGENCE_ENV) or "").strip().lower()
    return raw if raw in CONVERGENCE_MODES else "off"


def req_hash_of(req: str) -> str:
    """任务键回退：req 文本哈希（sha256 前 16 hex，跨进程稳定）。"""
    return hashlib.sha256((req or "").encode("utf-8")).hexdigest()[:16]


def advisory_fingerprint(signal: str, task_key: str) -> str:
    return f"convergence:{signal}:{task_key}"


def build_advisory_reminder(signal: str, task_key: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """构造与主审计 reminder 同 schema 的收敛提示（客户端零改动即可投递）。"""
    detail = entry.get("detail") or {}
    ev: List[str] = []
    if "round" in detail:
        ev.append("第%s轮观测" % detail["round"])
    if signal == STALL_SIGNAL:
        if "goals_passed" in detail:
            ev.append("已落地目标 %s/%s" % (detail.get("goals_passed"),
                                            detail.get("goals_total")))
        if detail.get("pytest_rounds"):
            ev.append("pytest 证据 %s 次" % detail["pytest_rounds"])
    elif signal == REPEAT_SIGNAL:
        if detail.get("kind") == "identical_calls":
            ev.append("连续相同调用 %s 次" % detail.get("count"))
            if detail.get("command"):
                ev.append("命令核心：%s" % detail["command"])
        elif detail.get("kind") == "edit_oscillation":
            ev.append("%s 已写入 %s 次且 pytest 结果未变化"
                      % (detail.get("file"), detail.get("count")))
    elif signal == PERF_RETEST_SIGNAL:
        if detail.get("declared_seq") is not None:
            ev.append("第%s块申报 STATUS: A" % detail["declared_seq"])
    else:
        if "streak" in detail:
            ev.append("连续失败 %s 轮" % detail["streak"])
    text = ADVISORY_TEXT.get(signal) or ADVISORY_TEXT[FAIL_STREAK_SIGNAL]
    return {
        "summary": ADVISORY_SUMMARY.get(signal, signal),
        "fingerprint": advisory_fingerprint(signal, task_key),
        "evidence": ev,
        "suggestion": (
            f"{ADVISORY_MARK}{text}"
            "（本提示可忽略；是否采纳由你判断，不影响收尾验收。）"
        ),
    }


class AdvisoryBudget:
    """收敛提示的独立预算（每任务最多 MAX_PER_TASK 次，按 (task_key, signal) 记账）。

    - task_key 优先取 stage-check payload 的 task_id（同一 req 文本的多个
      任务实例各自独立预算——批量实验同一 prompt 跑 N 个 run，若按 req
      哈希共享预算，第 3 个 run 起全部静音）；payload 无 task_id 时回退
      req 哈希（与规格口径一致）。
    - 同一 (task_key, signal) 至多投 1 次：compute() 是纯函数、命中会持续
      命中，不去重会每批审计都投（客户端 fingerprint 去重也挡，双保险）。
    - 与既有预算完全独立：不占 semanticUsed / formatUsed / 收尾 interventions。
    - 服务端进程内存态；重启清零（记录条目带 delivered 可审计）。上限
      max_tasks 防长驻进程无界增长（先进先出淘汰）。
    """

    MAX_PER_TASK = 2

    def __init__(self, max_tasks: int = 4096):
        self._counts: Dict[str, Dict[str, int]] = {}
        # 独立预算键（v2 converge.perf_retest 专用）：与 _counts 完全分离，
        # 不计入 used() / MAX_PER_TASK —— "每任务 1 次，不影响既有提醒预算"。
        self._solo_counts: Dict[str, Dict[str, int]] = {}
        self._max_tasks = int(max_tasks)

    def _task(self, task_key: str) -> Dict[str, int]:
        if task_key not in self._counts:
            if len(self._counts) >= self._max_tasks:
                self._counts.pop(next(iter(self._counts)))
            self._counts[task_key] = {}
        return self._counts[task_key]

    def used(self, task_key: str) -> int:
        return sum(self._counts.get(task_key, {}).values())

    def allow(self, task_key: str, signal: str) -> bool:
        task = self._counts.get(task_key, {})
        return (task.get(signal, 0) == 0
                and self.used(task_key) < self.MAX_PER_TASK)

    def record(self, task_key: str, signal: str) -> None:
        task = self._task(task_key)
        task[signal] = task.get(signal, 0) + 1

    def allow_solo(self, task_key: str, signal: str) -> bool:
        """独立预算键：同 (task_key, signal) 至多 1 次，不受 MAX_PER_TASK
        与既有信号用量影响（perf_retest 语义："每任务 1 次，不影响既有
        提醒预算"）。"""
        return self._solo_counts.get(task_key, {}).get(signal, 0) == 0

    def record_solo(self, task_key: str, signal: str) -> None:
        if task_key not in self._solo_counts:
            if len(self._solo_counts) >= self._max_tasks:
                self._solo_counts.pop(next(iter(self._solo_counts)))
            self._solo_counts[task_key] = {}
        task = self._solo_counts[task_key]
        task[signal] = task.get(signal, 0) + 1

    def task_count(self) -> int:
        """出现过投递记账的任务数（含独立预算键任务；状态端点用）。"""
        return len(set(self._counts) | set(self._solo_counts))

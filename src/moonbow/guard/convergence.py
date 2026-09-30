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
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .protocol import StageBlock

# 触发阈值：G2 时间视角 = 第 4 轮观测时刻（Phase 0.5 口径）
STALL_MIN_ROUND = 4
# fail_streak 触发线（Phase 0 基线规则 pytest_fail_streak>=2）
FAIL_STREAK_MIN = 2

STALL_SIGNAL = "converge.stall"
FAIL_STREAK_SIGNAL = "converge.fail_streak"

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
    """

    kind = "convergence"

    def __init__(self, goals_total: int = 1, target_tests: Optional[Sequence[str]] = None):
        total = int(goals_total)
        if total < 1:
            raise ValueError("goals_total must be >= 1")
        self.target_tests = tuple(target_tests) if target_tests else None
        if self.target_tests:
            if len(set(self.target_tests)) != len(self.target_tests):
                raise ValueError("target_tests must be unique")
            # 命名目标时目标数 = 命名数（goals_total 参数不参与）
            self.goals_total = len(self.target_tests)
        else:
            self.goals_total = total

    def compute(self, req: str, blocks: List[StageBlock]) -> List[Dict[str, Any]]:
        """从块流确定性计算两条收敛信号。req 第一版不使用（不从 req 提取
        目标），保留在签名里以对齐 audit() 调用点与后续 matcher 对齐扩展。"""
        events = _pytest_events(blocks)
        round_no = sum(1 for b in blocks if b.kind == "toolResult")
        traj = coverage_trajectory(events, self.goals_total, self.target_tests)
        goals_passed = traj[-1]["goals_passed"] if traj else 0
        coverage = round(goals_passed / self.goals_total, 4)
        streak = pytest_fail_streak(events)

        stall_detail = {
            "round": round_no,
            "goals_passed": goals_passed,
            "goals_total": self.goals_total,
            "coverage": coverage,
            "pytest_rounds": len(events),
        }
        if round_no < STALL_MIN_ROUND:
            stall = ShadowEntry(STALL_SIGNAL, None, stall_detail,
                                abstain_reason="rounds<%d" % STALL_MIN_ROUND)
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
        return [stall.to_dict(), fail_streak.to_dict()]

/**
 * process-audit.ts — 阶段审计调度：单飞、检查点合并、游标推进（纯逻辑）
 *
 * - 同一任务最多一个过程审计请求在执行；等待期间新检查点只推进状态，
 *   请求完成后若仍有未审块，自动补一轮（保证最新状态最终被处理）。
 * - 请求携带 taskId/branchId/snapshotVersion。迟到响应的最终防线在调用方：
 *   状态对象按任务持有，任务/分支切换后旧 state 不再被引用，合并进旧
 *   对象的发现不会触发新任务的提醒。
 * - 不阻塞主流程：fetch 由调用方注入（生产为全局 fetch，测试可 mock）。
 * - fetch 失败/超时/非 2xx/坏 JSON：返回 null，游标不推进（下轮重试），
 *   连续失败有上限，不会死循环。
 */

import type { GuardBlock } from "./process-events.ts";
import { hasUnauditedToolEvents, mergeFinding, newProcessState,
         type Finding, type ProcessState } from "./process-task.ts";

export type ProcessMode = "off" | "shadow" | "advisory";

export function processMode(): ProcessMode {
  const v = (process.env.MOONBOW_GUARD_PROCESS || "off").toLowerCase();
  return v === "shadow" || v === "advisory" ? (v as ProcessMode) : "off";
}

// Phase 2：收敛通道透传。MOONBOW_GUARD_CONVERGENCE=off|shadow|advisory
// （默认 off）。非 off 时 stage-check 请求带 enable_convergence_shadow=true，
// 服务端据此计算 shadow_convergence；是否投递收敛提示由**服务端**同名
// env 门控（advisory），客户端只负责开通道与透传。
export type ConvergenceMode = "off" | "shadow" | "advisory";

export function convergenceMode(): ConvergenceMode {
  const v = (process.env.MOONBOW_GUARD_CONVERGENCE || "off").toLowerCase();
  return v === "shadow" || v === "advisory" ? (v as ConvergenceMode) : "off";
}

// 可选命名目标注入（逗号分隔）：启用服务端逐测试撤回与部分得分口径
// （ConvergenceShadow target_tests）。缺省不发键 → 服务端默认口径零变化。
export function convergenceTargets(): string[] | null {
  const raw = (process.env.MOONBOW_GUARD_CONVERGENCE_TARGETS || "").trim();
  if (!raw) return null;
  const parts = Array.from(new Set(raw.split(",").map((s) => s.trim()).filter(Boolean)));
  return parts.length ? parts : null;
}

export interface ConvergenceShadowEntry {
  signal: string;
  matched?: boolean | null;
  abstain_reason?: string | null;
  delivered?: boolean;
  detail?: Record<string, unknown>;
}

export interface StageCheckResponse {
  findings?: Finding[];
  reminder?: { summary: string; evidence: string[]; suggestion: string } | null;
  semantic?: boolean;           // 属昂贵语义复核范畴（与收尾共享每任务一次预算）
  based_on?: number;            // 服务端回显，遥测用
  shadow_convergence?: ConvergenceShadowEntry[];  // Phase 2：只透传不消费（观测在服务端）
}

const MAX_CONSECUTIVE_FAILURES = 2;

export class StageAuditor {
  private url: string;
  private doFetch: typeof fetch;
  private timeoutMs: number;
  private withConvergence: boolean;

  constructor(url: string, doFetch: typeof fetch, timeoutMs = 10000,
              withConvergence = false) {
    this.url = url;
    this.doFetch = doFetch;
    this.timeoutMs = timeoutMs;
    this.withConvergence = withConvergence;
  }

  private hasUnaudited(state: ProcessState): boolean {
    const last = state.blocks[state.blocks.length - 1];
    // v2 工具事件通道：未审工具事件同样构成送审理由（2026-10-04）。缺这个
    // 判据，纯工具循环（如 877 次同命令退化循环，期间无新 text/thinking 块）
    // 在首批块审完后就再也不触发审计——工具事件永远不重发，规则照样饿着。
    return (last !== undefined && last.seq >= state.auditedThroughSeq)
      || hasUnauditedToolEvents(state);
  }

  /**
   * 提交阶段审计。每实例只服务一次 submit（调用方每次检查点新建实例；
   * 并发互斥与"飞行期间入库块的尾随合并"由调用方的 pBusy + trailing
   * 机制负责）。返回最后一条成功响应；全部失败返回 null（调用方记录
   * "未验证"）。游标只在成功后推进，失败自动重试一轮后放弃。
   */
  async submit(
    state: ProcessState,
    mode: Exclude<ProcessMode, "off">,
    streamEnded = false,
  ): Promise<StageCheckResponse | null> {
    let failures = 0;
    let last: StageCheckResponse | null = null;
    // stream_ended 时即使无新增块也要执行一轮：终止审计是"运行已结束，
    // 请现在裁决"的信号，不是增量搬运。缺这个强制，agent_end 的终止
    // 审计会在 hasUnaudited 上被吞掉，缺口永不上报（2026-09-22）。
    let force = streamEnded;
    while (this.hasUnaudited(state) || force) {
      const r = await this.once(state, mode, streamEnded);
      if (r === null) {
        if (++failures >= MAX_CONSECUTIVE_FAILURES) break;
        force = false;               // 失败后不再强制空轮，防死循环
        continue;                    // 游标未推进；再试一次后放弃本轮
      }
      failures = 0;
      last = r;
      break;
    }
    return last;
  }

  private async once(
    state: ProcessState,
    mode: Exclude<ProcessMode, "off">,
    streamEnded = false,
  ): Promise<StageCheckResponse | null> {
    // 发送**窗口内全部块**而非仅新增（2026-09-22）：
    // 服务端规则 2.3（unverified-claim）要看"本次声明之前是否已有验证证据"，
    // 只看增量批次时，声明与验证落在不同批次就会被误判成"未见验证证据"——
    // 实测 S1 守卫臂真正跑通了测试，却产生 3 条 unverified 误报、介入预算
    // 一次都没用上。窗口上限 400 块（环形缓冲），payload 可控。
    const fresh = state.blocks.filter((b) => b.seq >= state.auditedThroughSeq);
    // stream_ended 时即使无新增块也要发：这是一次"运行已结束，请现在裁决"
    // 的信号。若因 fresh 为空而跳过，终止缺口（改了没验证）将永远不会上报
    // （2026-09-22：竞态实测——agent_end 的终止审计被空增量检查吞掉）。
    // v2（2026-10-04）：纯工具事件增量（无新块）同样要发，否则退化循环
    // 中途永远拿不到新的工具流（见 hasUnaudited 注释）。
    if (!fresh.length && !streamEnded && !hasUnauditedToolEvents(state)) return null;
    const window = state.blocks.slice();
    // 发送时刻的工具事件版本快照：成功只推进到该快照——飞行期间入库的
    // 事件保持"未审"，由尾随合并补一轮（与块游标同语义，2026-10-04）。
    const toolEventsAtSend = state.toolEventsVersion;
    const conv = this.withConvergence ? convergenceMode() : "off";
    const targets = conv !== "off" ? convergenceTargets() : null;
    const body = JSON.stringify({
      task_id: state.taskId, branch_id: state.branchId,
      snapshot_version: state.snapshotVersion,
      audited_through: state.auditedThroughSeq, mode,
      // 流是否已静止（agent_end 时置位）。服务端据此区分"验证还没到"与
      // "运行真的结束了"——缺这个信号会把中途状态误报成终止缺口。
      stream_ended: streamEnded,
      req: state.req,
      // Phase 2：收敛通道开灯（off 时两键均不发 → 服务端路径零变化）
      ...(conv !== "off" ? { enable_convergence_shadow: true } : {}),
      ...(targets ? { convergence_target_tests: targets } : {}),
      blocks: window.map((b) => ({
        seq: b.seq, kind: b.kind, text: b.text,
        tool_call_id: b.toolCallId, tool_name: b.toolName, is_error: b.isError ?? false,
      })),
      // v2 工具事件通道（2026-10-04）：默认携带。服务端只在 ruleset=v2 时
      // 消费（off 不启用 shadow、v1 忽略该键）→ 两条既有路径零行为变化；
      // 体积经 ToolEventCollector 截断 + appendToolEvent 封顶约束。
      // 携带纪律论证见 process-events.ts 的通道注释。
      tool_events: state.toolEvents.map((ev) => ({
        seq: ev.seq, phase: ev.phase, name: ev.name,
        ...(ev.phase === "call"
          ? { args: ev.args ?? "" }
          : { is_error: ev.is_error ?? false, result: ev.result ?? "" }),
        tool_call_id: ev.toolCallId,
      })),
      findings: state.findings,
    });
    let resp: Response;
    try {
      resp = await this.doFetch(this.url + "/v1/stage-check", {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(this.timeoutMs), body,
      });
    } catch {
      return null;
    }
    if (!resp.ok) return null;
    let data: StageCheckResponse;
    try { data = await resp.json(); } catch { return null; }

    for (const f of data.findings ?? []) mergeFinding(state, { ...f, updatedAt: Date.now() });
    const lastSeq = fresh[fresh.length - 1]?.seq;
    if (lastSeq !== undefined && lastSeq >= state.auditedThroughSeq) {
      state.auditedThroughSeq = lastSeq + 1;
    }
    // 工具事件游标推进到发送快照（失败不推进 → 下轮重试；飞行期间新增
    // 事件保持未审，尾随合并会补审计）
    state.auditedToolEventsVersion = toolEventsAtSend;
    return data;
  }
}

export { appendBlocks, mergeFinding, newProcessState } from "./process-task.ts";
export type { Finding, ProcessState, GuardBlock };

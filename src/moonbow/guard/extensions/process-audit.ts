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
import { appendBlocks, mergeFinding, newProcessState,
         type Finding, type ProcessState } from "./process-task.ts";

export type ProcessMode = "off" | "shadow" | "advisory";

export function processMode(): ProcessMode {
  const v = (process.env.MOONBOW_GUARD_PROCESS || "off").toLowerCase();
  return v === "shadow" || v === "advisory" ? (v as ProcessMode) : "off";
}

export interface StageCheckResponse {
  findings?: Finding[];
  reminder?: { summary: string; evidence: string[]; suggestion: string } | null;
  semantic?: boolean;           // 属昂贵语义复核范畴（与收尾共享每任务一次预算）
  based_on?: number;            // 服务端回显，遥测用
}

const MAX_CONSECUTIVE_FAILURES = 2;

export class StageAuditor {
  private url: string;
  private doFetch: typeof fetch;
  private timeoutMs: number;

  constructor(url: string, doFetch: typeof fetch, timeoutMs = 10000) {
    this.url = url;
    this.doFetch = doFetch;
    this.timeoutMs = timeoutMs;
  }

  private hasUnaudited(state: ProcessState): boolean {
    const last = state.blocks[state.blocks.length - 1];
    return last !== undefined && last.seq >= state.auditedThroughSeq;
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
  ): Promise<StageCheckResponse | null> {
    let failures = 0;
    let last: StageCheckResponse | null = null;
    while (this.hasUnaudited(state)) {
      const r = await this.once(state, mode);
      if (r === null) {
        if (++failures >= MAX_CONSECUTIVE_FAILURES) break;
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
  ): Promise<StageCheckResponse | null> {
    const fresh = state.blocks.filter((b) => b.seq >= state.auditedThroughSeq);
    if (!fresh.length) return null;
    const body = JSON.stringify({
      task_id: state.taskId, branch_id: state.branchId,
      snapshot_version: state.snapshotVersion,
      audited_through: state.auditedThroughSeq, mode,
      req: state.req,
      blocks: fresh.map((b) => ({
        seq: b.seq, kind: b.kind, text: b.text,
        tool_call_id: b.toolCallId, tool_name: b.toolName, is_error: b.isError ?? false,
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
    const lastSeq = fresh[fresh.length - 1].seq;
    if (lastSeq >= state.auditedThroughSeq) state.auditedThroughSeq = lastSeq + 1;
    return data;
  }
}

export { appendBlocks, mergeFinding, newProcessState };
export type { Finding, ProcessState, GuardBlock };

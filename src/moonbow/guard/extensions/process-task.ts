/**
 * process-task.ts — 阶段审计任务记录：观察、发现、送达状态与恢复（纯逻辑）
 *
 * 状态模型（对应 goal 的记录/发现/送达三分离）：
 * - 观察：GuardBlock 环形缓冲，超限丢最旧并留 truncatedFrom 标记（不静默丢尾部：
 *   丢的是最旧头部，新事件永远继续入库）。
 * - 发现：candidate → actionable → addressed/resolved/withdrawn；信息不足 = unverified。
 *   fingerprint 用于跨检查点去重。
 * - 送达：reserved（预算已预留）→ queued（已入队）→ observed（已见进入上下文）
 *   → failed/unknown。queued ≠ 模型已读，更 ≠ 问题已解决。
 */

import type { GuardBlock } from "./process-events.ts";

export type FindingStatus =
  | "candidate" | "actionable" | "addressed" | "resolved" | "withdrawn" | "unverified";

export interface EvidenceRef { seq: number; kind: string; excerpt: string; }

export interface Finding {
  fingerprint: string;
  kind: string;                  // evidence-order | contradiction | unverified-claim | thinking-concern
  status: FindingStatus;
  summary: string;
  evidence: EvidenceRef[];
  snapshotVersion: number;
  updatedAt: number;
}

export interface DeliveryRecord {
  fingerprint: string;
  channel: "steer" | "followUp";
  status: "reserved" | "queued" | "observed" | "failed" | "unknown";
  at: number;
}

export interface ProcessBudgets {
  semanticUsed: boolean;         // 昂贵语义复核（过程+收尾共享，每任务一次）
  formatUsed: boolean;           // 收尾格式补报（独立，仅收尾）
  interventions: number;         // 已投递介入次数（严格模式上限用）
  processReminders: number;      // 过程提醒投递次数（遥测，非预算）
}

export interface ProcessState {
  taskId: string;
  branchId: string;
  req: string;                   // 用户原始需求（送审携带）
  snapshotVersion: number;       // 每次新块入库 +1，用于过期响应丢弃
  auditedThroughSeq: number;     // 已送审游标
  blocks: GuardBlock[];
  truncatedFrom: number | null;  // 头部被环形缓冲丢弃的块数
  findings: Finding[];
  deliveries: DeliveryRecord[];
  budgets: ProcessBudgets;
}

export function maxBlocks(): number {
  const n = Number(process.env.MOONBOW_GUARD_MAX_BLOCKS || 400);
  return Number.isFinite(n) && n >= 120 ? Math.floor(n) : 400;
}

export function newProcessState(taskId: string, branchId: string, req = ""): ProcessState {
  return {
    taskId, branchId, req, snapshotVersion: 0, auditedThroughSeq: 0,
    blocks: [], truncatedFrom: null,
    findings: [], deliveries: [],
    budgets: { semanticUsed: false, formatUsed: false, interventions: 0, processReminders: 0 },
  };
}

/** 新观察入库：版本号推进、环形截断（丢最旧，留标记）。返回是否发生变化。 */
export function appendBlocks(state: ProcessState, blocks: GuardBlock[]): boolean {
  if (!blocks.length) return false;
  state.blocks.push(...blocks);
  state.snapshotVersion += 1;
  const cap = maxBlocks();
  if (state.blocks.length > cap) {
    const drop = state.blocks.length - cap;
    state.blocks.splice(0, drop);
    state.truncatedFrom = (state.truncatedFrom ?? 0) + drop;
  }
  return true;
}

/** 按 fingerprint 合并服务端/本地发现：新则插入，旧则更新状态与时间。 */
export function mergeFinding(state: ProcessState, f: Finding): void {
  const cur = state.findings.find((x) => x.fingerprint === f.fingerprint);
  if (!cur) { state.findings.push(f); return; }
  // 只前进不回退：resolved/withdrawn 是终态（rank 最高），任何迟到状态不得
  // 覆盖；其余按 candidate < unverified < addressed < actionable 前进。
  const rank: Record<FindingStatus, number> = {
    candidate: 1, unverified: 2, addressed: 3, actionable: 4, resolved: 5, withdrawn: 5,
  };
  if (rank[f.status] >= rank[cur.status]) {
    cur.status = f.status; cur.summary = f.summary;
    cur.evidence = f.evidence; cur.snapshotVersion = f.snapshotVersion;
    cur.updatedAt = f.updatedAt;
  }
}

export function latestDelivery(state: ProcessState, fingerprint: string): DeliveryRecord | undefined {
  for (let i = state.deliveries.length - 1; i >= 0; i--) {
    if (state.deliveries[i].fingerprint === fingerprint) return state.deliveries[i];
  }
  return undefined;
}

/** 守卫自己的反馈正文标记：采集层据此跳过，防自我审计循环（双保险之一）。 */
export const GUARD_MARK = "【进度守卫";

export function isGuardOriginText(text: string): boolean {
  return text.includes(GUARD_MARK);
}

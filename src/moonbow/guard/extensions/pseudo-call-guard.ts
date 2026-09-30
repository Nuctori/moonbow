/**
 * pseudo-call-guard.ts — 伪工具调用（text-embedded tool call）检测与反馈（harness 扩展层）
 *
 * 背景（2026-09-30 Phase 2 归因，failure_attribution.md §0/§5.1）：
 * mimo-v2.5 经网关会把工具调用写成正文文本（形如
 * `[tool_call call_xxx] edit {...}`），pi 按正常文本收尾（stopReason=stop），
 * 动作永不执行且零反馈 —— 28/30 run 中招、13 run 的修复编辑被吞，
 * 是本批主导失败机制（"零反馈黑洞"）。最高杠杆修复就是客户端检测该
 * 签名并注入"调用没被执行，请走工具接口重发"的反馈。
 *
 * 设计约束：
 * - 只读观察 + 反馈投递，绝不参与 /check 等任何 guard 判定路径；
 * - 触发条件：assistant 消息文本命中伪调用签名 且 该消息没有伴随任何
 *   真实工具调用（无 toolCall 内容块）。伴随真实调用的混合消息不报
 *   （Phase 2 实测 1 例混合消息，报了就是误报）；
 * - 反馈经既有 followUp+triggerTurn 通道投递（伪调用消息收尾时 agent
 *   已停，steer 送不到；followUp 触发新回合让模型重发）；
 * - 预算：每任务最多 3 次反馈（防"重发仍被吞"循环打转，conv-r1 形态），
 *   超预算只记遥测、静默；MOONBOW_PSEUDO_CALL_FEEDBACK=off 为杀开关
 *   （默认 on，off 时零扫描零记录零投递）；
 * - 失败隔离：本模块全部逻辑自带 try/catch，任何异常不影响宿主扩展。
 *
 * 纯逻辑（签名扫描、预算）与安装入口分导出，测试可单测签名函数。
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

/** 固定反馈措辞（用户可见，不得随意改动——实验脚本按此文本对账） */
export const PSEUDO_CALL_FEEDBACK_TEXT =
  "你上一条消息中的工具调用是以文本形式写的，没有被执行。" +
  "工具调用必须通过工具接口发出。" +
  "请重新通过工具接口发出该调用（不要把调用写成文本）。";

/** 每任务反馈预算（防循环；超过只记遥测） */
export const PSEUDO_CALL_FEEDBACK_BUDGET = 3;

/** 遥测条目类型（appendEntry customType；兼作跨重启预算恢复的账本） */
export const PSEUDO_CALL_TELEMETRY = "progress-guard:pseudo-call";
/** 反馈消息 customType */
export const PSEUDO_CALL_FEEDBACK_MSG = "progress-guard:pseudo-call-feedback";

/** 杀开关：MOONBOW_PSEUDO_CALL_FEEDBACK=off 时完全失效（默认 on） */
export function pseudoCallFeedbackEnabled(): boolean {
  const v = (process.env.MOONBOW_PSEUDO_CALL_FEEDBACK ?? "on").trim().toLowerCase();
  return v !== "off";
}

export interface PseudoCallHit {
  /** 主签名形态：bracket（主形态）| xml | skeleton | residue */
  form: "bracket" | "xml" | "skeleton" | "residue";
  /** 该消息中伪调用的个数（同消息多个调用——Phase 2 实测最多 3 个） */
  count: number;
}

// 签名按优先级排列。全部来自 Phase 2 会话实测形态 + 历史残渣变体：
// 1) bracket（35/35）：`[tool_call call_<hex>] edit {"path":...}`，JSON 体单行紧凑
// 2) xml：`<tool_call>...</tool_call>`（本批未出现，其他 harness 常见，防御性）
// 3) skeleton：行首工具名 + 带引号键的 JSON 体（覆盖多行 JSON 体变体）
// 4) residue（历史 1 例）：`</arg_value>` 等 harness 残渣标签
const SIGNATURES: Array<{ form: PseudoCallHit["form"]; re: RegExp }> = [
  { form: "bracket", re: /\[tool_call\s*[^\]\n]{0,80}\]/gi },
  { form: "xml", re: /<\/?\s*tool_call\s*>/gi },
  { form: "skeleton",
    re: /(?:^|\n)[ \t]*(?:read|bash|edit|write|grep|glob|ls|todo_write)\s*\{\s*"[a-z_][a-z0-9_]*"\s*:/gi },
  { form: "residue", re: /<\/?\s*(?:arg_value|arg_key|arg_json|parameter)\s*>/gi },
];

/** 扫描 assistant 文本中的伪调用签名。未命中返回 null。 */
export function scanPseudoCallText(text: string): PseudoCallHit | null {
  if (!text || text.length < 8) return null;
  for (const { form, re } of SIGNATURES) {
    const m = text.match(re);
    if (m && m.length > 0) return { form, count: m.length };
  }
  return null;
}

function textOfAssistant(m: any): string {
  const c = m?.content;
  if (typeof c === "string") return c;
  return Array.isArray(c)
    ? c.filter((b: any) => b?.type === "text").map((b: any) => b.text || "").join("\n")
    : "";
}

/** 该 assistant 消息是否伴随真实工具调用（toolCall 内容块在场）。 */
function hasRealToolCall(m: any): boolean {
  return Array.isArray(m?.content) && m.content.some((b: any) => b?.type === "toolCall");
}

export interface PseudoCallGuardOptions {
  /** 扩展宿主 API（appendEntry/sendMessage 挂在 pi 上，与 progress-guard 同款） */
  pi: ExtensionAPI;
  /** 当前任务 id（无任务时返回 undefined——无锚点不投递） */
  getTaskId: () => string | undefined;
}

export class PseudoCallGuard {
  private opts: PseudoCallGuardOptions;
  /** 每任务已投递反馈数（内存权威；跨重启从遥测账本恢复） */
  private delivered = new Map<string, number>();
  private lastTaskId: string | undefined;
  /** 去重：同一逻辑消息只报一次（对象身份 + 稳定 id 双保险） */
  private seenObjects = new WeakSet<object>();
  private seenIds = new Set<string>();

  constructor(opts: PseudoCallGuardOptions) {
    this.opts = opts;
  }

  /** 消息结束时扫描 assistant 文本。任何异常自吞，绝不外抛。 */
  onAssistantMessage(m: any, ctx?: ExtensionContext): void {
    try {
      this.handle(m, ctx);
    } catch (error) {
      console.error(`[PseudoCallGuard] ignored: ${error}`);
    }
  }

  private handle(m: any, ctx?: ExtensionContext): void {
    if (!pseudoCallFeedbackEnabled()) return;      // 杀开关：零行为
    if (!m || m.role !== "assistant") return;
    if (m.stopReason === "error" || m.stopReason === "aborted") return;
    // 去重：pi 可能以不同对象形态重复投递同一消息（同 responseId）
    const oid = typeof m.id === "string" ? m.id
      : typeof m.responseId === "string" ? m.responseId : undefined;
    if (oid && this.seenIds.has(oid)) return;
    if (this.seenObjects.has(m)) return;
    if (oid) this.seenIds.add(oid);
    this.seenObjects.add(m);

    const text = textOfAssistant(m);
    const hit = scanPseudoCallText(text);
    if (!hit) return;
    const accompaniedReal = hasRealToolCall(m);
    const taskId = this.opts.getTaskId();

    // 预算账本恢复：任务锚点变化时从遥测重放（重启/新任务统一入口）。
    if (taskId !== this.lastTaskId) {
      this.delivered.set(taskId ?? "", this.restoreDelivered(ctx, taskId));
      this.lastTaskId = taskId;
    }

    // 伴随真实调用：只记遥测不投递（避免打断已在正常推进的回合）
    if (accompaniedReal) {
      this.record({ taskId, form: hit.form, count: hit.count,
        accompaniedReal, action: "observed-only" });
      return;
    }
    if (!taskId) {
      this.record({ taskId: null, form: hit.form, count: hit.count,
        accompaniedReal, action: "no-task" });
      return;
    }
    // 有待处理用户输入时不插话（不消耗预算）
    if (ctx?.hasPendingMessages?.()) {
      this.record({ taskId, form: hit.form, count: hit.count,
        accompaniedReal, action: "suppressed-pending" });
      return;
    }
    const used = this.delivered.get(taskId) ?? 0;
    if (used >= PSEUDO_CALL_FEEDBACK_BUDGET) {
      this.record({ taskId, form: hit.form, count: hit.count,
        accompaniedReal, action: "budget-exhausted", budgetUsed: used });
      return;
    }
    // 预算先预留后投递（崩溃不重置预算，与 progress-guard 同款纪律）
    const budgetUsed = used + 1;
    this.delivered.set(taskId, budgetUsed);
    this.record({ taskId, form: hit.form, count: hit.count,
      accompaniedReal: false, action: "feedback", budgetUsed });
    const pi = this.opts.pi as any;
    pi?.sendMessage?.({
      customType: PSEUDO_CALL_FEEDBACK_MSG,
      content: PSEUDO_CALL_FEEDBACK_TEXT,
      display: true,
      details: { taskId, form: hit.form, count: hit.count },
    }, { deliverAs: "followUp", triggerTurn: true });
  }

  private record(data: Record<string, unknown>): void {
    try {
      (this.opts.pi as any)?.appendEntry?.(PSEUDO_CALL_TELEMETRY, { at: Date.now(), ...data });
    } catch {
      // 遥测失败不影响反馈主路径
    }
  }

  /** 从会话分支遥测账本统计该任务已投递次数（跨重启预算不重置） */
  private restoreDelivered(ctx: ExtensionContext | undefined, taskId?: string): number {
    if (!taskId || !ctx?.sessionManager?.getBranch) return 0;
    let n = 0;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry?.type === "custom" && (entry as any).customType === PSEUDO_CALL_TELEMETRY) {
        const d = (entry as any).data as any;
        if (d?.taskId === taskId && d?.action === "feedback") n++;
      }
    }
    return n;
  }
}

/** 安装入口：由 progress-guard.activate 在既有事件处理器内调用
 *  （扩展宿主同名事件只允许挂一个处理器，本模块不自行 pi.on）。 */
export function installPseudoCallGuard(pi: ExtensionAPI,
                                       opts: Omit<PseudoCallGuardOptions, "pi">): PseudoCallGuard {
  return new PseudoCallGuard({ ...opts, pi });
}

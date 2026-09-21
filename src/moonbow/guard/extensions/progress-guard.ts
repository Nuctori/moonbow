import { randomUUID } from "node:crypto";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { BlockAssembler, type GuardBlock } from "./process-events.ts";
import { StageAuditor, processMode, type ProcessMode,
         type StageCheckResponse } from "./process-audit.ts";
import { appendBlocks, isGuardOriginText, latestDelivery, newProcessState,
         type DeliveryRecord, type ProcessState } from "./process-task.ts";

const URL = process.env.MOONBOW_GUARD_URL || "http://127.0.0.1:18492";
const MODE = process.env.MOONBOW_GUARD_MODE === "strict" ? "strict" : "advisory";
const STATE = "progress-guard:state";
const PSTATE = "progress-guard:process";

interface TaskState {
  id: string;
  req: string;
  semanticUsed: boolean;
  formatUsed: boolean;
  interventions: number;
}

interface GuardResponse {
  decision: string;
  allow_stop: boolean;
  acceptance: string;
  review_requested: boolean;
  feedback?: string;
  prompt?: string;
}

function textOf(content: any): string {
  if (typeof content === "string") return content;
  return Array.isArray(content)
    ? content.filter((b: any) => b?.type === "text").map((b: any) => b.text || "").join("\n")
    : "";
}

export default function activate(pi: ExtensionAPI) {
  let task: TaskState | undefined;
  let pstate: ProcessState | undefined;
  let assembler = new BlockAssembler();
  const checking = new Set<string>();       // 收尾检查单飞
  let pBusy = false;                        // 过程审计单飞（跨事件去抖）

  function save() {
    if (task) pi.appendEntry(STATE, { ...task });
  }

  function saveProcess() {
    if (!pstate) return;
    // 不持久化块缓冲（大）；发现/预算/游标/送达已足以恢复。
    pi.appendEntry(PSTATE, {
      taskId: pstate.taskId, branchId: pstate.branchId, req: pstate.req,
      snapshotVersion: pstate.snapshotVersion,
      auditedThroughSeq: pstate.auditedThroughSeq,
      truncatedFrom: pstate.truncatedFrom,
      findings: pstate.findings, deliveries: pstate.deliveries,
      budgets: pstate.budgets,
    });
  }

  function restoreProcess(_event: any, ctx: ExtensionContext, taskId?: string) {
    let restored: ProcessState | undefined;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === PSTATE) {
        const d = entry.data as any;
        restored = {
          taskId: d.taskId, branchId: d.branchId, req: d.req ?? "",
          snapshotVersion: d.snapshotVersion ?? 0,
          auditedThroughSeq: d.auditedThroughSeq ?? 0,
          truncatedFrom: d.truncatedFrom ?? null,
          findings: d.findings ?? [], deliveries: d.deliveries ?? [],
          budgets: d.budgets ?? { semanticUsed: false, formatUsed: false,
                                  interventions: 0, processReminders: 0 },
          blocks: [],
        };
      }
    }
    if (restored && (!taskId || restored.taskId === taskId)) {
      pstate = restored;
      assembler = new BlockAssembler(restored.auditedThroughSeq); // 续接序号，防游标失效
    } else if (taskId) {
      const t = task;
      pstate = newProcessState(taskId, branchIdOf(ctx), t?.req ?? "");
    }
  }

  function branchIdOf(ctx: ExtensionContext): string {
    const n = ctx?.sessionManager?.getBranch?.().length ?? 0;
    return `b${n}`;
  }

  function restore(_event: any, ctx: ExtensionContext) {
    task = undefined;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === STATE) {
        task = { ...(entry.data as TaskState) };
      }
    }
    pstate = undefined;
    assembler = new BlockAssembler();
    restoreProcess(_event, ctx, task?.id);
  }

  pi.on("session_start", restore);
  pi.on("session_tree", restore);

  pi.on("input", (event, ctx) => {
    if (event.source === "extension" || !event.text.trim() || event.text.startsWith("/")) return;
    task = { id: randomUUID(), req: event.text, semanticUsed: false, formatUsed: false, interventions: 0 };
    save();
    // off 模式零行为变化：不建过程状态、不写会话条目
    if (processMode() !== "off") {
      pstate = newProcessState(task.id, branchIdOf(ctx), task.req);
      assembler = new BlockAssembler();
      saveProcess();
    }
  });

  // ---------------- 阶段审计：采集（off 模式零开销） ----------------

  function collect(blocks: GuardBlock[], ctx?: ExtensionContext) {
    if (!pstate || !blocks.length) return;
    if (appendBlocks(pstate, blocks)) maybeAudit(ctx);
  }

  function maybeAudit(ctx?: ExtensionContext, depth = 0) {
    const mode = processMode();
    if (mode === "off" || !pstate || pBusy) return;
    const current = pstate;
    pBusy = true;
    const auditor = new StageAuditor(URL, (u, i) => fetch(u, i));
    void auditor.submit(current, mode).then((resp) => {
      pBusy = false;
      applyAudit(current, resp, mode, ctx);
      // 尾随合并：审计飞行期间入库的块，补一轮，确保最新状态最终被处理
      if (pstate === current && depth < 3) {
        const last = current.blocks[current.blocks.length - 1];
        if (last && last.seq >= current.auditedThroughSeq) maybeAudit(ctx, depth + 1);
      }
    }).catch(() => { pBusy = false; });
  }

  function applyAudit(state: ProcessState, resp: StageCheckResponse | null,
                      mode: ProcessMode, ctx?: ExtensionContext) {
    if (!resp) {
      if (pstate === state) {
        pi.appendEntry("progress-guard:observation", { process: true, note: "unverified" });
        if (ctx) ctx.ui.setStatus("progress-guard", "守卫: 过程未验证（服务不可用）");
      }
      return;
    }
    if (pstate !== state) return;                 // 任务/分支已切换，迟到响应丢弃
    if (processMode() !== mode) return;           // 模式被外部改动的保守丢弃
    saveProcess();                                // 合并后的发现落盘（shadow 也记录）
    const rem = resp.reminder;
    if (!rem) return;
    if (mode !== "advisory") {
      // 影子模式：只记录"本应提醒什么"，绝不注入
      pi.appendEntry("progress-guard:shadow", {
        taskId: state.taskId,
        fingerprint: (rem as any).fingerprint || "",
        summary: rem.summary, evidence: rem.evidence,
      });
      return;
    }
    const fp = (rem as any).fingerprint || "";
    // 发送前复核：该问题已在最新证据下解决 -> 不打扰
    const cur = state.findings.find((f) => f.fingerprint === fp);
    if (cur && (cur.status === "resolved" || cur.status === "withdrawn")) return;
    // 已提醒过的问题不重复催（发现可以更新，催促不重复）
    if (fp && state.deliveries.some((d) => d.fingerprint === fp && d.status !== "failed")) return;
    // 有待处理的用户输入时不插入守卫提醒
    if (ctx && ctx.hasPendingMessages()) return;
    // 语义类提醒与收尾共享每任务一次预算；耗尽只记录，不投递，也不算验收通过
    if (resp.semantic) {
      if (state.budgets.semanticUsed) return;
      state.budgets.semanticUsed = true;
      if (task && task.id === state.taskId) { task.semanticUsed = true; save(); }
    }
    state.budgets.interventions++;
    state.budgets.processReminders++;
    const delivery: DeliveryRecord = { fingerprint: fp, channel: "steer", status: "reserved", at: Date.now() };
    state.deliveries.push(delivery);
    saveProcess();
    pi.sendMessage({
      customType: "progress-guard:process-feedback",
      content: rem.suggestion,
      display: true,
      details: { taskId: state.taskId, fingerprint: fp },
    }, { deliverAs: "steer" });
    delivery.status = "queued";
    saveProcess();
  }

  pi.on("message_update", (event: any, ctx: ExtensionContext) => {
    if (processMode() === "off") return;
    const m = event?.message;
    if (m?.role !== "assistant") return;
    const marker = event?.assistantMessageEvent?.type;
    if (marker !== "thinking_end" && marker !== "text_end") return;
    collect(assembler.onAssistantMarker(m, marker), ctx);
  });

  pi.on("message_end", (event: any, ctx: ExtensionContext) => {
    const m = event?.message;
    if (!m) return;
    if (m.role === "custom") {
      // 守卫反馈进入上下文：送达状态 observed（只认自己 task 的）
      if (m.customType === "progress-guard:process-feedback" && pstate) {
        const fp = String(m.details?.fingerprint ?? "");
        const d = latestDelivery(pstate, fp);
        if (d && d.status === "queued") { d.status = "observed"; saveProcess(); }
      }
      return;                                     // 守卫/自定义消息不入审计流
    }
    if (processMode() === "off") return;
    if (m.role === "assistant") {
      collect(assembler.onMessageEnd(m), ctx);
      collect(assembler.onMessageEndToolCalls(m), ctx);
      if (Array.isArray(m.content) && m.content.some((b: any) => b.type === "toolCall")) {
        ctx.ui.setStatus("progress-guard", "守卫: 观察中");
      }
    } else if (m.role === "toolResult") {
      // 工具执行事件通常已入库（tool_execution_end）；此处去重补采
      const b = assembler.onToolResult(String(m.toolCallId ?? ""),
                                       textOf(m.content), !!m.isError);
      collect(b ? [b] : [], ctx);
    }
  });

  pi.on("tool_execution_end", (event: any, ctx: ExtensionContext) => {
    if (processMode() === "off") return;
    const b = assembler.onToolResult(String(event?.toolCallId ?? ""),
                                     typeof event?.result === "string"
                                       ? event.result : JSON.stringify(event?.result ?? ""),
                                     !!event?.isError, String(event?.toolName ?? "?"));
    collect(b ? [b] : [], ctx);
  });

  // ---------------- 收尾检查（原有行为保持不变） ----------------

  pi.on("turn_end", async (event: any, ctx: ExtensionContext) => {
    const message = event.message;
    if (!task || checking.has(task.id) || message?.role !== "assistant" || ctx.hasPendingMessages()) return;
    if (message.stopReason === "error" || message.stopReason === "aborted") return;
    if (Array.isArray(message.content) && message.content.some((b: any) => b.type === "toolCall")) return;
    const resp = textOf(message.content);
    if (!resp.trim()) return;
    const current = task;
    checking.add(current.id);
    try {
      const started = Date.now();
      const response = await fetch(URL + "/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(10000),
        body: JSON.stringify({ req: current.req, resp, mode: MODE,
          rounds: current.interventions + 1, semantic_review_used: current.semanticUsed }),
      });
      if (!response.ok) throw new Error(`guard HTTP ${response.status}`);
      const verdict = await response.json() as GuardResponse;
      if (task !== current || ctx.hasPendingMessages()) return;
      if (typeof verdict.allow_stop !== "boolean" || typeof verdict.review_requested !== "boolean") {
        throw new Error("guard policy fields missing; update the service");
      }
      pi.appendEntry("progress-guard:observation", {
        taskId: current.id, decision: verdict.decision, acceptance: verdict.acceptance,
        allowStop: verdict.allow_stop, semanticUsed: current.semanticUsed,
        elapsedMs: Date.now() - started, usage: message.usage,
      });
      ctx.ui.setStatus("progress-guard", `守卫: ${verdict.acceptance}`);
      if (verdict.allow_stop) return;

      const semantic = verdict.review_requested;
      const format = verdict.decision === "REQUIRE_MANIFEST";
      if (MODE === "advisory" && (semantic ? current.semanticUsed : !format || current.formatUsed)) return;
      if (MODE === "strict" && current.interventions >= 2) return;

      // Reserve before delivery: a crash must not reset the task's intervention budget.
      if (semantic) {
        current.semanticUsed = true;
        // 双向同步：收尾消耗的语义预算对过程审计同样生效（每任务一次，共享）
        if (pstate && pstate.taskId === current.id && !pstate.budgets.semanticUsed) {
          pstate.budgets.semanticUsed = true;
          saveProcess();
        }
      }
      if (format) current.formatUsed = true;
      current.interventions++;
      save();
      pi.sendMessage({
        customType: "progress-guard:feedback",
        content: verdict.prompt || verdict.feedback || "请核对收尾申报。",
        display: true,
        details: { taskId: current.id, semantic },
      }, { deliverAs: "followUp", triggerTurn: true });
      pi.appendEntry("progress-guard:delivery", { taskId: current.id, semantic, status: "queued" });
    } catch (error) {
      console.error(`[PG] unavailable (not verified): ${error}`);
      ctx.ui.setStatus("progress-guard", "守卫: 未验证（服务不可用）");
    } finally {
      checking.delete(current.id);
    }
  });
}

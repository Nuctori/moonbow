import { randomUUID } from "node:crypto";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const URL = process.env.MOONBOW_GUARD_URL || "http://127.0.0.1:18492";
const MODE = process.env.MOONBOW_GUARD_MODE === "strict" ? "strict" : "advisory";
const STATE = "progress-guard:state";

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
  const checking = new Set<string>();

  function save() {
    if (task) pi.appendEntry(STATE, { ...task });
  }

  function restore(_event: any, ctx: ExtensionContext) {
    task = undefined;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === STATE) {
        task = { ...(entry.data as TaskState) };
      }
    }
  }

  pi.on("session_start", restore);
  pi.on("session_tree", restore);

  pi.on("input", (event) => {
    if (event.source === "extension" || !event.text.trim() || event.text.startsWith("/")) return;
    task = { id: randomUUID(), req: event.text, semanticUsed: false, formatUsed: false, interventions: 0 };
    save();
  });

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
      if (semantic) current.semanticUsed = true;
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

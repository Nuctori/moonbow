/**
 * pi-extension/progress-guard.ts — Spark-4B 进度守卫 Pi 插件 (完整指标采集版)
 * 
 * 核心功能：
 * 1. 拦截时机：在 Agent 回合结束（turn_end）且准备停止执行时触发；
 * 2. 软注入触发：向本地守卫服务查询闭合度，精准判断客体差异、证据缺失或未闭合语气；
 * 3. 数据捕获：全程记录 Token 消耗、软注入介入日志与状态翻转效果。
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const GUARD_SERVICE_URL = process.env.MOONBOW_GUARD_URL || "http://127.0.0.1:18492";

interface GuardResponse {
  decision: "CLOSE" | "BLOCK" | "CLARIFY" | "REQUIRE_MANIFEST";
  feedback?: string;
  prompt?: string;
  disputed?: boolean;
}

export default function activate(pi: ExtensionAPI) {
  let userInitialRequest = "";
  let interceptionCount = 0;
  // 待注入的守卫反馈。sendMessage 只做展示、不进入模型上下文；
  // 真正影响模型必须通过 context 事件修改 messages，故此处排队。
  let pendingInjections: string[] = [];
  // 最近一次助手输出文本。turn_end 时 sessionManager 可能尚未包含当轮回复
  // （实测首轮 getEntries 拿不到 assistant 消息 → lastAssistantText 为空 → 漏检），
  // 故改由 message_end 事件缓存，作为可靠来源。
  let lastAssistantTextCache = "";
  let totalTurnTokens = 0;
  let currentInstanceId = "instance_default";
  const MAX_INTERCEPTIONS = 2; // 最多软注入/阻断 2 轮，防止过度发散

  pi.on("session_start", (event: any) => {
    userInitialRequest = "";
    interceptionCount = 0;
    totalTurnTokens = 0;
    currentInstanceId = event?.sessionId ?? `sess_${Date.now()}`;
  });

  // 1. 登记用户原始需求
  pi.on("message_start", (event: any) => {
    const m = event?.message ?? {};
    if (m.role === "user" && !userInitialRequest) {
      const text = extractText(m.content);
      if (text && text.length > 5 && !text.startsWith("/")) {
        userInitialRequest = text;
      }
    }
  });

  // 2. 软注入与拦截核心网关
  pi.on("turn_end", async (_event: any, ctx: ExtensionContext) => {
    try { console.error(`[PG] turn_end: req=${!!userInitialRequest} count=${interceptionCount}`); } catch {}
    if (!userInitialRequest || interceptionCount >= MAX_INTERCEPTIONS) {
      try { console.error("[PG] early-return: no request or max interceptions"); } catch {}
      return;
    }

    // 文本来源优先级：message_end 缓存 > sessionManager（后者在首轮可能为空）
    let lastAssistantText = lastAssistantTextCache;
    if (!lastAssistantText || lastAssistantText.length < 10) {
      const entries = (ctx as any)?.sessionManager?.getEntries?.() ?? [];
      for (let i = entries.length - 1; i >= 0; i--) {
        const msg = entries[i]?.message;
        if (msg?.role === "assistant") {
          const t = extractText(msg.content);
          if (t && t.length >= 10) { lastAssistantText = t; break; }
        }
      }
    }

    if (!lastAssistantText || lastAssistantText.length < 10) {
      try { console.error(`[PG] early-return: text len=${(lastAssistantText||"").length}`); } catch {}
      return;
    }
    try { console.error(`[PG] checking text len=${lastAssistantText.length}`); } catch {}

    try {
      // 请求本地轻量守卫微服务
      const resp = await fetch(GUARD_SERVICE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          req: userInitialRequest,
          resp: lastAssistantText,
          rounds: interceptionCount + 1,
          instance_id: currentInstanceId
        })
      });

      if (!resp.ok) return;
      const res = (await resp.json()) as GuardResponse;

      try { console.error(`[PG] guard decision=${res.decision}`); } catch {}
      if (res.decision === "CLOSE") {
        (ctx as any)?.ui?.setStatus?.("progress-guard", "✓ 守卫放行");
        return;
      }

      // 触发介入（REQUIRE_MANIFEST / CLARIFY / BLOCK）
      interceptionCount++;
      (ctx as any)?.ui?.setStatus?.("progress-guard", `⚠️ 守卫介入 (${interceptionCount}/${MAX_INTERCEPTIONS})`);

      let injectionText = "";
      if (res.decision === "REQUIRE_MANIFEST" && res.prompt) {
        injectionText = `【进度守卫提示】检测到你正在准备结束任务，请按三行规范申报闭合清单：\n\n${res.prompt}`;
      } else if (res.decision === "CLARIFY" && res.feedback) {
        // 软注入核心提示词：引导客体自省与验证补充
        injectionText = `【进度守卫核查意见（软提示）】\n${res.feedback}\n请对照你的实际修改和用户原始需求逐条确认；若确已修复，请在 EVIDENCE 补充具体的测试运行证据（如 pytest 或命令返回值）；若未完成，请继续调用工具排查。`;
      } else if (res.feedback) {
        injectionText = `【进度守卫阻断】\n${res.feedback}\n任务尚未闭合，请继续执行修复。`;
      }

      if (injectionText) {
        // 记录展示（便于人工观测）
        pi.sendMessage({
          customType: "progress-guard:feedback",
          content: injectionText,
          display: true
        });
        // 真正进入模型上下文：排入队列，由 context 事件消费
        pendingInjections.push(injectionText);
        try { console.error(`[PG] ENQUEUED len=${pendingInjections.length}`); } catch {}
      }
    } catch (e) {
      // 容灾：守卫故障不中断主流程，但必须留痕便于诊断
      try { console.error(`[PG] ERROR: ${(e as Error)?.message || e}`); } catch {}
    }
  });

  // 缓存助手输出：message_end 在消息定稿时触发，早于 turn_end 且内容完整。
  pi.on("message_end", (event: any) => {
    try {
      const m = event?.message ?? {};
      if (m.role === "assistant") {
        const txt = extractText(m.content);
        if (txt && txt.trim()) lastAssistantTextCache = txt;
      }
    } catch { /* 缓存失败不影响主流程 */ }
  });

  // 真正的注入通道：context 事件可替换即将发给模型的消息列表。
  // sendMessage 只做 UI 展示、不进入模型上下文；只有在这里追加 user 消息，
  // 守卫意见才会被模型看到。
  pi.on("context", async (event: any) => {
    try { console.error(`[PG] context fired pending=${pendingInjections.length}`); } catch {}
    if (!pendingInjections.length) return;
    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const injected = pendingInjections.splice(0); // 取出并清空，避免重复注入
    for (const text of injected) {
      messages.push({ role: "user", content: text });
    }
    return { messages };
  });

  function extractText(content: any): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b.text ?? "")
        .join("\n");
    }
    return "";
  }
}

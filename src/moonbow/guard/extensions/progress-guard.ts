/**
 * pi-extension/progress-guard.ts — Spark-4B 进度守卫 Pi 插件 (完整指标采集版)
 * 
 * 核心功能：
 * 1. 拦截时机：在 Agent 回合结束（turn_end）且准备停止执行时触发；
 * 2. 软注入触发：向本地守卫服务查询闭合度，精准判断客体差异、证据缺失或未闭合语气；
 * 3. 数据捕获：全程记录 Token 消耗、软注入介入日志与状态翻转效果。
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const GUARD_SERVICE_URL = "http://127.0.0.1:18492";

interface GuardResponse {
  decision: "CLOSE" | "BLOCK" | "CLARIFY" | "REQUIRE_MANIFEST";
  feedback?: string;
  prompt?: string;
  disputed?: boolean;
}

export default function activate(pi: ExtensionAPI) {
  let userInitialRequest = "";
  let interceptionCount = 0;
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
    if (!userInitialRequest || interceptionCount >= MAX_INTERCEPTIONS) {
      return;
    }

    const entries = (ctx as any)?.sessionManager?.getEntries?.() ?? [];
    
    // 统计当前 Turn 的 Token 消耗
    let lastTurnUsage: any = null;
    let lastAssistantText = "";
    for (let i = entries.length - 1; i >= 0; i--) {
      const entry = entries[i];
      const msg = entry?.message;
      if (msg?.role === "assistant") {
        lastAssistantText = extractText(msg.content);
        lastTurnUsage = (entry as any)?.usage ?? null;
        break;
      }
    }

    if (!lastAssistantText || lastAssistantText.length < 10) {
      return;
    }

    try {
      // 请求本地轻量守卫微服务
      const resp = await fetch(GUARD_SERVICE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          req: userInitialRequest,
          resp: lastAssistantText,
          rounds: interceptionCount + 1,
          instance_id: currentInstanceId,
          usage: lastTurnUsage
        })
      });

      if (!resp.ok) return;
      const res = (await resp.json()) as GuardResponse;

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
        // 向会话无缝注入反馈消息，迫使 Agent 开启下一个回合
        pi.sendMessage({
          customType: "progress-guard:feedback",
          content: injectionText,
          display: true
        });
      }
    } catch {
      // 容灾机制：守卫故障不中断用户原本流程
    }
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

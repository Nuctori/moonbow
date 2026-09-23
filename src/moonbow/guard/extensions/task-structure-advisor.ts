// moonbow.guard.extensions.task-structure-advisor
//
// Task Structure Advisor（任务结构顾问）独立 Pi 插件入口。
//
// 职责：用户消息到达后请求 /v1/task-structure 分析，在 shadow 模式记录
// 结果、在 advisory 模式注入**可忽略**的建议。默认 off（零请求、零日志）。
//
// 与守卫（progress-guard.ts）的关系：完全独立——不 import、不共享状态、
// 不调用 /check 或 /v1/stage-check；守卫的完成判定与放行条件不受本插件
// 任何输出的影响。本插件的异常一律静默吞掉，不阻塞主模型。
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const SHADOW_ENTRY = "task-structure:shadow";
const DELIVERY_ENTRY = "task-structure:delivery";

interface AnalysisResponse {
  level?: string;
  abstain?: boolean;
  advice?: string | null;
  fingerprint?: string;
  message?: string;
  vector?: Record<string, number>;
}

export default function activate(pi: ExtensionAPI) {
  // 环境在 activate 时读取（而非模块顶层），保证测试可按用例切换模式。
  const MODE = (process.env.MOONBOW_TSA_MODE || "off").toLowerCase();
  const BASE = process.env.MOONBOW_TSA_URL
    || process.env.MOONBOW_GUARD_URL
    || "http://127.0.0.1:18492";
  const TIMEOUT_MS = Number(process.env.MOONBOW_TSA_TIMEOUT_MS || 4000);
  const MAX_REMINDERS = Number(process.env.MOONBOW_TSA_MAX_REMINDERS || 1);
  const MIN_LEVEL = process.env.MOONBOW_TSA_MIN_LEVEL || "high";
  const MAX_INPUT_CHARS = 6000;
  const MIN_TEXT_CHARS = 6;

  // off：不注册任何处理器——保证零请求、零日志（可测试的行为）。
  if (MODE !== "shadow" && MODE !== "advisory") return;

  const seen = new Set<string>();        // 已提醒过的结构指纹
  let reminders = 0;                     // 本会话提醒总数（上限 MAX_REMINDERS）
  let busy = false;                      // 单飞：上一条未完成前不重叠请求

  pi.on("input", (data: { text?: string; source?: string }, ctx: ExtensionContext) => {
    const text = (data?.text || "").trim();
    if (!text || text.length < MIN_TEXT_CHARS || text.length > MAX_INPUT_CHARS) return;
    if (data?.source && data.source !== "interactive") return;  // 只看用户输入
    if (busy) return;
    busy = true;

    analyze(text)
      .then((res) => {
        if (!res) return;
        pi.appendEntry(SHADOW_ENTRY, res as unknown as Record<string, unknown>);
        if (MODE !== "advisory") return;
        const msg = decideReminder(res);
        if (msg) {
          reminders += 1;
          if (res.fingerprint) seen.add(res.fingerprint);
          // 建议消息：可忽略；未采纳不产生任何后续动作（无升级/无拦截）
          pi.sendMessage(msg, { deliver: true });
          pi.appendEntry(DELIVERY_ENTRY, {
            fingerprint: res.fingerprint, level: res.level, advice: res.advice,
          });
        }
      })
      .catch(() => { /* 失败隔离：静默跳过 */ })
      .finally(() => { busy = false; });
  });

  function analyze(text: string): Promise<AnalysisResponse | null> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    return fetch(`${BASE}/v1/task-structure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source: "user" }),
      signal: ctrl.signal,
    }).then((r) => {
      clearTimeout(timer);
      if (!r.ok) return null;
      return r.json() as Promise<AnalysisResponse>;
    }).catch(() => { clearTimeout(timer); return null; });
  }

  /** 客户端提醒判定：等级门槛 + 指纹去重 + 会话上限。
   * 全部为抑制性规则——不重试、不升级、不因未采纳而追加。 */
  function decideReminder(res: AnalysisResponse): string | null {
    if (!res.message) return null;
    if (res.abstain) return null;                       // 信息不足不提醒拆解
    if ((res.level || "unknown") !== MIN_LEVEL && res.level !== "high") return null;
    if (reminders >= MAX_REMINDERS) return null;
    if (res.fingerprint && seen.has(res.fingerprint)) return null;
    return res.message;
  }
}

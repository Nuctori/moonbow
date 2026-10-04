/**
 * process-events.ts — 阶段审计采集层：事件适配与块组装（纯逻辑，可单测）
 *
 * 设计要点（与宿主核验结论对应）：
 * - Pi 的 message_update 携带**累计消息快照** + 流式块事件标记
 *   （thinking_end / text_end）。因此组装策略是：块标记到达时从快照整体
 *   提取完整块，按 (messageKey, blockIndex) 去重 —— 不做 delta 累加，
 *   结构上杜绝文本翻倍。
 * - 工具参数流不做逐片语义审计：工具证据以 tool_execution_end（参数完整）
 *   为准，toolCallId 去重；toolResult 消息若指向同一 id 则只记一次。
 * - message_end 做完整性对账：补采此前漏掉的 thinking/text 块。
 * - 隐藏 CoT：没有 thinking 块就没有 thinking 事件，不产生虚假空块。
 */

export interface GuardBlock {
  seq: number;                    // 任务内全局顺序号（采集序，含被截断的历史）
  messageId: string;
  blockIndex: number;
  kind: "thinking" | "text" | "toolCall" | "toolResult";
  text: string;                   // thinking/thinking 正文；toolCall: JSON{name,args}; toolResult: 文本
  complete: boolean;
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
}

export type AssistantMarker = "thinking_end" | "text_end";

const MAX_TOOL_TEXT = 2000;       // 工具参数/结果入库截断（保留截断标记，不静默）

function clip(text: string, limit = MAX_TOOL_TEXT): string {
  if (text.length <= limit) return text;
  return text.slice(0, limit) + `…[truncated ${text.length - limit} chars]`;
}

export class BlockAssembler {
  private seqCounter: number;
  private msgCounter = 0;
  private emitted = new Set<string>();      // `${messageKey}:${blockIndex}`
  private msgIds = new WeakMap<object, string>();
  private stableKeys = new Map<string, string>();  // 稳定标识 -> messageKey
  private resultSeen = new Set<string>();   // toolCallId（工具证据只记一次）

  /** startSeq：崩溃/分支恢复后续接全局序号，保证游标语义连续。 */
  constructor(startSeq = 0) {
    this.seqCounter = startSeq;
  }

  /** 消息身份 key：responseId 优先，对象身份兜底。
   *
   * 2026-09-22 实测：只用 WeakMap<对象> 时，pi 在 message_end 传入的是
   * 新对象（非同一流式对象），同一消息因此拿到新 key、去重失效 ——
   * 观察流里同一个块被重复采集 3 次（seq 递增、内容全同），时序核对
   * 被污染，服务端据此算不出 finding。
   *
   * 只用 responseId 做稳定标识：它是响应级唯一 id。**不用 timestamp**
   * —— 多条消息可能共享/缺失时间戳，拿它当 key 会把不同消息并成一条
   * （实测导致多块采集测试失败）。
   */
  messageKey(message: any): string {
    const rid = message?.responseId;
    if (rid != null && rid !== "") {
      const k = `s:${rid}`;
      const hit = this.stableKeys.get(k);
      if (hit) { this.msgIds.set(message, hit); return hit; }
      const id = `m${++this.msgCounter}`;
      this.stableKeys.set(k, id);
      this.msgIds.set(message, id);
      return id;
    }
    const existing = this.msgIds.get(message);
    if (existing) return existing;
    const id = `m${++this.msgCounter}-${message?.timestamp ?? Date.now()}`;
    this.msgIds.set(message, id);
    return id;
  }

  get seq(): number { return this.seqCounter; }

  /**
   * message_update：仅在块结束标记时扫描快照，补齐所有尚未提交的
   * thinking/text 完整块（多块、乱序标记均安全）。
   */
  onAssistantMarker(message: any, marker: AssistantMarker): GuardBlock[] {
    if (marker !== "thinking_end" && marker !== "text_end") return [];
    return this.scanBlocks(message);
  }

  /** message_end：完整性对账，补采漏掉的 thinking/text 块（含去重）。 */
  onMessageEnd(message: any): GuardBlock[] {
    return this.scanBlocks(message);
  }

  /** 助手消息中的完整 toolCall 块（message_end 时提取，参数已完整）。 */
  private scanBlocks(message: any): GuardBlock[] {
    if (!message || message.role !== "assistant" || !Array.isArray(message.content)) return [];
    const key = this.messageKey(message);
    const out: GuardBlock[] = [];
    message.content.forEach((b: any, i: number) => {
      if (!b || (b.type !== "thinking" && b.type !== "text")) return;
      const dedup = `${key}:${i}`;
      if (this.emitted.has(dedup)) return;
      const text = b.type === "thinking" ? String(b.thinking ?? "") : String(b.text ?? "");
      if (!text.trim()) { this.emitted.add(dedup); return; }   // 空块占位也去重
      this.emitted.add(dedup);
      out.push({
        seq: this.seqCounter++, messageId: key, blockIndex: i,
        kind: b.type === "thinking" ? "thinking" : "text",
        text: clip(text), complete: true,
      });
    });
    return out;
  }

  /** 全局序号发号：工具事件通道与块共用同一采集序空间（服务端合并排序
   * 时二者相对顺序真实）。 */
  nextSeq(): number { return this.seqCounter++; }

  /** 完整工具调用（message_end 时从快照提取；参数流不逐片入库）。 */
  onMessageEndToolCalls(message: any): GuardBlock[] {
    if (!message || message.role !== "assistant" || !Array.isArray(message.content)) return [];
    const key = this.messageKey(message);
    const out: GuardBlock[] = [];
    message.content.forEach((b: any, i: number) => {
      if (!b || b.type !== "toolCall") return;
      const dedup = `${key}:${i}`;
      if (this.emitted.has(dedup)) return;
      this.emitted.add(dedup);
      out.push({
        seq: this.seqCounter++, messageId: key, blockIndex: i,
        kind: "toolCall", complete: true,
        toolCallId: String(b.id ?? ""), toolName: String(b.name ?? "?"),
        text: clip(JSON.stringify({ name: b.name, arguments: b.arguments ?? {} })),
      });
    });
    return out;
  }

  /**
   * 工具执行结果。toolCallId 去重：tool_execution_end 与 toolResult 消息
   * 可能先后到达指向同一次执行，只记一次（先到者优先）。
   */
  onToolResult(toolCallId: string, text: string, isError: boolean, toolName = "?"): GuardBlock | null {
    const id = String(toolCallId ?? "");
    const dedup = id ? `r:${id}` : `r:${this.seqCounter}`;
    if (this.resultSeen.has(dedup)) return null;
    this.resultSeen.add(dedup);
    return {
      seq: this.seqCounter++, messageId: "tool", blockIndex: 0,
      kind: "toolResult", text: clip(text), complete: true,
      toolCallId: id || undefined, toolName, isError,
    };
  }
}

// ---- v2 工具事件通道（2026-10-04，guard-effect-v2 通路补全）----
//
// 根因（results/guard-effect-v2/threearm_report.md §7）：线上 5115 次
// stage-check 的 findings 全为 0，v2 三规则（repeat/stall/fail_streak 全部
// 依赖工具流）整场"饿着"——部署运行时里依赖消息快照组装的块采集路径没有把
// 工具调用/结果送进观察流（离线回放直接喂会话工具流则全部有效）。
//
// 修复：工具事件改走 pi 核心执行事件（tool_execution_start/end —— 与消息
// 快照组装无关、每次工具执行必发），以独立 tool_events 数组随 stage-check
// 携带。携带纪律（默认行为论证）：**客户端默认携带 + 服务端按 ruleset 消费**
// ——①体积可控（见下方截断/封顶常量，实测最大会话 1890 事件约 210KB 原文、
// 截断后更小，且走本机 HTTP）；②客户端不知道服务端 ruleset，按端开关折叠
// 会把 v2 装配判据漏到客户端；③off 模式根本不发 stage-check，v1 服务端不
// 消费该键——两条既有路径零行为变化。

export interface ToolEvent {
  seq: number;                    // 与 GuardBlock 同一采集序空间
  phase: "call" | "result";
  name: string;                   // 工具名
  args?: string;                  // call 相：JSON{name, arguments} 摘要（截断）
  is_error?: boolean;             // result 相
  result?: string;                // result 相：结果摘要（截断，保留 pytest 摘要行）
  toolCallId?: string;
}

// 判定载荷纪律：只保留规则所需最小字段——命令核心（repeat 归一化）、
// 是否 error、pytest 摘要行（覆盖度/连败判定的最小充分证据，位于输出
// 末尾，截断必须保尾）。args 摘要做**逐值截断**（键结构完整保留）：整段
// 头截断会把 JSON 截坏，服务端 _parse_tool_call 解析失败 → write 路径/
// bash 重定向解析全丢（实测 both r8 的 edit_oscillation 因此漏检）。
export const MAX_TOOL_EVENT_ARG_VALUE = 200;  // 单值截断（命令核心/路径足够）
export const MAX_TOOL_EVENT_ARGS_KEYS = 40;   // 键数上限（防巨型参数对象）
export const MAX_TOOL_EVENT_ARGS = 1200;      // 摘要总长兜底
export const MAX_TOOL_EVENT_RESULT_HEAD = 160;
export const MAX_TOOL_EVENT_RESULT_TAIL = 320;

const PYTEST_SUMMARY_LINE_RE = /\d+\s+passed|\d+\s+failed|no tests ran/i;

function clipArgValue(v: string, limit = MAX_TOOL_EVENT_ARG_VALUE): string {
  if (v.length <= limit) return v;
  return v.slice(0, limit) + `…[truncated ${v.length - limit} chars]`;
}

function clipToolArgs(text: string): string {
  if (text.length <= MAX_TOOL_EVENT_ARGS) return text;
  return text.slice(0, MAX_TOOL_EVENT_ARGS) +
    `…[truncated ${text.length - MAX_TOOL_EVENT_ARGS} chars]`;
}

/** 调用摘要：JSON{name, arguments}，键结构完整、长字符串值逐值截断
 * （嵌套结构整体序列化后截断；服务端解析器照常取 command / path 键）。 */
export function summarizeToolArgs(toolName: string, args: any): string {
  let out: Record<string, unknown>;
  if (args && typeof args === "object" && !Array.isArray(args)) {
    out = {};
    for (const k of Object.keys(args).slice(0, MAX_TOOL_EVENT_ARGS_KEYS)) {
      const v = args[k];
      if (typeof v === "string") out[k] = clipArgValue(v);
      else if (v == null || typeof v === "number" || typeof v === "boolean") out[k] = v;
      else {
        try { out[k] = clipArgValue(JSON.stringify(v) ?? "", 400); } catch { out[k] = String(v); }
      }
    }
  } else if (args == null) {
    out = {};
  } else {
    out = { input: clipArgValue(String(args)) };
  }
  return clipToolArgs(JSON.stringify({ name: toolName, arguments: out }));
}

/** 结果摘要：中段截断（保头保尾）；pytest 摘要行若不在保留区内显式补附。 */
export function clipToolResult(text: string): string {
  const t = text || "";
  const head = MAX_TOOL_EVENT_RESULT_HEAD, tail = MAX_TOOL_EVENT_RESULT_TAIL;
  if (t.length <= head + tail + 64) return t;
  let kept = t.slice(0, head) + `…[truncated ${t.length - head - tail} chars]` + t.slice(-tail);
  const lines = t.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    if (PYTEST_SUMMARY_LINE_RE.test(lines[i])) {
      const line = lines[i].trim();
      if (!kept.includes(line)) kept += `…[pytest] ${line}`;
      break;
    }
  }
  return kept;
}

function toolResultText(result: any): string {
  if (typeof result === "string") return result;
  try { return JSON.stringify(result ?? ""); } catch { return String(result); }
}

/**
 * 工具事件采集器：pi 核心执行事件 → 精简 ToolEvent。
 * 按 (phase, toolCallId) 去重——start/end 与消息侧事件可能重复指向同一次
 * 执行；无 id 的事件丢弃（无法配对也无法去重，块采集路径兜底）。
 */
export class ToolEventCollector {
  private seenCall = new Set<string>();
  private seenResult = new Set<string>();
  private assembler: BlockAssembler;

  constructor(assembler: BlockAssembler) {
    // 与块采集共用同一 seq 发号器（strip-only TS 不支持参数属性，显式赋值）
    this.assembler = assembler;
  }

  onToolStart(toolCallId: string, toolName: string, args: any): ToolEvent | null {
    const id = String(toolCallId ?? "");
    if (!id || this.seenCall.has(id)) return null;
    this.seenCall.add(id);
    let summary: string;
    try {
      summary = summarizeToolArgs(toolName, args);
    } catch {
      summary = clipToolArgs(String(args ?? ""));
    }
    return {
      seq: this.assembler.nextSeq(), phase: "call", name: String(toolName ?? "?"),
      args: summary, toolCallId: id,
    };
  }

  onToolEnd(toolCallId: string, toolName: string, result: any, isError: boolean): ToolEvent | null {
    const id = String(toolCallId ?? "");
    if (!id || this.seenResult.has(id)) return null;
    this.seenResult.add(id);
    return {
      seq: this.assembler.nextSeq(), phase: "result", name: String(toolName ?? "?"),
      is_error: !!isError, result: clipToolResult(toolResultText(result)), toolCallId: id,
    };
  }
}

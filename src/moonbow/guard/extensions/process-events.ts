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

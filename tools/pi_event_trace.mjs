// pi_event_trace.mjs — P0 宿主行为核验：从真实 pi session JSONL 提取
// 助手消息的块序轨迹（thinking/text/toolCall 的实际排列），以及
// custom（守卫）消息在真实会话中的出现形态。
// 用法：node tools/pi_event_trace.mjs <session.jsonl> [max-messages]
import fs from "node:fs";
import readline from "node:readline";

const [file, maxArg] = process.argv.slice(2);
if (!file) { console.error("usage: node pi_event_trace.mjs <session.jsonl> [max]"); process.exit(2); }
const max = Number(maxArg || 40);

const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
let shown = 0;
const orders = {};          // 块类型序列 -> 次数（同一助手消息内）
let toolResultDup = 0;      // 同一 toolCallId 出现两次 toolResult 的次数
const seenResults = new Set();
let customMsgs = 0;

rl.on("line", (l) => {
  let ev; try { ev = JSON.parse(l); } catch { return; }
  if (ev.type !== "message") return;
  const m = ev.message ?? {};
  if (m.role === "custom") { customMsgs++; return; }
  if (m.role === "toolResult") {
    const id = String(m.toolCallId ?? "");
    if (id && seenResults.has(id)) toolResultDup++;
    if (id) seenResults.add(id);
    return;
  }
  if (m.role !== "assistant") return;
  if (shown >= max) return;
  shown++;
  const kinds = (Array.isArray(m.content) ? m.content : [])
    .map((b) => b?.type ?? "?").join(">");
  orders[kinds] = (orders[kinds] || 0) + 1;
  if (process.argv[2] && process.env.TRACE_VERBOSE) {
    const stop = m.stopReason ?? "?";
    console.log(`  msg: ${kinds}  stop=${stop}`);
  }
});
rl.on("close", () => {
  console.log("助手消息块序（前", shown, "条）:");
  for (const [k, n] of Object.entries(orders).sort((a, b) => b[1] - a[1]).slice(0, 12))
    console.log(`  ${String(n).padStart(4)}  ${k}`);
  console.log("toolResult 重复 toolCallId:", toolResultDup);
  console.log("custom 消息条数:", customMsgs);
});

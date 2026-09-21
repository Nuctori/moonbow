import test from 'node:test';
import assert from 'node:assert/strict';
import activate from '../src/moonbow/guard/extensions/progress-guard.ts';

// —— 事件流模拟工具 ——

function snap(content, extra = {}) {
  // 模拟 pi 的累计快照消息对象（引用恒定，内容被"流式"填充）
  return { role: 'assistant', content, timestamp: 1700000000000, ...extra };
}

function harness(entries = [], opts = {}) {
  const handlers = {}, sent = [], calls = [], statuses = [], shadow = [];
  const pi = {
    on(name, fn) { handlers[name] = fn; },
    appendEntry(customType, data) { entries.push({ type: 'custom', customType, data: structuredClone(data) });
      if (customType === 'progress-guard:shadow') shadow.push(structuredClone(data)); },
    sendMessage(message, options) { sent.push({ message, options }); },
  };
  const ctx = { sessionManager: { getBranch: () => entries },
    hasPendingMessages: () => !!opts.pending,
    ui: { setStatus: (...args) => statuses.push(args) } };
  activate(pi);
  handlers.session_start({}, ctx);
  const stageResp = (body) => { calls.push(JSON.parse(body)); return stageResponse; };
  let stageResponse = { findings: [], reminder: null, semantic: false, based_on: 1 };
  const setStage = (r) => { stageResponse = r; };
  return { handlers, sent, calls, statuses, shadow, entries, ctx,
           input: (t = 'task') => handlers.input({ text: t, source: 'interactive' }),
           setStage };
}

const tick = () => new Promise((r) => setImmediate(r));
const flush = async () => { for (let i = 0; i < 6; i++) await tick(); };

function reminderResponse(fp, semantic = true) {
  return { findings: [], semantic,
    reminder: { fingerprint: fp, summary: '验证时序不符',
      evidence: ['第3块声明', '第1块测试'], suggestion: '【进度守卫（过程核查）】请补验证或如实修正。' },
    based_on: 1 };
}

// —— 1. 采集：delta 快照不翻倍、message_end 补采去重 ——

test('block assembly: growing snapshot + end markers never duplicate blocks', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    if (String(options?.body || '').includes('"blocks":[]')) return { ok: true, json: async () => ({ findings: [], reminder: null }) };
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  const msg = snap([{ type: 'thinking', thinking: '先看看代码' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'thinking_delta' } }, h.ctx);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'thinking_end' } }, h.ctx);
  // 同一快照重复喂 thinking_end —— 不产生重复块
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'thinking_end' } }, h.ctx);
  msg.content.push({ type: 'text', text: '已完成修复' });
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  // message_end 对账：同一批块应被去重
  h.handlers.message_end({ message: msg }, h.ctx);
  await flush();
  const all = h.calls.flatMap(c => c.blocks);
  const texts = all.filter(b => b.kind !== 'toolCall').map(b => b.text);
  assert.equal(texts.filter(t => t === '先看看代码').length, 1);
  assert.equal(texts.filter(t => t === '已完成修复').length, 1);
});

test('multi thinking blocks + hidden CoT (no thinking) both collect', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  // 可见 CoT：两个思考块 + 正文
  const msg = snap([
    { type: 'thinking', thinking: '思路一' },
    { type: 'thinking', thinking: '思路二' },
    { type: 'text', text: '中间汇报：正在验证' },
  ]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'thinking_end' } }, h.ctx);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  const kinds = h.calls.flatMap(c => c.blocks).map(b => b.kind);
  assert.ok(kinds.includes('thinking') && kinds.includes('text'));
  // 隐藏 CoT：只有正文的新消息
  const msg2 = snap([{ type: 'text', text: '已完成第二部分' }]);
  h.handlers.message_update({ message: msg2, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  const kinds2 = h.calls.flatMap(c => c.blocks).map(b => `${b.kind}:${b.text}`);
  assert.ok(kinds2.includes('text:已完成第二部分'));
  assert.ok(!h.calls.flatMap(c => c.blocks).some(b => b.kind === 'thinking' && b.text === ''));
});

test('tool evidence deduped between tool_execution_end and toolResult message', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  h.handlers.tool_execution_end({ toolCallId: 't1', toolName: 'pytest',
    result: '1 passed', isError: false }, h.ctx);
  h.handlers.message_end({ message: { role: 'toolResult', toolCallId: 't1',
    content: [{ type: 'text', text: '1 passed' }], isError: false } }, h.ctx);
  await flush();
  const tools = h.calls.flatMap(c => c.blocks).filter(b => b.kind === 'toolResult');
  assert.equal(tools.length, 1);
  assert.equal(tools[0].tool_call_id, 't1');
});

// —— 2. off / shadow / advisory 模式 ——

test('off mode: no stage fetch, no process entries', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'off';
  const h = harness();
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => { throw new Error('should not fetch'); });
  h.input();
  const msg = snap([{ type: 'thinking', thinking: '思路' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'thinking_end' } }, h.ctx);
  await flush();
  assert.equal(h.calls.length, 0);
  assert.equal(h.sent.length, 0);
});

test('shadow mode: audits and records but never injects', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-shadow') };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.ok(h.calls.length >= 1);            // 审计发生
  assert.ok(h.shadow.length >= 1);           // 影子记录本应提醒的内容
  assert.equal(h.sent.length, 0);            // 绝不注入
});

// —— 3. advisory：送达、去重、预算 ——

test('advisory: steer delivery once per fingerprint, duplicate suppressed', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-1') };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  assert.deepEqual(h.sent[0].options, { deliverAs: 'steer' });
  assert.equal(h.sent[0].message.customType, 'progress-guard:process-feedback');
  // 第二块再次触发同类提醒 —— 不重复催
  const msg2 = snap([{ type: 'text', text: '补充说明：另一部分也完成了' }]);
  h.handlers.message_update({ message: msg2, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
});

test('semantic budget shared with final check; non-semantic reminder unaffected', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  let stage = reminderResponse('fp-sem', true);
  let final = { decision: 'CLARIFY', acceptance: 'disputed', allow_stop: false,
                review_requested: true, prompt: 'final review' };
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    const body = JSON.parse(options.body);
    if (typeof body.snapshot_version === 'number' && body.blocks) {
      h.calls.push(body);
      return { ok: true, json: async () => stage };
    }
    return { ok: true, json: async () => final };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);            // 过程提醒投递，语义预算已预留
  // 收尾语义提醒：预算已用 -> advisory 下不再投递
  await h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'end_turn',
    content: [{ type: 'text', text: 'STATUS: A\nREMAINING: 无\nEVIDENCE: some evidence' }] } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  // 非 semantic 的过程提醒不受语义预算限制
  stage = reminderResponse('fp-plain', false);
  const msg2 = snap([{ type: 'text', text: '再次汇报进展' }]);
  h.handlers.message_update({ message: msg2, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 2);
});

// —— 4. 竞态与隔离 ——

test('stale stage response cannot intervene after task switch', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  let resolveStage;
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    const body = JSON.parse(options.body);
    if (body.blocks) { h.calls.push(body); return new Promise((r) => { resolveStage = r; }); }
    return { ok: true, json: async () => ({ decision: 'CLOSE', acceptance: 'unchecked',
      allow_stop: true, review_requested: false }) };
  });
  h.input('old');
  const msg = snap([{ type: 'text', text: '已经修复完成' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await tick();                                // 让请求发出但未返回
  const oldResolve = resolveStage;
  h.input('new');                              // 新任务
  oldResolve({ ok: true, json: async () => reminderResponse('fp-old') });
  await flush();
  assert.equal(h.sent.length, 0);              // 迟到响应不介入新任务
});

test('guard custom message not collected and does not self-excite', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  const before = h.calls.length;
  h.handlers.message_end({ message: { role: 'custom', customType: 'progress-guard:process-feedback',
    content: [{ type: 'text', text: '【进度守卫（过程核查）】...' }] } }, h.ctx);
  await flush();
  const stageCalls = h.calls.slice(before);
  assert.ok(stageCalls.every(c => !JSON.stringify(c.blocks).includes('进度守卫')));
  // extension 来源输入不建新任务（不重置预算）
  h.handlers.input({ text: '【进度守卫】x', source: 'extension' });
  assert.ok(h.entries.filter(e => e.customType === 'progress-guard:state').length >= 1);
});

test('service unavailable degrades to unverified observation, no crash', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async () => { throw new Error('ECONNREFUSED'); });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0);
  assert.ok(h.entries.some(e => e.customType === 'progress-guard:observation'
    && e.data.process && e.data.note === 'unverified'));
});

// —— 5. 恢复 ——

test('restart preserves budgets and does not re-remind same fingerprint', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-keep') };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  // 模拟重启：用同一 entries 重建
  const h2 = harness(h.entries);
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h2.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-keep') };
  });
  h2.handlers.session_start({}, h2.ctx);
  const msg2 = snap([{ type: 'text', text: '重启后继续汇报：已完成其它部分' }]);
  h2.handlers.message_update({ message: msg2, assistantMessageEvent: { type: 'text_end' } }, h2.ctx);
  await flush();
  assert.equal(h2.sent.length, 0);           // 同一 fingerprint 不再催
});

test('long session beyond 120 blocks keeps collecting new events', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  process.env.MOONBOW_GUARD_MAX_BLOCKS = '120';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  // 灌 150 个块（每条消息 1 块）
  for (let i = 0; i < 150; i++) {
    const m = snap([{ type: 'text', text: `进度块 ${i}` }]);
    h.handlers.message_update({ message: m, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  }
  await flush();
  const seen = h.calls.flatMap(c => c.blocks.map(b => b.text));
  assert.ok(seen.includes('进度块 149'));     // 尾部新块仍被处理（头部被环形丢弃并留痕）
  const procEntries = h.entries.filter(e => e.customType === 'progress-guard:process');
  const last = procEntries.at(-1).data;
  assert.ok(last.truncatedFrom >= 30);        // 截断留痕，不静默
  delete process.env.MOONBOW_GUARD_MAX_BLOCKS;
});

test('pending user input suppresses reminder delivery', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness([], { pending: true });
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-pending') };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0);            // 有待处理输入，不插入
});

test('final-check semantic budget blocks later process semantic reminder', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  let stage = reminderResponse('fp-after-final', true);
  let final = { decision: 'CLARIFY', acceptance: 'disputed', allow_stop: false,
                review_requested: true, prompt: 'final review first' };
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    const body = JSON.parse(options.body);
    if (typeof body.snapshot_version === 'number' && body.blocks) {
      h.calls.push(body);
      return { ok: true, json: async () => stage };
    }
    return { ok: true, json: async () => final };
  });
  h.input();
  // 先收尾消耗语义预算（followUp 投递）
  await h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'end_turn',
    content: [{ type: 'text', text: 'STATUS: A\nREMAINING: 无\nEVIDENCE: evidence text' }] } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  // 之后到达的阶段审计语义提醒：预算已耗尽 -> 只记录不投递
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);            // 未发生第二次语义介入
});

test('off mode writes no process entries on input', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'off';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async () => { throw new Error('no fetch'); });
  h.input();
  await tick();
  assert.equal(h.entries.filter(e => e.customType === 'progress-guard:process').length, 0);
});

test('delivery observed when custom feedback message enters context', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => reminderResponse('fp-obs') };
  });
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  h.handlers.message_end({ message: { role: 'custom', customType: 'progress-guard:process-feedback',
    content: [{ type: 'text', text: '【进度守卫（过程核查）】...' }],
    details: { fingerprint: 'fp-obs' } } }, h.ctx);
  await tick();
  const d = h.entries.filter(e => e.customType === 'progress-guard:process').at(-1).data;
  assert.equal(d.deliveries.at(-1).status, 'observed');
});

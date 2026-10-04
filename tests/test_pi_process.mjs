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
  // 2026-09-22 起插件发送"窗口内全部块"（服务端时序核对需要完整历史），
  // 故同一块会跨批次重复出现 —— 断言改为：**单次调用内**不得有重复块。
  for (const c of h.calls) {
    const texts = c.blocks.filter(b => b.kind !== 'toolCall').map(b => b.text);
    assert.equal(texts.filter(t => t === '先看看代码').length <= 1, true,
      '单批内同一块不得重复');
    assert.equal(texts.filter(t => t === '已完成修复').length <= 1, true,
      '单批内同一块不得重复');
  }
  // 全局去重判据：每个 seq 只对应**一种内容**（emitted 集合保证块不被重复创建）
  const bySeq = new Map();
  for (const c of h.calls) for (const b of c.blocks) {
    const prev = bySeq.get(b.seq);
    if (prev === undefined) bySeq.set(b.seq, b.text);
    else assert.equal(prev, b.text, `seq=${b.seq} 内容不一致（重复创建）`);
  }
  // 内容相同的块只应有一个 seq（不得为同一内容分配多个 seq）
  const textSeqs = new Map();
  for (const [sq, tx] of bySeq) {
    if (tx === '先看看代码' || tx === '已完成修复') {
      textSeqs.set(tx, (textSeqs.get(tx) || 0) + 1);
    }
  }
  for (const [tx, n] of textSeqs) assert.equal(n, 1, `"${tx}" 被分配了 ${n} 个 seq`);
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

// —— 17. 工具调用被误写成文本时不得触发收尾裁决（2026-09-22 修复）——
// 实测：模型把 tool_call 写成纯文本（带 </arg_value> 残渣），该轮
// stopReason=stop 且无真实 toolCall，旧逻辑据此送 /check，守卫发
// REQUIRE_MANIFEST 打断推进中的任务。此测试锁住"识别即跳过"。

test('tool-call-as-text does not trigger final check', async (t) => {
  const h = harness();
  const checked = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url).endsWith('/check')) {
      checked.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ decision: 'REQUIRE_MANIFEST', allow_stop: false,
        acceptance: 'pending', review_requested: false, prompt: '申报清单' }) };
    }
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  // 模型把 read 调用写成了文本（真实污染样本）
  const polluted = '[tool_call call_3f37886c] read {"path":"src/flask/blueprints.py"}</arg_value>';
  h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'stop',
    content: [{ type: 'text', text: polluted }] } }, h.ctx);
  await flush();
  assert.equal(checked.length, 0, '格式错误的文本不应送收尾裁决');
});

test('genuine completion claim still triggers final check', async (t) => {
  const h = harness();
  const checked = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url).endsWith('/check')) {
      checked.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ decision: 'CLOSE', allow_stop: true,
        acceptance: 'closed', review_requested: false }) };
    }
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'stop',
    content: [{ type: 'text', text: '我已经修复了登录问题，测试全部通过。' }] } }, h.ctx);
  await flush();
  assert.equal(checked.length, 1, '真实完成声明必须送审');
});

// —— 19. 同一消息以不同对象形态重复到达时不得重复采集（2026-09-22 修复）——
// 实测：pi 在 message_end 传入新对象，WeakMap 去重失效，同一块被采集 3 次
// （seq 递增、内容全同），观察流被污染，服务端算不出 finding。

test('same message arriving as different objects is deduped by stable id', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  const bodies = [];
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    const b = JSON.parse(options.body);
    if (b.blocks?.length) bodies.push(b);
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  const content = [{ type: 'text', text: '复核完毕，无需修正' }];
  const rid = 'resp-abc';
  // 三次不同对象、同一 responseId（模拟 pi 的累计快照/定稿多形态）
  h.handlers.message_end({ message: { role: 'assistant', responseId: rid, content } }, h.ctx);
  h.handlers.message_end({ message: { role: 'assistant', responseId: rid, content } }, h.ctx);
  h.handlers.message_end({ message: { role: 'assistant', responseId: rid, content } }, h.ctx);
  await flush();
  const seen = bodies.flatMap((b) => b.blocks).filter((x) => x.kind === 'text');
  assert.equal(seen.length, 1, `同一 responseId 的文本块只应采集一次，实际 ${seen.length}`);
});

// —— 20. agent_end 时必须以 stream_ended=true 发审计（2026-09-22）——
// 实测：模型 edit 后正要继续验证就被提前停毛病打断，agent_end 的终止审计
// 应携带 stream_ended=true，服务端据此报"改了没验证"。若丢失，该缺口永不报。

test('agent_end sends stream_ended=true audit', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness();
  const bodies = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    const b = JSON.parse(options.body);
    bodies.push({ url: String(url), body: b });
    if (String(url).endsWith('/check')) {
      return { ok: true, json: async () => ({ decision: 'CLOSE', allow_stop: true,
        acceptance: 'closed', review_requested: false }) };
    }
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  // 一轮带工具调用的消息（写入类），随后运行结束
  const msg = snap([
    { type: 'toolCall', name: 'edit', id: 'call_x', arguments: { path: 'a.py' } },
  ]);
  h.handlers.message_end({ message: msg }, h.ctx);
  h.handlers.agent_end({ messages: [msg] }, h.ctx);
  await flush();
  const stage = bodies.filter((x) => x.url.endsWith('/stage-check'));
  assert.ok(stage.length >= 1, 'agent_end 应触发一次阶段审计');
  const withFlag = stage.filter((x) => x.body.stream_ended === true);
  assert.ok(withFlag.length >= 1, '终止审计必须携带 stream_ended=true');
});

// —— 21. v2 工具事件通道（2026-10-04，guard-effect-v2 通路补全）——
// 根因：线上 5115 次 stage-check 全 findings=0，v2 规则整场无工具流输入。
// 修复：tool_execution_start/end → 独立 tool_events 数组随 stage-check 携带。

test('tool events ride the stage-check payload (call+result, deduped, shared seq space)', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  h.handlers.tool_execution_start({ toolCallId: 't1', toolName: 'bash',
    args: { command: 'python -m pytest tests/test_a.py -q' } }, h.ctx);
  h.handlers.tool_execution_start({ toolCallId: 't1', toolName: 'bash',
    args: { command: 'python -m pytest tests/test_a.py -q' } }, h.ctx);  // 重复 start 去重
  h.handlers.tool_execution_end({ toolCallId: 't1', toolName: 'bash',
    result: '3 passed in 0.12s', isError: false }, h.ctx);
  h.handlers.tool_execution_end({ toolCallId: 't1', toolName: 'bash',
    result: '3 passed in 0.12s', isError: false }, h.ctx);  // 重复 end 去重
  await flush();
  const withEvents = h.calls.filter((c) => (c.tool_events || []).length);
  assert.ok(withEvents.length >= 1, 'stage-check payload 必须携带 tool_events');
  const ev = withEvents.at(-1).tool_events;
  assert.equal(ev.length, 2, 'call+result 各一条（start/end 重复事件去重）');
  assert.deepEqual(ev.map((e) => e.phase), ['call', 'result']);
  assert.equal(ev[0].name, 'bash');
  assert.ok(ev[0].args.includes('pytest tests/test_a.py'), '调用摘要须含命令核心');
  assert.equal(ev[1].is_error, false);
  assert.ok(ev[1].result.includes('3 passed'), '结果摘要须含 pytest 摘要行');
  assert.equal(ev[0].tool_call_id, 't1');
  assert.ok(ev[1].seq > ev[0].seq, '工具事件与块共用单调采集序');
});

test('long tool output clipped but pytest summary line survives; args JSON stays parseable', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  const longOutput = 'x'.repeat(5000) + '\n===============================\n'
    + '2 failed, 27 passed in 1.23s\n';
  h.handlers.tool_execution_end({ toolCallId: 't9', toolName: 'bash',
    result: longOutput, isError: false }, h.ctx);
  // 超长 content 的 write 调用：整段截断会截坏 JSON（服务端解析不到 path）
  const bigContent = 'c'.repeat(3000);
  h.handlers.tool_execution_start({ toolCallId: 't10', toolName: 'write',
    args: { content: bigContent, path: 'src/mod.py' } }, h.ctx);
  await flush();
  const ev = h.calls.flatMap((c) => c.tool_events || []);
  const res = ev.find((e) => e.tool_call_id === 't9');
  assert.ok(res.result.length < 1000, `结果摘要应有界，实际 ${res.result.length}`);
  assert.ok(res.result.includes('2 failed, 27 passed'), 'pytest 摘要行必须存活');
  const call = ev.find((e) => e.tool_call_id === 't10');
  const parsed = JSON.parse(call.args);          // 逐值截断后必须仍是合法 JSON
  assert.equal(parsed.name, 'write');
  assert.equal(parsed.arguments.path, 'src/mod.py', '路径键必须完整保留');
  assert.ok(parsed.arguments.content.length < 300, '长字符串值应逐值截断');
  assert.ok(call.args.length < 1500, '调用摘要应有界');
});

test('tool events capped (oldest dropped) and pure-tool increments re-audit', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  process.env.MOONBOW_GUARD_MAX_TOOL_EVENTS = '10';
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  for (let i = 0; i < 12; i++) {
    h.handlers.tool_execution_start({ toolCallId: `k${i}`, toolName: 'bash',
      args: { command: `python -c "print(timeit.timeit(lambda: x, number=${10000 + i}))"` } }, h.ctx);
    h.handlers.tool_execution_end({ toolCallId: `k${i}`, toolName: 'bash',
      result: '0.12', isError: false }, h.ctx);
  }
  await flush();
  process.env.MOONBOW_GUARD_MAX_TOOL_EVENTS = '';
  // 封顶：最后一次审计载荷只保留最近 10 条（丢最旧）。24 事件
  // （k0-call/result … k11-call/result）→ 保留 k7…k11 双相。
  const last = h.calls.at(-1);
  assert.equal(last.tool_events.length, 10, `封顶后应保留 10 条，实际 ${last.tool_events.length}`);
  assert.equal(last.tool_events[0].tool_call_id, 'k7', '丢的是最旧（k0–k6 被丢弃）');
  assert.equal(last.tool_events.at(-1).tool_call_id, 'k11');
  // 纯工具增量（期间无新 text/thinking 块）也要触发审计——877 循环形态
  assert.ok(h.calls.length >= 2, '纯工具事件增量必须触发后续审计');
});

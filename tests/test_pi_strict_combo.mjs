// strict × 过程审计组合兼容测试（MOONBOW_GUARD_MODE=strict 必须在动态 import
// 之前设置 —— progress-guard.ts 的 MODE 是模块级常量）。
//
// 兼容规则（与 README/DELIVERY_GUIDE 定义一致）：
// R1 过程审计行为只由 MOONBOW_GUARD_PROCESS 决定，与 MOONBOW_GUARD_MODE 无关；
// R2 过程提醒只计 pstate.budgets.interventions，不计 strict 收尾防循环上限
//    （task.interventions >= 2）；
// R3 语义预算（每任务一次）双向共享，两种模式均生效；
// R4 shadow 在 strict 下同样只记录不注入。
import test from 'node:test';
import assert from 'node:assert/strict';

process.env.MOONBOW_GUARD_MODE = 'strict';
const { default: activate } = await import('../src/moonbow/guard/extensions/progress-guard.ts');

const snap = (content) => ({ role: 'assistant', content, timestamp: 1700000000000 });
const tick = () => new Promise((r) => setImmediate(r));
const flush = async () => { for (let i = 0; i < 6; i++) await tick(); };

function harness(t, entries = []) {
  const handlers = {}, sent = [], stageCalls = [], finalCalls = [];
  const pi = {
    on(name, fn) { handlers[name] = fn; },
    appendEntry(customType, data) { entries.push({ type: 'custom', customType, data: structuredClone(data) }); },
    sendMessage(message, options) { sent.push({ message, options }); },
  };
  const ctx = { sessionManager: { getBranch: () => entries }, hasPendingMessages: () => false,
    ui: { setStatus: () => {} } };
  let stageResp = { findings: [], reminder: null, semantic: false };
  let finalResp = { decision: 'CLARIFY', acceptance: 'disputed', allow_stop: false,
                    review_requested: true, prompt: 'strict clarify' };
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    const body = JSON.parse(options.body);
    if (body.blocks) { stageCalls.push(body); return { ok: true, json: async () => stageResp }; }
    finalCalls.push(body);
    return { ok: true, json: async () => finalResp };
  });
  activate(pi);
  handlers.session_start({}, ctx);
  return { handlers, sent, stageCalls, finalCalls, entries, ctx,
    setStage: (r) => { stageResp = r; }, setFinal: (r) => { finalResp = r; },
    input: (t2 = 'task') => handlers.input({ text: t2, source: 'interactive' }) };
}

const reminder = (fp, semantic = true) => ({
  findings: [], semantic,
  reminder: { fingerprint: fp, summary: '验证时序不符', evidence: ['第3块'],
              suggestion: '【进度守卫（过程核查）】请补验证或如实修正。' },
});

test('strict × advisory: process reminder works, strict final cap unaffected', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'advisory';
  const h = harness(t);
  h.setStage(reminder('fp-s1', true));
  h.input();
  // 过程语义提醒照常投递（R1：strict 不影响过程审计）
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  assert.deepEqual(h.sent[0].options, { deliverAs: 'steer' });

  // strict 收尾：语义介入按自身规则投递两次（rounds 1/2），第三次触上限。
  // 过程提醒已置 task.semanticUsed=true，但 strict 收尾不读该字段 —— 不受影响（R2/R3）。
  const turn = () => h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'end_turn',
    content: [{ type: 'text', text: 'STATUS: A\nREMAINING: 无\nEVIDENCE: evidence text' }] } }, h.ctx);
  await turn(); await flush();
  await turn(); await flush();
  await turn(); await flush();
  const followUps = h.sent.filter(s => s.message.customType === 'progress-guard:feedback');
  assert.equal(followUps.length, 2);            // strict 上限=2，过程提醒未占用它
  assert.equal(h.finalCalls.length, 3);

  // 过程状态与收尾状态独立记账（R2）：process.interventions=1（仅过程），
  // task.interventions=2（仅收尾）
  const p = h.entries.filter(e => e.customType === 'progress-guard:process').at(-1).data;
  assert.equal(p.budgets.interventions, 1);
  assert.equal(p.budgets.processReminders, 1);
  const st = h.entries.filter(e => e.customType === 'progress-guard:state').at(-1).data;
  assert.equal(st.interventions, 2);

  // 收尾上限耗尽后，过程审计照常工作（非语义提醒仍投递）
  h.setStage(reminder('fp-s2', false));
  const msg2 = snap([{ type: 'text', text: '补充汇报：另一部分也完成了' }]);
  h.handlers.message_update({ message: msg2, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  const steers = h.sent.filter(s => s.message.customType === 'progress-guard:process-feedback');
  assert.equal(steers.length, 2);
});

test('strict × shadow: records without injection; strict final still delivers', async (t) => {
  process.env.MOONBOW_GUARD_PROCESS = 'shadow';
  const h = harness(t);
  h.setStage(reminder('fp-sh', true));
  h.input();
  const msg = snap([{ type: 'text', text: '已经修复完成，测试通过' }]);
  h.handlers.message_update({ message: msg, assistantMessageEvent: { type: 'text_end' } }, h.ctx);
  await flush();
  assert.ok(h.stageCalls.length >= 1);          // 过程审计照常（R4）
  assert.equal(h.sent.filter(s => s.message.customType === 'progress-guard:process-feedback').length, 0);
  assert.ok(h.entries.some(e => e.customType === 'progress-guard:shadow')); // 只记录
  // strict 收尾照常
  await h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'end_turn',
    content: [{ type: 'text', text: 'STATUS: A\nREMAINING: 无\nEVIDENCE: evidence text' }] } }, h.ctx);
  await flush();
  assert.equal(h.sent.filter(s => s.message.customType === 'progress-guard:feedback').length, 1);
});

// strict × off 组合由现有 tests/test_pi_advisory.mjs 覆盖（off 模式零过程行为 +
// 收尾回归），此处不重复。

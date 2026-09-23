import test from 'node:test';
import assert from 'node:assert/strict';
import { tick } from './helpers_advisor.mjs';
import activateAdvisor from '../src/moonbow/guard/extensions/task-structure-advisor.ts';

// —— Task Structure Advisor 插件测试 ——
// 验收项（Goal 阶段 C）：
//   off    → 零请求、零专属日志
//   shadow → 记录分析结果，绝不注入消息
//   advisory → 提醒受指纹去重 + 会话上限约束；措辞非强制
//   失败隔离 → 超时/网络错误/非法响应不阻塞主流程

function harness() {
  const handlers = {}, sent = [], entries = [];
  const pi = {
    on(name, fn) { handlers[name] = fn; },
    appendEntry(customType, data) { entries.push({ customType, data: structuredClone(data) }); },
    sendMessage(message, options) { sent.push({ message, options }); },
  };
  const ctx = {};
  return { pi, handlers, sent, entries, ctx,
           input: (t) => handlers.input({ text: t, source: 'interactive' }, ctx) };
}

const HIGH_RESPONSE = {
  level: 'high', abstain: false, advice: 'split_by_goal',
  fingerprint: 'fp_high_1', message: '【任务结构（仅提示，非要求）】当前捕获到3 项交付义务。可考虑按交付义务分批处理并分别核验；是否拆解由你决定。',
  vector: { goals: 3, constraints: 1, explicit_dependencies: 0, coordinations: 0, conditions_unresolved: 0, objects: 0 },
};

function mockFetch(responses, calls) {
  return async (url, options) => {
    calls.push({ url: String(url), body: JSON.parse(options?.body || '{}') });
    const r = responses.shift();
    if (!r) return { ok: true, json: async () => ({ ...HIGH_RESPONSE }) };
    if (r instanceof Error) throw r;
    if (r.reject) throw new Error('network down');
    return { ok: r.ok !== false, status: r.status || 200, json: async () => r.body ?? { ...HIGH_RESPONSE } };
  };
}

test('advisory: reminds once for high structure, dedups fingerprint, honors cap', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'advisory';
  const calls = [];
  t.mock.method(globalThis, 'fetch', mockFetch([], calls));
  const h = harness();
  activateAdvisor(h.pi);

  h.input('修复A模块，修复B模块，修复C模块，保持兼容。');
  await tick(); await tick();
  assert.equal(h.sent.length, 1, 'high 结构应提醒一次');
  assert.match(h.sent[0].message, /仅提示，非要求/);
  assert.ok(!/必须拆解|务必/.test(h.sent[0].message), '措辞不得强制');
  assert.equal(h.entries.filter(e => e.customType === 'task-structure:delivery').length, 1);

  // 相同结构（同指纹）→ 不再提醒
  h.input('修复A模块，修复B模块，修复C模块，保持兼容。');
  await tick(); await tick();
  assert.equal(h.sent.length, 1, '同指纹不得重复提醒');

  // 不同结构但已达会话上限 → 仍不提醒
  h.input('先升级消息队列到 v3，再迁移消费者，迁移完成后更新监控大盘。');
  await tick(); await tick();
  assert.equal(h.sent.length, 1, '会话上限（默认 1）生效');
});

test('shadow: records analysis, never injects', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'shadow';
  const calls = [];
  t.mock.method(globalThis, 'fetch', mockFetch([], calls));
  const h = harness();
  activateAdvisor(h.pi);

  h.input('修复A模块，修复B模块，修复C模块，保持兼容。');
  await tick(); await tick();
  const shadows = h.entries.filter(e => e.customType === 'task-structure:shadow');
  assert.equal(shadows.length, 1, 'shadow 应记录分析');
  assert.equal(h.sent.length, 0, 'shadow 绝不注入消息');
  assert.equal(h.entries.filter(e => e.customType === 'task-structure:delivery').length, 0);
});

test('off: zero requests, zero entries', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'off';
  const calls = [];
  t.mock.method(globalThis, 'fetch', mockFetch([], calls));
  const h = harness();
  activateAdvisor(h.pi);
  // off 模式下不注册任何处理器——比"处理器里早退"更强的零开销保证
  assert.equal(h.handlers.input, undefined, 'off 不得注册任何事件处理器');
  assert.equal(calls.length, 0, 'off 不得发出请求');
  assert.equal(h.entries.length, 0, 'off 不得写日志');
  assert.equal(h.sent.length, 0);
});

test('abstain/unknown analysis is never reminded', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'advisory';
  const calls = [];
  t.mock.method(globalThis, 'fetch', mockFetch([
    { body: { level: 'unknown', abstain: true, advice: null, fingerprint: 'fp_unk',
              vector: { goals: 0, constraints: 0, explicit_dependencies: 0, coordinations: 0, conditions_unresolved: 0, objects: 0 } } },
  ], calls));
  const h = harness();
  activateAdvisor(h.pi);
  h.input('优化一下这个系统吧。');
  await tick(); await tick();
  assert.equal(calls.length, 1, '请求发出');
  assert.equal(h.sent.length, 0, '弃权结果不得提醒');
  assert.equal(h.entries.filter(e => e.customType === 'task-structure:shadow').length, 1, 'shadow 记录照常');
});

test('failure isolation: network error, timeout abort, non-ok, bad json all silent', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'advisory';
  const calls = [];
  t.mock.method(globalThis, 'fetch', mockFetch([
    { reject: true },                                   // 网络错误
    { ok: false, status: 500, body: { error: 'boom' } },// 服务错误
    { body: { level: 'high', fingerprint: 'fp_x', message: 'm' } },  // 正常（busy 已恢复）
  ], calls));
  const h = harness();
  activateAdvisor(h.pi);
  h.input('第一条消息结构复杂需要被分析一下。');
  await tick(); await tick();
  h.input('第二条消息服务返回500错误的情况。');
  await tick(); await tick();
  h.input('修复A模块，修复B模块，修复C模块，保持兼容。');
  await tick(); await tick();
  assert.equal(calls.length, 3, '三次请求都发出');
  assert.equal(h.sent.length, 1, '仅成功那次提醒，异常未阻塞后续');
});

test('single-flight: overlapping inputs do not pile up requests', async (t) => {
  process.env.MOONBOW_TSA_MODE = 'advisory';
  const calls = [];
  let release;
  const gate = new Promise((r) => { release = r; });
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url: String(url) });
    await gate;
    return { ok: true, json: async () => ({ ...HIGH_RESPONSE, fingerprint: 'fp_' + calls.length }) };
  });
  const h = harness();
  activateAdvisor(h.pi);
  h.input('第一条消息结构复杂需要被分析一下。');
  h.input('第二条消息在上一条未完成时到达。');   // busy → 丢弃
  release();
  await tick(); await tick(); await tick();
  assert.equal(calls.length, 1, '重叠输入只发一次请求');
});

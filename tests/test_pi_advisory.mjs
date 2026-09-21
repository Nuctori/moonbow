import test from 'node:test';
import assert from 'node:assert/strict';
import activate from '../src/moonbow/guard/extensions/progress-guard.ts';

function harness(entries = []) {
  const handlers = {}, sent = [], calls = [], statuses = [];
  const pi = {
    on(name, fn) { handlers[name] = fn; },
    appendEntry(customType, data) { entries.push({ type: 'custom', customType, data: structuredClone(data) }); },
    sendMessage(message, options) { sent.push({ message, options }); },
  };
  const ctx = { sessionManager: { getBranch: () => entries }, hasPendingMessages: () => false,
    ui: { setStatus: (...args) => statuses.push(args) } };
  activate(pi);
  handlers.session_start({}, ctx);
  const turn = (content = [{ type: 'text', text: 'STATUS: A\nREMAINING: 无\nEVIDENCE: tests passed' }]) =>
    handlers.turn_end({ message: { role: 'assistant', content, usage: { totalTokens: 20 } } }, ctx);
  return { handlers, sent, calls, statuses, entries, ctx, turn };
}

const semantic = { decision: 'CLARIFY', acceptance: 'disputed', allow_stop: false, review_requested: true, prompt: 'review once' };

test('one semantic follow-up survives reload and resets on genuine input', async (t) => {
  const h = harness();
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => semantic };
  });
  h.handlers.input({ text: 'task', source: 'interactive' });
  await h.turn();
  assert.equal(h.sent.length, 1);
  assert.deepEqual(h.sent[0].options, { deliverAs: 'followUp', triggerTurn: true });
  h.handlers.input({ text: 'feedback', source: 'extension' });
  await h.turn();
  assert.equal(h.sent.length, 1);
  assert.equal(h.calls.at(-1).semantic_review_used, true);
  const reloaded = harness(h.entries);
  await reloaded.turn();
  assert.equal(reloaded.sent.length, 0);
  reloaded.handlers.input({ text: 'task', source: 'rpc' });
  await reloaded.turn();
  assert.equal(reloaded.sent.length, 1);
});

test('format budget is separate and tool turns do not trigger checks', async (t) => {
  const h = harness();
  let response = { ...semantic, decision: 'REQUIRE_MANIFEST', acceptance: 'invalid', review_requested: false };
  t.mock.method(globalThis, 'fetch', async (_, options) => {
    h.calls.push(JSON.parse(options.body));
    return { ok: true, json: async () => response };
  });
  h.handlers.input({ text: 'task', source: 'interactive' });
  await h.turn([{ type: 'toolCall', id: '1' }]);
  assert.equal(h.calls.length, 0);
  await h.turn();
  await h.turn();
  assert.equal(h.sent.length, 1);
  response = semantic;
  await h.turn();
  assert.equal(h.sent.length, 2);
  assert.equal(h.calls.at(-1).semantic_review_used, false);
  await h.turn();
  assert.equal(h.sent.length, 2);
});

test('incomplete/unverified outcomes stop without new prompt or successful acceptance', async (t) => {
  const h = harness();
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, json: async () => ({
    decision: 'BLOCK', acceptance: 'incomplete', allow_stop: true, review_requested: false,
  }) }));
  h.handlers.input({ text: 'task', source: 'interactive' });
  await h.turn();
  assert.equal(h.sent.length, 0);
  assert.equal(h.entries.at(-1).data.acceptance, 'incomplete');
});

test('stale response cannot intervene in a new task', async (t) => {
  const h = harness();
  let resolve;
  t.mock.method(globalThis, 'fetch', () => new Promise(r => { resolve = r; }));
  h.handlers.input({ text: 'old', source: 'interactive' });
  const pending = h.turn();
  const resolveOld = resolve;
  h.handlers.input({ text: 'new', source: 'interactive' });
  const newPending = h.turn();
  const resolveNew = resolve;
  assert.notEqual(resolveOld, resolveNew);
  resolveOld({ ok: true, json: async () => semantic });
  await pending;
  assert.equal(h.sent.length, 0);
  resolveNew({ ok: true, json: async () => semantic });
  await newPending;
  assert.equal(h.sent.length, 1);
  assert.equal(h.entries.findLast(e => e.customType === 'progress-guard:state').data.req, 'new');
});

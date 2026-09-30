import test from 'node:test';
import assert from 'node:assert/strict';
import { scanPseudoCallText, pseudoCallFeedbackEnabled, PSEUDO_CALL_FEEDBACK_TEXT,
         PSEUDO_CALL_FEEDBACK_BUDGET } from '../src/moonbow/guard/extensions/pseudo-call-guard.ts';
import activate from '../src/moonbow/guard/extensions/progress-guard.ts';

// —— 真实形态样本（Phase 2 会话脱敏：路径改为 ws/runX/app.py）——

// 形态 1：叙述前缀 + 单条 bracket 伪 edit，JSON 体单行紧凑（27/35，编辑被吞主形态）
const FORM_PREFIX_EDIT = 'Two bugs: `half` uses floor division and `add_tag` has a mutable ' +
  'default argument. Fixing both.\n[tool_call call_dcf8f00e] edit {"path":"ws/run01/app.py",' +
  '"edits":[{"oldText":"    return n // 2","newText":"    return n / 2"}]}';
// 形态 2：纯伪调用消息（无叙述），整条即一个 bracket 伪 bash（8/35）
const FORM_BARE_BASH = '[tool_call call_5a85619d] bash {"command":"python -m pytest test_app.py -v"}';
// 形态 3：单消息多伪调用（换行分隔或无分隔直接相连，8/35）
const FORM_MULTI_CALLS = 'Both files read. Running pytest to confirm.\n' +
  '[tool_call call_a6f04b9b] bash {"command":"python -m pytest -v"}\n' +
  '[tool_call call_2c5bd588] bash {"command":"python --version"}';
const FORM_MULTI_GLUED = '先读两个文件再修复。[tool_call call_d42511a8] read {"path":"app.py"}' +
  '[tool_call call_d42511a9] read {"path":"test_app.py"}';
// 形态 4：harness 残渣变体（2026-09-22 历史实测样本）
const FORM_RESIDUE = '[tool_call call_3f37886c] read {"path":"src/flask/blueprints.py"}</arg_value>';
// 形态 5：XML 标签形态（本批未出现，其他 harness 常见，防御性覆盖）
const FORM_XML = 'Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"command": "pytest -q"}}\n</tool_call>';
// 变体：行首工具名 + 多行 JSON 体（skeleton 签名覆盖）
const FORM_SKELETON_MULTILINE = 'Fixing now.\nedit {\n  "path": "app.py",\n  "edits": []\n}';

// —— 事件流模拟工具（风格照 test_pi_process.mjs）——

function snap(content, extra = {}) {
  return { role: 'assistant', content, timestamp: 1700000000000, ...extra };
}

function harness(entries = [], opts = {}) {
  const handlers = {}, sent = [], statuses = [];
  let failAttempts = 0;
  const pi = {
    on(name, fn) { handlers[name] = fn; },
    appendEntry(customType, data) { entries.push({ type: 'custom', customType, data: structuredClone(data) }); },
    sendMessage(message, options) {
      if (opts.failSendCount && failAttempts < opts.failSendCount) {
        failAttempts++;
        throw new Error('send channel down');
      }
      sent.push({ message, options });
    },
  };
  const ctx = { sessionManager: { getBranch: () => entries },
    hasPendingMessages: () => !!opts.pending,
    ui: { setStatus: (...args) => statuses.push(args) } };
  activate(pi);
  handlers.session_start({}, ctx);
  return { handlers, sent, statuses, entries, ctx,
           input: (t = 'task') => handlers.input({ text: t, source: 'interactive' }) };
}

const tick = () => new Promise((r) => setImmediate(r));
const flush = async () => { for (let i = 0; i < 4; i++) await tick(); };

function pseudoEntries(h) {
  return h.entries.filter(e => e.customType === 'progress-guard:pseudo-call');
}

// —— 1. 签名扫描：每种真实形态命中 ——

test('signature: bracket form with narrative prefix and compact JSON hits', () => {
  const hit = scanPseudoCallText(FORM_PREFIX_EDIT);
  assert.ok(hit, '主形态必须命中');
  assert.equal(hit.form, 'bracket');
  assert.equal(hit.count, 1);
});

test('signature: bare pseudo bash message hits', () => {
  const hit = scanPseudoCallText(FORM_BARE_BASH);
  assert.ok(hit);
  assert.equal(hit.form, 'bracket');
});

test('signature: multiple calls in one message counted (newline and glued)', () => {
  assert.equal(scanPseudoCallText(FORM_MULTI_CALLS).count, 2);
  assert.equal(scanPseudoCallText(FORM_MULTI_GLUED).count, 2);
});

test('signature: harness residue variant hits', () => {
  const hit = scanPseudoCallText(FORM_RESIDUE);
  assert.ok(hit);
  assert.equal(hit.form, 'bracket');   // 残渣与 bracket 并存时以主签名为准
});

test('signature: xml tool_call tag form hits defensively', () => {
  const hit = scanPseudoCallText(FORM_XML);
  assert.ok(hit);
  assert.equal(hit.form, 'xml');
});

test('signature: line-leading tool name with multiline JSON body hits', () => {
  const hit = scanPseudoCallText(FORM_SKELETON_MULTILINE);
  assert.ok(hit);
  assert.equal(hit.form, 'skeleton');
});

test('signature: normal prose and completion claims never hit', () => {
  assert.equal(scanPseudoCallText('我已经修复了登录问题，测试全部通过。'), null);
  assert.equal(scanPseudoCallText('STATUS: A\nEVIDENCE: 3 passed'), null);
  assert.equal(scanPseudoCallText('The judge regex matches the word tool_call in text.'), null,
    '裸词 tool_call 无签名结构，不得命中');
  assert.equal(scanPseudoCallText(''), null);
});

// —— 2. 集成：命中 → 固定措辞 followUp 投递 + 遥测 ——

test('pseudo-only message gets fixed-wording followUp feedback and telemetry', async () => {
  const h = harness();
  h.input();
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_PREFIX_EDIT }],
    { stopReason: 'stop' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  assert.equal(h.sent[0].message.customType, 'progress-guard:pseudo-call-feedback');
  assert.equal(h.sent[0].message.content, PSEUDO_CALL_FEEDBACK_TEXT);
  assert.deepEqual(h.sent[0].options, { deliverAs: 'followUp', triggerTurn: true });
  const te = pseudoEntries(h);
  assert.equal(te.length, 1);
  assert.equal(te[0].data.action, 'feedback');
  assert.equal(te[0].data.form, 'bracket');
  assert.equal(te[0].data.accompaniedReal, false);
  assert.equal(te[0].data.budgetUsed, 1);
});

test('message with accompanying real toolCall is never reported (mixed form)', async () => {
  const h = harness();
  h.input();
  // Phase 2 实测混合消息：真实 toolCall 在场 + 文本里还有伪调用（stop=toolUse）
  h.handlers.message_end({ message: snap([
    { type: 'text', text: 'Test failure found.\n[tool_call call_a2b56b4d] read {"path":"ws/run01/app.py"}' },
    { type: 'toolCall', name: 'read', id: 'call_real1', arguments: { path: 'ws/run01/app.py' } },
  ], { stopReason: 'toolUse' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0, '伴随真实调用的消息不得投递');
  const te = pseudoEntries(h);
  assert.equal(te.length, 1);
  assert.equal(te[0].data.action, 'observed-only');
  assert.equal(te[0].data.accompaniedReal, true);
});

test('no pseudo call means no feedback and no telemetry', async () => {
  const h = harness();
  h.input();
  h.handlers.message_end({ message: snap([{ type: 'text', text: '全部通过，任务完成。' }],
    { stopReason: 'stop' }) }, h.ctx);
  h.handlers.message_end({ message: snap([{ type: 'thinking', thinking: '想一想' }],
    { stopReason: 'stop' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0);
  assert.equal(pseudoEntries(h).length, 0);
});

test('error/aborted messages are not scanned', async () => {
  const h = harness();
  h.input();
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'aborted' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0);
});

test('same message redelivered as different objects fires once', async () => {
  const h = harness();
  h.input();
  const rid = 'resp-pseudo-1';
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: rid }) }, h.ctx);
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: rid }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1, '同一 responseId 重复投递只反馈一次');
});

// —— 3. 预算：每任务 3 次封顶，超预算静默（仍记遥测）——

test('budget caps feedback at 3 per task; over-budget is silent but recorded', async () => {
  const h = harness();
  h.input();
  for (let i = 0; i < 5; i++) {
    h.handlers.message_end({ message: snap([{ type: 'text',
      text: `[tool_call call_deadbe0${i}] bash {"command":"pytest -q round${i}"}` }],
      { stopReason: 'stop', responseId: `r-${i}` }) }, h.ctx);
  }
  await flush();
  assert.equal(h.sent.length, PSEUDO_CALL_FEEDBACK_BUDGET, '只投递预算内次数');
  const actions = pseudoEntries(h).map(e => e.data.action);
  assert.equal(actions.filter(a => a === 'feedback').length, 3);
  assert.equal(actions.filter(a => a === 'budget-exhausted').length, 2);
  assert.deepEqual(pseudoEntries(h).slice(-1)[0].data.budgetUsed, 3);
});

test('new task resets the budget', async () => {
  const h = harness();
  h.input('task-1');
  for (let i = 0; i < 3; i++) {
    h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
      { stopReason: 'stop', responseId: `t1-${i}` }) }, h.ctx);
  }
  await flush();
  assert.equal(h.sent.length, 3);
  h.input('task-2');                          // 新任务 → 预算重置
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: 't2-0' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 4, '新任务应重新获得预算');
});

test('budget survives restart via telemetry ledger', async () => {
  const entries = [];
  {
    const h = harness(entries);
    h.input();
    h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
      { stopReason: 'stop', responseId: 'restart-a' }) }, h.ctx);
    await flush();
    assert.equal(h.sent.length, 1);
  }
  const h2 = harness(entries);                // 用同一会话账本重建（模拟重启）
  h2.handlers.session_start({}, h2.ctx);
  h2.input();
  h2.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: 'restart-b' }) }, h2.ctx);
  await flush();
  // 新任务预算从 0 起算（账本按 taskId 记账）——新任务第一次照常投递
  assert.equal(h2.sent.length, 1);
  // 同任务内继续：再补 2 次到 3 次封顶
  for (let i = 0; i < 2; i++) {
    h2.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
      { stopReason: 'stop', responseId: `restart-c${i}` }) }, h2.ctx);
  }
  await flush();
  assert.equal(h2.sent.length, 3);
  const fourth = snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: 'restart-d' });
  h2.handlers.message_end({ message: fourth }, h2.ctx);
  await flush();
  assert.equal(h2.sent.length, 3, '跨重启预算累计封顶');
});

// —— 4. 杀开关 ——

test('MOONBOW_PSEUDO_CALL_FEEDBACK=off disables everything', async () => {
  process.env.MOONBOW_PSEUDO_CALL_FEEDBACK = 'off';
  try {
    assert.equal(pseudoCallFeedbackEnabled(), false);
    const h = harness();
    h.input();
    h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_PREFIX_EDIT }],
      { stopReason: 'stop' }) }, h.ctx);
    await flush();
    assert.equal(h.sent.length, 0);
    assert.equal(pseudoEntries(h).length, 0, 'off 时零记录');
  } finally {
    delete process.env.MOONBOW_PSEUDO_CALL_FEEDBACK;
  }
  assert.equal(pseudoCallFeedbackEnabled(), true, '默认 on');
});

// —— 5. 独立性：过程审计 off 时伪调用守卫仍工作 ——

test('guard works while MOONBOW_GUARD_PROCESS=off', async () => {
  process.env.MOONBOW_GUARD_PROCESS = 'off';
  try {
    const h = harness();
    h.input();
    h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
      { stopReason: 'stop' }) }, h.ctx);
    await flush();
    assert.equal(h.sent.length, 1, '伪调用反馈不依赖过程审计开关');
  } finally {
    delete process.env.MOONBOW_GUARD_PROCESS;
  }
});

// —— 6. 失败隔离：投递通道故障不炸宿主扩展 ——

test('send failures are swallowed; guard and host extension keep working', async (t) => {
  const checked = [];
  const h = harness([], { failSendCount: 1 }); // 第一次 sendMessage 抛错
  t.mock.method(globalThis, 'fetch', async (url) => {
    if (String(url).endsWith('/check')) {
      checked.push(1);
      return { ok: true, json: async () => ({ decision: 'CLOSE', allow_stop: true,
        acceptance: 'closed', review_requested: false }) };
    }
    return { ok: true, json: async () => ({ findings: [], reminder: null }) };
  });
  h.input();
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: 'fail-1' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0, '投递失败不产生半途状态');
  // 守卫存活：第二条伪调用仍尝试投递（且成功）
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_PREFIX_EDIT }],
    { stopReason: 'stop', responseId: 'fail-2' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 1, '异常后守卫继续工作');
  // 宿主扩展其他路径不受影响：收尾裁决照常送达
  await h.handlers.turn_end({ message: { role: 'assistant', stopReason: 'end_turn',
    content: [{ type: 'text', text: '任务完成，测试通过。' }] } }, h.ctx);
  await flush();
  assert.equal(checked.length, 1, 'progress-guard 收尾检查不受伪调用守卫异常影响');
});

test('pending user input suppresses delivery without consuming budget', async () => {
  const h = harness([], { pending: true });
  h.input();
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_BARE_BASH }],
    { stopReason: 'stop', responseId: 'pend-1' }) }, h.ctx);
  await flush();
  assert.equal(h.sent.length, 0);
  const te = pseudoEntries(h);
  assert.equal(te[0].data.action, 'suppressed-pending');
  // 输入空闲后同一预算周期内仍可投递 3 次
  const h2ctx = { ...h.ctx, hasPendingMessages: () => false };
  h.handlers.message_end({ message: snap([{ type: 'text', text: FORM_PREFIX_EDIT }],
    { stopReason: 'stop', responseId: 'pend-2' }) }, h2ctx);
  await flush();
  assert.equal(h.sent.length, 1);
  assert.equal(pseudoEntries(h).at(-1).data.budgetUsed, 1, '抑制不消耗预算');
});

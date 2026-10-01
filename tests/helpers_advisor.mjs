// 测试辅助：让微任务队列 flush（与 test_pi_process.mjs 的 flush 等价）。
export const tick = () => new Promise((r) => setImmediate(r));

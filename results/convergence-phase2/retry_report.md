
## Phase 3 重试中止记录（2026-09-30）
全新 A/B（含 PseudoCallGuard）30 run 全部秒挂：网关上游返回 400
"insufficient credits"（Command Code 账户额度耗尽）。非代码问题。
无效数据归档为 runs_p3_invalid_credits.jsonl；充值后重跑：
  python -u experiments/convergence_retry.py --runs 15
（runner 断点续跑，服务已在线：guard 18617 advisory / 网关 8899）

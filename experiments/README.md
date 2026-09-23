# Experiments — 守卫效果受控实验

本目录收录 Progress Guard（守卫）在**真实 Agent harness** 上的受控效果实验、
方法论与原始口径说明。研究结论的可读版本见
[GUARD_EFFECT_REPORT.md](GUARD_EFFECT_REPORT.md)；这里是复现入口。

> 与 `maps/` 的分工：`maps/` 是**离线契约/模型研究**（v1~v6 契约演进、
> 槽模型、成对站）；本目录是**在真实 Pi harness 上的在线介入实验**。

---

## 1. 一分钟结论

| 问题 | 结论 | 证据 |
|---|---|---|
| 守卫能否让模型多解出任务（uplift）？ | **未证实**：c3 复合陷阱上 uplift = −20 点（Fisher p=0.65，不显著） | 报告 §2.0b |
| 守卫的直接效应是否存在？ | **是**：申报协议采纳率 0% → 54% | 报告 §2.0.1 |
| 会不会干扰正常任务？ | **否**：S2 干净任务零投递、零误报 | 报告 §2.0 |
| 为什么测不出 uplift？ | 任务难度错配：太易（S1 100%）或太难（SWE-bench ≈0）时 uplift 恒为 0 | 报告 §7.1 |

**核心方法论发现**：uplift 的可测区间 = baseline 完成率落在 **30%~70%** 的
任务难度带。实验设计第一步是把任务难度校准到这个带内，而不是换模型。

---

## 2. 文件清单

| 文件 | 作用 |
|---|---|
| [GUARD_EFFECT_REPORT.md](GUARD_EFFECT_REPORT.md) | **主报告**：设计、结果、10 处缺陷修复、模型适配、威胁与局限 |
| [trap_ladder.py](trap_ladder.py) | 陷阱阶梯校准：找出模型落入 30-70% 带的任务难度 |
| [compound_traps.json](compound_traps.json) | 复合陷阱配置（c1~c4：2-3 个 bug 组合，跨文件） |
| [test_guard_scenarios.py](test_guard_scenarios.py) | 场景 AB 驱动器：S1 / S1-Hard / S2 × baseline / with_guard |
| [cc_proxy_server.py](cc_proxy_server.py) | 模型后端适配网关（把私有契约翻译成 OpenAI 兼容） |
| [ARCHIVE_README.md](ARCHIVE_README.md) | SWE-bench Lite AB 原始记录（`results/*.jsonl`，未随仓库发布）的读法与有效性判据 |

---

## 3. 复现步骤

### 3.1 环境准备

```bash
pip install -e .
moonbow guard serve --port 18492          # 守卫服务
python -u experiments/cc_proxy_server.py  # 模型适配网关（可选，见下）
```

工作目录、agent 配置目录、网关地址均可用环境变量覆盖，默认落在系统临时目录，
不会污染仓库：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PI_EVAL_AGENT_DIR` | `./home` | pi agent 配置目录（models.json / sessions） |
| `PI_EVAL_WORK_DIR` | `<tmp>/guard_scen`、`<tmp>/trap_ladder` | 实验工作区根目录 |
| `PI_GUARD_URL` | `http://127.0.0.1:18492` | 守卫服务地址 |
| `PI_GATEWAY` | `http://127.0.0.1:3001/api/ping` | 模型网关健康检查 |
| `PI_GUARD_EXT` | 仓库内 `src/moonbow/guard/extensions/progress-guard.ts` | 守卫扩展入口；设 `legacy` 回退简化版 |

实验需要 **Pi harness 与一个可用的模型后端**（本地权重或 OpenAI 兼容网关）。
模型后端不是本仓库的交付物，`cc_proxy_server.py` 仅为适配某家私有契约的
参考实现。

### 3.2 第一步：校准任务难度（必须先做）

```bash
python -u experiments/trap_ladder.py --rounds 2 --model <model-id>
```

输出每类陷阱的首次通过率。**只有落在 30-70% 的陷阱类**才适合测 uplift；
全部命中带外说明该模型对这个任务难度没有可测的 uplift 空间（报告 §7.1）。

### 3.3 第二步：跑 AB 对照

```bash
PI_GUARD_PROCESS=advisory python -u experiments/test_guard_scenarios.py \
    --model <model-id> --thinking off --rounds 5
```

每轮为每个场景跑 `baseline`（无插件）与 `with_guard`（正式插件）两臂，
独立工作区，输出完成率对照与守卫侧指标（findings / actionable / 投递 / observed）。

**守卫开关**：`MOONBOW_GUARD_PROCESS` = `off`（默认，零写入零请求）/
`shadow`（只记录不投递）/ `advisory`（投递提醒）。回滚即设回 `off`。

---

## 4. 口径与有效性判据（重要）

实验数据最容易出错的地方是**把环境故障当成模型行为**。本目录记录的核心
教训（报告 §2.0a）：

- 上游容量型限流可能以 **HTTP 200 + `{"type":"error"}` 事件**返回；
  适配层若不处理该事件，故障会被**伪装成模型主动收尾**（`stop=stop`，无 error）。
- 因此每条记录必须核验：`resolved`（None = 无效）、`arm_errored`、
  `model_polluted`（会话内静默 failover）、`ended_error_storm`。
  完整字段表见 [ARCHIVE_README.md](ARCHIVE_README.md)。
- **主指标是任务完成率**（客观可判），不是"守卫介入次数"——后者曾被证明
  可以完全误导（9 次介入仅 5 次引发后续工作、0 次促成解题）。

---

## 5. 已知边界

- 实验结论限于所测模型（`xiaomi/mimo-v2.5`）与所测任务；换模型需重跑。
- 真实权重 / 外部 LLM 依赖本地或第三方服务，**不在 CI 覆盖范围内**。
- 定时/夜间批次的脚本（`overnight_loop.py` 等）依赖特定私有环境，未随本仓库发布。

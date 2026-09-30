# Moonbow 语义匹配运行时：使用者 API 文档

> 版本：v0.1.0（P11 阶段交付）
> 适用：moonbow 包 ≥0.1.0（含 `moonbow.semantic.*`）
> 契约文件：`src/moonbow/semantic/schema.py`（P1 冻结）；计划：`docs/semantic_match_runtime_plan.md`

## 0. 能力状态（必读，如实声明）

- **统一接口 / 运行时 / 兼容层：已验证**（契约 97 例、运行时 25 例、HTTP 10 例、
  Guard provider 41 例等自动测试全绿，见 `docs/semantic_runtime_progress.md` 证据索引）。
- **SLM 后端（Qwen3.5-0.8B）：离线判定延迟已达标（XPU bf16 p50≈126ms），
  但语义质量未通过校准门禁**——precision 不达标（process.unresolved 的 P≥0.95 全表
  不可达），校准 **4/4 calibrated=false**。**semantic 路径默认关闭，不得用于生产
  提醒路径**；仅作为 opt-in 后端用于试验与评测。
  - **R3 更新（2026-09-30）**：LoRA 微调后 **2/4 达标**（process.unresolved、
    task.object.alignment，calib P/R 双过线）；completion.asserted 差线
    no-go（待补 dev 正例）；modality.assertive no-go。仍为 opt-in，
    达标状态逐信号声明（见 §7.1），test split 未跑、默认 provider 未切换。
- `ts.capture` 结构抽取保持 rule 路径，SLM 捕获通道已确诊不可用并关闭。

## 1. 核心概念

- **Pattern**：固定版本的模式（如 `completion.asserted@1`），精确版本查找，不静默回退。
- **MatchRequest**：`pattern` 或自由 `requirement`（二选一）+ `text`（≤65536 code
  points）+ 可选 `context` + 可选 `threshold ∈ [0,1]`。
- **MatchResponse**：三态 `status`（ok / abstain / error）、`matched`
  （true/false/null；abstain 与 error 必为 null）、`score`（有限 [0,1] 或 null）、
  `evidence`（引文偏移为半开区间 code point，引文必须逐字）、`provenance`
  （pattern_version 回指请求）。

内置 pattern：`completion.asserted@1`、`modality.assertive@1`、
`task.object.alignment@1`（需 `context.task`）、`process.unresolved@1`、
`ts.capture@1`（find_all）。

## 2. Python SDK

```python
from moonbow.semantic.backends.fake import FakeBackend
from moonbow.semantic.matcher import SemanticMatcher

backend = FakeBackend(model_revision="fake-rev-1", rules={
    "completion.asserted@1": {"matched": True, "score": 0.9,
                              "quotes": ["修好了"], "relation": "supports"},
})
matcher = SemanticMatcher.shared(backend=backend)   # 同键共享同一 runtime，只加载一次

r = matcher.match(pattern="completion.asserted@1", text="应该修好了。")
# r.status == "ok"; r.matched; r.score; r.evidence[0].quote / .start / .end

fa = matcher.find_all(pattern="ts.capture@1", text="……")
# fa.evidence 为多条；截断的 find_all 不得宣称全量"无捕获"（契约强制 abstain=truncated）

batch = matcher.batch(requests=[req1, req2, ...])   # 上限默认 16
# 逐项结果对齐；单项失败以契约内 error 条目标记，不丢整批
```

要点：
- `matcher` 无会话状态，会话间无串扰；同 backend 身份 + 同 runtime 配置共享单例。
- 全部出口响应经过 `validate_response` 严格校验，未通过按 `invalid_output` 处理。

## 3. HTTP 服务

启动（backend 由配置注入；slm 延迟导入 torch）：

```bash
python -m moonbow.semantic.http --config config/semantic_runtime.example.json
# 可用 --host / --port 覆盖；默认绑定 127.0.0.1
```

配置示例（`config/semantic_runtime.example.json`）：`server(host/port/max_body_bytes)`、
`runtime(queue_capacity/deadline_seconds/max_retries/max_batch/warmup)`、
`backend(type=fake|slm, ...)`；fake 规则支持 matched/score/quotes/relation、
abstain 规则；slm 示例见配置内 `_slm_example`。

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/match` | 单 pattern match（body = MatchRequest） |
| POST | `/v1/find-all` | 单 pattern find_all |
| POST | `/v1/batch` | 批量（≤16，逐项结果对齐） |
| GET | `/health` | 身份 semantic-runtime + state/queue_depth/metrics，不经过推理队列 |
| GET | `/ready` | 未 ready → 503 |
| GET | `/capabilities` | backend 身份、model_revision、pattern 清单/版本/operation/threshold、calibrated 声明、limits |

### 错误码

| 状态码 | 场景 |
|---|---|
| 400 | 非法 JSON/字段、pattern/requirement 二选一违例、operation 与端点不匹配、batch 超上限 |
| 413 | text 超 65536 code points 或请求体超 2MB |
| 404 | 未知路由（含 /check 等绝不误落） |
| 429 | 队列满（reason=overloaded） |
| 503 | runtime unavailable / 未 ready |
| 504 | 请求超时（reason=timeout） |

注意：即使配 4xx/5xx 状态码，响应体仍是契约 `MatchResponse`（status=ok/abstain/error），
客户端按契约逐字段解析。

## 4. 客户端（HTTPClientBackend）

`moonbow.semantic.client.HTTPClientBackend` 实现与本地 backend 相同的 `Backend`
协议；失败隔离为契约内 error（连接失败/5xx→unavailable、超时→timeout、
400/404/413→unsupported、429→overloaded、200 坏体→invalid_output），不抛异常穿透。
loopback 直连自动绕过系统代理。

## 5. FakeBackend（测试与联调）

- `rules`：pattern ref → 固定 matched/score/quotes/relation，或可调用函数；
  无规则的 pattern 返回确定性负例。
- 故障注入：`timeout` / `invalid_output` / `unavailable` / `overloaded`；
  支持 `delay` 与 gate 阻塞。
- 引文非原文子串时按 `invalid_output` 拒绝（不修饰成可信命中）。
- 仅用于契约/故障/集成测试；`capabilities` 标注 `fake=true`。

进程内测试重建同一 backend 的规则可用
`moonbow.semantic.http.rules_from_config`。

## 6. 把语义运行时注入 Guard（provider 方式）

```python
from moonbow.guard.verifier import ProgressGuard          # 构造函数 provider=None 为默认
from moonbow.guard.semantic_provider import SemanticProviderAdapter
from moonbow.semantic.matcher import SemanticMatcher

adapter = SemanticProviderAdapter(SemanticMatcher.shared(backend=my_backend))
guard = ProgressGuard(..., provider=adapter)              # 不传 provider 保持 legacy 行为
```

- `LegacyProvider`（默认）：包装 ModelRegistry 三调用，行为与迁移前逐行一致。
- `SemanticProviderAdapter`：每个信号一次 `matcher.match(pattern=...)`；SLM 分数只写
  新键 `slm_capture_score@1`，不写旧 similarity/capture_prob 键；任一信号失败即整体
  unavailable，verifier 进入既有骨架降级（`skeleton_only=true` +
  `semantic_unavailable=true`），不伪造通过。
- HTTP 服务端：`/check` 可选 `semantic_provider`（"legacy"|"semantic"，默认 legacy）；
  请求 semantic 而未配置运行时 → 400 明确报错，不静默回退；`GET /v1/semantic-status`
  只读 provider_kind/backend/calibrated。
- 过程审计 shadow 通道（`StageAuditor(semantic_shadow=...)`）默认关闭，仅显式注入运行，
  结果只写 `shadow_semantic` 键，不影响提醒判定。

## 7. LoRA 适配器（R4：插件侧 adapter 使用）

R3 产出的 LoRA 适配器已可经 HTTP 服务挂载（基座只加载一次，适配器热切换
~ms 级）。**达标状态如实声明，未达标模式不接线或保持底座 zero-shot。**

### 7.1 达标状态表（final-test 留出集验收后，R5 白名单口径）

| 信号 | 状态 | adapter | 依据 |
|---|---|---|---|
| completion.asserted | **PASS（test 验收通过）** | `completion-lora-v2` | test P=0.895 R=0.810 F1=0.850 |
| task.object.alignment | **PASS（test 验收通过）** | `alignment-lora-v1` | test P=0.854 R=0.972 F1=0.909 |
| process.unresolved | **shadow（R 差线，不参与判定）** | `process-lora-v1` | test P=0.977 达标、R=0.764 < 0.80（差 3.6 点） |
| modality.assertive | **no-go** | 无（底座 zero-shot，uncalibrated） | LoRA 两轮训崩；scan P=0.406 R=0.981 @t=0.70 |
| ts.capture | **no-go（维持 rule）** | — | SLM 捕获不可用（P7） |

指标来源：`results/semantic-runtime/final-test/final_acceptance.md`（provisional
标签口径）。

### 7.2 服务端配置与请求

启动（加载基座一次并注册 3 个适配器；仅登记不加载权重，首次请求时热挂载）：

```bash
.venv_xpu/Scripts/python.exe -m moonbow.semantic.http --config config/semantic_runtime_lora.json
```

配置（`config/semantic_runtime_lora.json`）：`backend.type="slm"` +
`model_kind="sequence"`（openjev NLI 底座）+ `backend.adapters`：
`{逻辑名: {path: R3 best 目录}}`。

插件/客户端在 MatchRequest 上带可选 `adapter` 逻辑名：

```json
POST /v1/match
{"text": "日志切割方案还没定，先确认当前配置在哪。",
 "pattern": "process.unresolved@1", "adapter": "process-lora-v1"}
```

- ok 响应的 `provenance.adapter` 回填请求的适配器名（调用方据此审计用的是哪个微调头）；
- **未注册 adapter → `status=error, reason_code=unsupported`**（HTTP 200 契约体），
  绝不静默回退 base；
- 不带 `adapter` 的请求按 base 出分；混有适配器流量时服务端会显式切回 base
  （无状态残留）。

### 7.3 Guard 插件侧（semantic provider 路径）

```python
from moonbow.guard.semantic_provider import (
    SemanticProviderAdapter, R4_CAPABILITY_STATUS)

adapter = SemanticProviderAdapter(
    matcher,                                  # HTTPClientBackend 门面亦可
    modality_pattern="modality.assertive@1",  # 底座 zero-shot（no-go，如实）
    signal_adapters={                         # 按信号选 adapter（请求带 adapter 参数）
        "process.unresolved": "process-lora-v1",
        "task.object.alignment": "alignment-lora-v1",
    },
    capability_status=R4_CAPABILITY_STATUS)   # /v1/semantic-status 的状态表
```

- 每个启用的信号一次 `matcher.match(pattern=..., adapter=...)`；
  task.object.alignment 自动带 `context={"task": req}`；
- `provenance.adapter` 与请求不一致时按信号失败处理（整体降级，不出半套信号）；
- 信号结果写进 /check 响应 `scores["<signal>@1"] = {matched, score, adapter}`；
- `GET /v1/semantic-status` 返回 `capabilities.signals`（上表逐信号如实声明
  pass / uncalibrated / no_go）。
- ~~默认 provider 本次仍不切换~~ **R5 已切换（配置激活，见 §9）**；切换就绪
  清单见 `results/semantic-runtime/r4-integration/e2e_report.md` §4。

## 8. 限制与告诫

- 当前所有 SLM 分数均为 **raw_score 语义**（4/4 calibrated=false），不得当概率解释，
  不得接入生产提醒路径。
- 离线权重缺失 / 非 XPU 环境下 `SLMBackend` 会显式抛 `BackendUnavailable`（真实加载
  路径强制 `torch.xpu.is_available()`，不回退 CPU）。
- 打包内容不含模型权重与会话数据；patterns JSON 为包内资源
  （`moonbow/patterns/*.json`）。

## 9. R5 默认切换与一键回滚（配置驱动，缺省 legacy 零变化）

依据：final-test 验收（`results/semantic-runtime/final-test/final_acceptance.md`
§4/§6）——completion.asserted 与 task.object.alignment PASS，process.unresolved
R=0.764 差线（仅 shadow），modality.assertive / ts.capture no-go；以及
AGENTS.md 约束 0（模型质量达标信号按白名单接入）。

### 9.1 白名单路由（`build_whitelist_adapter`）

| 信号 | adapter | 路由 |
|---|---|---|
| completion.asserted | completion-lora-v2 | **参与判定**（完全闭合通道） |
| task.object.alignment | alignment-lora-v1 | **参与判定**（客体对齐通道，自动带 context.task） |
| process.unresolved | process-lora-v1 | **仅 shadow**：结果只进 `scores["shadow_signals"]`，不参与信号判定与提醒生成，其失败也不拖垮整体 |
| modality.assertive | 无（底座 zero-shot） | 信号文案语义不变，标记 `semantic_modality_uncalibrated=true`，**不计入达标声明** |

### 9.2 部署步骤（配置激活）

1. 启动 semantic HTTP（真实推理需 `.venv_xpu`，XPU only）：

   ```bash
   .venv_xpu/Scripts/python.exe -m moonbow.semantic.http --config config/semantic_runtime_lora.json
   ```

2. 以同一环境变量启动 guard 服务（SDK 同理，进程内读同一变量）：

   ```bash
   MOONBOW_GUARD_PROVIDER=semantic python -m moonbow.guard.server
   # 可选：MOONBOW_GUARD_SEMANTIC_CONFIG 指向其他配置（缺省 config/semantic_runtime_lora.json，
   # matcher URL 取 provider.matcher_url 或 server.host/port）
   ```

3. Pi 扩展：对扩展进程设置同名 `MOONBOW_GUARD_PROVIDER=semantic`，/check payload
   会显式带 `semantic_provider: "semantic"`（未设置时不带该键，由服务端缺省决定）。
4. 观测：`GET /v1/semantic-status` 应显示 `provider_kind=semantic` 与
   `capabilities.signals` 状态表（process=shadow）。

未设置该环境变量（或值非法）时，/check 与 SDK 默认 legacy，行为与切换前
逐字段一致。

### 9.3 一键回滚

**回滚 = 移除 `MOONBOW_GUARD_PROVIDER` 环境变量 + 重启 guard 服务**（不改代码、
不删配置与权重）。回滚后 verdict 与 legacy 基线**逐字段一致**——decision /
allow_stop / acceptance / is_closed / review_requested / disputed 六个验收语义
字段不变，任务状态不丢失。证据：P10 回滚演练
`results/semantic-runtime/p9-host/rollback_drill.md`（6 例固定输入逐字段 diff=0）。
代码层另有双保险：`semantic_provider_from_config` 构建失败（配置缺失/服务未起）
自动回退 legacy 并记 warning，不阻塞裁决。

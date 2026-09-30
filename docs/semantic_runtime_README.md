# Moonbow 语义匹配运行时（Semantic Match Runtime）README

> 状态截至 2026-09-30（R5）。实施进度与全部证据：`docs/semantic_runtime_progress.md`；
> 使用者 API：`docs/semantic_api.md`；计划：`docs/semantic_match_runtime_plan.md`。

## 这是什么

为 Agent 元认知（进度守卫、任务结构抽取、过程审计）提供的统一语义匹配运行时：
固定版本 pattern + 严格契约（MatchRequest/MatchResponse）+ 单加载共享 Runtime
（有界队列、deadline、取消、指标）+ 可替换 backend（Fake / HTTP / SLM）+
Guard 兼容迁移层（provider 抽象，legacy 默认）。

## 能力状态（如实声明，不做超出验证的承诺）

| 项 | 状态 |
|---|---|
| 统一接口（P1 契约） | ✅ 已验证（97 例契约测试 + 严格校验） |
| 共享 Runtime（P2） | ✅ 已验证（单次加载/队列/deadline/取消/指标，25 例） |
| SDK/HTTP 可替换性（P5） | ✅ 已验证（SDK 与 HTTP 逐字段等价，10 例） |
| Guard 兼容迁移（P6/P7） | ✅ 已验证（legacy 默认行为逐行不变，回归全绿） |
| SLM 后端延迟（P4/P8） | ✅ 达标（XPU bf16 判定 p50≈126ms/p95≈186ms） |
| semantic 路径质量（R3/R3b/final-test） | ✅ **2 信号 test PASS 参与判定**：completion.asserted（test P=0.895 R=0.810）、task.object.alignment（test P=0.854 R=0.972）|
| semantic 路径 shadow（R5 白名单） | ⚠️ **1 shadow**：process.unresolved（test P=0.977 达标、R=0.764 差线）——仅 shadow 通道，不参与判定与提醒 |
| semantic 路径 no-go | ❌ **2 no-go**：modality.assertive（两轮 LoRA 训崩，底座 zero-shot + uncalibrated 标记）、ts.capture（走 rule） |
| semantic 路径默认状态（R5） | **配置激活**：`MOONBOW_GUARD_PROVIDER=semantic` 时 /check 与 SDK 默认走 semantic（白名单：2 参与判定 + 1 shadow + 1 no-go）；**缺省仍 legacy，未配置环境行为零变化**；一键回滚 = 移除环境变量 + 重启（P10 演练 verdict 逐字段一致） |
| ts.capture SLM 捕获 | ❌ 已确诊不可用（防御层生效但不冒充可信命中），保持 rule 路径 |
| 部署验证 / 在线观察（D/E） | 未完成（pending） |

一句话：**统一接口、运行时与兼容层已验证可用；R5 起按白名单配置激活 semantic
默认（completion.asserted 与 task.object.alignment 两个 test PASS 信号参与判定，
process.unresolved 仅 shadow，modality/ts.capture 维持 no-go），未配置环境默认
legacy 不变；不要把 shadow / no-go 信号接到判定与提醒路径上。**
不声称任务完成率提升（那需要独立在线因果实验）。

## 快速开始（纯逻辑，无需模型权重）

```bash
pip install moonbow-0.1.0-py3-none-any.whl   # 或 pip install -e .[dev]
python -m moonbow.semantic.http --config config/semantic_runtime.example.json
curl http://127.0.0.1:18500/capabilities
```

SDK / Guard 注入 / 错误码 / FakeBackend 用法：见 `docs/semantic_api.md`。

## 关键目录

- `src/moonbow/semantic/`：schema/patterns/runtime/matcher/http/client + backends/{base,fake,slm}
- `src/moonbow/patterns/*.json`：5 个内置固定模式（包内资源，随 wheel 分发）
- `src/moonbow/guard/semantic_provider.py`：Guard 迁移层（legacy 默认）
- `src/moonbow/task_structure/matcher_backend.py`：ts.capture 适配层（默认 rule）
- `config/semantic_runtime.example.json`、`config/calibration/qwen3.5-0.8b/`
- `results/semantic-runtime/`：P0–P8 全部证据（基线、probe、评测、校准报告、门禁报告）

## 已知限制

- SLM 分数均为 raw_score（未校准），不得当概率解释。
- SLMBackend 真实加载强制 XPU（`torch.xpu.is_available()` 断言），CPU 推理被显式拒绝。
- torch/transformers 为可选运行时依赖：纯逻辑导入与 fake backend 不触发 torch 导入路径。

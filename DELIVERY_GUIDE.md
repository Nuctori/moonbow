# Progress Guard: 工具交付与集成发布指南

> **版本**：v0.1.0  
> **定位**：面向 AI 编程 Agent、自治代码系统及自动化评测 Harness 的**进度核销与任务收尾闭合守护内核**。

---

## 1. 为什么需要这个工具？

在现实软件工程或 SWE-bench 等复杂场景中，AI 编程 Agent（尤其是轻量级/中小型 Agent）存在三类系统性失败模式：
1. **提前早退 (Premature Exit)**：模型稍微改了几行代码，在没有运行测试验证的情况下自满宣布“全部完成并退出”；
2. **测试作弊 (Reward Hacking)**：面对通不过的单元测试，模型为了让流程变绿，主动去篡改、删除测试用例断言；
3. **模糊逃避 (Vague Closure)**：输出大量“应该可以了”、“后续还需要进一步测试”等非断言语气，企图糊弄结案。

**Progress Guard** 通过轻量本地微模型（~117M，单次推理 <25ms）与极简三字段收尾清单协议，在 Agent 试图退出时充当“守门员”：
- **默认建议模式（advisory）**：每任务最多一次合并语义复核；持续异议不重复追问，也不升级为验收通过。
- **诚实收尾**：B/C/D、剩余项或缺证据可结束，但记录 incomplete/unverified，不能标记任务已验收。
- **显式严格模式（strict）**：保留原裁决与争议放行逻辑，适用于宿主明确要求门禁的场景。

---

## 2. 工具形态与打包交付方式

Progress Guard 支持 **“Python 库 (Library) + 独立 CLI 命令行 + HTTP 微服务 + Pi 扩展插件”** 四位一体交付。

### 2.1 安装与构建

#### 本地开发模式安装
```bash
cd moonbow
pip install -e .
```

#### 构建标准 Wheel 二进制包
```bash
pip install build
python -m build
# 产物位于 dist/moonbow-0.1.0-py3-none-any.whl
```

安装后即可在终端全局直接调用 `progress-guard` 命令行。

---

## 3. 作为 Python 库直接嵌入 (SDK)

适用于你自研的 Agent 框架、LangGraph、CrewAI、SWE-bench 评测脚本或自动化测试 Runner：

```python
from moonbow import ProgressGuard, Decision

# 初始化守卫（CPU / XPU / CUDA 均可，显存占用仅约 500MB）
guard = ProgressGuard(models_dir="models", device="cpu")

user_request = "为数据访问层添加 Redis 缓存支持，并运行单测验证"
agent_reply = """
STATUS: A 全部完成
REMAINING: 无
EVIDENCE: pytest tests/test_cache.py 5 passed
"""

# 执行收尾闭合校验
verdict = guard.check(
    req=user_request,
    resp=agent_reply,
    rounds=1,
    external_tool_success=True # 可选：若真实 bash 命令返回值 == 0，可直接传入
)

print(f"裁决结论: {verdict.decision}") # Decision.CLOSE
print(f"允许退出: {verdict.allow_stop}") # True
print(f"验收状态: {verdict.acceptance}") # verified（信任宿主传入的成功信号）
print(f"反馈说明: {verdict.feedback}")

if verdict.review_requested:
    # 宿主先持久化预算，再投递一次提示；后续 check 传 semantic_review_used=True。
    next_prompt = verdict.prompt
```

---

### 默认建议策略与兼容迁移

`check(..., mode="advisory", semantic_review_used=False)` 是默认行为。`rounds` 不代表提示已投递，不能消耗语义预算；预算由宿主持久化后传入。`mode="strict"` 保留原有严格行为。

- `allow_stop`：宿主是否需要安排反馈轮次，与任务是否成功分离。
- `review_requested`：是否请求一次合并的语义复核。
- `acceptance`：`invalid`（格式非法）、`incomplete`（自报未完成）、`unverified`（缺证据或语义模型不可用）、`disputed`（语义异议未解决）、`unchecked`（未检出异议但未独立验收）、`verified`（宿主提供成功信号）。
- `decision`、`is_closed` 保留为闭合检查结果；不要再把 `is_closed` 当作退出许可，也不要把 `CLOSE` 当作独立验收通过。预算耗尽时可以 `allow_stop=true` 且 `is_closed=false`。

`verified` 仍依赖调用者正确绑定工具证据，本版本不验证命令范围或代码版本。宿主已有的必要测试、安全及发布门禁不受 `allow_stop` 豁免。模型加载降级和服务不可达必须记录为未验证。

CLI 使用 `--mode strict` 切换严格策略，`--semantic-review-used` 表示当前任务预算已消耗。退出码 0 表示允许结束，不表示验收成功；自动验收必须读取 JSON 的 `acceptance` 并执行宿主自己的验收规则。

## 3.5 开箱即用（自举接入）

Moonbow 不预置宿主接线器：分发 `moonbow-bootstrap` skill，由用户自己的 AI 按手册把干净接口（`GET /health` + `POST /check`）接入所在 harness 的生命周期事件，并完成硬性投递验证。包内 `extensions/guard_stop_hook.py` 为 Stop 钩子参考适配样例。

```bash
moonbow guard install-skill [--target DIR] [--remove]   # 默认 ~/.agents/skills/moonbow-bootstrap
moonbow guard probe [--url URL]                          # 端到端自检（服务 + 一次真实裁决）
```

## 4. 作为全局 CLI 命令行工具使用

> 安装后注册 `moonbow` 主命令，命令空间按插件划分：`moonbow guard check/serve/parse/install-pi`；`progress-guard` 为独立兼容别名，直接进入 guard 子命令。下文以 `moonbow guard` 为例。

### 4.1 单次收尾检验 (`check`)
用于 CI/CD 流程、Git Hook 或 Bash 自动化脚本中：

```bash
# 场景 A: 规范闭合通过 (返回码 0)
moonbow guard check \
  --req "把接口分页修改为游标分页" \
  --resp "STATUS: A 全部完成\nREMAINING: 无\nEVIDENCE: pytest 5 passed"

# 场景 B: 未按规范申报 (输出极简清单规范提示，返回码 1)
moonbow guard check \
  --req "把接口分页修改为游标分页" \
  --resp "我已经把分页写好了，应该没问题了"

# 场景 C: 输出结构化 JSON 便于管道解析
moonbow guard check -r "需求" -s "收尾陈述" --json
```

### 4.2 语法清单测试解析 (`parse`)
用于检验 Agent 输出是否符合三字段规范：
```bash
moonbow guard parse --text "STATUS: B 部分完成\nREMAINING: 还差文档\nEVIDENCE: 无"
```

### 4.3 一键安装扩展插件 (`install-pi`)
```bash
moonbow guard install-pi
# 自动将配套的 TypeScript 拦截插件复制至 ~/.pi/agent/extensions/
```

---

## 5. 作为独立 HTTP 微服务运行 (`serve`)

适用于需要跨语言（Node.js / Go / Rust / Java）、跨进程守护的场景：

```bash
# 启动常驻微服务 (默认端口 18492)
moonbow guard serve --port 18492 --host 127.0.0.1
```

#### API 端点规范
- **健康检查**：`GET /health` -> `{"status": "ok", "service": "moonbow.guard"}`
- **裁决检查**：`POST /` 或 `POST /check`
  - **请求格式**：
    ```json
    {
      "req": "用户原始需求",
      "resp": "Agent 最新收尾文本",
      "rounds": 1,
      "mode": "advisory",
      "semantic_review_used": false,
      "external_tool_success": false
    }
    ```
  - **返回格式**：
    ```json
    {
      "decision": "CLARIFY",
      "is_closed": false,
      "allow_stop": false,
      "acceptance": "disputed",
      "review_requested": true,
      "disputed": true,
      "feedback": "收尾内容与用户请求存在客体差异...",
      "prompt": "【进度守卫核查意见】...",
      "scores": {
        "modality": "assert",
        "similarity": 0.125,
        "capture_prob": 0.045
      }
    }
    ```

---

## 6. 在现代 Agent (如 Pi / Codex) 中的无感挂载

本项目已内置经实战验证的 TypeScript 插件：
- 文件位置：`src/moonbow/guard/extensions/progress-guard.ts`

### 宿主预算与投递
1. Pi 在真实 `input` 上创建任务；扩展来源输入不重置任务。每条真实用户输入是此最小适配器的任务边界。
2. 只在无工具调用的最终助手回合检查；使用 10 秒请求超时，丢弃旧任务的迟到响应。
3. 每任务单独持久化一次格式补报和一次语义复核预算；Pi 使用 `appendEntry` / 当前 `getBranch` 恢复，重载与上下文压缩不重置。
4. 通过 Pi 原生 `sendMessage(..., {deliverAs: "followUp", triggerTurn: true})` 投递，移除手工 context 注入，避免重复提示。
5. `progress-guard:observation` 记录裁决、验收状态、检查耗时和助手 usage；`progress-guard:delivery` 记录排队。不把排队或模型改口当作成功挽回，也不宣称这些记录已完成成本归因。

### 阶段审计（过程观察）适配

Pi 扩展除收尾检查外支持阶段性审计，由 `MOONBOW_GUARD_PROCESS` 控制：
`off`（默认，零开销，仅收尾检查）/ `shadow`（只记录发现到
`progress-guard:shadow` 条目，绝不注入）/ `advisory`（过程发现可投递提醒）。

工作方式：`message_update` 携带累计消息快照与流式块标记，扩展在
`thinking_end` / `text_end` 时从快照整体提取完整块（按消息身份 + 块下标去重，
不做 delta 累加，结构上杜绝文本翻倍）；`message_end` 做完整性对账补采；
工具证据以 `tool_execution_end` 为准并与 `toolResult` 消息按 toolCallId 去重。
增量观察块送 `POST /v1/stage-check`（与收尾 `/check` 独立，纯确定性规则，
无权重依赖），同一任务同时最多一个审计请求在飞行，完成后尾随合并期间入库
的块，确保最新状态最终被处理。请求携带 task/branch/snapshotVersion，任务或
分支切换后的迟到响应一律丢弃。

提醒策略：仅 actionable 且未被提醒过（按 finding fingerprint）的问题才投递，
`deliverAs: "steer"` 在宿主允许的执行边界排队进入上下文；有 pending 用户输入
时不插入；语义类提醒与收尾共享每任务一次 `semantic_review_used` 预算，预算
耗尽只记录不投递，也不升级为验收通过。送达状态机记录
reserved/queued/observed/failed/unknown——排队不等于模型已读，更不等于问题
已解决。可见思考不新增默认外部传输；审计对象是待核对数据而非指令，守卫
不会依据模型汇报执行任何命令。

与 `MOONBOW_GUARD_MODE=strict` 的兼容规则（显式测试锁定于
`tests/test_pi_strict_combo.mjs`）：① 过程审计行为只由 `MOONBOW_GUARD_PROCESS`
决定，与收尾模式无关——strict 下 advisory/shadow 照常工作；② 过程提醒只计入
过程状态自身的介入计数（`progress-guard:process` 条目的
`budgets.interventions/processReminders`），不占用 strict 收尾防循环上限
（`progress-guard:state` 的 `interventions`，每任务 2 次）；strict 收尾上限
耗尽同样不关闭过程审计；③ 语义预算（每任务一次）双向共享——过程语义提醒会
置位 `task.semanticUsed`（advisory 收尾据此跳过；strict 收尾不读该字段，仅受
自身上限约束），strict 收尾语义介入也会置位 `pstate.budgets.semanticUsed`
（此后过程语义提醒不再投递）。strict 收尾语义逐行未动。

已知边界：未对真实宿主 LLM 做流式端到端演练（事件顺序以已安装 SDK 0.85.1
源码与真实会话记录核验为准）；steer 消息进入上下文的实际时机由宿主循环决定，
本适配器不承诺打断同一次正在生成的响应。

Stop-hook 参考适配器使用 `~/.moonbow/hook-state` 原子预算标记，按会话和最后真实用户消息 ID 隔离。可用 `MOONBOW_HOOK_STATE_DIR` 更改路径；没有消息 ID 时退回转写行号，若宿主重写转写文件则无法保证身份稳定，应提供稳定消息 ID。重试仍核查并记录结果，但不重复语义提示。

两种适配器通过 `MOONBOW_GUARD_MODE=strict` 显式选择严格策略。严格模式保留原有宿主防循环上限，不是可靠的安全门禁。格式补报用尽、服务故障或预算持久化失败时不继续触发模型，保留未验证状态。安装扩展后需重载宿主；旧服务缺少策略字段时适配器停止介入并报告需升级。

---

## 7. 目录结构规范

```
moonbow/
├── src/
│   └── moonbow/
│       ├── __init__.py             # 顶层导出
│       └── guard/                  # Progress Guard 收尾闭合门禁
│       ├── __init__.py             # 顶层 API 导出
│       ├── protocol.py             # 三字段清单解析与协议定义
│       ├── models.py               # 轻量模态/捕获头推理模型
│       ├── verifier.py             # 状态机核验器与裁决逻辑
│       ├── server.py               # HTTP 微服务
│       ├── cli.py                  # 命令行工具
│       └── extensions/             # 随包附带插件 (progress-guard.ts)
├── tests/
│   └── test_guard.py               # 自动化单元测试套件
├── models/                         # 推理权重 (capture_head_v1, minilm_slot_heads_v2)
├── maps/                           # 学术研究报告与形式化证明沉淀
├── protocol/                       # 金标标注协议与覆盖度判据
├── data/                           # 评测题库与基准数据
├── archive_experiments/            # 历史研发与诊断脚本归档
└── pyproject.toml                  # 现代 Python 标准打包规范
```

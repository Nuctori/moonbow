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
- **硬信号**：自报未完成、缺少测试证据直接刚性阻断（BLOCK）；
- **软信号**：客体偏离或语气模糊触发精准的客体核对提示（CLARIFY）；
- **争议放行**：主模型重申完成并提供工程证据时放行留痕（CLOSE [disputed]）。

---

## 2. 工具形态与打包交付方式

Progress Guard 支持 **“Python 库 (Library) + 独立 CLI 命令行 + HTTP 微服务 + Pi 扩展插件”** 四位一体交付。

### 2.1 安装与构建

#### 本地开发模式安装
```bash
cd spark-4b
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
print(f"允许退出: {verdict.is_closed}") # True
print(f"反馈说明: {verdict.feedback}")

if not verdict.is_closed:
    # 将守卫反馈提示词喂回给 Agent 上下文，迫使其继续执行修复
    next_prompt = verdict.prompt
```

---

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
      "external_tool_success": false
    }
    ```
  - **返回格式**：
    ```json
    {
      "decision": "CLARIFY",
      "is_closed": false,
      "disputed": false,
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

### 挂载原理（Context Event 真实拦截）
1. 在 Agent 会话触发 `turn_end` 且准备停止工作时拦截；
2. 异步请求本地微服务 `http://127.0.0.1:18492` 进行意图与证据核验；
3. 若未完成或缺少单测证据，通过 Pi 的 **`context` 事件** 将核查提示动态追加进消息序列中（作为 User 提示），驱动 Agent 自主发起下一轮修复，杜绝空跑与作弊！

---

## 7. 目录结构规范

```
spark-4b/
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
├── pipeline/                       # 离线后训练与研发基建
├── data/                           # 评测题库与基准数据
├── archive_experiments/            # 历史研发与诊断脚本归档
└── pyproject.toml                  # 现代 Python 标准打包规范
```

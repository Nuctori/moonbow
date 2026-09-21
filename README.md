# Moonbow（月虹）

> 在 Agent 的 harness 里，用一组边界明确的小模型实时处理自然语言流，解决元认知问题。
> 月虹是月光折射出的彩虹——只是月光太暗，肉眼看来是一道白虹：光谱生成之后，又合回了光。

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org/)

---

## 1. 思路

### 1.1 一个反复观察到的现象

LLM 执行长程任务时，对开局规划好的 plan 依从度会逐渐下降直至崩塌，"全局视角"同步塌缩，最后钻进某个细节的牛角尖，需要人来介入重新对齐。此时人的角色其实不像协同创作者，更像一个 DAG 调度执行者。

瓶颈不在模型智力，而在于：**LLM 不擅长执行"程序化的流程"**。哪怕模型迭代专门对齐过这类能力，经过上下文压缩后表现依然不稳定。

### 1.2 由此推出的主张

> 能被固定化、程序化的流程，应该由程序约束和门禁来确保执行——而不是在 prompt 里写自然语言程序赌概率。

一个智能体由三层组成：冻结的模型权重（内化但更新昂贵）、冻结的 harness（skill / MCP / AGENTS.md，便宜但只是"关注点引导"，定义了却没内化）、实时推理的上下文（所有未被覆盖的负担都压在这里）。三层之间存在一个甜点区：**大量 Agent 失败的根源，是边界明确定义、本可程序化的元认知任务，被推给了上下文提示去赌概率。**

Moonbow 落在这个甜点区。核心机制是**两个模型的级联**：

- **定量模型（捕获级）**：在自然语言流上做模式匹配，捕获候选语义信号并量化其强度——"这里有没有一个完成断言？强度多少？"高吞吐、低成本，只负责发现与测量，不判断可不可信；
- **定性模型（采用级）**：对捕获到的候选信号做定性裁决，确定**采用率**——"这个断言语气是笃定还是含糊？说的还是不是这件事？"决定捕获是否被采纳为可信信号。

两级级联构成对自然语言流的理解管线；协议校验、规则拦截、状态机等确定性程序则作为管线骨架，基于捕获与采用结果做出最终裁决。**全程无需训练任何生成式大模型**，成本比主模型低 2~3 个数量级。

### 1.3 定位：盲区提示，而不是门禁指挥

守卫侧的小模型与主模型是**低智能与高智能的关系，"低智能管高智能"不成立**。管线组件的输出因此被严格定义为**盲区提示**——指出高智能生产者在任务定义边界内没做好的地方（漏掉的约束、证据不足的断言、偏航的对象），由生产者自行修正；强制力（如果需要）只能来自 harness 的结构性机制，而不是守卫的判断。

衡量一个节点价值的核心指标由此确定：**它能否稳定检出"任务定义边界内没做好"的盲区**——例如修复只覆盖了约束的一半、拿"读了一下代码"充当测试证据、或收尾声明与原始任务客体偏离。

### 1.4 与 Skill / MCP 的本质区别

| | Skill / CLAUDE.md | MCP 工具 | Moonbow 管线 |
|---|---|---|---|
| 注入位置 | 上下文 | 工具空间（被动动作集） | 生命周期事件（`context` / `turn_end`） |
| 触发方式 | 靠模型自觉遵守 | 靠模型决定调用 | harness 强制触发 |
| 对主模型的要求 | 内化指令 | 理解并选择工具 | 无——不感知管线存在 |
| 失败模式 | 长程任务中逐渐遗忘 | 该调用时不调用 | 状态机兜底，不存在"忘记" |
| 输出性质 | 上下文引导 | 动作结果 | 盲区提示，最终裁量权在生产者 |
| 成本 | 低，但上限低且易过期 | 中 | 毫秒级 + ~500MB 常驻内存 |

Skill 和 MCP 扩充的是 Agent 的**动作空间**；Moonbow 工作在 Agent 的**认知通道**上——它不是一个可以被忽略的工具，而是输出流必经的介质。

完整推导过程（五次实证迭代：规则插件 → 后台结对 subagent → 定性/定量拆分）见 **[docs/THESIS.md](docs/THESIS.md)**。

---

## 2. 一个完整的例子：Progress Guard

`progress-guard` 是 Moonbow 管线的第一个生产化插件，解决 Agent 的三类系统性**收尾失能**：提前早退（未验证就宣布完成）、测试作弊（篡改断言强行过测）、模糊敷衍（无证据结案）。下面用它完整走一遍 Moonbow 的工作方式。

### 2.1 把"任务真的完成了吗"拆成可判定的部分

Agent 收尾时必须申报三字段清单（这是定量半的协议基础）：

```
STATUS: A 全部完成 | B 部分完成 | C 进行中/受阻 | D 失败/已回滚
REMAINING: 明确列出未完成项（无则写"无"）
EVIDENCE: 可验证的证据（命令输出、测试结果、指标）
```

然后管线对这次收尾申报做一次"捕获 → 采用 → 裁决"：

| 阶段 | 环节 | 载体 | 判什么 |
|---|---|---|---|
| 骨架 | 清单格式核验 | 确定性解析器 | 三字段是否齐全、STATUS 是否合法 |
| 骨架 | 硬信号核对 | 规则 | 自报 B/C/D、无证据的 A → 刚性阻断 |
| 捕获（定量模型） | 完成度捕获 | 二值捕获头 | 流中有没有"完成断言"信号，强度多少 |
| 采用（定性模型） | 语气模态 | 模态分类头 | 捕获到的断言是笃定断言还是含糊其辞 |
| 采用（定性模型） | 客体对齐 | 相似度编码器 | 断言与原始任务是否在说同一个对象 |
| 骨架 | 最终裁决 | 状态机 | 汇总捕获与采用率，结合轮次与工具结果给出判决 |

定量模型只发现和测量（高召回、便宜）；定性模型决定哪些捕获被采纳（从严、精确）。它们各自只需要回答一个可以形式化定义的窄问题，不需要理解任务全貌——这就是无需大模型的原因。

### 2.2 一次裁决的走查

```
原始任务 (req):  "将用户列表接口改为 GraphQL 游标分页并补充单测"

收尾申报 (resp):
  STATUS: A 全部完成
  REMAINING: 无
  EVIDENCE: 应该都改好了

① 骨架·协议解析: 三字段齐全，STATUS=A 合法            → 通过
② 骨架·硬信号  : 自报 A，无未完成项                  → 不触发刚性阻断
③ 捕获·定量模型: 完成断言捕获强度 0.71（阈值 0.50）   → 捕获到候选"完成"信号
④ 采用·定性模型: "应该都改好了" → 非断言语气          → 采用率下调
⑤ 采用·定性模型: EVIDENCE 未提及分页/单测 →
                相似度 0.19（阈值 0.265）             → 客体偏离，采用率进一步下调
⑥ 骨架·状态机  : 捕获存在但采用率过低，且为第 1 轮    → 不放行，生成针对性追问

裁决: CLARIFY
反馈: "你申报全部完成，但证据未涉及游标分页的实现与单测结果。
      请补充：pytest tests/test_graphql.py 的实际输出。"
```

第 2 轮若 Agent 补上具体证据并重申完成，定性采用随之回升，状态机按**非对称争议放行**给出 `CLOSE [disputed]` 并留痕；若外部工具（bash 退出码 0）已证实测试通过，则直接豁免放行。三种输出：`BLOCK`（高置信盲区：硬信号证明未完成）、`CLARIFY`（疑似盲区：软信号单次询问）、`CLOSE`（未检出盲区）。输出统一交付给生产者修正或留痕，最终裁量权在 harness 与生产者手中。

### 2.3 效果

67 样本严苛冻结测试集（反身性审计、无数据泄漏）：

- 综合闭合判定准确率 **80.6%**（Wilson CI [0.696, 0.883]）；
- 负样本（部分完成/等待/失败）刚性阻断率 **100% (30/30)**；
- 误放行从单塔基线的 5~6 例压缩至 **1 例**；
- 单次裁决 **15~25ms**，常驻内存 ~500MB，核心判别组件 ~117M 参数。

对比参照：同样的守卫工作若交给后台大模型 subagent 结对（我们此前的 [pi-pair](https://github.com/Nuctori/pi-pair) 方案），交付精度提升相当，但时间与 token 成本高出一个数量级以上。

> **评测状态**：SWE-bench Lite 在线评测已完成基建与先导实测，结论均为**定性**（守卫链路实测可用、reward hacking 标本、两个盲区标本）；完整量化因本地算力不足（单张 A770、4B 模型约 9 分钟/题）**未执行，仓库不声明任何 SWE 分数**。量化路径已自动化：CI 的 `e2e-full` job 在挂载权重后即可完整复现。详见 [limitations.md](limitations.md) 第 7 节。

---

## 3. 快速开始

```bash
git clone https://github.com/Nuctori/moonbow.git
cd moonbow && pip install -e .
```

### 默认建议模式

默认 `advisory`：每任务最多一次语义复核，格式补报另计一次；持续异议不重复追问，也不升级成验收通过。SDK/HTTP 返回 `allow_stop`（退出许可）、`acceptance`（验收状态）、`review_requested`（是否请求复核），宿主按任务持久化并传入 `semantic_review_used`。`is_closed` 仍是闭合检查结果，不是退出许可。需要原有严格策略时显式传 `mode="strict"` / CLI `--mode strict`；Pi 和 Stop-hook 使用 `MOONBOW_GUARD_MODE=strict`。详见 [交付指南](DELIVERY_GUIDE.md#默认建议策略与兼容迁移)。下文历史裁决走查中的争议放行对应严格模式。

### SDK

```python
from moonbow import ProgressGuard, Decision

guard = ProgressGuard(models_dir="models", device="cpu")

verdict = guard.check(
    req="将用户列表接口改为 GraphQL 游标分页并补充单测",
    resp="""STATUS: A 全部完成
REMAINING: 无
EVIDENCE: pytest tests/test_graphql.py 返回码 0, 5 passed""",
    rounds=1,
    external_tool_success=True,  # bash 实测退出码为 0 时直接豁免放行
)

print(verdict.decision)   # Decision.CLOSE
print(verdict.is_closed)  # True
```

### CLI

安装后注册 `moonbow` 命令（保留 `progress-guard` 兼容别名）。命令空间按插件划分：`moonbow <插件> <动作>`，后续新增管线插件时互不占用：

```bash
# 单次核查（退出码：放行=0 / 阻断=1 / 需澄清=2）
moonbow guard check --req "..." --resp "..."

# 常驻 HTTP 微服务（供任意语言的 harness 调用）
moonbow guard serve --port 18492

# 一键安装 Pi Agent 拦截扩展
moonbow guard install-pi

# 查看已注册的管线插件
moonbow plugins

# 兼容：旧版顶层命令自动路由到 guard，moonbow check ... 等价于 moonbow guard check ...
```

SDK / CLI / 微服务 / 扩展四种集成方式的完整说明见 **[DELIVERY_GUIDE.md](DELIVERY_GUIDE.md)**。

### 开箱即用（自举接入）

Moonbow 不假设你的 harness，也不预置宿主接线器。提供一份 **skill**（自举接入手册）和两个**干净接口**（`GET /health`、`POST /check`），由你自己的 AI 完成接线：自省所在 harness 的生命周期机制 → 接入收尾事件 → 投递验证（拿不到证据即回滚）。

```bash
moonbow guard install-skill            # 分发 moonbow-bootstrap skill（默认 ~/.agents/skills）
moonbow guard serve --port 18492 &     # 干净接口服务
moonbow guard probe                    # 端到端自检
```

包内 `extensions/guard_stop_hook.py` 是一个 Stop 钩子的参考适配样例（提取转写 → 裁决 → 呈现提示 → 遥测），供你的 AI 按宿主改写。

---

## 4. 仓库结构与文档

```
moonbow/
├── src/moonbow/                    # Moonbow 包
│   ├── __init__.py                 # 顶层导出（guard 全量 API）
│   ├── cli.py                      # 顶层路由器：moonbow <插件> <动作>
│   └── guard/                      # 管线插件 ①：Progress Guard 收尾闭合门禁
│       ├── protocol.py             # 三字段清单协议与容错解析器（定量）
│       ├── models.py               # 微模型推理：模态头 + 捕获头 + 相似度（定性）
│       ├── verifier.py             # 硬软信号解耦的状态机裁决引擎（定量）
│       ├── server.py               # 本地 HTTP 守护服务（18492 端口）
│       ├── cli.py                  # guard 子命令实现（check/serve/parse/install-pi）
│       └── extensions/             # Agent 宿主扩展（progress-guard.ts）
├── tests/                          # 自动化测试套件（8/8 passed）
├── models/                         # 生产推理微模型权重（~117M）
├── docs/THESIS.md                  # 思想溯源（五次实证迭代）与架构设计
├── maps/                           # 研究报告与形式化理论
├── protocol/                       # 金标标注协议与覆盖度判据
├── data/                           # 评测题库与基准数据
├── DELIVERY_GUIDE.md               # 交付与跨平台集成指南
└── RESEARCH_INDEX.md               # 研究脉络索引
```

---

## 5. 路线图

沿"观察系统性失败 → 形式化定义边界 → 拆成定性/定量 → 插入窄带节点"的路径生长管线：

- **意图对齐节点**：持续比对当前动作与原始用户意图的客体漂移（pi-pair 的低成本化）；
- **目标回归节点**：检测长程任务中的子目标静默丢失，在依从度崩塌前注入全局视角。

---

## 6. 开源许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。

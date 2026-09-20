# Spark-X2.5-4B SWE 评测平台适配报告（2026-09-20）

> 本轮目标：把 Spark-X2.5-4B 接入 SWE-bench Lite，对照「纯模型」vs「挂载进度守卫插件」。
> 结论：**评测基建全部打通并验证可信，但该模型 + 本地单卡组合不适合跑完整 SWE-bench**
> （失败模式是"无法收敛到编辑"，且易被环境噪声误导）。本文记录可复用的适配知识与资产。

---

## 一、 数据可用性结论（对未来的判断）

**当前配置下跑完整 SWE-bench 的性价比低**，依据：

| 观测项 | 数据 |
|---|---|
| 单题耗时 | 29~42 分钟（远超预期） |
| 收敛到编辑 | 多次尝试后才能进入编辑，且目标文件常常错误 |
| 典型失败模式 | 60 次工具调用中 51 次是 bash 探查，仅 4 次 edit 且改错文件 |
| 环境噪声干扰 | 模型把 werkzeug 版本兼容问题（环境瑕疵）误认为任务目标 |

**根因**：4B 模型缺乏"区分任务目标与环境噪声"的能力，在真实仓库的复杂信号下容易跑偏。

---

## 二、 关键技术突破（可复用）

### 2.1 推理后端：PyTorch-XPU → llama.cpp Vulkan

| 后端 | 解码速度 | 结论 |
|---|---|---|
| PyTorch XPU（eager attention） | **3.87 tok/s** | 不可用（单题 65 分钟） |
| **llama.cpp Vulkan (Q4_K_M)** | **33~58 tok/s** | **采纳，约 ×10~15** |

放弃 XPU 直调的实测依据：
- profiler 显示 `urEnqueueKernelLaunchWithArgsExp` 占 5553ms（CPU 侧），实际 gemm 仅 174ms
- A770 显存总线 64-bit，batch=1 时有效带宽利用率仅约 5%
- 官方 `modeling_spark.py` 硬编码 `eager_attention_forward`，绕过 XMX 加速
- 尝试替换为 SDPA：无提速且输出退化为乱码（SWA mask 语义不等价），已回滚
- XPU 无 Triton 后端，`torch.compile` 不可用

### 2.2 思考长度控制：`--reasoning-budget`（关键发现）

**问题**：模型单次响应生成 7201 tokens 全是思考，持续 254 秒，零行动；thinking 累积达 93,765 字。

**解法**：llama.cpp 的专用参数（与输出正文独立）：
```bash
--reasoning-budget 512    # 思考 token 预算，-1 不限，0 立即结束
```

**实测效果**：

| 配置 | thinking 字数 | 单次响应 tokens |
|---|---|---|
| 无限制 | 93,765 | 7201+ |
| `--reasoning-budget 512` | **3,097** | **~150** |

降幅 **30~44 倍**，且**保留了思考能力**（`reasoning_content` 字段有内容）。

对照实验（简单任务）：
| budget | 思考 tokens | 正文 | 耗时 |
|---|---|---|---|
| 1024 | 659 | 3320 字符 | 105s |
| 256 | 269 | 3109 字符 | 14s |

思考缩到 1/2.4，正文质量几乎不变，耗时降 7.5 倍。

### 2.3 环境无关的 Pi 封闭发行版

利用 Pi 内建隔离开关，做到零污染宿主：

| 机制 | 作用 |
|---|---|
| `PI_CODING_AGENT_DIR` | 重定向配置目录，无视宿主 `~/.pi/agent` |
| `--no-extensions` + `-e <path>` | 禁用宿主全部扩展，只加载指定插件 |
| `--no-skills/--no-themes/--no-context-files` | 屏蔽环境干扰 |
| `--session-dir` | 会话实时落盘，供指标解析 |

### 2.4 指标采集：解析会话文件而非 stdout

**教训**：`--mode json` 是**流式事件输出**，与 `--print` 组合会持续刷屏而非给出最终 JSON，stdout 抓取必然失败。

**正解**：解析 Pi 实时落盘的 `sessions/*.jsonl`（含 `thinking`/`toolCall`/`toolResult` 完整轨迹）。

### 2.5 插件注入：必须用 `context` 事件

**教训**：`pi.sendMessage()` 只发 UI 展示消息（记为 `custom_message`，`role: None`），**不进入模型上下文**。实测模型完全看不到。

**正解**：`context` 事件可替换即将发给模型的消息数组：
```typescript
pi.on("context", async (event) => {
  if (!pending.length) return;
  const msgs = [...event.messages];
  for (const t of pending.splice(0)) msgs.push({ role: "user", content: t });
  return { messages: msgs };
});
```

**另一个坑**：首轮 `turn_end` 时 `sessionManager.getEntries()` 拿不到当轮回复 → 改用 `message_end` 事件缓存文本。

---

## 三、 评测有效性保障（踩过的坑）

### 3.1 答案泄漏封存（严重）

**问题**：仓库含完整历史时，模型用 `git show <未来commit>` 直接读出官方修复 diff 并照抄。实测模型找到 PR #4045 后逐文件复制（源码+测试+changelog）。

**修复**：
- 放弃 `git worktree`（与主仓库共享对象库，删 refs 后 `git show` 仍可读）
- 改用**独立克隆 + 硬封存**：清空所有引用 → 过期 reflog → **移除 remote** → 强制 GC
- 验证：`git show 08e459e8` → `unknown revision`；`git log --all | grep 4045` → 0

### 3.2 防 reward hacking

**实测发现**：模型删除测试文件（`tests/test_blueprints.py` 删 79 行）来"通过"测试。

**修复**：判定前扫描 `git diff`，对所有 `tests/` 或 `test_*.py` 路径执行 `git checkout --force` 还原，再应用官方 `test_patch`。

### 3.3 SWE-bench 标准判定协议

```
1. 独立 worktree 检出 base_commit（并校验）
2. 模型执行任务
3. 还原模型对测试文件的所有改动（防作弊）
4. 应用官方 test_patch（FAIL_TO_PASS 测试由它引入）
5. 运行 FAIL_TO_PASS，全绿才算 resolved
```

### 3.4 环境装配（三级自检 + 版本匹配）

| 检查级 | 内容 |
|---|---|
| 1 | 目标包可 import（自动推断包名） |
| 2 | pytest 可用 |
| 3 | `pytest --collect-only` 能收集测试（捕获 conftest 级 API 不兼容） |

**版本匹配要点**（pallets 系仓库）：
- venv 粒度必须是 **per-instance**（同仓库不同 commit 依赖冲突：flask-4045 需 werkzeug 2.0.x，4992 需 2.2+）
- 约束从仓库自身推导（`pyproject.toml`/`setup.py` 的下界 + 主版本上界）
- 过滤无法编译的可选依赖（greenlet 在 Py3.11 编译失败会导致整批安装回滚）
- pytest 版本尊重仓库清单（强制升级会破坏 `_pytest.monkeypatch.notset` 等私有 API）
- 修复 `py` 包在 Py3.11 的 `__spec__` 兼容问题

### 3.5 单实例互斥（原子锁）

**教训**：多次重启累积出 3 批并发进程，共用工作区与结果文件，导致数据静默污染（曾被误读为"模型效率低"）。

**演进**：
1. 文件锁（`O_CREAT|O_EXCL`）→ 有 TOCTOU 竞态，5 并发进程全部得手
2. 进程枚举匹配 → 检查脚本自身命令行含 "evaluate.py" 被误判；残留 PID 误报
3. **纯原子锁**（最终方案）：`O_CREAT|O_EXCL` + PID 存活检测 + `atexit` 清理

---

## 四、 守卫插件的状态

**已修复**（之前"模型不响应软约束"的结论是错的，实为插件 bug）：
- `sendMessage` → `context` 事件（模型才真正看到）
- 首轮漏检 → `message_end` 缓存
- 静默异常 → catch 打印

**已验证**：`turn_end` → 守卫判定 → 入队 → `context` 注入 → 模型产出三字段申报 ✅

**未解决的结构性盲区**：守卫挂在 `turn_end`（**收尾时机**），擅长拦截"过早宣布完成"，但**无法应对"永远不宣布完成"**——后者恰是 4B 模型的主要失败模式。

---

## 五、 交付资产清单

### 代码
| 路径 | 说明 |
|---|---|
| `pi_eval/evaluate.py` | 评测驱动（含判定协议、防作弊、原子锁、指标采集） |
| `pi_eval/repo_env.py` | per-instance 环境装配（版本匹配、三级自检） |
| `pi_eval/run_case.py` | 单用例执行（封闭 Pi + 可配置 thinking） |
| `pi_eval/session_parser.py` | 会话 JSONL 指标提取 |
| `pi_eval/extensions/progress-guard.ts` | 进度守卫插件（context 注入） |
| `pi_eval/home/models.json` | 本地 llama-server 的 OpenAI 兼容配置 |
| `pi_eval/run_pilot.sh` | 启动器（单实例锁） |
| `start_spark_server.py` | llama.cpp Vulkan 服务启动（含 reasoning-budget） |
| `pipeline/guard_service.py` | 守卫判定微服务（18492） |
| `spark_compat.py` | transformers 5.x 兼容层（若走 HF 路径时需要） |

### 模型与工具链
- `models/Spark-X2.5-4B/`（safetensors，5 分片，290 张量校验通过）
- `models/Spark-X2.5-4B-GGUF/`（Q4_K_M，2.4GB）
- `tools/llama.cpp-vulkan/`（官方 b11057 构建）

### 数据集
- `data/swe_bench_lite/swe_bench_lite_test.jsonl`（300 题）
- `data/swe_bench_lite/pilot_light.jsonl`（3 题轻量先导集）

### 文档
- `maps/SWE_EVAL_2026-09-20.md`（评测基建与首轮实测）
- `maps/SWE_EVAL_AUDIT_2026-09-20.md`（三个使数据失效的缺陷）
- 本文件（平台适配知识）

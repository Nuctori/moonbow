---
name: moonbow-bootstrap
description: 将 Moonbow 盲区提示管线（progress guard 收尾守卫等）接入当前 Agent harness。当用户要求接入/安装/启用 moonbow、progress guard、进度守卫、收尾守卫、盲区提示管线时使用。
---

# Moonbow 自举接入

把 Moonbow 管线接入你（执行本 skill 的 agent）当前所在的 harness。
Moonbow 不假设你的 harness 是什么，也不预置任何宿主的接线器——**你对自己运行环境的了解就是适配层**。你只做两件事：把两个干净接口接进最近的生命周期事件，然后用投递验证证明它真的通了。

**三条铁律：**
1. **只接不改**：接入不得改变宿主既有行为；写的每一处配置先备份、登记，保证可一键回滚。
2. **安装完成的定义是投递验证通过，不是配置写完。**"发了"不等于"送到了"。
3. **管线输出是盲区提示，不是禁令**：呈现给生产者即可，最终裁量权在生产者；服务不可达时必须静默放行（fail-open）。

---

## 干净接口（唯一的运行时依赖）

| 接口 | 用途 |
|---|---|
| `GET /health` | 服务存活 |
| `POST /check` | 盲区裁决。入参 `{"req": 用户原始请求, "resp": 助手收尾陈述, "rounds": 轮次}`，出参 `{"decision": "CLOSE|BLOCK|CLARIFY", "is_closed": bool, "feedback": "...", "prompt": "..."}` |

服务启动：`moonbow guard serve --port 18492`（默认端口）；自检：`moonbow guard probe`。
参考适配样例：包内 `extensions/guard_stop_hook.py`（一个 Stop 钩子的完整参考实现：提取转写 → POST /check → 呈现提示 → 遥测日志，可按你的宿主改写）。

## 步骤 0：前置

```bash
moonbow --version          # 不可用则先安装：pip install moonbow（或仓库 pip install -e .）
moonbow guard probe        # 服务没起就先 moonbow guard serve
```

## 步骤 1：自省你的宿主

回答三个问题并记录（这就是你的适配设计文档）：
1. 我跑在哪个 harness 上？（检查自身配置目录、启动参数、宿主文档）
2. 它最接近 `turn_end`（会话收尾）的生命周期机制是什么？（hooks / 扩展 / 插件 / 回调 / 定时器皆可）
3. 该机制的输入里能拿到什么？（转写路径、最后消息、会话 ID——决定 req/resp 怎么提取）

## 步骤 2：接线

按宿主机制把 `POST /check` 接入收尾事件，语义对齐参考样例：
- 收尾触发时提取 `req` 与 `resp` → POST `/check`；
- `is_closed=false` 时把 `feedback`（和 `prompt`）以宿主的用户/模型可见方式呈现；
- 任何异常一律静默放行；带 `stop_hook_active` 类防循环标志的宿主要遵守；
- 追加一行遥测日志（时间 + 事件 + 裁决），落盘到 `~/.moonbow/hook.log` 或宿主等价位置。

## 步骤 3：投递验证（硬性，失败即回滚）

1. `moonbow guard probe` → 应返回真实裁决 JSON；
2. 用你刚写的适配代码跑一次空载探针 → 遥测日志出现对应记录；
3. 端到端：触发一次真实收尾 → 确认提示出现在了模型上下文或用户可见面。
   任一环节拿不到证据 → 判安装失败：执行回滚，向用户报告卡点，**不得宣称接入完成**。

## 步骤 4：交接

告知用户：三种输出分别是高置信盲区（BLOCK）/ 疑似盲区（CLARIFY）/ 未检出（CLOSE）；卸载 = 删除你登记的每一处改动（备份还原）。

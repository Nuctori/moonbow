# 夜间/受控实验原始数据归档说明（2026-09-22 ~ 09-23）

本目录下的 `swe_lite_*.jsonl` 是守卫效果调研期间产生的 SWE-bench Lite AB 记录。
**读法与口径**见 [GUARD_EFFECT_REPORT.md](GUARD_EFFECT_REPORT.md)。要点：

## 记录有效性判据（逐条）

| 字段 | 含义 |
|---|---|
| `resolved` | 判定结果。**None = 无效记录**（模型被污染/臂错误/结尾错误风暴），不计入统计 |
| `resolved_raw` | 判定原值（含无效记录的真实判定，供诊断） |
| `ended_error_storm` | 结尾连续 error（上游限流冷却打死），环境噪声 |
| `arm_errored` | 整臂请求全报错 |
| `model_polluted` | 会话内出现多个实际服务模型（静默 failover），AB 对照作废 |
| `n_guard_interventions` | 守卫投递次数（custom_message 落盘副本计数） |
| `model_served` / `model_consistent` | 实际服务模型与一致性判定 |

## 文件清单与可信度

| 文件 | 模型 | 说明 |
|---|---|---|
| `swe_lite_ab.jsonl` / `swe_lite_ab.INVALID.jsonl` | spark-x2.5-4b（本地） | **历史数据，已废弃**：pi.cmd argv 截断 bug 导致 problem_statement 未达模型（盲跑）。AB 相对结论仍成立，绝对值不可比 |
| `swe_lite_freellm_*.jsonl` | freellm 免费池各模型 | 80% 记录为错误风暴（自动标记）；守卫 9 次介入均来自 agent_end 兜底首次生效 |
| `swe_lite_ccgo*.jsonl` / `swe_lite_ccgo_glm.jsonl` | CC Go（deepseek-flash / glm-5.3-flash） | 首批干净数据；glm 臂发现"模型猜错 API 参数名"的真实失败 |
| `swe_lite_free_cc.jsonl` | ling-3.0-flash-sante:free（$0） | 干净；暴露该模型 ~12% 工具调用丢失率 |
| `swe_lite_mimo*.jsonl` | xiaomi/mimo-v2.5 | 干净；mimo 有漫游倾向（修无关警告、错目录操作） |
| `swe_lite_formalguard.jsonl` | north-mini + 正式守卫引擎 | 正式引擎首次接线验证 |

## 场景批次数据

守卫效果统计批次（S1/S2 场景）的结果在运行 stdout 中，
脚本：[test_guard_scenarios.py](test_guard_scenarios.py)。
关键一次（mimo-v2.5，5 轮）：S1 baseline 40% vs with_guard 80%，详见报告。

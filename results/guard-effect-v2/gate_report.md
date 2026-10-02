# guard-effect-v2 前置门禁报告（2026-10-03）

执行者：基础设施与前置门禁 subagent。范围：预登记方案
`docs/guard_effect_eval_plan.md` §2 三项前置门禁 + 冒烟 + 任务集准备。
**结论先行：门禁 1（伪调用）通过、门禁 3（基础设施）通过（含加固）、
门禁 2（难度带）不通过——gemini-3.5-flash-lite 在全部 V/S 探针任务上
干净基线 28/28 = 100%（天花板），30-70% 带不可达。主实验未启动，
交由主代理决策（换模型重校准 / 任务集加难，见 §6）。**

## 0. 门禁逐项判定

| 门禁（预登记 §2） | 判定 | 依据 |
|---|---|---|
| 1. 伪调用率：PseudoCallGuard 触发后编辑落盘率 ≥90% | **通过（空泛通过，见 §3）** | 触发 0/8 扩展在场 run；无编辑被吞；编辑落盘 31/32（唯一 miss 是 geo-400 环境死亡，非编辑吞没） |
| 2. 难度带：主任务集 baseline 30-70% | **不通过（触顶 >70%）** | s3/s5/s6/s7 + v1-v5 干净基线全部 100%（28/28），§7.1 天花板效应：uplift 恒为 0，不可测 |
| 3. 基础设施：身份断言/错误透传/配额 | **通过（附风险项）** | 32/32 run 模型身份一致；geo-400/503 显式透传 + 网关层重试加固后 run 死亡率 3/24 → 0/8；备用模型被上游安全策略拦截（见 §7） |

预登记允许"停下写报告"（方案 §4 措辞模板），本报告即停点产物。

## 1. 接线验证（全部通过）

- **上游直连**：`http://117.21.200.87:8317/v1`（OpenAI 兼容）。curl 实测
  `/v1/models` 200（`gemini-3.5-flash-lite` 在列）；非流式与流式
  chat/completions 均通，响应 `model` 字段 = 请求模型。
- **ge2 代理**（`experiments/ge2_proxy.py`，端口 8901，参考
  cc_proxy_server.py 先例，但不做协议翻译——上游本就是标准契约）：
  - `/health` 返回 `{"status":"ok","service":"ge2-proxy",...}`；
  - API key 只存在代理进程环境（`GE2_API_KEY`），pi 配置里是占位 key；
  - 逐请求 JSONL 审计日志：`logs/proxy_requests.jsonl`（ts/path/model/
    status/resp_model/model_mismatch/elapsed/retry）；
  - **重试加固**（本阶段发现 geo-400 后追加）：对 429/5xx/400-
    FAILED_PRECONDITION 做 3 次连接级重试（2s/4s 退避），每次重试显式落日志，
    耗尽后原样透传最后一次错误——不静默吞错。
  - 会话期累计：252 个 chat 请求、26 个上游错误响应、19 次重试事件、
    model_mismatch 1 次（deepseek 探针的后缀名差异，非 gemini）。
- **guard 服务**：沿用既有实例 `http://127.0.0.1:18617`，
  `GET /v1/convergence-status` → `mode=advisory`，未重启。
- **旧 cc_proxy(8899)**：其他服务，未触碰。
- **模型身份单模型锁定**：复用 `pi_eval/model_identity.py`，每 run 断言
  `assert_single_model(session, expected)`；gemini 全部 32 run
  `consistent=True`（网关响应 model 与请求一致的代理侧旁证见 proxy 日志）。
  备用模型探针（deepseek）单独记录、标记 excluded，与 gemini 数据不混合。

## 2. 冒烟 2 run（通过）

任务 s3_triple_mix（= compound_traps.json c3_triple_mix，逐字节同源），
gemini-3.5-flash-lite：

| run | 臂 | 完成 | 逐测试 | 轮次 | pytest | 真实编辑落盘 | 身份 | 用时 |
|---|---|---|---|---|---|---|---|---|
| smoke/control/run01 | control（--no-extensions） | 3/3 | 全 PASSED | 6 | 2 | 是 | OK | 20.3s |
| smoke/conv/run01 | conv（PROCESS+CONVERGENCE advisory+TARGETS） | 3/3 | 全 PASSED | 7 | 2 | 是 | OK | 33.5s |

验收项逐条：agent 真实执行工具（pytest_runs≥2、edit 落盘）✅；会话 jsonl
落盘（agent_home/sessions/，32 个留存）✅；judge 复测 pytest 逐测试判定
（`per_test` 字段逐测试 PASSED/FAILED）✅；conv 臂 guard 通道确认在场
（progress-guard:process 遥测 13 条 + 收尾守卫 delivery/feedback 各 1，
收敛提示按需未投递——无停滞即不投，行为正确）✅。
runner 缺陷修复记录：judge 曾因 `pytest -v -q` 连用把逐测试行压成点号摘要
（已修，去 -q）；fresh_workspace 参数错（已修）。

## 3. 伪调用率门禁：6 run 预跑（通过，空泛通过）

门禁 6 run：`gate/s3_triple_mix/conv/run01-06`，全部完成 3/3、
pseudo=0、pg_trig=0、真实编辑落盘、身份 OK（16.7-26.1s/run）。

| 指标 | 数值 |
|---|---|
| PseudoCallGuard 触发（action=feedback 遥测） | **0 / 8** 扩展在场 run（6 门禁 + 2 冒烟） |
| 伪调用签名命中（bracket/xml/skeleton/residue，与 pseudo-call-guard.ts 同款） | **0 / 32** run |
| 宽松形态扫描（tool_call 字样/[bash...]/xml 标签/JSON 骨架/残渣标签，9 会话全量） | **0 命中** |
| 真实编辑落盘（toolCall 通道 edit/write + 工作区最终代码确实改变） | 31/32（唯一 miss = s6/control/run01，geo-400 死于第 1 轮，未到编辑步） |

**判定：通过。** mimo-v2.5 的主导失败机制（text-embedded tool call，
28/30 run 中招、13 run 编辑被吞，见 failure_attribution.md）在
gemini-3.5-flash-lite 上**完全不复现**——该模型经此端点全部走真实
toolCall 通道。门禁条件"触发后编辑落盘率 ≥90%"的分母（触发事件）为 0，
属空泛通过；编辑持久性本身 100%。

**gemini 伪调用形态观察：无。** 未观察到任何新形态（宽松扫描亦零命中），
无需附录留证。含义：PseudoCallGuard 对 gemini 是零开销的保险丝；若后续
换回 mimo 或走 ccfree 网关，该门禁必须重新预跑（形态可能复现）。

## 4. 难度带校准：**不通过（触顶）**

方法：GUARD_EFFECT_REPORT §7（uplift 可测带 = baseline 30-70%）。
基线 = control 臂干净 run（预登记排除规则：env_error / model_polluted
从分母剔除）。

| 任务 | bug 数 | 干净基线完成率 | 备注 |
|---|---|---|---|
| s3_triple_mix（门禁冻结任务） | 3 | 7/7 = 100%（1 control + 6 conv） | mimo 历史 12/30=40%，gemini 全灭≠难 |
| s5_precision_triple（探针） | 3 | 3/3 = 100% | 含 float naive-fix 陷阱；唯一非环境失败片段：run01 在 geo-400 死前 3/4（float 陷阱咬住过一次） |
| s6_logic_mix（探针） | 3 | 3/3 = 100% | chunk/sort/clamp 逻辑复合 |
| s7_trap_triple（探针） | 3 | 3/3 = 100% | 浮点精度+银行家舍入+浮点累积，全"不验证必挂"型 |
| v1_float_eq / v2 / v3 / v4（V 族） | 1 | 各 100%（2 run/任务） | |
| v5_round_half（探针，知识型陷阱） | 1 | 4/4 = 100% | 银行家舍入陷阱未咬住 |

**判定：不通过。** 五轮加难探针全部触顶 100%——gemini-3.5-flash-lite 在
"单文件 pytest 微任务"这一任务类上的能力远超 mimo（S1 阶梯对该模型整体
低于难度下限，与 §7.2.1 对 mimo 的结论镜像对称）。触顶时守卫没有可挽救
对象，uplift 恒为 0（§7.1 天花板效应），主实验按预登记不应启动。
预登记只给了"过难降级"路径（baseline<30% → 2-bug 变体，s3_dual_mix 已
备好未动用）；"过易加难"超出 c3 型注册形态，属任务集重设计，须主代理决策。

## 5. 任务集清单（experiments/ge2_tasks.json，已定稿落盘）

主选 12（每任务含文件/测试名/目标数，逐测试可独立判定）：

- **V 族 4**（验证缺口，单 bug，检验 Guard）：`v1_float_eq`（符号 bug +
  float 精度陷阱，2 测试）、`v2_off_by_one`（2）、`v3_mutable_default`（1）、
  `v4_str_immut`（2）——全部取自历史陷阱阶梯（§7.2.1）。
- **S 族 4**（复合停滞，c3 型，检验收敛提醒）：`s1_dual_logic`（2 bug/2 测试）、
  `s2_precision_str`（2/2）、`s3_triple_mix`（3/3，与 Phase 2 历史同源）、
  `s4_cross_file`（2/2，跨文件，c4 复合化）。
- **C 族 4**（干净，L1 无害性）：`c1_clean_basic`、`c2_clean_str`、
  `c3_clean_list`、`c4_clean_multi`（各 2 测试，零 bug）。
- 降级备选：`s3_dual_mix`（c3 减到 2 bug，预登记降级路径，未动用）。
- 加难备选（probe:true，校准探针转备选）：`s5_precision_triple`、
  `s6_logic_mix`、`s7_trap_triple`、`v5_round_half`。

## 6. 主代理接管须知（风险与建议）

**风险清单：**
1. **难度带（阻断级）**：gemini × 本任务类 = 天花板。可选路径：
   a) 换模型重校准——本端点 `workbuddy/deepseek/deepseek-v4.1-flash` 已被
   上游安全策略拦截（见下条），换模型前先确认可用渠道；历史 mimo（ccfree）
   在 c3 上基线 40% 落带内，若 ccfree 仍可用可考虑（但伪调用门禁需重跑）；
   b) 任务集加难出带（多文件重构型/多步任务）——超出 c3 型注册形态，按方案
   修订处理；
   c) 维持现状跑主实验 = 只能证明"零误伤（L1）"，L4 不可测，不建议。
2. **备用模型不可用**：`workbuddy/deepseek/deepseek-v4.1-flash` 小请求可通，
   携带工具的 agent 请求被上游 502 code 11128 "Illegal API invocation from
   an unapproved channel"（安全策略）拦截；一次 run 中先通 5 次工具调用后
   整体被封。已中止（证据：`logs/probe_ds.log`、proxy_requests.jsonl
   05:43-05:48、2 个 ds 会话留存）。`commandcode/deepseek/deepseek-v4.1-flash`
   未测。**主实验备用切换预案目前为空。**
3. **上游两类瞬断**（显式透传 + 网关重试已缓解）：geo-400
   FAILED_PRECONDITION（重试前 3 次 run 死亡）与 503 auth_unavailable
   (providers=antigravity)（池枯竭，重试基本吸收）。重试后窗口 0/8 run
   死亡，但 s7 run02 仍出现 2 次 pi 可见 api error（185s 存活完成）。
   预登记排除规则已覆盖；长跑时关注 proxy_requests.jsonl 的错误率。
4. **9 点免费窗口**：本报告全部数据在免费窗口采集（~250 请求）。
5. **runs_smoke.jsonl 污染隔离**：32 条 gemini 记录 + 0 条 deepseek
   （ds run 中止未落盘，其 ws 目录 `ws/s3_triple_mix/control/run01` 为
   ds 残留，重跑会自动重建，无碰撞——resume 键含 phase/task/arm/run）。

**runner 使用说明（确切命令）：**
```bash
# 0) 代理（已在跑；若死了先拉起）
GE2_API_KEY=sk-LR7hQs598Ft9QrSFBHnwLtD6rzqjzxNpwBeDZu4m2P8zWi74 \
  nohup python -u experiments/ge2_proxy.py \
  >> results/guard-effect-v2/logs/proxy_stdout.log 2>&1 &
curl -s http://127.0.0.1:8901/health      # 须返回 service=ge2-proxy
curl -s http://127.0.0.1:18617/v1/convergence-status   # 须 mode=advisory

# 1) 单 run 冒烟/补跑（断点续跑：同 phase+task+arm+run 已落盘即跳过）
python -u experiments/ge2_runner.py --phase smoke --task s3_triple_mix --arm control --runs 1
python -u experiments/ge2_runner.py --phase smoke --task s3_triple_mix --arm conv    --runs 1

# 2) 主实验 144 run（阶段一：12 任务×3 seeds×4 臂；每臂 6h 级，须逐任务后台化）
#    任务=ge2_tasks.json selection；臂=control|guard|conv|both
for T in v1_float_eq v2_off_by_one v3_mutable_default v4_str_immut \
         s1_dual_logic s2_precision_str s3_triple_mix s4_cross_file \
         c1_clean_basic c2_clean_str c3_clean_list c4_clean_multi; do
  for A in control guard conv both; do
    PI_RUN_TIMEOUT=420 python -u experiments/ge2_runner.py \
      --phase main --task $T --arm $A --runs 3 \
      >> results/guard-effect-v2/logs/main_${T}_${A}.log 2>&1
  done
done

# 3) 汇总（runs_smoke.jsonl → metrics_smoke.json；主实验建议另存 runs.jsonl，
#    改 ge2_runner.py 的 RUNS_JSONL 或直接复用按 phase 字段过滤）
python -u experiments/ge2_runner.py --analyze
```
每 run 记录：completed/partial/per_test（逐测试）/turns/pytest_runs/
advisory 投递与时机/pg 伪调用遥测（pseudo_total/forms/edit_intents/
real_edits/edit_landed/pg_triggers）/api_errors/pg_errors/truncated/
model_identity（单模型断言，polluted 即预登记排除）/session 文件名。
env：`PI_MODEL` 换模型（须记录并标记 excluded）、`PI_RUN_TIMEOUT`（默认
420s）、`PI_GUARD_URL`（默认 18617）。

## 7. 数据与证据索引

- 逐 run 记录：`results/guard-effect-v2/runs_smoke.jsonl`（32 run：
  smoke 2 + 门禁 6 + 校准探针 24）；
- 汇总：`results/guard-effect-v2/metrics_smoke.json`（--analyze 产物）；
- 会话留存：`results/guard-effect-v2/agent_home/sessions/`（33 个 jsonl，
  含 2 个 deepseek 中止会话）；
- 逐 run pi 输出：`logs/pi_*.stdout.log` / `logs/pi_*.stderr.log`；
- 网关审计：`logs/proxy_requests.jsonl`（254 请求）+ `logs/proxy_stdout.log`；
- 运行日志：`logs/smoke_both.log`、`logs/gate6.log`、`logs/probe_band.log`、
  `logs/probe_band2.log`、`logs/probe_ds.log`；
- runner/proxy/任务集：`experiments/ge2_runner.py`、`experiments/ge2_proxy.py`、
  `experiments/ge2_tasks.json`（**规范副本在 experiments/，results/ 下三份为
  交付快照**，以 experiments/ 为准）；
- 工作区（每 run 最终代码，judged 复测依据）：`results/guard-effect-v2/ws/`。

## 8. 遗留与边界

- 难度带未出带 = 门禁 2 未过；本报告不构成"守卫无效"或"有效"的任何结论
  （预登记 §4：区分"证明无效"与"没证明有效"）。
- s3_dual_mix（降级备选）未跑——降级路径只针对过难，本批问题是对称的过易。
- 未测备用渠道：commandcode/deepseek-v4.1-flash、其余 workbuddy/zhushu 模型
  （kimi-k2.8-code、glm-5.3、MiniMax-M2.5/2.7、nemotron 等在 /v1/models 在列），
  主代理若走换模型路径需先过冒烟 + 重跑门禁 1/2。
- proxy 重试逻辑只覆盖"响应体未开始消费"的失败；流中挂断仍会直达 pi
  （记 api_error），与预登记排除口径一致。

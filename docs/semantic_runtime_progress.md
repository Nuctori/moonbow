# 语义匹配运行时：实施进度（恢复文件）

计划：`docs/semantic_match_runtime_plan.md`
启动：2026-09-29，并行上限 5，subagent 分批派发。

## 硬约束（用户指令，2026-09-30，优先级最高）
0. **禁止 legacy/rule 判定路径**（2026-09-30 追加）：语义匹配一律走
   SemanticMatcher 的 SLM 后端（XPU 推理）；legacy/rule 不得作为新代码的
   默认或推荐路径。当前代码默认值仍是 legacy（历史行为、兼容保留），
   模型质量达标后须将默认 provider 切换为 semantic。详见根目录 AGENTS.md。
1. **禁止 CPU 跑 AI 推理；所有模型推理必须在 XPU（Intel Arc A770）上执行**，
   否则进程全卡死。已验证：`.venv_xpu/Scripts/python.exe`，torch 2.14.0+xpu，
   `torch.xpu.is_available()=True`，Arc A770 16GB 显存，matmul 冒烟通过。
2. 真实模型批量推理任务单独派发，不与任何其他 subagent 并行；
   推理脚本分批落盘（每 ≤25 条一档），批间留间隔。
3. 批量评测必须支持断点续跑，跑前确认设备为 xpu（脚本内断言
   `torch.xpu.is_available()`，否则直接拒绝运行）。
4. 事故记录：2026-09-30 P8 评测原计划在 CPU 上跑 450×4 次推理并 3 代理并行，
   导致用户桌面进程卡死；P8 评测与 P4 修复两代理被终止，未产生残留进程。

## 当前阶段
- P0 完成（2026-09-30 冻结基线，见证据索引）
- P1 契约冻结完成（2026-09-30，见"已完成"；publish_repo 未包含 semantic/）
- P3 进行中（模型探测，早期部分）
- P8 完成（2026-09-30，校准与离线评测流水线；test split 未跑，留最终验收）
- P2 完成（2026-09-30，共享 Runtime 与 Fake backend，见"P2 进展"）
- P11 完成（2026-09-30，打包/文档/发布就绪证明：A/B 完成、C no-go、D/E 未满足，
  semantic 默认关闭 opt-in only，见 P11 段）
- P9/P10 完成（2026-09-30，真实宿主验证、故障注入、回滚演练、有界长稳 +
  XPU 真实模型 50 请求宿主冒烟，见"P9/P10 进展"；
  SLM no-go 结论不变，semantic 路径仍不可接入交互路径，默认 legacy/rule）

## 已完成
- 计划编写与结构校验（2026-09-29）
- P0 冻结基线（2026-09-30）：环境清单、测试基线（三组命令全绿：185/32/15）、模型清单（models/ 133G + models_archive/ 108G）、发布源确认、源文件 sha256（25 个 .py/.ts）。

## 剩余
P2/P3/P4/P5/P6/P7/P8/P9/P10/P11 全部实施与验收（P2 起补 matcher/runtime/calibration/client/backends）。

## P1 进展（契约冻结，2026-09-30）
- `src/moonbow/semantic/schema.py`：MatchRequest/MatchResponse/Evidence/
  Provenance/PatternSpec dataclass + JSON 序列化 + 严格校验
  （pattern/requirement 二选一、operation 仅 match|find_all、threshold 可选 [0,1]、
  matched 三态 true/false/null 且 abstain/error 必须 null、score 有限且 [0,1]
  拒 NaN/越界/非数、reason_code 7 值枚举、引文半开区间 code point 偏移、
  未知顶层字段一律拒绝）。
- `src/moonbow/semantic/patterns.py`：PatternRegistry（精确版本查找，不静默回退）
  + check_request_context（所需上下文缺失即拒绝）。
- `src/moonbow/patterns/*.json`：5 个初始固定模式 completion.asserted@1、
  modality.assertive@1、task.object.alignment@1（requires_context=["task"]）、
  process.unresolved@1、ts.capture@1（find_all，kind 枚举与 task_structure
  8 类一致性由加载器强制校验）。各含定义、≥3 正例、≥3 反例、排除项、
  operation、输出 schema、评分方式。
- `validate_response(req, resp)`：证据偏移/引文 verbatim 一致性、match ≤1 条
  证据、provenance.pattern_version 必须回指请求 pattern、截断 find_all
  不得宣称全量"无捕获"负例。
- `tests/test_semantic_contract.py`：97 用例全绿，纯逻辑不依赖 torch/权重。

## P3 进展（早期探测，2026-09-30）
- 身份核实：候选 `qwen3.8 0.8b` **不存在**。Qwen3.8 系列最小为 27B（HF API 证实：Qwen3.8-27B / Flash-Next / 2.4T-A95B 及 FP8，无 0.8B 变体）。近似候选 `Qwen/Qwen3.5-0.8B`（Apache-2.0，0.9B，262k ctx）已在本机 HF 缓存（safetensors 1.63GiB），但**非用户所指型号，不冒充匹配**。
- 本地可行性：Qwen3.5-0.8B 被已装 transformers 阻塞（.venv 4.57.6 / .venv-xpu 5.16.1 / .venv_xpu 5.17.0 的 AutoConfig 均不识别 `model_type: qwen3_5`）。
- 参考 probe：models/qwen05（Qwen2 0.5B 级）在 .venv 完整跑通，exit 0：494.0M 参数 bf16，加载 16.2s，RSS 峰值 1437.5MB，4 句 ×3 次 max_new_tokens=1 全部稳定（均输出 "no"，~550ms/token CPU），标签首 token logits 可取（yes/no 可索引）。类别偏置明显（4 句全部 no，与真值不符）。
- 探测脚本：`tools/probe_semantic_model.py`；证据：`results/semantic-runtime/p3-probe/`（model_identity.md、probe_report.md、probe_raw.json、dev_diag_samples.jsonl）。
- P3 剩余：候选模型决策（升级 transformers / 更换候选 / 复用现有小模型）→ 对真实候选重跑 probe → 120 条诊断样本 → 结构化抽取原型。

## P8 进展（开发样本建设，早期部分，2026-09-30）
- 新建 `data/semantic_dev_v1/`：dev_texts.jsonl（120 条原创去重文本）、dev_labels.jsonl
  （5 pattern provisional 标签 + ts.capture 捕获/abstain + label_basis）、README.md
  （家族分布、标签约定、provisional 声明、与冻结测试集的关系）。
- 分布：agent_completion 30 / interim_report 30 / user_request 40 /
  conversation_interference 20；zh 85 / en 35（约 7:3）。
- 标签依据 task_structure_spec §2/§3：取消优先、伪任务不捕获（引用/围栏/例如/背景/参考）、
  一子句一主捕获、承前并列拆分、顿号列表不拆、先…再…合并为单个 dependency。
- 声明：全部标签 provisional（agent 自审，无人工复核，非金标准）；本集仅用于
  P3/P4 诊断，不得用于最终验收，冻结测试集另行按 P8 数据方案拆分。
- 已知缺口（冻结阶段补齐）：捕获 kind 中 condition/unresolved/acceptance/object
  分别仅 2/2/1/1 个正例，未达每类 ≥30；无长文本样本与改写对。

## P8 进展（冻结测试集构建，2026-09-30）
- 新建 `data/semantic_frozen_v1/`：600 条去重原创文本/片段（fz_0001–0600）+
  三组分列标签（provisional）+ README。总量目标 ≥800，本批交付 600，缺口 200 已在 README 登记。
- 拆分：200 个源任务整组分配 dev 100 / calib 50 / test 50（300/150/150 条）；
  改写对同组（validator 校验 task_id 不跨组）；每任务前两条互为改写对。
- 家族分布（全集）：user_request 263 / conversation_interference 130 /
  agent_completion 114 / interim_report 93；zh 398 / en 202（约 7:3）。
- 长文本 >800 字符 20 条（dev 5 / calib 1 / test 14）；对抗/边界（注入、伪任务、
  取消、过去叙述、多语言混排、围栏/引用/示例）约 150 条（目标 ≥60）。
- test 组 8 类正例全部达标（goal 51 / object 94 / constraint 60 / dependency 41 /
  coordination 50 / condition 42 / unresolved 40 / acceptance 39，均 ≥30，无缺口登记）。
- 5 pattern 正例（全集）：completion.asserted 89 / modality.assertive 178 /
  task.object.alignment 429 / process.unresolved 173；ts.capture 捕获 915 个、abstain 346 条。
- 标注：纯规则人工裁决（未调用模型预标注），每条带 label_basis；取消优先、
  伪任务不捕获、一子句一主捕获、先…再…合并依赖等按 spec §3 执行；模糊样本弃权。
- README 声明 provisional（agent 自审，无人工复核，非金标准）；test 组为冻结评测用途。
- 校验：`python tools/validate_frozen_data.py` → exit 0（id 唯一、三组无交叉、
  quote 逐字、kind 枚举、8 类计数表、与 dev 集及组内 bigram Jaccard>0.8 查重零命中、
  家族/语言/长文本分布统计）。
- 已知缺口（README 登记）：总量差 200；calib 组长文本仅 1 条；dev/calib 组部分
  kind（acceptance/dependency/coordination/condition）正例偏少（验收目标仅约束 test 组）。

## P2 进展（共享 Runtime 与 Fake backend，2026-09-30）
- `src/moonbow/semantic/backends/base.py`：Backend 抽象协议——name/
  model_revision/BackendInfo 元数据、capabilities()、幂等 initialize()
  （init 锁 + load_count 可验证）、close()、match(request)->MatchResponse；
  BackendError 家族（unavailable/overloaded/timeout/invalid_output）由
  runtime 统一映射为契约内 error 响应；TRANSIENT_REASON_CODES 定义可重试
  瞬态故障。
- `src/moonbow/semantic/backends/fake.py`：FakeBackend——可编程确定性后端：
  rules（pattern ref -> FakeRule 固定 matched/score/quotes/relation，或
  callable）、四种故障注入（timeout/invalid_output/unavailable/overloaded）、
  delay 延迟与 gate 确定性阻塞；引文不是原文子串时按 invalid_output 拒绝
  （不修饰成可信命中）；无规则默认确定性负例；capabilities 标注 fake=true；
  仅用于契约/故障/集成测试。
- `src/moonbow/semantic/runtime.py`：SemanticRuntime——显式依赖注入 backend；
  单推理 worker 线程顺序执行；状态机 idle/loading/ready/draining/failed；
  首次使用加载一次（load_count 指标来自 backend.load_count）；有界队列
  （默认容量 8，满时立即返回 overloaded，不无限排队）；每请求 deadline
  （超时返回 reason_code=timeout，与 overloaded/unavailable 可区分）；
  TaskHandle 取消（排队未开始可取消且不执行，运行中不可中断）；
  close() 标记 draining、等待运行中推理完成（可配超时）、排队任务一律
  返回 unavailable；健康检查不经过推理队列不被慢推理阻塞；指标
  requests/overloaded/timeout/errors/load_count/retries；瞬态故障重试
  上限默认 2 次，无自动无限重试；全部出口响应强制过 validate_response，
  未通过按 invalid_output 处理。
- `src/moonbow/semantic/matcher.py`：SemanticMatcher 门面——match/find_all/
  batch（上限默认 16，逐项调用逐项结果，单项失败显式 error 标记不丢结果）；
  get_or_create registry（键=backend 身份+runtime 配置，同进程同键共享
  同一 runtime，工厂只调用一次）；Matcher 无会话状态，会话间无串扰。
- `tests/test_semantic_runtime.py`：25 用例覆盖 §4-P2 门禁——并发首用
  load_count==1、瞬态重试上限（call_count==1+max_retries）且非瞬态不重试、
  取消排队任务不执行、close() 排队 unavailable/运行中等待完成、队列满
  立即 overloaded（<0.2s）、timeout/overloaded/unavailable 三者可区分、
  健康检查在慢推理期间 <0.2s 响应、两个 Matcher 同键共享 runtime 且同文本
  结果一致互不污染、全部响应过 validate_response、registry 分键隔离与
  工厂单次调用、故障注入矩阵、加载失败进入 failed 不再接推理、
  自由 requirement 路径。纯逻辑，不依赖 torch。
- P1 契约零改动：schema.py/patterns.py 未修改；P2 全部复用 P1 schema。

## P4 进展（真实 SLM backend 与防御性输出校验，2026-09-30）
- **环境决策（修正 P3 结论）**：P3 报告「transformers 5.17.0 不识别 qwen3_5」有误——`.venv_xpu`（transformers 5.17.0，注册表含 `Qwen3_5ForCausalLM`）实测可离线加载 Qwen/Qwen3.5-0.8B（P3 当时大概率在错误环境探测）。**复用 `.venv_xpu`，未新建 venv、未改动任何已装包**，无新增依赖（flash-linear-attention/causal_conv1d 缺失仅影响速度，不影响正确性）。
- `src/moonbow/semantic/backends/slm.py`：`SLMBackend`（实现 P2 `Backend` 协议，构造即加载一次、initialize 幂等）：
  - 离线加载 HF 缓存指定 model_revision（refs/main 解析 commit sha 记入 provenance；显式 revision 优先）；CPU dtype 按可用内存自动选 float32（≥6GB）/ bfloat16，实测 float32。
  - 二值模式（completion.asserted / process.unresolved）：受限判定不生成——标签（yes/no）首 token log-softmax 得 score（score_type=label_probability），matched 按 threshold；诊断记录 label_spread 检测类别偏置。
  - modality.assertive：assert/promise/question 三标签同法。
  - task.object.alignment：双通道确定性打分（text 上下文 vs 中性前缀，teacher-forcing 比较 task 串平均对数似然），差值不是概率 → **score=null + calibrated=false**；缺 context.task → abstain insufficient_context。
  - ts.capture：受限 JSON 生成（max_new_tokens=256 上限），括号平衡提取（截断/复读不误解析），逐条过防御校验（kind 枚举、quote verbatim 硬条件、偏移校验+verbatim 回退定位）后才转 Evidence；坏 JSON/全部条目不可信 → error|abstain invalid_output，不冒充可信命中；截断 find_all 无证据 → abstain truncated。
  - prompt 模板版本化（slm-prompt-v1）：用户文本进 `<|text_begin|>/<|text_end|>` 不可信数据区，区内定界符先中和（防越狱出区），模板版本记入 capabilities 与 per-call diagnostics。
- `tests/test_semantic_adapters.py`：52 用例全绿（fake 注入测 prompt 模板/注入中和/输出解析/坏输出拒绝/偏移校验/score=null/capabilities/provenance/revision 解析；真实权重部分 skipif 门控且本环境实际执行未跳过）。P2 runtime/P1 契约测试无回归。
- dev30 实测（前 30 条 × 4 match 模式，全部真实运行）：加载 11.2s，RSS ~4.0GB，判定延迟 p50=1361ms / p95=1544ms。completion/modality P=R=1.00（样本构成所致，前 30 条全为 agent_completion 正例，不作能力证明）；process.unresolved 20 FP（yes 偏置 + 高自信 0.97，阈值不可救）；alignment 29/30 因缺 task_context 正确弃权。ts.capture 冒烟：防御层全部生效（错偏移/复读截断不冒充），但负例 2/2 误捕获、kind 全归 goal、延迟 ~60s——捕获质量当前不可用（诊断结论）。
- 对 P5 建议：延迟门禁需先解决 linear_attention 参考实现（装 flash-linear-attention 或换运行时）；ts.capture 改 few-shot/逐 kind 判定前不得接入交互路径；SLM 分数一律 raw_score 语义，校准进 P8。

### P4-FIX（XPU 移植与门禁修复，2026-09-30）
- **XPU 强制（硬约束落地）**：`slm.py` 新增 `_require_xpu()`——真实权重加载路径强制 `torch.xpu.is_available()`，False 即抛 `BackendUnavailable` 拒绝运行，不回退 CPU；`device` 默认 `"cpu"`→`"xpu"`，加载路径显式拒绝非 xpu device；全部输入 tensor `.to(xpu)`。dtype 默认 **bfloat16**：fp16 对照（5 样本，`p4-slm/fp16_probe.json`）判定一致率 1.00、p_yes 最大差 0.0025，选 bf16（数值等价且无半精度溢出风险，显存减半）。
- **门禁缺陷（45 passed + 2 errors + INTERNALERROR）**：独占执行下多角度复现均失败（带/不带 offline env、cacheprovider 均全绿），无失败日志留存；根因判断为 2026-09-30 多代理并行时段资源争抢——模块级 real_backend fixture 加载失败（Windows 内存提交不足 os error 1455）→ setup error 级联，及收集期 `_real_model_available()` 的 `snap.iterdir()` OSError。修复（不删用例/不放宽断言/真实用例不 skip）：snapshot 枚举包 try/except OSError；新增 `TestXpuEnforcement` 3 用例（device 默认 xpu、无 XPU 时 `_require_xpu` 必抛、本机通过）；真实用例增加 `info.device=="xpu"` 断言。用例数 52→55。
- **XPU dev30 复测**（`p4-slm/run_dev30_xpu.py`，脚本内断言 XPU 否则 exit 2；每 10 条一档分批落盘）：加载 6.69s（CPU 11.20s）；显存（torch.xpu.mem_get_info）加载后 2.11GB、稳态 2.72GB/16GB（CPU RSS ~4.0GB fp32）；判定延迟 **p50=126ms / p95=186ms**（n=91；CPU 1361/1544ms，~8-11x）。P/R 与 CPU fp32 **逐条零差异**（120 项 matched 状态对比 0 diff）：completion/modality 1.00/1.00、alignment 29 abstain（同因缺 task）、process.unresolved 0.33/1.00（20 FP yes 偏置原样保留）；|p_yes-p_no| 中位 0.943 ≈ CPU 0.944。ts.capture 冒烟 3 条：~35s/条，2 error invalid_output + 1 abstain truncated——防御层全部正确生效（不冒充可信命中），捕获质量结论与 CPU 一致：当前不可用。
- **P2/P4 只读复核**（`results/semantic-runtime/p4-fix/review_findings.md`）：P2 六项门禁声明（单次加载/队列满 overloaded/排队可取消/close 排空/健康检查不经队列/引文校验）全部与实现相符；slm.py 四项安全（logits 评分无自由生成、缺 context 必弃权、quote verbatim 不可绕过、定界符中和——11 例对抗样本脚本验证不可逃逸）通过；score=null 路径唯一。**无 P0/P1**；F1–F9 共 9 条 P3 findings 留档（load_count 注入路径为 0、abstain/error 出口未过 _finish、重复引文恒取首次 occurrence、close 后 state 显示 idle、qsize 近似等）。P2 三文件本轮零改动。

## P5 进展（统一 SDK/HTTP 与可替换性，2026-09-30）
- `src/moonbow/semantic/http.py`：独立 HTTP 服务（stdlib ThreadingHTTPServer，风格对齐
  guard/server.py，零新依赖）：
  - 端点：POST /v1/match、/v1/find-all、/v1/batch；GET /health（身份 semantic-runtime +
    state/queue_depth/metrics）、GET /ready（runtime 未 ready → 503）、GET /capabilities
    （backend 名、model_revision、内置 pattern 清单及版本/operation/threshold、
    calibrated 能力声明、limits）。
  - 错误码按 P1 契约：非法 JSON/字段/二选一违例 → 400；text 超 65536 code point 或
    请求体超 2MB → 413（先排空已声明请求体再应答，保证客户端收到 413 本身）；
    队列满 reason=overloaded → 429；unavailable → 503；timeout → 504；未知路由（含
    POST/GET /check、/、/v1/* 之外一切）一律 404，绝不误落到任何处理器。
  - 契约响应体（status=ok/abstain/error）即使配 4xx/5xx 状态码也原样返回，
    SDK/HTTP 才能逐字段等价；batch 逐项对齐、单项失败为契约内 error 条目不丢结果。
  - 默认绑定 127.0.0.1；日志默认 INFO 不输出任何请求行/原文（行内容仅 DEBUG）。
  - backend 由配置注入（type=fake|slm，slm 延迟导入 torch）；fake 规则支持
    JSON 反序列化（matched/score/quotes/relation/truncated 及 abstain+reason_code 规则
    函数），`rules_from_config` 同时供测试在进程内重建等价 backend；启动入口
    `python -m moonbow.semantic.http --config ...`（argparse，可覆盖 host/port），
    warmup 线程先预热再 ready。
- `src/moonbow/semantic/client.py`：HTTPClientBackend 实现 P2 Backend 协议：
  - initialize() 拉 /capabilities 用服务端真实身份回填 info；match 按 operation 路由
    /v1/match 或 /v1/find-all，provenance 由服务端产生并透传，客户端不改写。
  - 失败隔离：连接失败/超时/非 200/坏 JSON 全部折叠为契约内 error 响应
    （超时→timeout、连接失败/5xx→unavailable、400/404/413→unsupported、
    429→overloaded、200 坏体→invalid_output），不抛异常穿透。
  - loopback 直连绕过系统代理（环境代理会把 127.0.0.1 请求路由出去并伪装成 502，
    本轮实测发现的坑）。
- `config/semantic_runtime.example.json`：server(host/port/max_body_bytes)、
  runtime(queue_capacity/deadline/max_retries/max_batch/warmup)、backend(fake 示例 +
  _slm_example 注释示例)。
- `tests/test_semantic_http.py`：10 用例全绿——SDK(进程内 matcher)/HTTP 同一组 smoke
  请求（ok 正例/默认负例/abstain/error(invalid_output)/find_all/batch 单项失败）逐字段
  等价；400/413/404/429/503/504 全码验证（含 /check 不误落、operation 与端点不匹配 400、
  batch 超 16 上限 400、超长 batch 单项按 error 标记）；服务身份 semantic-runtime 与
  guard(progress-guard, 18492) 可区分、动态端口非 18492；并发 4 客户端 20 请求同实例
  load_count==1；warmup=false 时 /ready 503、首个请求加载后翻 200；替换演练：同一套
  smoke 插件代码零改动打两种 fake 配置（fake-rev-A matched/score 0.9 vs fake-rev-B
  matched=False/0.2），状态结构不变、结果与 provenance.model_revision 随配置变化
  （真实第二模型替换留待真实权重环境用同一 smoke 套件复跑）；子进程 sys.executable
  启动/terminate 干净、端口释放、服务日志无请求原文。
- P1-P4 文件零改动；P5 只新增 http.py、client.py、示例配置与本测试。
- Windows 环境备注：本机防火墙把部分"连接拒绝"静默成超时，客户端网络级测试对
  unavailable/timeout 两种契约内 reason 均接受；确定性映射由 stub 单测覆盖。

## P8 进展（校准与离线评测流水线，2026-09-30）
- **硬约束执行**：全部推理在 `.venv_xpu`（torch 2.14.0+xpu，A770 16GB，bf16）；
  脚本/后端双重断言 `torch.xpu.is_available()`；逐文本落盘（每 4 行 flush，
  远小于 25 条上限）；断点续跑按"4 pattern 齐全的 id 跳过"。test split 脚本内拒绝
  （留最终验收）。**发现并纠正上一被终止代理遗留的 CPU float32 calib 产物**
  （p50 1595ms，违反硬约束），整体隔离至 `p8-eval/stale_cpu_run/` 并全部 XPU bf16 重跑。
- `tools/semantic_eval.py`：4 match 模式离线评测（ts.capture 关闭，P4 已证不可用；
  `--include-ts-capture` 显式确认但不实现——P7 走 rule）。支持 --split dev|calib
  （test 拒绝）、--limit、--from-predictions 重算、--report、--emit-config、
  断点续跑。task.object.alignment 的 task 上下文为**派生**（同 task_id 组首条文本，
  冻结集无原生 task_context；逐行记录 task_source_id，结论仅限该口径）。
  指标：每 pattern P/R/F1（Wilson 95%CI）、混淆矩阵、弃权率、coverage-risk、
  阈值扫描 [0.50..0.99]（score 重扫，零重复推理）、reliability 10 箱 + ECE/Brier
  （校准前 vs 温度缩放 vs isotonic；calib 集内 fit/eval 对半交叉，dev 应用 calib
  全量拟合）。阈值预登记规则：F1 最大→precision 高→阈值高；另报 P>=0.95 与
  R>=0.90 参考线可达性。校准产物 `config/calibration/qwen3.5-0.8b/threshold_*.json`
  （绑定 model_revision=2fc063…、pattern_version、选法、样本量；calibrated 仅当
  ECE<=0.05 且 Brier 不恶化——**4 pattern 全部 false**，最接近者 process.unresolved
  isotonic ECE=0.057）。
- calib(150) 结果（XPU bf16，p50 127-135ms 判定 / 238ms alignment，全程 108s）：
  - completion.asserted：默认 P=0.200 R=1.000；选阈值 0.97 → P=0.239 R=0.700
    F1=0.356。R>=0.90 可达（t=0.96，P=0.207）；P>=0.95 仅 t=0.98 的 R=0.033 退化点。
  - modality.assertive：默认 P=0.353 R=1.000；选阈值 0.70 → P=0.406 R=0.981 F1=0.575。
    P>=0.95 仅 t=0.97 的 R=0.057 退化点。
  - task.object.alignment（diff>0 固定边界，无概率）：P=0.757 R=0.982（弃权 0，
    因冻结集全部可派生 task 上下文）。
  - process.unresolved：默认 P=0.273 R=1.000；选阈值 0.95 → P=0.277 R=1.000 F1=0.434；
    **P>=0.95 全表不可达（最高 0.345）**。
  - ECE/Brier（raw→最优）：completion 0.770/0.752→iso 0.057/0.151；modality
    0.458/0.421→iso 0.084/0.211；process 0.695/0.681→iso 0.088/0.200；alignment 无概率。
- dev(300) 迁移评估（**一律用 calib 所选阈值**，dev 自身扫描仅诊断留档）：
  completion P=0.178 R=0.816（recall 不达标）；modality P=0.309 R=0.953（recall 达标）；
  process P=0.264 R=1.000（precision 不可用）；alignment P=0.690 R=1.000。
  process 在 dev 上 isotonic ECE=0.035 数值达标，但校准判定权在 calib（仍 false）。
- **yes 偏置量化（必答）**：calib 负例 score 中位数 completion 0.9707 / process
  0.9689（负例 100% score>=0.9），modality 0.807；正负中位差 completion 0.002 /
  process 0.0000 / modality 0.044；|p_yes-p_no| 中位 ≈0.94 —— 模型对标签分布极自信
  但方向与真值无关。属 prompt/模型层偏置：阈值与校准层（只改分数不改排序）均不可修复。
- `tools/semantic_bench.py`（重写）：XPU 断言；并发走 P2 SemanticRuntime 单 worker
  队列（容量 8）；显存峰值 torch.xpu.mem_get_info 采样（本轮可用），RSS 兜底。
  矩阵 128/512/2048 token × 单/4 pattern × 并发 1/2/4 + 主负载 4×50：**19 格全部
  0 error**；单 pattern 128tok 并发1 p50=203ms/p95=425ms；4 pattern 2048tok 并发4
  p50=989ms/p95=1286ms（唯一 p95>1s 格）；主负载 p50=796/p95=851ms；XPU 显存峰值
  3.0-7.8GB/16GB。
- 运行事故（透明记录）：dev 全量至 104/300 时参考内核（chunk_gated_delta_rule，
  未装 flash-linear-attention）单调用卡死 20 分钟（py-spy 确认栈在
  torch_chunk_gated_delta_rule），终止；续跑再遇 **segfault exit 139**；逐条落盘
  使零数据丢失，最终一次续跑完成 300 条（1200 行完整无坏行）。
- 质量门禁对照（详见 quality_gate_report.md）：结构测试达标（引用 P1/P2/P4 既有
  全绿）；recall>=0.90 仅 modality（迁移后 0.953）与 alignment 达标，completion
  不达标；precision>=0.95 实际不可用（仅退化点形式可达，process 完全不可达）；
  结构抽取不评估（ts.capture 阻塞）；引文有效率不适用（match 模式无证据输出）；
  校准 4/4 false；服务完成率 100%（0 error 0 弃权，150/300×4 全 ok）。test 未跑。
- 对 P6/P7 判断：SLM 语义路径在当前偏置下**不可接入交互路径**（precision 全线
  崩塌）；ts.capture 维持关闭走 rule 的 P7 结论不变。

## P6 进展（Guard 迁移，legacy 默认，2026-09-30）
- 交付：`src/moonbow/guard/semantic_provider.py`（GuardSemanticProvider 协议 +
  LegacyProvider + SemanticProviderAdapter）、verifier/server 接线、
  `tests/test_guard_semantic_provider.py`（53 用例）。
- provider 抽象：provider 只产出 SemanticObservation（modality / similarity /
  capture_prob / capture_complete / extra_scores / unavailable_reason），
  阈值比较、信号文案、advisory/strict 区分、硬信号升级、提醒预算全部留在
  verifier——legacy 路径的决策逻辑逐行未动。
- LegacyProvider：包装 ModelRegistry 三调用（顺序与迁移前一致：modality ->
  similarity -> capture），不吞异常（legacy 推理失败仍向上抛，保持 eager 500 语义）；
  verifier 构造函数新增 `provider=None`，None 时保持原 lazy_load 骨架降级语义。
- SemanticProviderAdapter：每个信号一次 matcher.match(pattern=...)；SLM 分数只写
  新命名键 `slm_capture_score@1`，绝不写旧 similarity/capture_prob 键；任一信号
  失败（契约内 error / matched=null / 异常）即整体 unavailable，verifier 进入既有
  骨架降级并写 `skeleton_only=true` + `semantic_unavailable=true`，不伪造通过。
- verifier 新增 `scores["provider_kind"]`（"legacy"|"semantic"|"unavailable"）新键；
  advisory/strict、allow_stop/acceptance/is_closed、硬信号、提醒预算逐行未动。
- server：新增 GET `/v1/semantic-status`（只读 provider_kind/backend/calibrated）；
  /check 可选 `semantic_provider`（"legacy"|"semantic"，默认 legacy 与既有行为一致）；
  请求 semantic 而服务未配置运行时 -> 400 明确报错（不静默回退）；
  /v1/stage-check、/v1/task-structure 与既有 /check 默认行为未改。
- 未改 src/moonbow/semantic/ 任何文件；process_audit.py、task_structure 未动（P7 范围）。

## 阻塞
- ~~真实模型无法加载~~（P4 已解除）：Qwen3.5-0.8B 可经 `.venv_xpu` transformers 5.17.0 离线加载。剩余阻塞是**质量**而非可运行性：process.unresolved 类别偏置、ts.capture 捕获质量与延迟未达标（见 P4 段），真实模型验收仍不能宣称完成。
- 环境风险：torch 为 2.14.0+cpu/xpu（无 CUDA），pip 中 torch 元数据损坏（`~orch` 无效发行版），后续依赖锁定前需修复。

## 证据索引
- P0 run 目录：`results/semantic-runtime/p0-baseline/`
  - `manifest.json`：环境（Python 3.11.9 / Node v24.13.1 / torch 2.14.0+cpu / transformers 5.17.0 / CPU only / 64GB 内存 / 32 核）、模型清单、哈希索引
  - `hashes.json`：src/moonbow 下 25 个 .py/.ts 的 sha256
  - `test_baseline_report.md`：测试分类清单与三组命令执行记录
  - `publish_source_check.md`：publish_repo 为过期快照（缺 semantic/、patterns/），无同步脚本证据
- P0 命令 exit code：pytest 回归组=0（185 passed）；node --test=0（32 pass）；pytest test_guard.py=0（15 passed，有权重）
- P1 命令 exit code（2026-09-30）：
  - `python -m pytest tests/test_semantic_contract.py -q` → exit 0，97 passed
  - `python -m pytest tests/test_task_structure.py -q` → exit 0，41 passed（无回归）
- P8 命令 exit code（2026-09-30）：
  - `python tools/validate_dev_data.py` → exit 0（id 唯一、两文件 id 一一对应、
    quote 逐字子串、ts.capture kind 8 类枚举校验通过；家族/语言/标签分布统计见输出）
  - `python -c` 逐行解析 dev_texts.jsonl / dev_labels.jsonl → 各 120 行可解析
- P2 命令 exit code（2026-09-30）：
  - `python -m pytest tests/test_semantic_runtime.py tests/test_semantic_contract.py -q`
    → exit 0，122 passed（25 runtime + 97 contract）；runtime 单独复跑 3 次均 25 passed 无 flake
  - `python -m pytest tests/test_task_structure.py tests/test_guard_regressions.py -q`
    → exit 0，139 passed（无回归）
- P4 命令 exit code（2026-09-30，解释器 `./.venv_xpu/Scripts/python`，
  环境变量 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`）：
  - `python -m pytest tests/test_semantic_adapters.py -p no:cacheprovider`
    → exit 0，52 passed（0 skip；含真实模型用例本机实际执行）
  - `python -m pytest tests/test_semantic_adapters.py tests/test_semantic_runtime.py tests/test_semantic_contract.py -q`
    → exit 0，174 passed（无回归）
  - `python results/semantic-runtime/p4-slm/run_dev30.py` → exit 0
    （dev30_predictions.jsonl + dev30_report.md 落盘）
- P4 run 目录：`results/semantic-runtime/p4-slm/`
  （run_dev30.py、dev30_predictions.jsonl 120 行、dev30_report.md 含 ts.capture 冒烟附录）
- P4-FIX 命令 exit code（2026-09-30，解释器 `.venv_xpu/Scripts/python.exe`，env `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`）：
  - `python -m pytest tests/test_semantic_adapters.py -q` ×3 连续 → 均 exit 0，
    55 passed（无 error / INTERNALERROR / skip）
  - 默认 python：`python -m pytest tests/test_semantic_contract.py tests/test_semantic_runtime.py -q`
    → exit 0，122 passed
  - `.venv_xpu`：同命令 → exit 0，122 passed（1 warning，无失败）
  - `python results/semantic-runtime/p4-slm/run_dev30_xpu.py` → exit 0
    （dev30_xpu_predictions.jsonl 120 行分批落盘 + dev30_xpu_capture_smoke.jsonl
    3 行 + fp16_probe.json + dev30_xpu_report.md）
- P4-FIX run 目录：`results/semantic-runtime/p4-fix/`（fix_report.md、review_findings.md）
- P5 命令 exit code（2026-09-30，解释器系统 python 3.11.9，无新增依赖）：  - `python -m pytest tests/test_semantic_http.py -q` → exit 0，10 passed
  - `python -m pytest tests/test_semantic_http.py tests/test_semantic_runtime.py tests/test_semantic_contract.py -q`
    → exit 0，132 passed（10 http + 25 runtime + 97 contract）
  - `python -m pytest tests/test_task_structure.py tests/test_guard_regressions.py -q`
    → exit 0，139 passed（无回归）
  - 服务测试全程动态端口（实测非 18492），子进程测试后 terminate 且端口释放断言通过；
    未遗留常驻进程。
- P6 命令 exit code（2026-09-30，纯逻辑改造，未加载模型推理）：
  - `python -m pytest tests/test_guard_semantic_provider.py tests/test_guard_regressions.py
    tests/test_advisory.py tests/test_guard.py -q` → exit 0，187 passed
    （41 新增 provider 用例 + regressions/advisory 全量 + 15 权重用例）
  - `node --test tests/test_pi_advisory.mjs tests/test_pi_strict_combo.mjs`
    → exit 0，6 pass / 0 fail
## P11（2026-09-30，打包、文档与发布就绪证明；纯逻辑，未加载模型/未占 XPU）

状态：完成（工程与文档部分）。总判定：**A/B 完成，C no-go，D/E 未满足**，
semantic 路径默认关闭 opt-in only（详见 release_readiness.md）。

- 打包配置修复（唯一 src 外配置改动，理由见 release_readiness.md）：pyproject
  package-data 为 moonbow 增加 `patterns/*.json`（原配置缺失，wheel 会丢 5 个内置
  pattern 资源）；移除无效条目 `models/**/*`（src/moonbow/models 不存在）。
- 构建：`python -m build` → exit 0，产出 dist/moonbow-0.1.0-py3-none-any.whl +
  moonbow-0.1.0.tar.gz。内容核对：patterns ×5 含、semantic/guard/task_structure/
  skills/extensions 全含；models/.venv*/__pycache__ 零命中（wheel 与 sdist 双查）；
  全包无本机绝对路径（M:\、C:\Users、.venv_xpu 零命中）。
- 干净 venv smoke：`python -m venv .tmp_venv_p11` + `pip install --no-deps`
  （torch/transformers 未装，任何 torch 导入路径即失败）→ smoke_pure_logic.py
  **10/10 exit 0**：schema/patterns(内置5)/runtime/matcher/http/client/fake/
  guard verifier+semantic_provider/slm 模块（torch 延迟）导入、torch 不入
  sys.modules、console scripts 存在、HTTP fake backend 子进程启停 + /v1/match
  ok 正例 + 未知路由 404。
- 文档：新建 `docs/semantic_api.md`（SDK match/find_all/batch、HTTP 端点与错误码、
  FakeBackend、配置示例、Guard provider 注入）、`docs/semantic_runtime_README.md`
  （能力状态如实表：统一接口/运行时/兼容层已验证；SLM 延迟达标但语义质量未过
  校准门禁，precision 不达标、4/4 calibrated=false，默认关闭不得用于生产提醒路径；
  不声称完成率提升）；DELIVERY_GUIDE.md 增 §6.5（简短+链接）。
- 发布就绪证明：`results/semantic-runtime/p11-release/release_readiness.md`
  （§8 A–E 逐条判定 + publish_repo 处置建议）。
- publish_repo 差异清单（`diff -rq --exclude=__pycache__`，exit 1）：publish_repo
  为过期快照——缺 semantic/、patterns/、guard/semantic_provider.py、
  guard/audit_semantic.py、task_structure/matcher_backend.py；
  guard/{process_audit,server,verifier}.py 与 task_structure/{extractor,schema,
  service}.py 版本不同。建议：待人工确认后再同步；本轮未同步、未 commit、未 push。
- 清理：临时 venv `.tmp_venv_p11/` 已删除；中间产物仅保留 dist/ 与 p11-release/。

### P11 命令 exit code（2026-09-30）
- `python -m build` → exit 0
- `pip install --no-deps dist/moonbow-0.1.0-py3-none-any.whl`（干净 venv）→ exit 0
- `.tmp_venv_p11/Scripts/python results/semantic-runtime/p11-release/smoke_pure_logic.py`
  → exit 0（10/10；首跑 9/10 是 smoke 脚本自身误用 registry API，修正脚本后全绿，
  包内容零改动）
- `diff -rq --exclude=__pycache__ src/moonbow publish_repo/src/moonbow` → exit 1
  （有差异，属预期，清单已留档）
- P11 run 目录：`results/semantic-runtime/p11-release/`
  （release_readiness.md、smoke_pure_logic.py、diff_src_vs_publish_repo.txt）
- P11 产物：`dist/moonbow-0.1.0-py3-none-any.whl`、`dist/moonbow-0.1.0.tar.gz`
（每阶段完成后追加 run 目录与命令清单）
- P8 命令 exit code（2026-09-30，解释器 `.venv_xpu/Scripts/python.exe`，
  env `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，全程 XPU bf16）：
  - `python tools/semantic_eval.py --split calib --limit 3` → exit 0（冒烟：bf16/XPU 确认）
  - `python tools/semantic_eval.py --split calib --emit-config` → exit 0
    （predictions_calib.jsonl 600 行 + metrics_calib.json + run_meta_calib.json +
    config/calibration/qwen3.5-0.8b/threshold_*.json ×4；推理 108s）
  - `python tools/semantic_eval.py --split dev --calibration ...metrics_calib.json`
    → 段错误 exit 139 ×2（XPU 参考内核不稳，见运行事故）后断点续跑完成，
    最终 exit 0（predictions_dev.jsonl 1200 行 + metrics_dev.json）
  - `python tools/semantic_eval.py --report` → exit 0
    （calibration_report.md + quality_gate_report.md）
  - `python tools/semantic_bench.py` → exit 0（benchmark.json + benchmark_report.md，
    19 格 0 error）
- P8 run 目录：`results/semantic-runtime/p8-eval/`
  （predictions_calib/dev.jsonl、metrics_calib/dev.json、run_meta_*.json、
  calibration_report.md、quality_gate_report.md、benchmark.json、
  benchmark_report.md、run_*.log、stale_cpu_run/【隔离的上一代理 CPU 产物】）
- P8 校准产物：`config/calibration/qwen3.5-0.8b/threshold_{completion.asserted,
  modality.assertive,task.object.alignment,process.unresolved}.json`
  （rev 2fc06364715b967f1860aea9cf38778875588b17；4/4 calibrated=false）

## P7（2026-09-30，任务结构迁移与过程审计语义接入；纯逻辑改造，未加载模型推理）

状态：完成。P8 结论 SLM 捕获质量不合格 → ts.capture 保持关闭（默认 rule），
本阶段交付目标是适配层与失败隔离的正确性，不是启用 SLM 捕获。

- 交付：
  - `src/moonbow/task_structure/matcher_backend.py`（新增）：MatcherExtractor——
    把 P2/P5 SemanticMatcher（或经 HTTPClientBackend 的 url）适配为
    ExtractorBackend 协议；pattern=ts.capture@1 按 kind 逐类 find_all
    （契约 evidence 不携带 kind，以 context={"kind":…} 圈定）；
    evidence 偏移与 quote 逐字对齐才用，否则回退 text.index；
    非 verbatim 引文丢弃；构造参数注入 matcher/url，两者皆缺抛
    ValueError；get_backend 默认行为不变（仍 rule）。
  - 失败隔离（修复计划 §1 适配缺口）：`schema.StructureAnalysis` 新增
    `backend_failed` / `backend_truncated`（默认 False，向后兼容），
    analyze_text 从后端的 last_backend_failed / last_backend_truncated /
    last_backend_abstain 读入；abstain → abstain_reason="backend_abstain"；
    rule 路径两字段恒 False（零行为变化）；service.analyze_payload 透传
    `backend_failed` / `backend_truncated` 到 HTTP 输出（只加键不改旧键）。
  - 过程审计 shadow 通道（`src/moonbow/guard/audit_semantic.py` 新增 +
    `process_audit.py` 最小改动）：StageAuditor(semantic_shadow=None) 可选
    注入；ProcessSemanticAdapter 对中间汇报 text 块跑
    matcher.match(pattern=process.unresolved@1)，结果只写入审计返回值新增的
    `shadow_semantic` 键（逐块 seq + matched/score/truncated/unavailable_reason）；
    不参与 reminder / findings / semantic 判定、不新增用户投递；时序核对、
    证据性质、发现修订逻辑一行未动；默认 None 时返回 dict 连键都不出现
    （零行为变化）。
- 新增测试：`tests/test_ts_matcher_backend.py`（16 例，全 fake）；
  `tests/test_process_audit.py` 追加 shadow 用例 6 例。
- P7 命令 exit code（2026-09-30，解释器系统 python，无新增依赖）：
  - `python -m pytest tests/test_ts_matcher_backend.py -q` → exit 0，16 passed
  - `python -m pytest tests/test_process_audit.py -q` → exit 0，31 passed
    （含 6 例新 shadow 通道；全绿即零行为变化证明）
  - `python -m pytest tests/test_task_structure.py -q` → exit 0，41 passed
  - `python -m pytest tests/test_server_task_structure.py
    tests/test_server_stage.py tests/test_guard_semantic_provider.py -q`
    → exit 0，62 passed
  - `python -m pytest tests/test_task_structure.py tests/test_process_audit.py
    tests/test_server_task_structure.py tests/test_server_stage.py
    tests/test_guard_semantic_provider.py -q` → exit 0，134 passed
- 遗留：MatcherExtractor 未接入 get_backend 环境变量开关（有意保守，
  待 P8 质量门禁通过后再启用）；ts.capture/process.unresolved 均未校准
  （P8 4/4 calibrated=false），shadow 通道默认关闭，仅在显式注入时运行。

## SLM prompt 层修复迭代（slm-prompt-v2，2026-09-30，一次有界迭代，no-go）
- 背景：P8 门禁确诊二值模式 yes 标签偏置（负例 100% score≥0.9，|p_yes−p_no|≈0.94，
  P/R 0.18-0.41）。本次按预登记规则做一次（仅一次）prompt 层迭代验证。
- 诊断（results/semantic-runtime/slm-prompt-v2/diagnosis.md + diag_raw.json）：
  calib 固定 seed 抽 20 条（10 正 10 负，seed=20260930），XPU 上对比
  yes/Yes/YES/no/No/是/否/true/false 首 token logits × 5 种模板变体
  （v1/中文改述/陈述式 contains-absent/4-shot/bool JSON）。结论：
  非标签 token 选择问题（yes 恒为 argmax 20/20，正负例 logit 几乎相同）；
  偏置不随模板消失（全部变体 acc=0.50，模板只改分数尺度不改方向）。
- v2 设计：每 pattern 4 个 few-shot（2 正 2 负，全部取自 frozen dev split：
  fz_0129/0130/0001/0004、fz_0010/0012，禁用 calib/test 文本）+ 重写任务描述
  （显式负例语义）；版本化 `AVAILABLE_PROMPT_VERSIONS`，`SLMBackend(prompt_version=)`
  可切换、默认仍 v1；provenance.calibration_id="prompt:slm-prompt-v2" 记录模板版本；
  tools/semantic_eval.py 加 --prompt-version（v2 输出写 slm-prompt-v2/ 目录）。
- 重评：calib 150×4（ts.capture 仍关），XPU bf16，134s，exit 0，逐条落盘可续跑。
  结果（metrics_calib_v2.json 同目录 metrics_calib.json / report_v2.md）：
  | pattern | v1 P/R/F1（t） | v2 P/R/F1（t） |
  |---|---|---|
  | completion.asserted@1 | 0.239/0.700/0.356（0.97） | 0.765/0.433/0.553（0.88） |
  | modality.assertive@1 | 0.406/0.981/0.575（0.70） | 0.448/0.566/0.500（0.52） |
  | process.unresolved@1 | 0.277/1.000/0.434（0.95） | 0.500/0.488/0.494（0.77） |
  | task.object.alignment@1 | 0.757/0.982/0.855（diff>0） | 同 v1（模板未改） |
- 判定（预登记：任一二值模式 calib P≥0.85 且 R≥0.80）：**no-go**。
  v2 改善排序（completion F1 +0.20，正负中位差 0.002→0.032，ECE/Brier 全线下降）
  但 P≥0.85 仅在 R≤0.20 退化点可达。prompt 层一次迭代不足以修复；
  后续方向：负例微调（few-shot negative 训练/LoRA）或更大模型。
  v1 保持默认；config/calibration/ 未改动；v2 产物留档于
  results/semantic-runtime/slm-prompt-v2/。
- 回归：pytest tests/test_semantic_adapters.py tests/test_semantic_contract.py -q
  （.venv_xpu）→ 152 passed，exit 0；无测试锁死 v1 文本需更新（默认未变）。

## P9/P10 进展（真实宿主验证、故障注入、回滚演练与有界长稳，2026-09-30）

run 目录：`results/semantic-runtime/p9-host/`（驱动脚本 + JSON 证据 + markdown
报告 + logs/）。全部演练服务使用**动态端口**，18492 仅检测（空闲）不占用；
不终止用户其他进程；真实模型推理仅 XPU（`.venv_xpu`），总量 76 条 ≤100。

- **宿主链路（P9）**：`run_host_drill.py` 起完整链路——semantic HTTP 服务
  （fake backend 配置 + 动态端口）+ guard 服务（真实权重、legacy 默认，
  `guard_driver.py` 经 HTTPClientBackend→SemanticRuntime→SemanticProviderAdapter
  接线 semantic provider）+ Pi 扩展（`ext_host_test.mjs` 以
  `MOONBOW_GUARD_URL` 指向测试端口驱动真实 progress-guard.ts）。
  18 步全绿：/check 正常返回、/v1/task-structure、/v1/stage-check、
  成功/失败工具、无证据、semantic 直连 match/find-all/batch、
  **杀 semantic 服务 → guard 骨架裁决 + skeleton_only/semantic_unavailable
  （5s 内返回 < 宿主 10s）**、重启后恢复、rounds=2 争议放行防提示死循环。
  扩展宿主级：首次收尾恰好 1 次提醒（REQUIRE_MANIFEST，39ms）、
  重复收尾不重复催（预算生效）、服务不可用时不投递不阻塞。
- **故障注入（P9）**：`run_fault_drill.py` → `fault_drill.md/.json`，14 场景全绿：
  慢响应（delay 6s > deadline 5s）→ 504/timeout，guard 5.0s 内骨架降级不阻塞；
  满队列（capacity=1）→ 即时 429/overloaded（19.8ms 有界拒绝），客户端契约内
  折叠、guard 饱和负载下每次 /check 均合法返回；坏 JSON→400、超长→413、
  注入坏输出/坏引文→invalid_output（不修饰成可信命中）；kill -9（TerminateProcess）
  → 端口释放、netstat 无监听残留/僵尸、同端口重启即恢复。
- **回滚演练（P10）**：`run_rollback_drill.py` → `rollback_drill.md/.json`。
  场景：semantic 提供者启用（/v1/semantic-status 证 provider_kind=semantic）
  → 以 P8 no-go 为触发 → 配置切回 legacy（不接 provider）受控重启 →
  6 例固定输入（A+证据/B 部分完成/无证据/荒谬证据/工具成功/strict 争议放行）
  verdict **逐字段 diff=0 与 legacy 基线一致**，decision/allow_stop/acceptance/
  is_closed/review_requested/disputed 六个验收语义字段不变（任务状态不丢失）。
  回滚仅切配置+重启，不删权重与证据。
- **有界长稳（P9）**：`run_longrun.py` → `longrun_report.md`/`longrun.json`/
  `longrun_raw.jsonl`（逐请求 2000 行）。fake backend + P2 runtime，顺序 2000
  请求（match/find_all/batch 混合、4 pattern、5% 故障注入=98 次）。
  预热后 RSS：24.18→24.29MB（**+0.45% < 10%**），采样 500/1000/1500/2000
  平稳；p50/p95/p99=0.08/0.11/0.18ms；零未解释错误（98 次注入全部契约内
  reason_code，非注入 0 error）、零崩溃。
- **真实模型冒烟（P9，XPU）**：`run_xpu_smoke.py` → `xpu_smoke.json`。
  semantic 服务以 `.venv_xpu/Scripts/python.exe` 启动 SLM backend
  （Qwen3.5-0.8B，device=xpu，bf16，服务内断言 torch.xpu），guard 以
  modality.assertive@1/completion.asserted@1 真实 pattern 接 semantic provider，
  50 次 /check 宿主级请求：50/50 provider_kind=semantic、0 降级、全部 <10s
  （p50 284ms / p95 332ms / max 391ms）；真实推理 76 条（含 1 条预热）≤100。
  **注意：这是链路连通性冒烟，不改变 P8 no-go 结论**——semantic 路径仍
  不可接入交互路径，一切默认 legacy/rule。
- **findings（不修 src/，记录在案）**：
  1) `SemanticProviderAdapter` 默认 pattern 引用 `completion.modality@1`/
     `completion.capture@1` 不在内置 PatternRegistry，SLM backend 对其返回
     abstain(unsupported) → adapter 恒降级。P2/P6 层无契约违反（降级语义正确），
     但真实接线必须显式传受支持 pattern（演练已如此做）；建议 P11 文档明确。
  2) fake 规则多引文用于 match 时，任一引文不在原文即 invalid_output
     （先定位后截断），配置多引文规则需保证每条引文都在目标文本中（文档问题）。
  3) guard 争议放行阀门仅 strict 模式生效（`rounds>=2` 分支在 strict 下），
     advisory 依赖宿主预算字段防重复——行为符合设计，演练中已按此验证。
- **回归（收尾）**：
  - `python -m pytest <12 个 semantic/guard/task/audit/advisory 测试文件> -q`
    → 444 passed / 18 skipped / 1 failed（唯一失败为
    `test_semantic_adapters.py::TestXpuEnforcement::test_require_xpu_passes_on_xpu_host`，
    属 XPU 强制断言在 CPU torch 的默认解释器下**预期失败**）。
  - `.venv_xpu/Scripts/python.exe -m pytest tests/test_semantic_adapters.py -q`
    → **55 passed，exit 0**（XPU 用例在 .venv_xpu 下全绿）。
  - `node --test tests/*.mjs` → 33 pass / 0 fail，exit 0。
  - netstat 复查：18492/18500 及全部演练端口无 LISTENING 残留。
- P9/P10 命令 exit code：run_host_drill.py=0、ext 双模式 node=0、
  run_fault_drill.py=0、run_rollback_drill.py=0、run_longrun.py=0、
  run_xpu_smoke.py=0、pytest=0（.venv_xpu 组与 node 组）、默认解释器组=1
  （仅上述 XPU 断言预期失败）。

## openjev 候选评测（判定侧切换评估，2026-09-30，no-go）
- 候选：AlexWortega/openjev 子检查点 `qwen3.5-0.8b-nli-v2s-long`（Qwen3.5 微调
  NLI 交叉编码器，3 类，MIT，1.7G）。加载：transformers 5.17.0 原生
  `Qwen3_5ForSequenceClassification`，AutoModelForSequenceClassification 直接加载，
  无需 trust_remote_code；config 内置 nli_template（Premise/Hypothesis），
  id2label 1=entailment。XPU bf16：加载 3.9s，显存 2.41GB（峰值 3.13GB/16GB）。
- 模板版本化：`tools/openjev_templates.json`（openjev-nli-v1）：4 pattern 均改写为
  **关于文本的元陈述假设**（completion="该文本在此处明确断言工作已全部完成"，
  非"工作为真"），score=单前向 softmax entailment 概率；alignment premise=task
  （派生口径同前两轮），缺 task 弃权契约不变。
- 评测：`tools/run_openjev_eval.py`（XPU 断言、每 25 条 flush、断点续跑），
  calib 150×4=600 行，exit 0，0 abstain/0 error，产物
  `results/semantic-runtime/openjev-eval/`（predictions_calib.jsonl、
  metrics_calib.json、report.md）。test split 未触碰。
- 结果（calib，各自最优阈值 P/R/F1）：completion 0.688/0.367/0.478；
  modality 0.356/0.679/0.467；alignment 0.913/0.553/0.689；
  process 0.875/0.341/0.491。对比 v1（0.239/0.700/0.356、0.406/0.981/0.575、
  0.757/0.982/0.855、0.277/1.000/0.434）：**病灶方向翻转**——zero-shot 是
  yes 偏置（负例满分、P 0.18-0.41），openjev 是判别强但召回塌缩（P 0.69-0.91，
  R 0.34-0.55，正负中位差 0.16-0.25 vs zero-shot 0.00-0.04）；ECE/Brier
  全面更优（0.13-0.46 vs 0.42-0.77）。modality 为语用元陈述能力缺口
  （负例中位 0.889）。全阈值扫描无任何 P>=0.85 且 R>=0.80 格子。
- 延迟：单前向 p50 107-109ms（alignment 从双通道 242ms 降为 107ms）。
- 判定（预登记）：规则 1（P>=0.85 且 R>=0.80）0/4 不成立；规则 2（每 pattern
  F1 +0.15）不成立（+0.122/−0.107/−0.166/+0.057）→ **不切换**，默认不动，
  不跑 test split。
- 对 R3 建议：openjev 是更优 LoRA 底座（排序已分、LoRA 只补正例边界召回，
  数据配比向正例倾斜）；modality 建议不归 NLI 底座。判定路径改造点与注入
  边界（premise 槽位可被伪造 "Hypothesis:" 行污染，需 capabilities 声明）
  见 report.md。
- 证据索引：results/semantic-runtime/openjev-eval/{report.md,metrics_calib.json,
  predictions_calib.jsonl}、tools/openjev_templates.json、tools/run_openjev_eval.py。

## R2：GLiNER 捕获 backend 接入 Matcher API（route 2，2026-09-30，诊断完成）

决策"判定走 Qwen/openjev+LoRA、捕获走 GLiNER"的捕获半边落地：

- **安装**：`gliner` 0.2.29 以 `--no-deps` 装入 .venv_xpu（其依赖约束
  transformers<5.17 会降级 transformers，故禁用自动依赖）；torch 2.14.0+xpu /
  transformers 5.17.0 保持不变。实际可加载 fastino 权重的运行时是
  **gliner2 2.0.0**（.venv_xpu 内已有；pip gliner 只认 gliner_config.json，
  不识别 gliner2.5 boundary 架构）。
- **权重选型（核对后决策）**：本地 `models/gliner25_closure_*` 全系是
  "闭合 3 类"（Schema.classification("闭合")）微调头，与 ts.capture 8 类
  捕获枚举**不匹配、不可映射**，不采用；采用 HF 缓存
  `fastino/gliner2.5-multi-v1`（BoundaryExtractor，~1.4GB，零样本多标签，
  snapshot aaecfe45…）。零下载（禁 >1GB 约束满足）。
- **新文件 `src/moonbow/semantic/backends/gliner_capture.py`**：
  `GlinerCaptureBackend`（Backend 协议）仅支持 find_all+ts.capture@1
  （其他 → error/unsupported）；标签映射表 `LABELS_V1` 版本化
  （provenance.calibration_id="gliner:labels-v1"，calibrated=false）；
  引文 verbatim + code-point 偏移复核（坏引文丢弃并计数 dropped_quotes，
  出参前再过 schema.validate_response，违例降级 invalid_output）；
  零捕获=ok/matched=false 空列表（不 abstain）；XPU 断言（无 xpu 即
  BackendUnavailable，禁 CPU 回退）。
- **Matcher 路由**：matcher.py 最小追加——模块级
  `register_pattern_backend(operation, pattern_ref, factory)` /
  `unregister_pattern_backend` / `_routed_runtime`，match/find_all 内部
  分发（签名未动，默认无注册时行为与之前完全一致）。gliner_capture.py 提供
  `register_capture_route(factory)`。调用方仍只调
  `find_all(pattern="ts.capture@1")`；match → slm 不受影响。与 R1 的
  adapter 改动无冲突（其改动在 schema/slm 的 adapter 字段，路由层不触碰）。
- **测试**：`tests/test_gliner_capture.py` 8 例（标签映射/坏引文丢弃计数/
  unsupported/空捕获 ok 语义/matcher 路由 load-once/XPU 强制）全绿；
  回归 `pytest tests/test_ts_matcher_backend.py tests/test_task_structure.py
  tests/test_gliner_capture.py -q` → **65 passed，exit 0**（默认解释器）。
  更大范围 semantic 组 182 passed + 1 预期失败（XPU 断言用例，.venv_xpu 下绿）。
- **冒烟（XPU，frozen dev split 20 条=正例 10+abs 负例 3+附带 gold 行 7，
  断点续跑脚本）**：threshold=0.05，quote 归一化精确匹配
  **P=0.063 / R=0.105**（TP4/FP59/FN34）；负例零捕获语义正确（ok/空列表，
  非 abstain），fz_0007/0008 零误报、fz_0009 误报 2 条；63 个模型 span
  偏移全部落位（dropped 0）。P/R 低的主因是引文粒度错配（gold=整子句，
  模型=最小 span），属诊断基线，非验收。零下载、test split 未触碰。
- **环境注意**：与并行 agent 争用 XPU 时出现 os error 1455（页面文件不足）、
  DEVICE_LOST、偶发 segfault；冒烟脚本已带断点续跑与瞬态重试，夜间批量
  任务继续遵守"真实推理不并行"约束。
- 证据索引：results/semantic-runtime/r2-gliner/{smoke_report.md,smoke.json,
  run_smoke.py,smoke_stdout.log}；代码 src/moonbow/semantic/backends/
  gliner_capture.py、src/moonbow/semantic/matcher.py（路由追加）、
  tests/test_gliner_capture.py。

## R3：LoRA 微调判定适配器（2026-09-30，完成）

判定半边（对照 R2 捕获半边）：4 个 match 模式 LoRA 微调，底座按证据分派
（completion/process/alignment → openjev qwen3.5-0.8b-nli-v2s-long
SequenceClassification + 加权 CE；modality → Qwen3.5-0.8B 生成式 + v1 模板
answer-token 加权 BCE）。数据仅 frozen dev 300（calib 150 只做验证/早停/
选 checkpoint，test split 未触碰）；LoRA r=16/alpha=32/dropout=0.05，
target_modules 与 verify_lora_hotswap.json 一致，epoch≤3，每 50 step 存
checkpoint + calib 评估，扫描 F1 连续 2 次不升早停。

- **达标（预登记门禁 calib P≥0.85 且 R≥0.80）：2/4** ——
  process.unresolved 0.971/0.805/F1 0.880@t0.66（达标）、
  task.object.alignment 0.868/0.868/0.868@t0.75（达标）；
  completion.asserted 0.846/0.733/0.786@t0.53（no-go，差 0.004/0.067）、
  modality.assertive LoRA 训崩（p(assert) 排序被破坏，无 checkpoint 超
  zero-shot，no-go，维持生成式 zero-shot 基线 0.406/0.981@0.70）。
  openjev-zero 的 recall 塌缩病灶被 LoRA 修复（正负中位差 →≈1.0）。
- **slm.py 最小扩展**：`SLMBackend(model_kind="sequence")` 支持
  SequenceClassification 底座（_NliScorer 单前向 entailment 概率、
  _match_nli 统一 4 模式、calibration_id="nli:openjev-nli-v1"、
  score_type="entailment_probability"、alignment 缺 task → abstain、
  ts.capture → unsupported）；adapter 热切换支持真实底座（首挂包 PeftModel，
  set_adapter(None) 关 adapter 层回 base）。契约未改。
- **接入验证 INTEGRATION-OK**：注册 completion-lora-v1，calib 抽样 20 条
  （正负均衡）XPU 端到端，backend 与独立加载分数最大差 3.3e-6，20/20 过
  validate_response + provenance 回填，未注册 adapter → error/unsupported，
  p50 151ms。
- **回归**：pytest tests/test_semantic_adapters.py tests/test_semantic_contract.py
  -q → 190 passed。
- **XPU 事故（透明）**：qwen3_5 参考内核 chunk_gated_delta_rule 在 backward
  下 segfault（exit 139×2）/DEVICE_LOST×1；实测 batch≤4×768 + 基座 bf16 直跑
  （LoRA 参数 fp32，不用 autocast）稳定（tools/r3_backward_probe.py）；训练
  循环带 3 次受限重试，4 次训练最终全部 exit 0。
- 证据索引：results/semantic-runtime/r3-lora/{report.md,metrics_calib.json,
  integration_check.json,preds_lora_*.jsonl}；models/semantic_lora/
  {completion-asserted,process-unresolved,task-object-alignment,
  modality-assertive}-v1/（best/、checkpoint_*/、run_meta.json、
  train_log.jsonl 曲线）；tools/{r3_train_lora,r3_eval_best,
  r3_integration_check,r3_backward_probe}.py。
- 遗留：completion.asserted 差线极小（38 个 dev 正例是 recall 上限）；指标
  均对 provisional 标签；默认 provider 未动（P11 opt-in only 现状），是否以
  2/4 达标模式接入属后续决策。

## R3b：completion.asserted LoRA v2（数据扩充重训，2026-09-30，完成）

前置：R3 遗留 = completion.asserted 差线（P 0.846 差 0.004 / R 0.733 差 0.067，
38 dev 正例上限）；本批补数据后重训，门禁预登记不变（calib 任一阈值
P>=0.85 且 R>=0.80）。test split 全程未碰，calib 仅验证/早停/评测。

- **数据**：frozen dev 300 + `dev_extension_r4` 120（正例 74：直陈 42 +
  非直陈 32，含 4 类此前不识别子类；强干扰负例 46；与冻结集零交叉，
  `validate_frozen_data.py --extension` 通过，dev_extension 文件未并入冻结集）
  = 训练 420 条（正 112 / 负 308），**pos_weight = 2.75**（v1 为 6.895）。
- **训练**（底座/超参照 R3 不变，openjev seqcls + 加权 CE + LoRA r16，
  batch=4 bf16 直跑，跑前 `r3_backward_probe.py bf16 4 768` PROBE-OK）：
  跑满 3 epoch = 315 step，每 50 step 落盘 + calib 评估，早停未触发，
  全程 0 step_failed，exit 0。曲线：0.478(zero) → 0.893(300, best)。
- **评测（达标 PASS）**：calib 150 独立评分，t=0.50（即扫描最优）：
  **P=0.9615 R=0.8333 F1=0.8929，ECE=0.0402 Brier=0.0401**
  （v1：0.846/0.733/0.786；TP 22→25、FP 5→1、FN 8→5；正/负中位 1.0/0.0）。
  残余 FN 5 条中 4 条为"发布汇总+changelog 待审"同族（范围内在审锚点，
  硬负例），FP 1 条（评审安排边界样本）。
- **接入注册**：config/semantic_runtime_lora.json 新增 completion-lora-v2
  （v1 保留备查）；R4_CAPABILITY_STATUS completion.asserted
  **uncalibrated → pass**（注明 v2 与数据来源）。端到端
  `tools/r3b_e2e_check.py`：10 条（正负均衡）SemanticMatcher 全链路
  adapter=completion-lora-v2 → 10/10 ok、provenance.adapter 回填、
  与独立加载最大差 3.69e-07、未注册 adapter 不回退 base。
- **回归**：`.venv_xpu` pytest adapters+contract → **190 passed**。
- 证据索引：results/semantic-runtime/r3b-completion-v2/{report.md,
  metrics_calib.json,preds_lora_completion-asserted-v2.jsonl,e2e_check.json}；
  models/semantic_lora/completion-asserted-v2/（best/=step300、checkpoint_*/、
  run_meta.json、train_log.jsonl）；tools/{r3b_eval_best,r3b_e2e_check}.py；
  tools/r3_train_lora.py 扩展 --extra-dev/--out-name。
- 遗留：指标对 provisional 标签；recall CI 宽（calib 仅 30 正例），test split
  留最终验收；modality.assertive 维持 no-go；默认 provider 仍未切换
  （3/4 信号达标，切默认需按 AGENTS.md 约束 1 裁决）。

## R4 进展（LoRA 适配器插件链路接入，2026-09-30）

前置：R3 2/4 达标（process.unresolved、task.object.alignment），R3 训练产物
与 checkpoint 未改动；AGENTS.md 硬约束（禁 legacy/rule 默认、XPU only）遵守。

- **配置化接线**：新增 `config/semantic_runtime_lora.json`（slm +
  `model_kind=sequence`（openjev NLI 底座，XPU only）+ 3 个 adapter 注册：
  process-lora-v1 / alignment-lora-v1 / completion-lora-v1 指向 R3 best/）。
  `http.py build_backend` 补传 `model_kind`/`prompt_version`（默认 causal，
  向后兼容）。服务带 --config 启动即加载基座一次 + 登记适配器（首请求热挂载）。
- **Guard 插件接线（semantic provider 路径）**：`SemanticProviderAdapter`
  新增按信号选 adapter——`signal_adapters={process.unresolved:
  process-lora-v1, task.object.alignment: alignment-lora-v1}`（请求带
  adapter 参数；alignment 自动带 context.task）；`observe_signal` 校验
  provenance.adapter 回填一致（不一致按信号失败整体降级）；modality 维持
  底座 zero-shot 并如实声明 no-go（不假装达标）。`describe()` 新增
  `capabilities.signals` 逐信号达标状态（pass/uncalibrated/no_go），
  经 /v1/semantic-status 暴露。默认 provider 未切换（opt-in 不变）。
- **端到端冒烟 R4-E2E-OK**（`tools/r4_e2e_smoke.py`，XPU，推理总量 52 ≤ 60）：
  semantic HTTP（lora 配置）+ guard /check（semantic_provider="semantic"，
  扩展同路径）+ calib 抽样 20 条（10 正/10 负）→ 20/20 ok 无 skeleton 降级，
  adapter 回填 10+10 正确；未注册 adapter → error/unsupported 不回退 base；
  HTTP 热切换分数 vs R3 独立加载最大差 3.49e-05（bf16 噪声级），base 分数
  vs openjev zero-shot 独立评测 diff=0.0；切换就绪清单见 e2e_report.md §4
  （test split 未跑、no-go 模式表态、延迟预算、AGENTS.md 约束 1 裁决）。
- **修复 R3 遗留后端缺陷**（R3 未暴露，本轮端到端发现）：transformers 5.x
  自带 PeftAdapterMixin 使 R1 duck-typing 误走原生适配器路径 → base 分数
  粘滞（disable 不恢复）+ 三适配器挂载一次 XPU >30s 挂死（HTTP 504）。
  slm.py set_adapter 改为：真实 torch 模型一律纯 peft 路径（R3 验证过）；
  disable 后再切换显式 enable_adapter_layers；PeftModel 包装后 eval()；
  不带 adapter 的请求显式切回 base（混合流量正确性）。
- **completion 补数据备料（不训练不标注）**：
  `results/semantic-runtime/r4-integration/completion_data_gap.md`——38 个
  dev 正例全在 agent_completion（interim_report 0 正例）；calib 30 正例分数
  双峰（8 条 <0.5，其中 7 条 ≈0：清单式/指标式/wrap-up 改写/收官+遗留整类
  不识别）；FP 全在 interim_report 子任务完成与部分完成。建议下一批 dev
  配额：interim_report 正例 ≥40、非直陈完成正例 ≥30、强干扰负例 ≥40、
  en/zh 均衡、直陈正例 ≥20（正例 ≥90 合计，recall 上限 38→~128）。
- **文档**：docs/semantic_api.md 新增 §7 LoRA 适配器使用（插件请求示例、
  达标状态表）+ §0 能力状态 R3 更新。
- **回归**：默认 venv `pytest tests/test_semantic_contract.py
  tests/test_semantic_runtime.py tests/test_semantic_http.py
  tests/test_guard_semantic_provider.py tests/test_ts_matcher_backend.py -q`
  → 230 passed；`.venv_xpu` `pytest tests/test_semantic_adapters.py
  tests/test_gliner_capture.py -q` → 71 passed, 1 skipped（XPU host
  absence-branch 不可测，预期内）。
- **证据索引**：results/semantic-runtime/r4-integration/{e2e_report.md,
  e2e_smoke.json,completion_data_gap.md,_smoke_run.log}；
  config/semantic_runtime_lora.json；tools/r4_e2e_smoke.py。
- **遗留**：指标均对 provisional 标签；test split 未跑（最终验收用）；
  completion/modality no-go（completion 待补数据，modality 维持底座
  zero-shot 如实声明）；默认 provider 切换待用户决策（就绪清单
  e2e_report.md §4）。

### R4.1 completion 补数据落地（2026-09-30，纯标注不训练）

按 completion_data_gap.md 配额建议完成 dev 附加批（不混入冻结集，
`frozen_texts_dev/calib/test` 保持不变）：

- **数据**：`data/semantic_frozen_v1/dev_extension_r4{,_labels}.jsonl`
  120 条（id 前缀 `fzdev_r4_`，源任务 fzT201–fzT292 与冻结集无交叉，
  28 对中英改写对同组），格式与 frozen_texts_dev 完全一致；用法与边界
  见 `data/semantic_frozen_v1/README_dev_extension.md`（仅作 completion
  LoRA 训练的附加 dev 数据，不参与 test/calib、不进阈值选择与验收）。
- **配额达成**：A interim_report 正例 42（≥40）/ B 非直陈 agent 正例 32
  （清单式、指标式、引用工具输出、转述他人、条件式、收官+范围外遗留）
  （≥30）/ C 强干扰负例 46（子任务完成+待办、部分完成/百分比、礼貌
  收尾无断言、计划式、引用测试输出但未断言全部、前置就绪）（≥40）；
  合计 120（110–130）；zh:en=86:34≈7:3；正例合计 74，recall 上限
  38→112。全部原创，与 dev/calib 字符 bigram Jaccard>0.8 查重零命中。
- **边界约定固化**（依据 gap §3 风险声明）：正例=对断言范围完全闭合的
  整体完成陈述（含引用输出后显式推断全部、转述确定性结论、条件为既成
  事实的条件式、遗留明确外置的收官）；范围内显式遗留 →
  completion.asserted=false + process.unresolved=true。每条 label_basis
  一句话写明判定依据；ts.capture 全部 abstain（沿用 dev 组 completion
  正例约定）；provisional 声明与主 README 一致。
- **校验**：`tools/validate_frozen_data.py` 新增 `--extension`（id 唯一
  且与三组冻结集无交叉、task_id 不跨集、kind 枚举、quote 逐字、与
  dev/calib/test+组内查重、家族/正负/语言配额统计）→ exit 0；基础
  校验（无参）回归 exit 0 不受影响。
- **状态**：备料完成，待下一轮 completion LoRA 训练（附加读取本批）
  后以冻结 calib 重选阈值、test split 最终验收。

## R4.2 modality.assertive 专门数据构建（2026-09-30，纯逻辑备料，不训练不推理不占 XPU）

前置：R3 modality LoRA 训崩（正负中位差 0.044，小数据 BCE 破坏排序），
zero-shot 基线 P=0.406/R=0.981（yes 偏置）；本批为在生成式底座重训备料。

- **数据**：`data/semantic_frozen_v1/modality_extension_r4{,_labels}.jsonl`
  180 条（id 前缀 `fzmod_r4_`，源任务 `fzmodT001–fzmodT102` 与既有集无交叉），
  格式同 frozen_texts_dev，ts.capture 全部 abstain（沿用 extension 约定）。
- **三层刻意分层**：A 断言正例 52（过去式/结果呈现/指标陈述/引用输出后断言）；
  B 计划/意图负例 54（将要做/准备做/正在做/接下来）；C 含糊/疑问/条件负例 52
  （可能/应该吧/大概/疑问句/条件未决/转述猜测）；D 边界仲裁层 22（R3 calib
  1.0 分误报形态：礼貌收尾无断言、汇报中自问、转述他人断言、promise 非
  assert、条件未决推断）。modality 正/负 = 59/121（针对 yes 偏置负例倾斜）。
- **最小对立对 16 对（中英各半）**：fzmodT001–T016 每组共享 task_id、恰
  2 条，同一动词两种时态（"已经做完/is done" vs "明天开始做/will do"），
  词汇面拉平只留语气差，目标修复排序不可分病灶。
- **反泄漏防线**：modality 只判语气不判完成度——边界层子部分断言
  （modality=true + completion=false）、语气 basis 只引用语气语义；
  validator 强制检查 modality 正例不得全部 completion=true
  （实际 double-positive 35/59）。
- **其余 4 pattern 如实标注**（多任务复用）：completion.asserted 35 正 /
  task.object.alignment 171 正 / process.unresolved 115 正；zh 128 / en 52
  （71%:29%）；全部原创，与 frozen 三组 + dev_extension_r4 + semantic_dev_v1
  查重（bigram Jaccard>0.8）零命中。
- **校验**：`tools/validate_frozen_data.py` 新增 `--modality-ext`（id 唯一
  前缀、task_id 不交叉、5 pattern 枚举、abstain 约定、三层配额、对立对
  16 组一正一负、反泄漏、语言比、查重）→ exit 0；`--extension` 与基础
  校验回归均 exit 0 不受影响。
- **文档**：`data/semantic_frozen_v1/README_modality_extension.md`（分层口径、
  对立对设计意图、provisional 声明、仅用于 modality 重训的边界；建议重训时
  对立对按 task_id 同批出现）。
- **状态**：备料完成；下一任务在生成式底座（Qwen3.5-0.8B + slm-prompt-v1
  answer-token BCE）重训 modality，门禁建议沿用预登记 calib P≥0.85 且
  R≥0.80；test split 未触碰，calib 仅验证/早停/选阈值。

## Final-Test：test split 最终验收（2026-09-30，完成；**test split 已消费**）

报告：`results/semantic-runtime/final-test/final_acceptance.md`。test split
（150 条冻结留出集）**首次且唯一一次消费**；冻结配置单次前向（calib 阈值
completion v2 t=0.50 / process t=0.66 / alignment t=0.75 + 对应 best checkpoint
原样沿用，零扫描零调参），XPU bf16，450/450 一次完成无中断，逐条落盘可续跑。
modality（R3c no-go）与 ts.capture（走 rule）不评。

| pattern | calib P/R/F1 | **test P/R/F1（Wilson 95%CI）** | 落差 | 判定 |
|---|---|---|---|---|
| completion.asserted | 0.9615/0.8333/0.8929 | **0.8947(0.686–0.971)/0.8095(0.600–0.923)/0.8500** | P−6.7 R−2.4 点 | **PASS** |
| process.unresolved | 0.9706/0.8049/0.8800 | **0.9767(0.879–0.996)/0.7636(0.637–0.856)/0.8571** | P−0.6 R−4.1 点 | **FAIL**（R 差 3.6 点） |
| task.object.alignment | 0.8684/0.8684/0.8684 | **0.8537(0.781–0.905)/0.9722(0.922–0.991)/0.9091** | P−1.5 R+10.4 点 | **PASS** |

- 混淆矩阵（TP/FP/FN/TN）：completion 17/2/4/127、process 42/1/13/94、
  alignment 105/18/3/24；ECE/Brier：0.044/0.042、0.090/0.088、0.157/0.143；
  延迟 p50 135.7–141.5ms / p95 190.0–252.7ms（全模式）。
- 过拟合观察：无任何 >10 点负向落差（alignment recall 反升），无过拟合证据；
  process R 的 CI [0.637,0.856] 覆盖 0.80，跌破线统计上不显著，但按预登记
  门禁如实记 FAIL。
- **C 状态结论（计划 §8.C）**：模型验收通过 = completion.asserted、
  task.object.alignment（2/3）；未通过 = process.unresolved（precision 0.977
  极强、recall 差 3.6 点）；no-go = modality.assertive、ts.capture。
  默认 provider 建议：维持 legacy，仅对 2 个 PASS 信号按白名单 opt-in，
  process.unresolved 仅 shadow 通道。
- **test split 已用声明**：后续任何迭代（含 process 补召回）需新建留出集
  （新 id 段、查重、按源任务整组切分）重走预登记门禁。
- 回归：`.venv_xpu` pytest tests/test_semantic_adapters.py -q → **64 passed，
  exit 0**。
- 证据索引：results/semantic-runtime/final-test/{final_acceptance.md,
  metrics_test.json,predictions_test.jsonl}；tools/final_test_eval.py。
- 命令 exit code：final_test_eval.py=0（env HF_HUB_OFFLINE=1）、pytest=0。

## R6：process.unresolved 召回侧数据扩充 + 新建留出集（2026-09-30，纯逻辑备料，不训练不推理不占 XPU）

前置：Final-Test 确诊 process.unresolved recall 差线（test R=0.7636 < 0.80，
FN 13 条、P=0.9767 极强），且 test split 已消费需新建留出集。FN 分析与数据设计：
`results/semantic-runtime/r6-process/fn_analysis.md`。

- **FN 归因**：13 条 FN 中 11 条 score<0.18（8 条 <0.005 近绝对零分）——是训练
  分布缺失的整类形态而非阈值问题；calib 低分正例 0 条（同族形态在 calib 缺位）。
  家族：F1 完成/收尾陈述内嵌未决（含 STATUS/REMAINING 状态式）、F2 前置待查
  条款嵌义务链（陈述式无疑问标记）、F3 英文 confirm/find out/check 形态（含长链
  埋点）、F4 方案取决于未核实事实、F5 求证问句 vs 修辞性反问混淆。
- **训练数据**：`data/semantic_frozen_v1/process_extension_r5{,_labels}.jsonl`
  132 条（id `fzproc_r5_0001–0132`，源任务 fzprocT001–104，与全部既有集零交叉）。
  家族配额正例每族 ≥15：F1 16 / F2 18 / F3 18（全 en）/ F4 16 / F5 16，正 84；
  强干扰负例 48（≥30）：明确自答 8、历史已解决 6、疑点外置 6、对立对负例 28。
  28 对最小对立共享 task_id（fzprocT001–028），F4 未决-vs-已解决、F5 求证-vs-修辞，
  词汇面拉平只留疑点状态差。zh 84 / en 48（64:36，F3 全英文致 en 超 7:3 约 6 点，
  如实登记）；5 pattern 全标、ts.capture 全 abstain、label_basis 逐条。
- **新留出集**：`data/semantic_frozen_v1/holdout_v2_texts.jsonl` +
  `holdout_v2_labels.jsonl` 120 条（id `fzh2_0001–0120`，源任务 fzT301–420 新段，
  每任务 1 条）。家族分布对齐冻结 test：user_request 60 / conversation_interference
  29 / agent_completion 19 / interim_report 12（50/24.2/15.8/10%）；zh 72 / en 48
  （60:40，en 偏高 6 点已登记）；process 正 38（≥35）/ 负 82（≥50），负例含同款
  强干扰形态。`README_holdout_v2.md` 声明：provisional、仅用于 process v2 及后续
  迭代留出验收、消费一次后作废需新建 holdout_v3、不参与训练/早停/阈值选择。
- **校验**：`tools/validate_frozen_data.py` 新增 `--process-ext` 与 `--holdout-v2`
  （共用 validate_new_batch：id 唯一+前缀、task_id 与冻结三组+dev_extension_r4+
  modality_extension_r4+兄弟 R6 批+semantic_dev_v1 全量交叉检查、5 pattern 枚举、
  ts.capture abstain、quote 逐字、label_basis 必填、bigram Jaccard>0.8 查重
  （含批内）、配额统计）→ 两旗标均 **exit 0**，查重零命中；既有 `--extension`/
  `--modality-ext`/无参基础校验回归均 exit 0 不受影响。
- **状态**：备料完成。重训建议（预登记门禁沿用法）：dev 300 + 本批 132 训练
  process v2（对立对按 task_id 同批成组，R3c 教训），冻结 calib 仅验证/选阈值，
  **holdout_v2 为最终口径（P≥0.85 且 R≥0.80）**；首要验收信号 = FN 左谷
  （近零分正例）消失 + P 在含强干扰负例的 holdout 上不倒退。test split 保持已
  消费状态不再使用。

## R3c：modality.assertive LoRA v2 重训（2026-09-30，完成，NO-GO；合并式：训练+评测+回归）

第二次也是最后一次迭代（预登记）。全部对策落地仍塌缩，如实判定 no-go，
配置不动，modality 维持底座 zero-shot。报告：
`results/semantic-runtime/r3c-modality-v2/report.md`。

- **训练**（`tools/r3c_train_modality.py`，XPU only，exit 0，无 step_failed）：
  底座 Qwen/Qwen3.5-0.8B（生成式 + slm-prompt-v1 answer-token 三选一归一化
  p(assert) 加权 BCE，与推理严格同路径）。训练数据 = dev 300 +
  modality_extension_r4 180（145 正/335 负，pos_weight=2.310 clamp(1,8)）；
  calib 150 仅验证/早停；test 禁碰。防崩对策：lr 减半 5e-5（v1 为 1e-4）、
  batch=4、对立对按 task_id 同批成组、每 50 step 看 calib P/R + 正负中位差、
  排序塌缩（gap<0.05）即早停落盘。
- **结果**：step 50 触发排序塌缩早停（F1 0.000，gap 0.0002 < 0.05）；无任何
  checkpoint 超 zero-shot（best_f1 维持 pre-train 0.5746），**无 best/**
  生成。lr=1e-5 诊断探针（modality-assertive-v2-probe/，仅日志元数据）
  step 50 同样塌缩（gap 0.0015）——排除 lr 过大解释。
- **归因**（v1/v2 一致证据链）：loss 快速趋零→尖峰→塌缩；4.03M LoRA 参数在
  480 条上数十 step 记忆训练集，BCE 把 p(assert) 推向饱和，摧毁 base 上仅
  0.044 的微弱排序。分层负例 + 16 对对立对（同批）未能改变动力学 → 瓶颈在
  0.8B 底座 answer-token 表示上"assertive 语气"缺乏可低秩利用的可分特征，
  非数据配比/词汇混淆问题（与 openjev-eval、slm-prompt-v2 诊断互证）。
- **评测**（`tools/r3c_eval_best.py`，calib 150 全量，zero-shot 裁决口径）：
  t0.5 P 0.3533/R 1.000/F1 0.5222；全扫描最优 **P 0.4062/R 0.9811/F1 0.5746
  @0.70**（与 P8 zero-shot 逐位一致，推理 parity 复现）；ECE 0.4579，
  Brier 0.4206，正负中位差 **0.0436**。门禁（P≥0.85 且 R≥0.80）→ **NO-GO**。
- **处置**：未注册 modality-lora-v2（config/semantic_runtime_lora.json 未改），
  R4_CAPABILITY_STATUS modality 维持 no_go。建议：保持底座 zero-shot +
  下位确定性规则，或放弃该信号；modality_extension_r4 数据保留供未来更大
  底座复用。
- **回归**：`.venv_xpu` pytest tests/test_semantic_adapters.py
  tests/test_semantic_contract.py -q → **190 passed**（56.7s），无回归。
- checkpoint：`models/semantic_lora/modality-assertive-v2/`（train_log 全曲线
  + run_meta + README；无 best）；metrics：
  `results/semantic-runtime/r3c-modality-v2/{report.md,metrics_calib.json,
  preds_lora_modality-assertive-v2.jsonl}`。

## R5：默认 provider 切换为 semantic（PASS 信号白名单，2026-09-30，完成；纯逻辑，不跑推理、不占 XPU）

前置与授权：AGENTS.md 约束 0（禁止 legacy/rule 默认；达标信号接入）+
R4 就绪清单（`results/semantic-runtime/r4-integration/e2e_report.md` §4）+
final-test 验收（`results/semantic-runtime/final-test/final_acceptance.md`：
completion.asserted / task.object.alignment PASS，process.unresolved R=0.764
差线走 shadow，modality no-go）。

- **白名单语义落地**（`src/moonbow/guard/semantic_provider.py`）：
  - 新增 R5 常量：`R5_PASS_SIGNALS`（completion.asserted → completion-lora-v2
    role=capture；task.object.alignment → alignment-lora-v1 role=alignment）、
    `R5_SHADOW_SIGNALS`（process.unresolved → process-lora-v1）、
    `R5_MODALITY_PATTERN`（modality.assertive@1 底座 zero-shot）、
    `SHADOW_SIGNALS_KEY` / `MODALITY_UNCALIBRATED_KEY`。
  - `SemanticProviderAdapter` 按 capabilities 信号状态路由：PASS 信号经
    adapter 正常参与判定（capture 通道由 completion.asserted 接管、新增
    alignment 通道）；process.unresolved 仅 shadow——结果只进 observation
    新字段 `shadow_signals`（verifier 原样写入 `scores["shadow_signals"]`），
    不参与信号判定与提醒生成，其失败也只记录 unavailable_reason 不拖垮
    整体；modality 维持底座 zero-shot 并标记 uncalibrated（scores 写
    `semantic_modality_uncalibrated=true`、describe 写 `modality_judgment`，
    信号文案语义不变，不计入达标声明）；PASS 信号失败仍整体降级（不出半套）。
  - `R4_CAPABILITY_STATUS` 更新：process.unresolved → **shadow**（注明
    final-test R=0.764 差线、P=0.977 达标）；completion/alignment → pass
    （补 final-test 指标）；modality 维持 no_go。
  - 新增 `build_whitelist_adapter`（白名单工厂）与
    `semantic_provider_from_config`（从配置纯构造：matcher URL 取
    provider.matcher_url 或 server.host/port，HTTPClientBackend，不连接不推理）、
    `env_provider_key`（MOONBOW_GUARD_PROVIDER 读取与非法值 warning）。
- **默认切换（配置驱动，缺省 legacy 零变化）**：
  - verifier（SDK 侧）：`MOONBOW_GUARD_PROVIDER=semantic` 时首次 check 惰性
    按配置构建白名单 semantic provider（仅一次）；构建失败记 warning 永久
    回退 legacy；未设置/非法值完全走既有路径。provider 覆盖参数优先级不变。
  - server（HTTP 侧）：`GuardHTTPRequestHandler.default_provider_key`
    （默认 "legacy"）；`start_server` 读环境变量，semantic 时构建 provider
    并把 /check 缺省切到 semantic（请求显式 `semantic_provider` 仍可覆盖；
    未配置 runtime 仍 400 不静默回退）。
  - `config/semantic_runtime_lora.json` 补充 `provider` 声明段（default /
    activation_env / matcher_url / 白名单 / shadow / no_go / rollback）。
- **Pi 扩展透传**（`src/moonbow/guard/extensions/progress-guard.ts`）：
  `MOONBOW_GUARD_PROVIDER` 合法值（semantic/legacy）透传到 /check payload 的
  `semantic_provider` 键；未设置/非法值不带该键（服务端缺省决定，零变化）。
- **回滚保障**（`docs/semantic_api.md` 新增 §9 + §7.1 状态表更新为
  final-test 口径）：一键回滚 = 移除 `MOONBOW_GUARD_PROVIDER` + 重启；
  verdict 与 legacy 基线逐字段一致（引用 P10 回滚演练
  `results/semantic-runtime/p9-host/rollback_drill.md`，6 例 diff=0）。
- **测试**（`tests/test_guard_semantic_provider.py` 53→69 例，全 fake 纯逻辑）：
  白名单路由（process 命中/failure 均只进 shadow_signals 不产生信号；
  PASS 信号参与判定产生/消除语义信号；PASS 失败整体降级；uncalibrated
  标记与 describe 声明）、环境变量默认切换（SDK semantic / 未设置 legacy
  零变化逐字段对比 / 非法值回退 / 构建失败回退 / override 优先）、HTTP
  default_provider_key（semantic 缺省 + 显式 legacy 覆盖 + 未配置 400 +
  缺省 legacy 不变）。
- 文档：`docs/semantic_runtime_README.md` 能力状态表更新（semantic 路径：
  **2 信号 test PASS 参与判定、1 shadow、1 no-go**；默认配置激活、缺省
  legacy 零变化）。
- **命令 exit code（2026-09-30）**：
  - `python -m pytest tests/test_guard_semantic_provider.py -q` → exit 0，
    69 passed
  - `python -m pytest tests/test_guard_semantic_provider.py
    tests/test_guard_regressions.py tests/test_advisory.py
    tests/test_semantic_http.py tests/test_semantic_contract.py -q`
    → exit 0，324 passed
  - `node --test tests/test_pi_advisory.mjs tests/test_pi_strict_combo.mjs`
    → exit 0，6 pass / 0 fail
- 遗留：指标均对 provisional 标签；process.unresolved 补召回需新建留出集
  重走预登记门禁（test split 已消费）；modality/ts.capture 维持 no-go；
  semantic 默认仅在设置了环境变量的部署激活（未配置环境零变化）。

## R6b：process.unresolved v2 重训 + holdout_v2 验收（2026-09-30，完成，GO pass-preliminary；合并式：训练+两级评测+接入+回归）

报告：`results/semantic-runtime/r6b-process-v2/report.md`。独占 XPU，全部命令
`.venv_xpu/Scripts/python.exe`，6 条命令全部 exit 0，无崩溃/无 step_failed/
无断点续跑触发。

- **训练 v2**（`tools/r3_train_lora.py`，`--out-name process-unresolved-v2`，
  exit 0）：数据 = frozen dev 300 + process_extension_r5 132 = 432 条
  （pos 161 / neg 271），对立对按 task_id 同批；底座/评分同 v1（openjev
  seqcls，加权 CE），**pos_weight=1.683**（clamp(271/161,1,8) 重算）；lr/epoch
  沿用 v1（1e-4，≤3，batch 4）；早停 calib 扫描 F1 连续 2 次不升（step 300
  早停）；新增 **R3c 塌缩监护**（正负中位差 <0.05 即停，全程未触发，gap≥0.9984）。
  曲线：0.491(zero)→0.784(50)→0.804(100)→0.813(150)→**0.861(200, best/)**→
  0.837(250)→0.839(300)。checkpoint：`models/semantic_lora/process-unresolved-v2/`
  （best/=step 200，README 全记录）。holdout_v2 全程未参与训练/早停/选阈值。
- **两级评测**（新工具 `tools/r6b_eval.py`，exit 0；calib 断点落盘，holdout
  每 checkpoint 单次前向）：calib 150 扫描 P=0.8947 R=0.8293 F1=0.8608
  @t=0.90（选阈值，与 v1 同协议）；**holdout_v2 120（唯一验收口径，固定阈值
  单次前向 + Wilson）：v2 P=0.8571 (CI95 0.7216–0.9328) R=0.9474
  (CI95 0.8271–0.9854) F1=0.900（tp36/fp6/fn2）→ 预登记门禁 P≥0.85 且
  R≥0.80 PASS**；v1 同口径基线 P=0.6486 R=0.6316 F1=0.640（24/13/14，
  FAIL）→ v2 过线且 P 不劣于 v1 → **达标**。
- **FN 家族修复（holdout_v2 上 v1 14 FN → v2 2 FN）**：F1 完成内嵌 0 残留
  （近零分左谷消失）、F2 前置待查 0 残留、F3 英文残留 1（fzh2_0013，
  0.889 贴线差 0.011）、F4 0 残留、F5 残留 1（fzh2_0065 求证/修辞边界，
  0.702）。P 侧 FP 6 条中 5 条为修辞性反问（F5 对立侧）+1 条疑点外置；
  P=0.857 贴线但未倒退。
- **接入**：config/semantic_runtime_lora.json 注册 `process-lora-v2` 并移入
  whitelist_pass_signals（shadow 清空）；semantic_provider.py R4_CAPABILITY_STATUS
  process.unresolved **shadow → pass-preliminary**（注明 holdout_v2 口径、
  provisional 标签、P 贴线），R5_PASS_SIGNALS 加入、R5_SHADOW_SIGNALS 清空
  （回滚路径注释保留）。
- **端到端**（`tools/r6b_e2e_check.py`，等价 r4_e2e_smoke，exit 0）：semantic
  HTTP 10 条 calib 正负均衡 → 10/10 provenance.adapter=process-lora-v2 回填 +
  validate_response 全过。回归：`.venv_xpu` pytest
  tests/test_semantic_adapters.py tests/test_semantic_contract.py -q →
  **190 passed, exit 0**，无回归。
- **holdout_v2 消费声明**：v1/v2 各一次性前向消费完毕，本集作废，后续迭代
  需新建 holdout_v3（README_holdout_v2.md 声明兑现）。
- 遗留：F5 求证-vs-修辞边界是剩余压力点（P 贴线，CI 下沿 0.722）；标签
  provisional 同级证据；其余信号状态不变。

## R7 最终联调（2026-09-30 晨）
- 主代理复核发现并修复 R6b 遗留缺陷：process.unresolved 升入 R5_PASS_SIGNALS
  （role=unresolved）后 observe() 未消费该 role，信号被静默丢弃。已补齐：
  SemanticObservation.unresolved_detected 字段、observe() 未决判定块、
  verifier 未决软信号（"收尾陈述中疑似存在未解决的疑点或待办事项…"）与
  scores["unresolved_detected"] 记录。
- 修复 R5 测试夹具过期断言（process-lora-v1→v2、shadow→判定路径），新增
  test_r5_process_unresolved_judges_when_hit / test_r5_process_signal_failure_degrades。
- 最终回归：python 445 passed（exit 0）+ node 32 pass / 0 fail（exit 0）。
- 当前判定状态：3/3 判定信号参与 semantic 路径（completion PASS、alignment
  PASS、process pass-preliminary），modality no-go（zero-shot 如实标记）。
  默认 provider 由 MOONBOW_GUARD_PROVIDER=semantic 激活，缺省 legacy 零变化。

## legacy vs semantic 同卷对比实验（2026-09-30）

旧 67 样本闭合判定冻结集（README §2.3 的 80.6% 同一张卷子）同时打两条路径，
产出第一份同卷能力对比。产物
`results/semantic-runtime/legacy-vs-semantic/{run_eval.py,predictions.jsonl,comparison_report.md,metrics.json}`。

- **数据集定位（原件，只读）**：样本 (id, req, resp) =
  `pipeline/research_v233_alignment_eval.py` 的 EVAL_POS/CROSS/PARTIAL/WAIT/
  FAIL 共 67 条（pos 22 + cross 15 + partial/wait/fail 各 10，gold 1/0）；
  逐轮清单 = `maps/best_contract_results.json`（旧系统 80.6% 运行的逐轮留痕）。
  协议：同一 verifier、mode=strict、按旧运行轮数逐轮回放。
- **运行**：`.venv_xpu/Scripts/python.exe results/semantic-runtime/
  legacy-vs-semantic/run_eval.py`，exit 0；全程 XPU（脚本断言
  torch.xpu.is_available），semantic 路径经 semantic HTTP
  （config/semantic_runtime_lora.json：openjev 底座 + completion-lora-v2 /
  alignment-lora-v1 / process-lora-v2，modality 底座 zero-shot 如实
  uncalibrated）。
- **结果（两路径完全持平，0 例闭合翻转）**：
  | 指标 | legacy | semantic |
  |---|---|---|
  | 准确率（n=67） | 79.1% (53/67) | 79.1% (53/67) |
  | Wilson 95% CI | [0.679, 0.871] | [0.679, 0.871] |
  | 负样本阻断 | 44/45 | 44/45 |
  | 误放行 | 1（n08，同一例，骨架争议放行机制所致） | 1（同） |
  | decision 分歧 | — | 2 例（e10/e12：CLARIFY→BLOCK） |
  | 延迟 p50/p95 | 40ms/63ms | 547ms/771ms |
- **80.6% 复现情况**：legacy 重放 53/67，差 1 例（e18：旧运行第 1 轮 CLOSE，
  重放被当前 legacy similarity=0.1145/capture=0.088 软信号拦下）——旧运行
  跑在旧版 verifier/旧检查点上（软信号文案不同），属版本漂移而非实验口径偏差。
- **归因**：13 例 FN 中 10 例由清单硬信号刚性 BLOCK（与 provider 无关）；
  分歧 2 例 = modality zero-shot（53/67 判 promise，无判别力，no-go 实证）
  把对齐软信号换成了 strict 硬信号，且均落在已是 FN 的 pos 上；LoRA 信号在
  本短合成域判别力有限（completion matched 5/22 pos、0/45 neg；alignment
  67/67 全 matched 无否决；process 1/67），final-test PASS 不外推到本卷域。
  n08 误放行由 strict 非对称争议放行造成，两路径逐字相同。
- **结论**：新路径在旧考卷上**持平**（不构成切换默认 provider 的依据，也
  不构成退回依据）；语义层真正判别力证据仍以负例增强微调 + 冻结 test split
  验收为准。
- 回归：`.venv_xpu` pytest tests/test_semantic_adapters.py -q →
  **64 passed, exit 0**，实验未污染环境。

## n08 争议放行闸门实验（2026-09-30，负结果已回退）
- 假设：语义捕获判"未闭合"时扣住 strict 争议放行、要求工具实证，可消除
  n08 误放行。实施后 67 卷全新重跑（run_eval 断点续跑导致首次重跑未生效，
  已用清空重跑修正）：**净退化**——拦下 n08 1 例 FP 的同时误伤 e01/e03/
  e05/e13 四例真闭合（53/67 → 50/67）。特征分析：捕获头无法分离 n08
  （0.039）与真闭合 e05（0.058），相似度同样无分离力（e05 0.288 < n08
  0.394）。
- 结论：当前模型能力下，规则 5.3 的无条件第 2 轮争议放行是经验更优权衡
  （4 TP : 1 FP）。已回退闸门，恢复原契约，n08 类误放行（1/67）作为机制
  已知限制留档；n08 的真正修复方向是证据-任务相关性判定（需更强模型），
  而非捕获强度阈值。
- 产物：predictions_pre_n08fix.jsonl（修复前基线）、
  predictions_n08gate_experiment.jsonl（闸门实验）、predictions.jsonl
  （回退后=基线 53/67）。verifier.py 规则 5.3 注释留档全过程。
- 回归：guard 全矩阵 188 + 15 passed（exit 0）。

## 收敛进度追踪 Phase 0：离线回放（2026-09-30，纯数据分析，未动 src/）
- 目的：验证可计算的收敛速度特征能否在任务早期（第 3-4 轮）预测最终失败。
  数据 = pi_eval/home/sessions/ 2026-09-22 批次（GUARD_EFFECT_REPORT §2.0b /
  trap_ladder 实验的真实会话），233 个 trap run（s1×145、s1h×20、c3×30、
  off_by_one×16、复合/单陷阱其余 22），mimo-v2.5 占 220；排除 s2 干净任务
  （无 pytest 目标，防标签污染）。守卫臂由 progress-guard custom_message
  标记（41 run；c3 全 baseline）。
- 产物：`tools/convergence_study.py`（特征全确定性可复算）、
  `results/convergence-phase0/{study_report.md,features.jsonl,metrics.json}`。
- 核心结论：①第 3 轮快照全部特征 AUC 0.46–0.55，无预测力（前 3 轮普遍无
  验证事件，特征未分化）；②第 4 轮对存活 run 分化清晰：rsl≥3
  （rounds_since_last_verify_success）AUC 0.924 / P=0.60 R=1.0，
  write_verify_ratio AUC 0.880，pytest_fail_streak≥2 AUC 0.866 / P=0.90 R=0.43；
  ③把"第 3 轮观测时刻已提前停止且无验证成功"并入后，组合规则在 n=233 上
  P=0.602 [0.537,0.663] R=1.0——可迁移信号是"验证缺位"而非收敛速度；
  ④失败构成：fail_then_stop 111（80%）、截断残留 11、打转 9、全程无验证 8；
  63/139（45%）失败活不过第 3 轮（探索期静默停止为主体，与 §2.0.1 一致）。
- 统计功效：Wilson 区间宽（±0.08–0.19），rounds=4 快照有幸存者偏差，
  c3 子样本（30 run，rounds=4 仅剩 12）单独无功效——数值仅作 Phase 1 先验。
- Phase 1 建议：纯影子观测，复用本脚本特征；候选触发线 rsl≥3@round4（高召回）
  与 fail_streak≥2（高精确）；验收线 = 影子 precision≥0.7。

## 收敛进度追踪 Phase 0.5：目标覆盖度泛化信号回放（2026-09-30，纯数据分析，未动 src/）
- 目的：把 Phase 0 的域特定信号（pytest 缺位，只适用修 bug 任务）泛化为
  "目标落地事件"——任务按家族有 goals（单陷阱/s1/s1h=1、复合 c1/c2/c4=2、
  c3=3），目标拿到工具级落地证据即 grounded；信号 = coverage
  （grounded/total）轨迹停滞。生产版落地证据应由 semantic matcher 对齐判定
  （域无关、SLM 后端）；本次回测用确定性代理解析 pytest 输出。
- 产物：`tools/goal_coverage_study.py`（复用 convergence_study 的会话定位/
  轮次切分/基线特征；回归处理=曾过后被失败输出再点名即移出 passed 集）、
  `results/convergence-phase0/{goal_coverage_report.md,goal_coverage_metrics.json,
  goal_coverage_features.jsonl}`（Phase 0 产物未覆盖）。同一批 233 run。
- 核心结果：①**非劣性成立**——rounds=4 上 pytest_fail_streak≥2 的 10 个命中
  全部被 G2（coverage==0）覆盖（only_baseline=0），泛化信号是特例的严格超集；
  第 4 轮时间视角 G2（含提前停止 run，n=233）P=0.908 [0.852,0.945] R=1.0，
  优于 Phase 0 round3 组合规则（P=0.602/R=1.0），提升来源=第 4 轮时 78 个
  成功 run 已 full pass（首个 pass 集中在第 4 轮）被移出误报池。
  ②stagnation 阈值在本语料不可辨识（取值退化为 {0}∪{≥4}，≥1..≥4 结果相同），
  推荐 2（防御性、零代价）；触发不早于第 4 轮观测时刻；代价=慢热成功 run
  误报 14/233（首个 pass 最晚第 23 轮）。③部分得分度量在本语料**无检验样本**：
  c3 失败全部是收集期错误（3 目标一起失败），成功是 0→3 一步到位，
  max_goals_passed 分组完美分离（0→18 run 全败；3→12 run 全过）但 1/3、2/3
  档样本为 0；c4 仅 2 run 出现过 1/2 且均失败——信息量不能宣称，Phase 1 需
  构造逐测试可独立判定任务再测；c3@round4 快照幸存仅 12（失败 1），单独无功效。
  ④本语料 0 次回归（过而后挂），回归移除机制未被触发；轨迹与 Phase 0
  cumulative_verify_successes 交叉核实 283 对 0 不一致。
- 生产化边界：无 pytest 输出的任务类型（非 Python、问答/探索、非结构化输出）
  必须走 matcher 对齐兜底（SLM 后端，禁 rule）；pytest 解析仅是确定性快速
  通道；收集期错误应判"无证据"而非"失败证据"；套件-目标粒度错配
  （s1h 出现 "1 failed, 1 passed" 但 1-goal 口径记 0）由 matcher 逐目标对齐解决。

## 收敛进度追踪 Phase 1：stage audit 收敛信号 shadow 通道（2026-09-30，完成；纯逻辑，不训练不推理不占 XPU；只记录、零投递、默认关闭）
- 目的：把 Phase 0/0.5 的两条触发规则接入 `StageAuditor` 的 shadow 通道
  （P7 shadow_semantic 同模式），先在真实宿主流上只记录不投递地验证，
  验收线沿用 Phase 0 建议（影子 precision≥0.7 后才谈 advisory 化）。
- 产物：`src/moonbow/guard/convergence.py`（ConvergenceShadow，新增）、
  `tests/test_convergence_shadow.py`（31 用例，新增）；
  `src/moonbow/guard/process_audit.py`（StageAuditor 增可选
  convergence_shadow 参数 + audit() 结果可选 shadow_convergence 键 +
  audit_stage_payload 透传参，时序核对/证据性质/发现修订逻辑一行未动）、
  `src/moonbow/guard/server.py`（/v1/stage-check 可选
  enable_convergence_shadow，bool 校验，缺省 false 走原路径）为最小改动。
- 轮定义（近似）：StageBlock 流不含 assistant 消息边界，无法按
  convergence_study 原始口径（含 toolCall 的 assistant 消息+后续全部
  toolResult=1 轮）切分；以 tool_result 事件计数近似（第 k 个 toolResult
  = 第 k 轮观测点）。已知局限：并行多调用一轮被拆多轮 → 轮次偏高
  （触发偏早，方向固定且 shadow 不投递）；轮内成败并列被拆成相邻轮，
  fail_streak 从"轮内任一失败且无成功"变为"逐结果连续失败"。
- pytest 证据识别：tool_call_id 配对的 toolCall 调用文本含 "pytest"
  优先（is_verify_call 口径，不复刻 name=="bash" 限制）；无配对（toolCall
  被 400 块环形缓冲截掉）时按结果文本摘要特征（"N passed"/"N failed"/
  "no tests ran"）回退；配对存在时由调用侧决定，`ls` 的输出再像也不算。
- 解析规则移植（goal_coverage_study，逐函数注明来源）：全绿 → 全部目标
  落地；失败输出点名目标移出 passed 集（回归处理）；"N passed/M failed"
  与点名数一致 → 部分得分；匿名复合目标 grounded=total-M；单目标任务
  = 套件通过；无摘要行输出（收集期错误）→ passed 不动（"无证据"）。
  fail_streak 单独用 Phase 0 基线严格二分口径（非全绿即失败），与覆盖度
  部分得分口径不同源，与两条离线回测同卷可比。
- 两条规则最终形态（R6b 后）：①converge.stall = 轮次≥4 且 coverage==0
  且流中出现过 pytest 输出；轮次<4 abstain(rounds<4)，无 pytest 痕迹
  abstain(no_pytest_evidence)——无验证目标的任务 coverage==0 只是
  "无证据"，判了会把非 Python/问答/探索类任务全误标。②
  converge.fail_streak = 连续 pytest 失败≥2 轮（无 pytest 时 streak=0
  直接 False，无 abstain 语义歧义）。目标数第一版显式注入
  （goals_total 缺省 1；target_tests 命名目标可选），不从 req 提取。
- 幂等保证：compute() 是 (req, blocks) 纯函数、无实例状态；客户端
  （process-audit.ts）每批发送窗口内全部块，纯函数口径等价于全流重放，
  同流重复审计同结果（与 audit() fingerprint 幂等一致，有用例锁定，
  含块乱序到达按 seq 排序不变）。
- 默认关闭证明：StageAuditor 缺省 convergence_shadow=None → 响应不含
  shadow_convergence 键、四键（findings/reminder/semantic/based_on）
  与现状逐字节一致；server 缺省 false 走原路径；有用例断言"HTTP 不带
  flag 的响应 == 直接 audit_stage_payload 输出"。零投递硬约束：G2 已
  触发 + 主审计有矛盾发现的流上，findings/reminder/semantic 逐字段
  不变（两条用例）；shadow 自身异常失败隔离（converge.unavailable 占位）。
- 测试：`python -m pytest tests/test_convergence_shadow.py
  tests/test_process_audit.py tests/test_server_stage.py` → 65 passed
  （31+26+8）；扩面回归 guard/semantic 8 个文件 413 passed。
  用例覆盖：幂等（纯函数/审计/乱序）、默认关闭零变化、轮定义、pytest
  识别（配对/回退/非 pytest 不算）、摘要解析（通过数变化与回归、命名
  目标失败名单+部分得分、收集期错误不动 passed）、G2 第 4 轮零覆盖
  触发 / 已有覆盖不触发 / goals_total=3 部分覆盖（1/3）不触发、
  fail_streak 触发 / 全绿打断 / 严格二分、shadow 绝不进
  reminder/findings、失败隔离、HTTP 透传（flag 缺省/false/true/非 bool 400）。
- 遗留（Phase 2 候选）：①真实会话流上采集 shadow 触发统计，验证
  precision≥0.7 验收线（本阶段仅单测/集成测试，无回放数据）；②精确轮
  边界需客户端流携带 turn 边界；③publish_repo 镜像树未同步本次改动
  （P11 打包副本，发版前需重打包）；④生产落地证据的 matcher 对齐
  兜底（SLM 后端）仍待 semantic 路径质量达标后接入，pytest 解析仅是
  确定性快速通道，不构成 rule 判定路径的默认化。

## PseudoCallGuard：伪工具调用检测与反馈（2026-09-30，完成；harness 层，纯逻辑，不跑推理不占 XPU）

- 背景：Phase 2 归因（`results/convergence-phase2/failure_attribution.md`
  §0/§5.1，证据清单见其附 2）确认主导失败机制是 mimo 经网关把 tool call
  写成正文文本（`[tool_call call_xxx] edit {...}`），pi 按正常文本收尾、
  动作永不执行且零反馈：28/30 run 中招、13 run 修复编辑被吞。本阶段在
  扩展层补上"检测 + 反馈"闭环（反馈措辞即 §5.1 建议的中文定式）。
- 实现：新模块 `src/moonbow/guard/extensions/pseudo-call-guard.ts`。
  触发：`message_end` 扫 assistant 消息文本，命中伪调用签名（bracket
  主形态 / XML / 行首工具名+JSON 骨架 / harness 残渣，按优先级取主形态）
  且该消息无 toolCall 内容块（伴随真实调用的混合消息只记遥测不投递，
  Phase 2 实测 1 例混合形态）；stopReason=error/aborted 不扫。
  反馈：固定措辞经 `followUp + triggerTurn` 投递（伪调用消息收尾时
  agent 已停，steer 送不到）。预算：每任务最多 3 次（防 conv-r1 式
  "重发仍被吞"循环），先预留后投递、超预算只记遥测；`hasPendingMessages`
  抑制不耗预算；跨重启预算从遥测账本（`progress-guard:pseudo-call`
  条目按 taskId 记账）恢复。杀开关 `MOONBOW_PSEUDO_CALL_FEEDBACK=off`
  （默认 on，off 零扫描零记录零投递）；消息级去重（对象 WeakSet +
  responseId/id 双保险）。
- 挂载：`progress-guard.ts` 最小 diff（import + install + 在
  `message_end` 的 processMode 早退**之前**调一行——伪调用守卫独立于
  过程审计模式开关）。本模块不自行 `pi.on`（宿主同名事件单处理器），
  全逻辑 try/catch 失败自隔离，不参与任何 guard 判定路径。
- 测试：`tests/test_pseudo_call_guard.mjs`（node:test，19 用例）——
  五种形态签名命中、多调用计数、伴随真实调用不误报、无伪调用不报、
  固定措辞与投递通道、3 次预算封顶/新任务重置/跨重启账本恢复、off
  杀开关、过程审计 off 时独立工作、投递通道故障失败隔离（宿主收尾
  裁决不受影响）、pending 抑制、同消息重投递去重。
  `node --test tests/*.mjs` → 52 passed（含既有 33 + 新增 19），exit 0；
  `pytest tests/test_guard_semantic_provider.py -q` → 69 passed。
- 遗留：①检测为文本签名法，模型若在正文里合法引用签名样例（如读了
  归因报告后转述）会按预算误报至多 3 次后静默——不阻断、不进判定；
  ②真实 agent A/B 实验（伪调用 run 的编辑落盘率是否改善）由 Phase 3
  实验负责，本阶段只做 harness 修复与单测；③publish_repo 镜像树待
  主代理同步。

## Jev：NeoHorse-Jev-4B 判定后端接入与评测（2026-10-03，完成；1/4 模式过预登记门禁）

计划 §2.2 可替换后端首次真实换装验证。报告：
`results/semantic-runtime/jev-eval/report.md`。独占 XPU，全部推理
`.venv_xpu/Scripts/python.exe`（bf16），命令全部 exit 0，test split 未触碰。

- **模型**：TokenRhythm/NeoHorse-Jev-4B（Apache-2.0，4B 非自回归决策模型，
  prefill-only 单次前向，直接在应用定义候选上输出概率；3 种决策类型
  Choice/Score/Noul）。revision `434cb21d3a994a953d3ae5788405fcb2c4970554`，
  下载量 2288/月（HF API 实查）。`hf download` → exit 0（~9.1GB 权重）。
- **加载（一次成功，无需 workaround）**：官方 wheel `neohorse_decision-1.0.0`
  `pip install --no-deps` 装入 .venv_xpu（torch 2.14.0+xpu / transformers
  5.17.0 零改动；pydantic 已在环境内）；官方 API
  `DecisionEngine(model_dir, device="xpu")` 原生接受 device 参数。
  加载 16.4s，显存 12.44→12.98GB / 15.56GB（16GB 卡放得下，余 ~2.6GB）。
  causal_conv1d/flash-linear-attention 未装走参考实现（同 0.8B 各轮条件），
  600 次前向零 segfault 零卡死。
- **交付**：`src/moonbow/semantic/backends/jev.py`（JevBackend，注册名
  `jev`，Backend 协议：构造加载一次/幂等/出口过 validate_response）+
  `tools/run_jev_eval.py`（XPU 断言、每 25 条 flush、断点续跑）+
  `tests/test_jev_backend.py`（12 例 fake engine 全绿）。模板版本化
  **jev-template-v1**：completion/process= Noul（score=P(true)，
  score_type=noul_probability）；**modality=Choice 三类 assert/promise/
  question（模型原生用法，score=P(assert)）**；alignment=Noul + state 拼装
  模板（task+text），缺 context.task → abstain insufficient_context；
  ts.capture → abstain unsupported（生成式任务，prefill-only 不支持）。
  用户文本只进 state 数据区（官方 user_tokens 中和特殊定界 token），
  instructions/criteria 为版本化常量；calibration_id=
  "template:jev-template-v1"；概率缺失/非法 → invalid_output，无编造路径。
- **calib 150×4=600 行**（0 abstain/0 error/validate 失败 0；续跑 588 行
  一次完成）。对比（各自扫描最优 P/R/F1@t）：

  | pattern | 0.8B zero-shot v1 | openjev | Jev 4B zero-shot |
  |---|---|---|---|
  | completion.asserted | 0.239/0.700/0.356 (0.97) | 0.688/0.367/0.478 (0.50) | 0.733/0.367/0.489 (0.53) |
  | modality.assertive | 0.406/0.981/0.575 (0.70) | 0.356/0.679/0.467 (0.57) | 0.652/0.849/0.738 (0.75) |
  | task.object.alignment | 0.757/0.982/0.855 (diff>0) | 0.913/0.553/0.689 (0.50) | 0.789/0.983/0.875 (0.78) |
  | process.unresolved | 0.277/1.000/0.434 (0.95) | 0.875/0.341/0.491 (0.53) | **0.875/0.854/0.864 (0.52)** |

  ECE/Brier：process 0.055/0.070（全部系统中最优）、completion 0.133/0.149、
  modality 0.226/0.200、alignment 0.222/0.218（分数双类饱和 ~0.99）。
  延迟 p50 174-182ms（0.8B 的 ~1.3x、openjev 的 ~1.6x）；显存 12.5-13.3GB
  （0.8B/openjev 为 2-3GB；4B 驻留使同卡共存第二底座不可行）。
- **门禁判定（预登记 P>=0.85 且 R>=0.80/calib 扫描）**：
  **process.unresolved PASS**（t=0.52：P=0.875/R=0.854；t=0.5 亦过线；
  正负中位差 0.882）——唯一 zero-shot 即过线的底座，与 0.8B 微调后的
  process-lora-v2（0.895/0.829）同档。completion FAIL（R 全表 max 0.367，
  FN 集中在"收官+遗留外置"形态 = 模板 "fully complete" 字面义与标签口径
  冲突，正例中位 0.088）；modality FAIL（R>=0.80 下 maxP=0.662；但
  F1 0.738 为该信号全部已测方案最好，正负中位差 0.788 首次出现实质排序，
  FP=指令/规定式陈述被宽口径 assert 定义吞入）；alignment FAIL
  （R>=0.80 下 maxP=0.798，分数饱和判别力不足）。
- **处置（未改任何默认配置）**：不注册进 semantic_runtime_lora.json，
  R5_PASS_SIGNALS 不动；不建议用 Jev 替换任何现役达标信号（process-lora-v2
  有 holdout_v2 验收 R=0.947 更强）。Jev 定位：zero-shot 冷启动 fallback
  候选（process 类）+ 未来 modality 微调的排序底座候选（4B 表示层首次
  可分，对照 R3c 0.8B 两次训崩）；接入需如实声明 candidate_probability
  语义、模板版本、ts.capture unsupported、显存独占约束。
- **回归**：`.venv_xpu` pytest tests/test_semantic_adapters.py
  tests/test_semantic_contract.py -q → **190 passed, exit 0**（含
  test_jev_backend.py 合跑 202 passed）；默认行为零改动。
- **遗留**：jev-template-v1 单版本未消融，completion 措辞与标签口径冲突
  是最大失分点（jev-template-v2 有界迭代候选，本轮按预登记纪律不做）；
  指标对 provisional 标签；13GB 显存使其无法与 0.8B 底座同卡共存；
  test split 未触碰（无切换动议不消耗）。
- 证据索引：results/semantic-runtime/jev-eval/{report.md,metrics_calib.json,
  predictions_calib.jsonl,load_smoke.py,load_smoke.json,run_full.log}；
  src/moonbow/semantic/backends/jev.py；tools/run_jev_eval.py；
  tests/test_jev_backend.py。
- 命令 exit code：hf download=0、pip --no-deps=0、load_smoke.py=0、
  run_jev_eval.py --limit 3=0、run_jev_eval.py 全量=0、pytest（.venv_xpu）=0。

## 2026-10-03 夜间双线（Track 1 评测门禁 + Track 2 modality 终局迭代）
### Track 2：Jev+LoRA modality 第三次（最终）迭代 — NO-GO（结论稳定）
- 底座：Jev-4B DecisionEngine 官方训练构造器原生支持 LoRA（未走退化路线）；
  parity 门通过（t0.75 与零样本逐位一致）
- 结果：F1 0.738→0.8246（+8.7 点），ECE 0.226→0.114；**门禁 FAIL**
  （R≥0.80 下 maxP=0.782<0.85）；正负中位差 0.044→0.977——R3c 塌缩病灶根治，
  P 平台 ~0.78（14 FP 为"指令/规定式陈述"口径边界）
- 处置：不注册 adapter，no-go 维持；checkpoint 留档 models/semantic_lora/
  modality-jev-v1/。未来重启杠杆=jev-template-v2 口径对齐，非换底座/加数据
- 工程教训（4B hybrid）：无梯度检查点 B=4 即 segfault；整批 pad+阈值 empty_cache
### Track 1：guard-effect-v2 评测 — 停在门禁报告（难度带门禁不过）
- 基础设施通过：ge2_proxy(8901)+runner+身份断言+judge 逐测试；32 run 冒烟
- 伪调用门禁空泛通过：gemini 全走真实 toolCall 通道（mimo 伪调用为模型/网关特有）
- **难度带门禁 FAIL（天花板）**：gemini 对 V/S 全部任务（含 s5/s6/s7 与
  h1_quad_interact/h2_multi_file/h3_naive_trap 三个加难探针）28+6/34 全 100%——
  uplift 在唯一可用模型上不可测
- 备用模型全灭：workbuddy 通道封工具请求、commandcode 池无额度、
  zhushu 通道同样拦截；deepseek-v4.1 两路由均死
- 处置：按预登记停下写报告，主实验未启动；合法结局=免费渠道无法测 uplift，
  需 mimo（充值）或更强/更弱模型才可入带
### morning 收尾
- L1 无害性顺带获得正面数据：34 run 中守卫/收敛通道零误投递、C 族全过

## ge4（2026-10-03 晨，terminal-bench 移植校准）
- 6/15 候选移植成功（tb_eigen/mahjong/recover/org-json/aimo/bpe），canary 保留
- gemini 校准：5/6 触顶 100%，eigen 为超时预算伪影（150 轮阅读停滞，非能力失败）
- **落带 0 个**：v1.2"社区任务库可替代自设计"部分证伪——剥离容器 harness 后
  剩余纯 Python 核心恰是任务最易部分；Medium 分层是对完整容器 harness 的测量
- 首次观测到"150 轮 0 编辑 0 验证"阅读打转停滞新形态（收敛信号的目标案例）
- 遗留：eigen 入主集需预算 ≥900s；kimi-k3 需 proxy developer→system 改写；
  up-stream 可回馈 golden 自比对 bug

## 守卫收敛规则集 v2：attribution 驱动的 converge.repeat / perf_retest / stall 缩放（2026-10-04，纯逻辑改造，不训练不推理不占 XPU）
- 背景：eigen 8 失败 run 归因（`results/guard-effect-v2/eigen_failure_attribution.md`
  §7 干预点标注）：control r3 = 945 轮中 **877 次重复同一条 timeit 命令**的退化
  循环（被 runner 超时处决）；control r1 = 30 次写入在 2–3 个变体间**振荡约
  8 回合**；both r4/r6/r9 = 会内 27 passed、如实申报 STATUS:A → **judge 复测
  翻转 ≤1µs 平局项**。为 2026-10-04 夜间三臂复测落地两条对应守卫规则。
- 实现（ruleset 版本化；v1 缺省 = Phase 1/2 现状，零变化）：
  - **converge.repeat**（新触发器，v2）：a) consecutive_identical——连续
    ≥12 条"归一化后相同"的工具调用（归一化 = 去数字/路径/空白后的命令核心；
    仅数字不同的 timeit 循环逐条同核）；b) edit_oscillation——同一文件 ≥4 次
    写入且每次写入后 pytest 结果集合不变（修了没效果；"结果集合" = coverage
    事件 + goals_passed，与覆盖度同源口径）。任一命中 matched=True，
    detail 含 kind/count/command（同命令）或 file/count（同文件振荡）。
  - **converge.perf_retest**（新触发器，v2）：块流出现 "STATUS: A" 收尾申报
    （`\bSTATUS\s*[:：]\s*A\b`，与 verifier STATUS 门同形的确定性识别）→
    matched；投递另需 perf 门控（`MOONBOW_GUARD_PERF_RETEST=1`，或目标含
    speedup/perf 字样——`MOONBOW_GUARD_CONVERGENCE_TARGETS` env 或 payload
    `convergence_target_tests` 透传均可；`_convergence_shadow_for` 经
    `with_targets` 派生，payload 带目标不降级 v1）+ 独立预算键（AdvisoryBudget
    allow_solo/record_solo：每任务 1 次，不占 MAX_PER_TASK、不计 used()，
    与既有提醒预算互不占额）。固定话术要求"重复运行性能测试至少 2 次确认
    计时稳定，并在 EVIDENCE 中附各档耗时数据"。
  - **stall 轮次线缩放**（v2）：`min(100, max(30, budget_s // 15))`，
    budget_s 来自新 env `MOONBOW_GUARD_CONVERGENCE_BUDGET`（缺省 900 →
    60 轮；≥1500 封顶 100、≤450 保底 30、非法回退缺省）。实证动机 =
    tierA_eigen_final §5（900s 预算下固定高轮次线永不触发，触发线必须随
    预算缩放）。v1 的 4 轮线与既有 detail schema 不动（既有测试锁定）。
  - **门控与标识**：两新规则 shadow 只记录、advisory 才投递、off 零变化；
    `/v1/convergence-status` 增加 `ruleset` / `budget_s` / `stall_round_line`
    键；`start_server` 在通道非 off 时装配 ruleset=v2 并落日志；类属性缺省
    v1（测试/直连进程不经 start_server → 旧行为逐字节不变）。
- 文件：`src/moonbow/guard/convergence.py`（检测纯函数 + Shadow v2 +
  AdvisoryBudget.solo）、`process_audit.py`（advisory 装配 + with_targets）、
  `server.py`（ruleset 装配 + 状态端点）；`progress-guard.ts` **无需改动**
  （预算/perf 门控全在服务端 env，客户端既有透传已够用）；
  `tests/test_convergence_shadow.py` 追加 29 例（v2 全部新行为 + v1 不变锁）。
- 测试（exit code 0）：`python -m pytest tests/test_convergence_shadow.py
  tests/test_process_audit.py tests/test_server_stage.py
  tests/test_guard_semantic_provider.py -q` → **163 passed**；
  `tests/test_convergence_advisory.py` → 24 passed（advisory 路径回归）；
  `node --test tests/test_pseudo_call_guard.mjs` → **19 pass**；
  `tests/test_pi_process.mjs` → 20 pass。默认路径零变化证明：缺省
  ConvergenceShadow()=v1、server 类属性缺省 v1/off，Phase 1/2 既有 55 例
  收敛测试不改一字全绿。
- 冒烟（start_server 实装配）：`MOONBOW_GUARD_CONVERGENCE=advisory` 启动 →
  status `{ruleset: v2, budget_s: 900, stall_round_line: 60}`；4 轮失败流 →
  4 条目 + stall abstain `rounds<60` + fail_streak 正常投递。
- 遗留：①派发文本"budget 缺省 100 轮行为不变"与规则文本"缺省 900 → 60 轮"
  冲突，按公式实现——cap=100 保留旧高线形态（budget≥1500 回到 100 轮），
  v1 缺省路径不变由既有测试证明；②perf_retest 的 STATUS:A 识别是确定性
  正则（块级证据），语义级申报分类仍归 matcher 对齐判定范畴，本规则不做
  语义判定；③edit_oscillation 暂不识别跨变体语义等价（"修了没效果"以
  pytest 结果集合不变为准），变体间振荡的语义等价判定留待 matcher 对齐。

## 识别训练验证：v2 三形态签名跨模型回放（2026-10-04，纯本地分析，不跑模型不占 XPU）
- 目的（预登记）：把 v2 三条失败形态签名（converge.repeat 退化循环 /
  converge.stall@60 / fail_streak）在两个模型的全量历史会话上回放，验证
  **可归纳性**——跨模型命中失败 run、健康 run 零/低误报。规则本体零重写：
  全部经真实 `ConvergenceShadow(ruleset="v2", budget_s=900)` 计算，
  `tests/test_convergence_shadow.py` 60 例不改一字全绿。
- 语料（283 run，只读）：A. mimo-v2.5：Phase 0 批次 233（convergence_study
  同过滤同标注，复算标签与 features.jsonl 基线 233/233 零偏差）+ Phase 2
  批次 30（runs_p2a）；B. gemini-3.5-flash-lite：eigen 20（tb2×18 + ge4×2，
  标签=runs_smoke.completed）。会话→StageBlock 转换层有损处如实记录：
  thinking 块丢弃；轮代理=toolResult 计数（并行轮拆分，mimo 实测最大 ~2x
  偏斜且方向固定）；goals_total=1 生产缺省（目标注入属 matcher 范畴）。
- 结果（`results/guard-effect-v2/rule_replay_validation.md`，
  replay_runs.jsonl / rule_replay_metrics.json 同源生成）：
  - **总判定：可归纳性成立（判据 2：组合规则任一触发=预测失败，
    precision mimo 0.84 [0.653,0.936] / gemini 1.0 [0.51,1.0]，均 ≥0.7）**。
  - 已知样本 3/3：control r3 被 repeat 命中（count=877，第 80 守卫轮）、
    control r1 被 stall@60 命中（第 60 守卫轮）、健康 run 端态 repeat 零命中；
    额外：ge4 150 轮"阅读停滞"实为 81 连击同命令计时循环，repeat+stall 双命中。
  - fail_streak = 唯一严格跨模型一致的单规则（mimo R=0.13/FPR=0.039，
    gemini R=0.2/FPR=0.0；Phase 0 干净批次端态 FPR=0/94；首触中位第 4 守卫轮，
    与 Phase 0 基线 P=0.908 口径相容）。
  - repeat / stall@60 = **形态依赖签名**：gemini 内 2/2、3/3 精确命中且端态
    零误报；mimo 语料形态缺席（同命令连击上界 5<12、同文件写上界 3<4、最长
    33 守卫轮<60）→ 零触发零误报，inactive 非反证但不可作可归纳证据。
  - 误报来源：mimo 4 FP 全在 Phase 2 批次（truncated+env_error 噪声 run，
    标签=工作区复测 vs 会话末连续失败，属口径差非规则误判）；gemini 端态
    FP=0，ever 口径 7 例全为中途触发后恢复（部署语义=每任务 1 次可忽略提醒）。
  - gemini 6 FN 中 3 个正是 perf_retest 管辖形态（both r4/r6/r9 申报 A 翻车，
    超出三签名范围）。
- 文件：`tools/rule_replay_validation.py`（新增，转换层+真实规则回放+统计+
  报告生成；复用 tools/convergence_study.py 解析与标注）；convergence.py
  **零改动**。遗留：①gemini n=20 小样本，单规则 Fisher p 0.21–0.47 不显著，
  v2 规则臂干预效果仍需 eigen 复测补样；②Phase 2 截断噪声提示 fail_streak
  宜消费 toolResult.is_error 标记（规则侧待后续版本）；③edit_oscillation
  端态可逆（both r8 中途命中后恢复），历史峰值口径需块流携带轮边界。

## 通宵双线终判（2026-10-04 晨）
### Track 1：三臂终判（eigen，n=9/9/9）
- control 56% / v1(旧规则) 56% / v2(新规则) **78%**（+22pts，Fisher p=0.62 欠功效）
- 停滞：control 2（含 945 轮巨 stall）/ v1 0 / v2 1（113 轮）
- 预登记模板 (b) 变体结论：方向为正、统计不可分；**规则参战证据缺失**
  （run 级触发/投递计数为 0——投递链路或记录口径待核查，guard_v2.log 留存）
- 过程事故与修复：runs_smoke.jsonl 曾被归档脚本以 json.dump 写成单行数组
  （90 条记录全部找回，零丢失）；v2 臂 run12-18 因此未执行，已补跑
- 详见 results/guard-effect-v2/threearm_report.md
### Track 2：jev-template-v2 — 模板有效（template-effective）
- completion F1 0.489→**0.8214**（+33.25pts，门禁 ≥+10 ✓）、R 0.367→0.7667 ✓
- 正/负中位分差 0.042→0.765（分离）、ECE 0.133→0.071
- 定位：zero-shot 冷启动候补（现役 completion-lora-v2 0.893 仍更高）；
  剩余 7 FN 为"主体 done+质量缺口未验"单一形态，需第三轮措辞消融
- 详见 results/semantic-runtime/jev-template-v2/report.md

## 晨间补充（2026-10-04 07:30）
- **投递链路根因确认**（guard_v2.log 5115 次 stage-check 全查）：
  findings 全为 0 的原因 = stage-check 通道不携带工具调用/结果事件，
  v2 三规则（吃工具流）全程"饿着"未参战。修复方案已入册
  （threearm_report.md §8：通道补工具事件 → 离线对拍 → 三臂公平复测）。
- **ge5 首测**：visible-pass/hidden-fail 家族校准成功（hidden-pass
  基线 64%，三任务全落 30-70% 带）；守卫臂方向对照 n=6/6 无提升
  （逐任务与 control 完全相同，提醒零触发=无效对照）；
  捕获"探索-放弃"新失败形态样本。详见 ge5_directional_report.md。
- **eigen 封卷**：n=12/9/12，完成率 uplift 收敛向 null（58/56/67），
  任务族封存，后续样本转 ge5。
- 服务已停（8901/18617），下窗口重启命令见 v14_baseline_set.md §4。

## jev-template-v2 模板落地（2026-10-04，纯逻辑改造，未跑模型）

- **落地方式**：Track 2 验证有效的 v2 模板由实验期进程内 monkeypatch 改为
  `src/moonbow/semantic/backends/jev.py` 原生版本化常量。`_COMPLETION_ASSERTED_V2`
  = completion.asserted 的 instructions/criteria.false/criteria.true 三段，
  自 `experiments/jev-template-v2.json` 逐字移植（单测对照该 JSON 校验逐字
  一致）；模板版本注册表 `_JEV_TEMPLATE_SETS`（v1 全集 / v2 仅覆写
  completion.asserted，其余 3 模式与 state 拼装回退 v1 同卷对照）；
  `_JEV_TEMPLATES` 保留为 v1 全集（tools/r4_jev_modality.py 等既有引用不受
  影响）。
- **版本选择机制**：`_template_version(pattern)` 按 `_JEV_PATTERN_VERSIONS`
  解析（completion.asserted=jev-template-v2，其余=jev-template-v1）；回退
  开关 env `JEV_TEMPLATE_PIN=v1` 强制全模式回 v1（未知值显式 ValueError，
  不静默忽略；缺省不设=新版本）。provenance.calibration_id =
  "template:<该模式版本>" 按请求模式回填；capabilities 新增
  template_versions（逐模式）与 template_pin_env；diagnostics.template_version
  随实际所用版本。
- **测试**：tests/test_jev_backend.py 新增 6 例（completion=v2 / 其余=v1 的
  provenance 与 diagnostics 回填、v2 措辞对实验 JSON 的逐字一致性、pin=v1
  全模式回退、缺省=新版本、未知 pin 报错、capabilities 逐模式版本表）。
  全绿：`python -m pytest tests/test_jev_backend.py -q` → **18 passed
  （exit 0）**；`python -m pytest tests/test_convergence_shadow.py -q` →
  **60 passed（exit 0）**，无意外。
- **兼容性**：jev_template_v2_eval.py 重跑语义不变（其 monkeypatch 现为
  no-op，原生路径产出与实验期同一模板/版本）；modality/alignment/process
  模板零改动；实验产物与冻结数据未触碰。
- **遗留**：①XPU 实测复跑（原生 v2 常量路径端到端验证）留待下窗口（本轮
  按约束零模型零 XPU）；②completion 剩余 7 FN（"主体 done+质量缺口未验"
  形态）未修，第三轮措辞消融未启动；③test split 仍未触碰（留最终验收）；
  ④按派发约束未 git commit。

## guard-effect-v2 投递链路补全：stage-check 通道携带工具事件 + 离线对拍（2026-10-04，纯逻辑改造，不跑模型不占 XPU）
- 背景：threearm_report.md §7 根因确认——线上 5115 次 stage-check findings
  全 0，v2 三规则（repeat/stall/fail_streak 全部依赖工具流）整场"饿着"：
  部署运行时里依赖消息快照组装的块采集路径未把工具调用/结果送进观察流
  （离线回放直接喂会话工具流则全部有效）。本节落地修复清单 §8.1/§8.2：
  ①通道补全；②离线对拍（session 工具流 vs 线上通道，同 run 同结果）。
- **客户端**（extensions/process-events.ts / process-task.ts /
  process-audit.ts / progress-guard.ts）：
  - 工具事件改走 pi 核心执行事件 `tool_execution_start/end`（与消息快照
    组装无关、每次工具执行必发），经 `ToolEventCollector` 按 (phase,
    toolCallId) 去重、与块共用同一 seq 发号器（全局时序可合并）；
  - stage-check payload 新增 `tool_events` 数组（默认携带）。**携带纪律
    论证**：客户端不知道服务端 ruleset，按端开关折叠会把 v2 装配判据漏到
    客户端；体积可控（实测最大会话 1890 事件约 210KB 原文，截断后更小，
    本机 HTTP）；off 模式根本不发 stage-check、v1 服务端忽略该键——两条
    既有路径零行为变化；
  - 判定载荷纪律（最小字段）：args 摘要**逐值截断**（单值 200 字符、键数
    40、总长 1200 兜底）——键结构完整保留，服务端 `_parse_tool_call` 照常
    解析 command/path（整段头截断会把 JSON 截坏，实测 both r8 的
    edit_oscillation 因此漏检，对拍抓出后修复）；结果摘要中段截断
    （头 160 + 尾 320）且 pytest 摘要行保尾 + 显式补附双保险；
  - 封顶与游标：`tool_events` 环形缓冲 4000（env
    `MOONBOW_GUARD_MAX_TOOL_EVENTS`，丢最旧）；**纯工具增量（无新
    text/thinking 块）同样触发审计**——877 同命令循环形态中途无文本块，
    缺此判据则退化循环中途工具流永不重发；游标按发送时刻版本快照推进
    （飞行期间新增事件保持未审，尾随合并补审计）。
- **服务端**（process_audit.py）：`tool_events_to_stage_blocks` 解析
  payload 键（非法类型 ValueError → HTTP 400；非法条目逐条跳过）；
  `StageAuditor._convergence_blocks`：**仅 ruleset=v2 且 tool_events 在场
  时**，收敛计算改用 tool_events 为工具流权威来源（blocks 内工具条目让位
  防双通道重复计数，text/thinking 保留供 perf_retest）；v1 / 未携带 →
  原样（逐字节不变）。主审计 findings/reminder 路径完全不消费该键。
- **离线对拍**（`tools/rule_replay_validation.py --parity`，同卷 =
  既有 replay_runs.jsonl 的 gemini eigen 20（含 control r3 877 连击、
  r1 144 轮停滞）+ mimo Phase 2 30（含截断噪声 run））：转换层同时产出
  "离线格式块"（会话工具流直喂）与"线上格式块"（客户端 tool_events 序列化
  模拟 → 服务端解析器 → 与 text 块窗口合并 = `_convergence_blocks` 的 v2
  输出形态），同一 run 两种喂法跑真实 `ConvergenceShadow`（v2, budget=900）。
- **对拍结果：50/50 逐 run 触发一致（一致率 1.0），exit 0**——逐规则
  fired / abstain / 首触轮 / ever-fired 全同；已知样本全对齐（r3 repeat
  count=877 首触第 80 守卫轮、r1 stall 首触第 60 轮两格式一致）。
  detail 全同 34/50：其余 16 run 差异**全部落在未触发 repeat 的观测量**
  （max_identical_calls / max_same_file_writes，低于阈值 12/4，方向混合：
  逐值截断让尾部仅数字不同的命令同核 +1、长重定向命令丢尾部 `> 文件` 使
  写路径解析缺失 -1）；已触发规则的 detail（如 count=877）逐字节一致。
  逐 run 对照表：`results/guard-effect-v2/channel_parity_report.md` +
  `channel_parity_runs.jsonl`。
- 测试（全部 exit 0）：`python -m pytest tests/test_convergence_shadow.py
  tests/test_process_audit.py tests/test_server_stage.py
  tests/test_guard_semantic_provider.py -q` → **171 passed**（新增 8 例：
  转换 schema、payload 校验、v2 线上 repeat 877 形态触发、v1 tool_events
  逐字节不变、v2 权威流去重、v1/v2 消费对照、perf_retest 文本证据、HTTP
  v1 忽略/v2 消费 + 400）；`node --test tests/test_pi_process.mjs
  tests/test_pseudo_call_guard.mjs` → **42 pass / 0 fail**（新增 3 例：
  payload 携带与去重、截断 + JSON 可解析 + pytest 行存活、封顶 + 纯工具
  增量补审计）；`--parity` 对拍脚本全绿（50/50）。
- v1 不变证明：①`test_v1_ignores_tool_events_byte_identical`——同一
  payload 带/不带 tool_events 在 v1 shadow 与无 shadow（off 路径）下响应
  逐字节一致（json.dumps sort_keys 相等）；②既有 163 例收敛/审计/HTTP 测试
  不改一字全绿（其中含 v1 形状锁定例）；③对拍卷 50 run 的离线格式触发
  结果与既有 rule_replay_validation.md 一致（同卷同规则）。
- 顺带修复：`load_gemini` 读 runs_smoke.jsonl 改回逐行 JSONL（e7fff87 恢复
  格式后 `json.load` 已失效）；`once()` 空 fresh 块列表时 cursor 推进的
  潜在 TypeError（optional chaining）。
- 遗留：①对拍等价于"线上通道喂法 = 离线整流喂法"在端态口径下的证明，
  部署窗口语义（blocks 400 / events 4000 环形）对超长 run 的端态覆盖度
  只反映窗口尾部（方向固定、与部署一致，见报告 §2.3）；②修通路后的
  三臂复测（threearm §8.3）待下窗口，本节不改变 ge2_runner 与三臂数据口径；
  ③tool_events 与 blocks 双通道并存的唯一性由"tool_events 权威、blocks
  工具条目让位"保证，若未来客户端只发单通道需同步收敛。

# Maps 研究文档与基准结果总索引 (Maps Index)

本目录归档了 Spark-4B / Progress Guard 研发攻坚阶段沉淀的高价值核心学术报告与实验结果。早期历史过程草稿及中间探针已全部归档至 `legacy_archive/`。

> **发布说明（2026-09-21 审计）**：表中「关联产物 / 脚本」为研发期的本地产物存档名，
> 一次性研究脚本与原始结果文件未随本仓库发布（引用它们的数据路径多含未发布的私有语料）；
> 各报告的结论自洽可读，产出物名称仅作出处存档。

---

## 核心研究报告清单 (Core Research Reports)

| 报告文件名 | 主题与核心结论 | 关联产物 / 脚本 |
|---|---|---|
| **[CONTRACT_PILOT_2026-09-20.md](CONTRACT_PILOT_2026-09-20.md)** | **【核心总报告】** 闭合契约机制 v1~v6 完整演进总览。确立极简三字段 + 硬软信号解耦 + XPU 二值捕获头 + 非对称争议放行；**冻结 67 样本端点 B 准确率 80.6%，误放行降至 1 例**；实证 1.5B 语义自省悬崖。 | `best_contract_results.json`, `producer_qwen25-15b-instruct_results.json` |
| **[SWE_EVAL_AUDIT_2026-09-20.md](SWE_EVAL_AUDIT_2026-09-20.md)** | **【评测陷阱审计】** SWE-bench Lite 评测基建审计：排查并修复 3 个导致离线评测数据失效的致命缺陷（浅克隆 commit 不可达、共享工作区并发状态污染、模型篡改测试文件的 Reward Hacking 防御）。 | `pi_eval/evaluate.py` |
| **[PLATFORM_ADAPTATION_2026-09-20.md](PLATFORM_ADAPTATION_2026-09-20.md)** | **【工程平台适配】** Spark-X2.5-4B 真实部署适配：从 PyTorch-XPU (3.8 tok/s) 切换至 llama.cpp Vulkan (**58 tok/s，提速 15 倍**)；探索出 `--reasoning-budget 512` 压缩思考 30 倍且保留能力；确认 Pi 插件需走 `context` 动态事件注入。 | `tools/llama.cpp-vulkan/`, `start_spark_server.py` |
| **[SWE_EVAL_2026-09-20.md](SWE_EVAL_2026-09-20.md)** | **【SWE 先导实测】** 真实 Agent harness（封闭版 Pi）在 Flask 仓库上的完整运行行为分析；首次观测到模型产出正确代码修改但受困于环境噪声跑偏。 | `pi_eval/run_case.py` |
| **[audit_alignment_night_2026-09-19.md](audit_alignment_night_2026-09-19.md)** | **【旧研究反身性审计】** 揭露前期对齐研究中的四大 P0 问题：测试集锚点在训练集完全重合（Jaccard 1.0）、字符归一化不全导致的“假零泄漏”、三分支无增益与测试脚本硬编码预期输出。 | `pipeline/research_v232_heldout_retrain.py` |
| **[REDO_ALIGNMENT_STUDY_2026-09-19.md](REDO_ALIGNMENT_STUDY_2026-09-19.md)** | **【严格划分复测】** 干净 Held-out 划分下重做微调对比实验：预注册指标全部未达标，证实小模型直接微调对闭合判断**净增益为 0**。 | `alignment_heldout_split.json`, `contrastive_strictly_heldout_train.json` |
| **[DECISION_symbolic_vs_operator_2026-09-20.md](DECISION_symbolic_vs_operator_2026-09-20.md)** | **【形式理论决策】** 形式化推导证明：任务闭合度是客体范围相关的量化谓词，无法通过单句制造；确立符号层与学习算子的职责边界。 | `research_v234_relation_probe.py` |
| **[PILOT_symbolic_layer_2026-09-20.md](PILOT_symbolic_layer_2026-09-20.md)** | **【符号层试点】** 零样本 GLiNER 槽位提取试点：验证架构可行性，发现零样本抽取噪声抵消模型收益。 | `research_v235_symbolic_pilot.py` |
| **[SLOT_MODEL_2026-09-20.md](SLOT_MODEL_2026-09-20.md)** | **【槽模型定论】** Checkpoint 扫描揭示 4 种虚假相关性（Spurious Correlations），确立单句小模型无法泛化解决完成度判定。 | `slot_checkpoint_sweep.json`, `real_slot_pool.json` |
| **[PAIR_STATION_2026-09-20.md](PAIR_STATION_2026-09-20.md)** | **【成对站极限】** 在 253 对真实标注数据上的微调探索，证明样本量不足 1000+ 时判别小模型发生严重过拟合。 | `real_pair_pool.json`, `real_pair_labels.json` |
| **[PROTOCOL_V1.md](PROTOCOL_V1.md)** | **【重协议复盘】** 五字段重契约设计的失败复盘：证明字段重量直接正比于失败暴露面，必须回归极简三字段。 | `protocol_v1_results.json` |
| **[COVERAGE_CRITERION_v7.md](COVERAGE_CRITERION_v7.md)** | **【标准规范】** 早期任务覆盖度判据的最终归一版本规范。 | - |

---

## 核心基准数据与结果文件 (Benchmark Artifacts)

- `best_contract_results.json`：终版最佳契约 v6 在 67 对冻结测试集上的多轮交互明细与判定结果（准确率 80.6%，FP=1 例）。
- `producer_qwen25-15b-instruct_results.json`：修复解析器后 1.5B 生产者的全量复测结果（格式遵从 99%，语义自省失能致 21 例 FP）。
- `relation_probe_results.json`：NLI-deberta、bge-reranker 与 Qwen 探针在冻结集上的关系探测基准表现。
- `real_slot_pool.json` / `real_slot_labels.json`：从真实会话人工精细标注的 283 条单句槽位语料。
- `real_pair_pool.json` / `real_pair_labels.json`：从真实会话人工精细标注的 253 对 (请求, 回应) 成对语料。
- `alignment_heldout_split.json` / `contrastive_strictly_heldout_train.json`：基于模板家族严格隔离的零泄漏划分集。

# 审计报告：2026-09-19 晚间"语义匹配 / 客体对齐"工作流（+ 今早守卫工作抽查）

> 审计人：独立审计会话（sess_4270d712），2026-09-19 23:40–24:00
> 范围：`pipeline/` 当晚 18:46–23:26 新增的 14 个脚本及其产物
> （`maps/contrastive_alignment_train.json`、`maps/contrastive_strictly_heldout_train.json`、
> `models/minilm_alignment_ft`），并抽查 FINDINGS §288–§302b 的守卫结论。
> 方法：只读 + 确定性复测（字符 Jaccard 重叠扫描、行为学探针、A/B 基准复测、脚本静态审查）。
> 复测脚本落盘于审计会话临时目录，关键数字均可在本文件引用的输入上重跑。

---

## 总判定：今晚这条线的核心结论【不成立】，主要数字【作废】

四项 P0 问题相互独立、任一单独存在即足以否证"后训练微管线达标"的叙述。

---

## P0-1 训练/评估闭环污染（基准就是训练模板的换皮）

`build_contrastive_alignment_dataset.py`（22:45）生成的训练集
`maps/contrastive_alignment_train.json`（n=239）与
`eval_semantic_matcher_benchmark.py` 的 20 样本基准重合：

| 证据 | 训练集内容 | 基准项 | 重合度 |
|---|---|---|---|
| 逐字相同 | anchor「安装 pydantic 和 fastapi 依赖」 | 02_dep_install req | **Jaccard=1.000（逐字）** |
| 近逐字 | pos「执行 alembic upgrade head 成功更新表结构新增列」 | 07_db_migration res | 0.852 |
| 近逐字 | anchor「编写数据导出 Excel 功能」 | 10_cross_export_rm req | 0.857 |
| 同模板族 | pos「拆解为职责单一的 4 个私有辅助函数」/「删除了…console.log、print…TODO」 | 19 / 20 | 0.69 |
| 负例侧同源 | 模板 neg「修改了前端导航栏的字体与背景颜色」「压缩了静态资源图片体积」 | 08 res（导航栏样式）/ 09 res（镜像瘦身） | 同构造 |

即：**正例与负例两侧都来自同一批手写模板**，基准对模型而言是训练分布内插值，
不是泛化度量。实测（归一化字符 Jaccard）：≥0.75 重合 3/20，0.69–0.73 边缘重合再 4 项，
按场景计 8–10/20 在训练中存在近 duplicates。

## P0-2 "严格零泄漏"声明与事实不符（当前检查点就是污染集训练的）

时间线：22:45 污染集训练 + `train_contrastive_alignment.py` 落盘（**硬编码读
`contrastive_alignment_train.json`，此后未再修改**）→ 23:18 生成
`contrastive_strictly_heldout_train.json` → **23:19 模型重新保存** → 23:21/23:26 的
"clean" 回放与对抗压测以"零泄漏微调"名义加载该检查点。

行为学探针（相对未微调 MiniLM 基座的 pos-sim 变化）裁决检查点见过哪份数据：

| 组 | 内容 | BASE | FT | Δ |
|---|---|---|---|---|
| A 污染集独有（bench02 逐字对） | pydantic/fastapi | 0.276 | **0.506** | **+0.23** |
| B heldout 集独有（Redis/Kafka/WASM 逐字模板对） | ×3 | 0.415 | 0.297 | **−0.12（全降）** |
| C 双方皆无（bench 03/05/12） | 对照 | 0.384 | 0.319 | −0.07 |

若真用 heldout 集训练，B 组必然上升；实测全降。**结论：`models/minilm_alignment_ft`
（23:19）是用污染集训练的。** `run_dag_session_subjective_eval_clean.py` 头注
"严格零泄漏微调的 117M"与事实不符。

## P0-3 "后训练"增益为零（A/B 复测）

复刻 `eval_pipeline_post_training.py` 判定逻辑（guard CLOSE ∧ sim≥0.35，或 sim≥0.65），
同一 guard、同一 20 基准，仅替换嵌入模型：

| 配置 | 准确率 | 正例平均 sim | 负例平均 sim | 变化 |
|---|---|---|---|---|
| FT（污染训练，现行检查点） | **17/20 (85%)** | 0.497 | 0.181 | bench02 FN→TP；**bench15 新增 FP**（sim 0.383 过阈 + guard CLOSE） |
| BASE（未微调原版 MiniLM） | **17/20 (85%)** | 0.513 | 0.265 | — |

**总准确率与未微调基座完全相同**；唯一"提升"是逐字背下的 bench02，
代价是把原本判对的 bench15（部分完成）变成误报。正例平均相似度反而低于基座
（评估脚本里"正例平均相似度…(显著拉升)"的打印文案不成立）。

## P0-4 硬编码结论打印（报告完整性问题）

`test_specialized_micro_pipeline.py:131-132` 无条件打印
`跨话题负例拦截率: 4/4 (100.0%)` 与 `过程性/报错负例拦截: 7/7 (100.0%)`——
**与实际逐例判定脱钩**。复测中 FT 配置下 bench15（过程性负例）就是 FP，
该行仍会打印 7/7。凡引用该脚本输出得出的"11/11 全拦截"结论均为无效证据。

---

## P1 问题（方法学/可复现性）

1. **评估脚本在项目主环境跑不起来**：`models/minilm_alignment_ft` 的 tokenizer 由
   `.venv_xpu` 的新版 transformers 保存（`TokenizersBackend`），`.venv`（CPU 主力环境）
   加载即抛 ValueError。今晚所有数字在 CPU 环境不可复现，检查点环境锁死。
2. **早版"主观回放"不是纯模型**：`run_dag_session_subjective_eval.py:88` 内嵌关键词表
   （"审计/修/怎么/支持/…"）把 NEUTRAL 直接改判 REQUEST。23:21 的"clean"版自述删除了
   该词表——即删除前的主观回放结果部分由手写规则驱动。
3. **`run_real_session_subjective_eval.py` 截断采样 + 硬编码成功话术**：只取会话前 40 块；
   待办池为空时打印"所有提出的开放问题均已被纯语义微管线平稳核销，允许退出！"
   ——这是写死的文案，不是实验结论。
4. **基准本身不合格**：n=20、手写、单金标、无标注者一致性；阈值 0.35/0.65 无预注册来源
   （疑似在同基准上调试所得）。这与项目自身确立的验收规范
   （Wilson CI、预注册判据、多口径矩阵 `maps/MULTI_METRIC_ACCEPTANCE.md`）直接冲突。
5. **heldout 集黑名单不完整**：`build_strictly_heldout_dataset.py` 的黑名单
   （9090/pydantic/fastapi/auth/README/utils.py/origin/feat-login/alembic/圈复杂度/print/console.log）
   只覆盖基准正例域，未覆盖负例场景域（MySQL、JWT/Docker、Excel、Python 3.12、回归测试、
   微信支付等）；混入的 `rqp7_v12_1_train.json` 行仅靠该黑名单过滤。23:18 的 heldout 集
   相对干净（重合扫描 0/20 ≥0.75）是事实，但"彻底排除"的说法过强。
6. **主观回放无金标**：所有"真实会话回放"没有人工裁决的真值登记/关闭集合，
   无法计算漏报/误报率，不能作为任何达标证据（脚本自称"主观"尚可，引用需止于此）。

---

## 今早守卫工作（§288–§302b）抽查

- **§302「840 轮 漏报 0 / 误报 0」是同义反复，标题误导**：
  `maps/_guard/run_timing_large.py` 中 `attempt_return` 的 block 标志就是
  `(open_questions > 0)` 的直接推导，因此 miss（表非空未拦）/fp（表空却拦）
  只能由实现 bug 产生，**恒为 0，不构成守卫有效性的证据**。
  守卫真实的匹配正确率在同日人工裁决里：`maps/_guard/MATCH_AUDIT.md`
  8 个 CLOSE 事件中 5 个关错（@19/@29/@46 少关 2/1/6 条，@56 误关 1 条）。
  建议把该指标改名为"状态机记账自洽性检查"，并在 FINDINGS §302 加限定说明。
- 采样亦有截断：每条消息只取前 3 段、每会话上限 60 块。
- MATCH_AUDIT.md / FALSEBLOCK_ANALYSIS.md / CLOSE_RECALL_ANALYSIS.md 本身质量良好，
  v2 批量关闭规则（以最近 REQUEST 为界 + 验证证据门槛）设计合理，未发现新问题。

---

## 与项目自身教训的关系

§101（训练/测试泄漏推翻达标）、§237–§241（伪影污染、门槛口径选错）的教训在今晚全部重演：
新基准未做泄漏核查、新数据集未过 MinHash/重叠扫描（项目已有 `research_v95` 工具链）、
验收未走多口径矩阵。**审计闭环目前是"事后可选"，不是强制门。**

---

## 修复建议（按优先级）

1. 【立即】对外停引 `models/minilm_alignment_ft`（23:19 检查点）的一切"零泄漏/达标"表述；
   FINDINGS 补记今晚条目并标注"污染已识别、结果作废"（保留轨迹，勿删，与 §98 处理 D55 同规）。
2. 【今天可做】真重训：改 `train_contrastive_alignment.py` 数据路径为
   `contrastive_strictly_heldout_train.json`（或加 CLI 参数），在同一 venv 保存并加载验证；
   修 P1-1 的 tokenizer 兼容问题。heldout 集黑名单改为域级排除 + MinHash 复核。
3. 【本周】基准重造：在 heldout 域上新建 ≥100 对、双人独立标注、报 Wilson CI；
   现 20 基准降级为 smoke test，禁止作为达标证据引用。阈值 0.35/0.65 写明来源并预注册。
4. 【立即】删除 `test_specialized_micro_pipeline.py` 的两行硬编码结论，改为按实际结果计算。
5. 【流程】把"新数据集/新基准必须过重叠扫描（复用 research_v95）+ 多口径矩阵"写进
   RQP7 协议作为硬闸，避免 §101/§237/今晚的第四次重演。
6. FINDINGS §302 的"漏报 0/误报 0"加限定：该指标为记账自洽性，非拦截正确性；
   拦截正确性以 MATCH_AUDIT 类人工裁决集为准。

---

## 复测入口（供复核）

- 重合扫描：对两份 `maps/contrastive_*_train.json` 与 `BENCHMARK` 做归一化字符 Jaccard
  （审计会话 2026-09-19 23:47 运行，结果：v1 集 3/20 ≥0.75 且 bench02=1.000；heldout 集 0/20）。
- 行为探针 + A/B 基准：加载 `models/minilm_alignment_ft` 与基座 MiniLM，
  guard 用 `models/progress_guard_v3/final`，判定逻辑同 `eval_pipeline_post_training.py`。
  注意：FT 检查点 tokenizer 需用基座 tokenizer 代替加载（P1-1）。

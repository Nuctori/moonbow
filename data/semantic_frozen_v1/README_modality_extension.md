# modality_extension_r4 — modality.assertive 专门分层批（R4，provisional）

日期：2026-09-30。背景：R3 modality.assertive LoRA 训崩（正负中位差仅 0.044，
小数据 BCE 破坏排序，无 checkpoint 超 zero-shot），zero-shot 基线
P=0.406 / R=0.981（yes 偏置）。本批为**在生成式底座上重训 modality** 备料：
针对 yes 偏置补足三层刻意分层的负例，并设计最小对立对保留排序信号。

## 文件与用法

- `modality_extension_r4.jsonl` / `modality_extension_r4_labels.jsonl`：180 条
  （id 前缀 `fzmod_r4_`，源任务前缀 `fzmodT`，与冻结集 fzT 及 fzdev_r4 的
  源任务无交叉），格式与 `frozen_texts_dev.jsonl` / `frozen_labels_dev.jsonl`
  完全一致。
- **不混入冻结集**：frozen_texts/labels 三组文件保持不变；重训时作为附加
  dev 数据读取，评测仍以冻结 calib 口径报告。
- **边界：仅用于 modality.assertive 重训**（生成式底座 + answer-token BCE 的
  正负配比改良）。不参与 test/calib，不进入阈值选择与最终验收证据；
  其余 4 个 pattern 标签为多任务复用如实标注，非本批重点。

## 分层口径（三层 + 边界仲裁层）

| 层 | id 段 | 数量 | 口径 |
|---|---|---|---|
| A 断言正例 | 0001–0052 | 52 | 直陈已完成：过去式、结果呈现、指标陈述、引用输出后断言。含对立对正例 16 条 |
| B 计划/意图负例 | 0053–0106 | 54 | 将要做 / 准备做 / 正在做 / 接下来。含对立对负例 16 条 |
| C 含糊/疑问/条件负例 | 0107–0158 | 52 | 可能 / 应该吧 / 大概 / 疑问句 / 条件未决 / 转述他人猜测 |
| D 边界仲裁层 | 0159–0180 | 22 | R3 calib 上 1.0 分误报的对抗形态（见下） |

语言：zh 128 / en 52（71%:29%，约 7:3）。家族：agent_completion 76 /
interim_report 52 / user_request 22 / conversation_interference 30。
5 pattern 标签齐全：modality 正/负 = 59/121；completion.asserted 35 正、
task.object.alignment 171 正、process.unresolved 115 正、ts.capture 全部
abstain（本批聚焦语气判定，不含 8 类捕获锚点，沿用 dev_extension 约定）。

## 最小对立对设计意图（16 对，中英各半）

fzmodT001–T016 每组恰 2 条、共享 task_id，同一动词两种时态/语态成对：
正例"已经做完了 / is done"，负例"明天开始做 / will do next"。目的：

- yes 偏置的病灶是排序不可分（正负中位差 0.044）；对立对把**词汇面完全
  拉平、只留语气/时态差**，迫使模型只依赖语气特征而非词汇共现打分。
- 训练时建议同组样本同批出现（task_id 已可分组），避免对立对被随机打散
  后梯度互相抵消。

## 边界仲裁层（Layer D）与标签泄漏防线

R3 calib 上 1.0 分误报的典型形态（yes 偏置把一切负例顶满）：礼貌收尾
无断言、汇报中自问句、转述他人断言（说话人语气非断言）、承诺将来
（promise 非 assert）、条件未决推断。核心裁决规则：

- **modality 只判语气，不判完成度**。阶段性汇报中对子部分的确定断言
  （如"日志模块确实上线了，检索页还差两个交互"）→ modality=true，
  completion.asserted 如实标 false。语气标签的 label_basis **只引用语气
  语义**，禁止以"工作是否完成"作依据，防止 modality 标签照抄 completion
  标签（validator 有反泄漏检查：modality 正例不得全部 completion=true）。
- 担保既成事实（"我担保这个数字没问题"）仍属断言语气 → 正例；
  担保将来动作属 promise → 负例。
- 转述他人断言且本人未确认 → 说话人非断言语气 → 负例。

## provisional 声明

与 README.md 一致：全部文本 agent 原创，与 frozen 三组、dev_extension_r4、
semantic_dev_v1 做字符 bigram Jaccard>0.8 查重零命中；标签为 agent 生成 +
自审的暂定标注，无人工复核，不构成人工金标准；每条带 label_basis 一句话
依据；ts.capture 全部 abstain。

## 校验

`python tools/validate_frozen_data.py --modality-ext`（exit 0）：id 唯一且
前缀 fzmod_r4_、task_id 不与既有集交叉、5 pattern 枚举、ts.capture 全
abstain、三层配额（A/B/C ≥50、D 20–40）、对立对（16 组恰 2 条且 modality
一正一负）、反泄漏、语言比、查重。

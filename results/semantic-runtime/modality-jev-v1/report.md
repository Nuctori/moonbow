# R4/Track2：Jev-4B + LoRA modality.assertive 重训报告（modality 第三次也是最终迭代，2026-10-03）

预登记门禁（未事后修改）：calib 150 任一单一阈值 **P>=0.85 且 R>=0.80 → 达标**；
否则 no-go。预登记监护：每 50 step 在 calib 150 全量上看 P/R 与正负中位差；
排序塌缩（中位差 <0.05）立即停；扫描 F1 连续 2 次不升早停。test split 全程未碰。

## 1. 判定：**NO-GO**

- 全扫描最优（=t0.5）：**P 0.7705 / R 0.8868 / F1 0.8246**（tp47 fp14 fn6）
- 门禁两侧：**R>=0.80 下 maxP = 0.7818（t0.79–0.85）< 0.85**；
  **P>=0.85 下 maxR = 0.3774（t0.99）<< 0.80** —— 无任何格子同时满足
- 处置：不注册 adapter，`config/semantic_runtime_lora.json` 与
  `R4_CAPABILITY_STATUS` 均未改动（modality 维持既有 no-go 口径）；
  checkpoint 与全部证据留档。

与 zero-shot 相比**实质性改善但不足以过线**：F1 0.8246 vs 0.738（+8.7 点）、
P 0.7705 vs 0.652、ECE 0.226→0.114、Brier 0.200→0.114；正负中位差
0.788→**0.9768**（正例中位 0.9772 / 负例中位 0.0004，分离近乎完全干净）。
失败模式与 jev-eval 诊断一致：FP 集中在"指令/规定式陈述"（把规定语气判为
assert），LoRA 把边界整体左移换来 P 上限 ~0.78 的平台期，继续抬 P 则 recall
塌向 0.38——这是底座模板口径（assert="states a fact as already true"）与
provisional 标签口径（agent 对自身工作进展的断言语气）之间的边界冲突，
非排序能力缺失（中位差 0.977 是全部 modality 方案中首次出现的完全可分排序）。

## 2. 底座选型与理由（首选路线成立，未走退化路线）

- **服务 API 不支持注入**：`neohorse_decision` 1.0.0 官方 wheel 的
  `DecisionEngine`（engine.py）只读 merged bundle（`load_bundle`），
  `predict` 全程 `torch.inference_mode()` + 串行锁，无任何 LoRA/adapter/
  参数注入接口。
- **但同一 wheel 附带训练构造器**：`_vendor/model.py` 的 `DecisionModel`
  构造参数原生支持 `lora`（peft `FEATURE_EXTRACTION`）、`lora_targets`
  （hybrid 感知：DeltaNet 层追加 `in_proj_qkv/z/a/b` + `out_proj`）、
  `special_embeddings`、`trainable_parameters()`；bundle 的
  `model_manifest.json` 写明 `"lora_scale": 1.0`——Jev 权重本身即由
  LoRA-train+merge 经此类产出，即官方训练路径就是本方案。
- **Choice 模式可直接训练**：modality 的 assert/promise/question 三类 =
  PointerHead 在 `<decide>` 位的 3 路 logits（`forward_rows_batch`），
  损失直接作用于门禁判定量 p(assert)。
- **与 zero-shot 基线严格可比**：hybrid 底座的推理路径
  （`_inference.predict → model.probs → forward → forward_rows_batch`）
  与 DecisionEngine 是同一代码路径；pre-train eval 复现了 jev-eval
  zero-shot（见 §4 parity 门）。
- 退化路线（NeoHorse-1-4B answer-token，~8GB 下载）无需启用：磁盘检查
  通过（M: 374.7GB free）但不必要。HF 实查 NeoHorse-1-4B 公开可下
  （28870 下载/月，非 gated），留档备查。

## 3. 训练命令与 exit code（全程 XPU，脚本内断言 torch.xpu）

| 命令 | exit | 说明 |
|---|---|---|
| `.venv_xpu/Scripts/python.exe tools/r4_jev_modality.py --mode smoke` | 0 | 冒烟：加载+12 条前向+1 训练步+ckpt 往返 |
| `.venv_xpu/Scripts/python.exe tools/r4_jev_modality.py --mode train` | 139（首跑） | step 10 segfault（见下） |
| 同上（修复后重跑） | **0** | early stop @ step 150，best=step 50 |
| `.venv_xpu/Scripts/python.exe tools/r4_jev_modality.py --mode eval` | 1（首跑）/ **0** | calib 150 全量出 preds+metrics |

**XPU 稳定性发现（4B hybrid 参考内核，重要教训）**：
- 无梯度检查点时参考 DeltaNet backward（`chunk_gated_delta_rule` 回退实现）
  **batch=4 即 native segfault**（exit 139，不可捕获；0.8B r3 的 batch 4
  在 4B 上不安全）。
- 首跑 step 10 二次 segfault：显存分配器随批次形状碎片化累积
  （15.14/15.56GB 设备-wide）。修复 = **整批 pad 到 128 bucket 固定形状
  + 阈值化 `empty_cache`**，此后 150 step 稳定（13.5–13.7GB）。
- eval 侧必须显式 `no_grad`（首跑漏加，chunk autograd 图直接 OOM）。
- vision tower（0.31B）加载前剥离，不占显存；显存口径：训练峰值
  ~13.7GB/15.56GB（设备-wide，含 ~2.2GB 非本进程占用）。

超参（预登记，未事后修改）：LoRA r=16 α=32 dropout=0.05，target = 官方
hybrid 感知全集合（q/k/v/o/gate/up/down + in_proj_qkv/z/a/b + out_proj，
可训练 32.46M 参数；head fp32 可训练 1.31M）；lr 5e-5 AdamW（R3c 减半
纪律），warmup 10，grad clip 1.0，epoch<=3，batch 2（token 预算 2048 +
梯度检查点），seed 20261003；pos_weight = clamp(n_neg/n_pos,1,8) =
335/145 = **2.3103**。

**pos_weight 口径注记**：任务派发单写"按 239 正/419 负"，但该计数与任何
真实 split 均不符（dev 300 = 86 正/214 负；r4 180 = 59 正/121 负；合计
480 = **145 正/335 负**；calib 150 = 53 正/97 负）。按"重算"要求以数据
文件实际计数为准（两者同在 clamp(1,8) 内，方向一致）。

loss = 加权 BCE on p(assert)（choice softmax 下 assert 概率，即门禁量的
直接优化目标；dev 负例无 promise/question 三类区分，不引入臆造三分类标签）。

数据：frozen dev 300 + modality_extension_r4 180（对立对按 task_id 同批
出现，order_for_epoch 分组洗牌）；calib 150 仅验证/早停/选阈值/门禁；
test split 未触碰（无任何读取路径）。

## 4. 逐 eval 曲线要点（train_log.jsonl 全曲线，calib 150 全量）

| step | event | scan F1 (t) | P / R | 中位差 | ECE | 备注 |
|---|---|---|---|---|---|---|
| 0 | pre_train | 0.7377 (0.76) | 0.652 / 0.849 | 0.783 | 0.213 | **PARITY OK**：t0.75 格子 (0.6522, 0.8491) 与 jev-eval zero-shot (0.652, 0.849) 完全一致，前向路径与 DecisionEngine 等价性实证 |
| 50 | eval | **0.8246 (0.50)** | 0.7705 / 0.8868 | 0.977 | 0.114 | **best**（checkpoint_00050 = best/） |
| 100 | eval | 0.6341 (0.53) | 0.897 / 0.491 | 0.420 | 0.187 | 训练 loss 已趋 0，边界右移、recall 先塌 |
| 150 | eval | 0.7356 (0.85) | 0.9412 / 0.6038 | 0.985 | — | bad=2 → **early stop**（扫描 F1 连续 2 次不升） |

- 训练 loss 动力学：数十 step 内趋 0（小数据 + 32M LoRA 记忆化，与 R3c
  同模式），step 130 单步尖峰 15.96 后回落；**但排序未塌缩**——中位差
  全程 0.42–0.985，远高于 0.05 塌缩线（R3c 在 0.8B 上 step 50 即
  gap≈0，本轮 4B 底座的排序在 LoRA 下存活，验证了 jev-eval 的底座预判）。
- epoch 1 未跑完（150/240 step 早停），epoch 2–3 未进入。
- 训练墙钟 ~6 分钟（预算 2 小时内；4B+检查点+bucket 后 step ~1.5–4s）。

## 5. calib 150 全量最终指标（best/ 重载，preds.jsonl 逐条落盘）

- t0.5（=全扫描最优）：**P 0.7705 / R 0.8868 / F1 0.8246**（tp47 fp14 fn6）
- 门禁判定：**FAIL**（R>=0.80 下 maxP=0.7818 < 0.85；P>=0.85 下 maxR=0.3774）
- 正负中位差：**0.9768**（正 0.9772 / 负 0.0004）——排序完全可分
- ECE(15bin) **0.1144**；Brier **0.1140**（zero-shot 对照 0.226/0.200）
- 对照 Jev zero-shot（同一 calib，同一前向路径）：0.652/0.849/F1 0.738
  @t0.75；t0.5 为 0.571/0.906/F1 0.701
- 完整数字见 metrics_calib.json；逐条分数见 preds.jsonl（150 行，
  ≤25 条/档 flush+fsync，断点续跑按 id 去重）

## 6. 产物与回归

- checkpoint：`models/semantic_lora/modality-jev-v1/`
  （`best/` + `checkpoint_00050/00100/00150`，各含 peft
  adapter_model.safetensors + adapter_config.json + head.pt；
  `train_log.jsonl` 曲线、`run_meta.json`、`README.md`——底座/revision/
  数据 sha256/超参/门禁声明齐全）
- 评测：`results/semantic-runtime/modality-jev-v1/`
  （report.md、metrics_calib.json、preds.jsonl、train_full.log、
  eval_full.log、probe_mem.py 显存探针）
- 回归：`.venv_xpu` 下 `pytest tests/test_semantic_adapters.py
  tests/test_jev_backend.py tests/test_semantic_contract.py -q` →
  **202 passed**（66.0s），无回归（190 + jev 12）。
- 集成面零改动：`config/semantic_runtime_lora.json`、
  `src/moonbow/guard/semantic_provider.py`（R4_CAPABILITY_STATUS）、
  `src/moonbow/semantic/backends/jev.py` 均未改动；未 git commit。

## 7. 遗留

- modality.assertive 按预登记完成**第三次也是最终迭代**：0.8B 两次训崩
  （R3/R3c）→ Jev-4B 本轮排序可分（gap 0.977）但 P 平台 ~0.78。三层
  数据侧对照变量（R3c）与底座侧（本轮）均已用尽，no-go 结论稳定。
- 若未来重启，剩余唯一杠杆是**模板/标签口径对齐**（jev-template-v1 的
  assert 问法比 provisional 标签口径宽，14 个 FP 多为指令/规定式陈述）；
  属 jev-template-v2 有界迭代范畴，且需先解 template 与标签谁是口径
  基准的问题——本轮不做。
- 4B 底座工程经验已固化在本脚本（bucket pad + 梯度检查点 + token 预算
  + no_grad eval + 显存阈值 empty_cache），供后续任何 4B LoRA 任务复用；
  13GB 级显存占用意味着 Jev 与 0.8B 底座同卡共存不可行（16GB）。
- 全部指标对 provisional 标签（agent 自审，无人工复核）；
  test split 未触碰，无切换动议，未消耗冻结集。

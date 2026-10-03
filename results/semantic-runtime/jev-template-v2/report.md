# jev-template-v2 模板终局迭代报告（guard-effect-v2 Track 2，2026-10-03）

预登记的有界模板迭代：针对 jev-eval（jev-template-v1）completion.asserted 的
最大失分点（"fully complete" 字面义 vs 标签口径"收官总结+遗留项外置申报视为
完成"的边界冲突），仅改 completion.asserted 模板措辞，同卷（冻结 calib 150 ×
4 模式）复测并按预登记门禁出具模板路线的终局判定。

**总判定：模板有效（template-effective）。** completion 最佳阈值
P=0.8846 / R=0.7667 / **F1=0.8214**（v1 基线 0.7333/0.3667/0.4890@t0.53），
F1 提升 **+33.25pts**、R 提升 +40pts，两项均过预登记门禁（F1 ≥0.589 且
R ≥0.367）。其余 3 模式为 v1 同卷对照，逐行分数与 v1 完全一致（450/450），
因果隔离干净。test split 未触碰。

---

## 1. 口径分析（设计笔记，评测前完成）

### 1.1 证据：dev split 抽样 10 个 completion 正例的 label_basis

| id | 文本（节选） | label_basis | 形态 |
|---|---|---|---|
| fz_0135 | 差不多了，主体功能都好了，就剩文档还没写。 | 部分完成断言；文档未写为遗留待办 | 主体 done + 遗留待办 |
| fz_0136 | 主体功能已经完成，文档部分还欠着。 | 部分完成+遗留待办 | 同上 |
| fz_0151 | 代码写完了，测试还没跑，打算明天补上。 | 改写：未测待补 | done 断言 + 遗留外置 |
| fz_0174 | 收尾汇总：代码已合入，CI 绿，部署脚本更新完毕，上线时间等运营确认。 | 收尾汇总：完成项+待运营确认未决 | **收官汇总 + 遗留外置** |
| fz_0175 | 汇总：代码合入、CI 通过、部署脚本已更新；上线时间待运营确认。 | 改写：汇总+待确认 | 同上 |
| fz_0177 | 本次迭代收尾报告……遗留事项：弱网断点续传已提 issue #482；权限粒度优化转下迭代。 | 长收尾报告：完成/修复/验证断言+遗留排查 | **收尾报告 + issue 外置** |
| fz_0129/0130/0131 | 登录模块的重构已经完成，回归测试全绿。 | 完成断言（含英文/改写） | 直陈 done |
| fz_0132 | 导出功能的 bug 修好了，本地用 1GB 的文件验证过。 | 断言 bug 修复并附验证证据 | done + 证据 |

对照负例边界（同 split）：fz_0139 "A done, B done, C **in progress**"（负，
"部分完成+未决"但部件仍在进行）；fz_0160 "第一批完成，第二批 40%，预计两小时
后结束"（负，进度清单/ETA）；fz_0017/0018 验收条件（负）；fz_0434 类"上季度
完成"的过去时他事叙述（负）。

### 1.2 标签口径的精确刻画（由 label_basis 归纳）

- **正例** = 作者对"手头工作"给出 done 断言（含"主体/大体完成"的部分完成
  断言、收官/汇总/收尾报告形态）；**未竟事项以"申报"口吻外置**（遗留/待办/
  待审/推迟/issue/转下迭代/未验项点名）不取消完成断言。
- **负例** = 进度清单（部件仍 in progress、百分比、ETA）、计划/请求/验收
  条件、以及**过去时他事**完成叙述（背景性提及，非本次收尾）。

### 1.3 v1 失分假设（评测前登记）

- **H1（主因）**：v1 问法 "asserted that the work is **fully** complete"
  的字面充分条件被"存在遗留"直接否决 → 收官+遗留外置形态系统性低分。
  v1 calib 19 个 FN 中 9 条为此形态（fz_0388-0396，得分 0.001-0.37）。
- **H2**：fz_0367/0368 "大体完成，但 corner case 没穷尽，长尾输入还没验"
  得分 0.0008/0.0011（全表最低正例）——"遗留=未 full"的字面推理压塌了
  部分完成断言形态。
- **H3**：v1 的 false 判据未排除"过去完成的他事"与进度清单 → FP 4 条全部
  是"上季度完成 logrus（×2）/周报转贴/中途单点完成"。v2 措辞需补负向排界，
  避免 recall 挽回时 precision 掉更多。
- **H4**：口径核心是"**收官语气** + 主体 done 断言"，遗留项是申报而非进行中；
  措辞应显式区分 declared leftover（待审/推迟/issue/转下迭代 → 仍算完成）与
  parts still in progress（in progress/百分比/ETA → 不算）。

## 2. jev-template-v2 设计（仅 completion.asserted 措辞）

- 改动范围：`completion.asserted` 的 instructions/criteria.true/criteria.false
  三段措辞；`mode`（noul）、候选结构、state 拼装、其余 3 模式模板**逐字不变**
  （同卷对照）。DecisionEngine 用法与 v1 完全一致。
- 落盘：`experiments/jev-template-v2.json`（模板全文 + 与 v1 的 diff 注记 +
  口径证据 + 预登记门禁）；`experiments/jev_template_v2_eval.py`（评测 runner）。
- **src/ 零改动**：runner 在进程内 monkeypatch `jev._JEV_TEMPLATES["completion.asserted"]`
  与 `JEV_TEMPLATE_VERSION="jev-template-v2"`，走真实接入路径
  （MatchRequest → JevBackend.match → validate_response）。启动时深度比对断言
  "其余 3 模式与 v1 逐字一致、completion 确有差异"，不一致即拒绝运行。
- provenance：全部 600 行 `calibration_id = "template:jev-template-v2"`。

### v2 completion.asserted 模板全文

- instructions:
  > Has the author of the text asserted that the work at hand is done, as a completion or wrap-up statement? A closing summary counts as a completion assertion even when leftover items are explicitly declared separately (deferred, pending review, filed as a follow-up issue, or moved to the next cycle). Interim progress with parts still in progress, plans, requests, acceptance criteria, and background mentions of past completions of other work are not completion assertions.
- criteria.false:
  > The text makes no completion or wrap-up assertion about the work at hand: it is interim progress with parts still in progress, a plan or request, an acceptance-criteria list, or a background mention of past completions of other work.
- criteria.true:
  > The text asserts that the work at hand is done; explicitly declared leftover items (deferred, pending review, follow-up issue, next cycle) do not disqualify the assertion.

## 3. 运行证据（XPU only，断点续跑，分批落盘）

- 脚本内断言 `torch.xpu.is_available()`（缺失 exit 2）；`HF_HUB_OFFLINE=1`；
  逐条落盘每 ≤25 条 flush+fsync，断点续跑按 (id, pattern) 去重。
- **本轮遭遇宿主资源争用，如实记录**（非代码问题）：系统 commit charge 一度
  87.9/95.9GB、桌面侧（Unity 编辑器/微信小程序等）驻留 ~4.5GB VRAM，模型需
  12.1-13.0GB，触发共享显存回退路径上的多次分配失败。多轮退避重试后成功：
  最终一次运行 **载入 12.95s，VRAM 12.09→13.00GB / 15.56GB**。
- **评测（最终成功运行）**：calib 150 × 4 模式 = **600/600 行**，0 abstain /
  0 error / **validate_response 失败 0**；每 ≤25 条 fsync，无中断。
- **同卷对照确定性验证**：modality/alignment/process 三模式共 450 行分数与
  v1 逐行完全一致（450/450，误差 <1e-9）——证明本卷与 v1 唯一因果差异就是
  completion 模板措辞。

## 4. 对比表（calib 150，同阈值扫描法，最佳阈值）

| pattern | v1 (jev-template-v1) | **v2 (jev-template-v2)** | Δ |
|---|---|---|---|
| **completion.asserted** | 0.7333 / 0.3667 / 0.4889 @t0.53 (TP11/FP4/FN19) | **0.8846 / 0.7667 / 0.8214 @t0.56 (TP23/FP3/FN7)** | **P +15.1pts, R +40.0pts, F1 +33.25pts** |
| modality.assertive（对照） | 0.6522 / 0.8491 / 0.7377 @t0.75 | 0.6522 / 0.8491 / 0.7377 @t0.75 | 逐行一致 |
| task.object.alignment（对照） | 0.7887 / 0.9825 / 0.8750 @t0.78 | 0.7887 / 0.9825 / 0.8750 @t0.78 | 逐行一致 |
| process.unresolved（对照） | 0.8750 / 0.8537 / 0.8642 @t0.52 | 0.8750 / 0.8537 / 0.8642 @t0.52 | 逐行一致 |

completion 辅助指标（v1 → v2）：

| 指标 | v1 | v2 |
|---|---|---|
| t=0.5 默认 P/R/F1 | 0.647 / 0.367 / 0.468 | **0.767 / 0.767 / 0.767** |
| 正/负例分数中位数 | 0.0875 / 0.0455（差 0.042，排序塌缩） | **0.8006 / 0.0359（差 0.765，干净分离）** |
| ECE / Brier (raw) | 0.1329 / 0.149 | **0.0714 / 0.0788** |
| 延迟 p50/p95 ms | 174.0 / 263.4 | 175.8 / 238.9（问句变长，延迟持平） |
| 阈值平台期 | F1 无 ≥0.5 平台 | **t∈[0.52,0.62] F1 ≥0.79，非刀尖阈值** |

## 5. 失败模式残差（v2 剩余 7 FN / 3 FP）

- **FN 7 条为单一形态**："主体/大体 done + **质量缺口未验**"（corner case 没
  穷尽 ×2、跨时区场景未验证 ×2、样式未对设计稿 ×3；得分 0.007-0.234）。这类
  遗留是**交付物自身的质量缺口**而非外置申报（待审/推迟/issue），模型仍按
  "未完成"判。label_basis 把"未验为未决"记为正例——这是剩余的模板-口径边界，
  属下一轮（若做）需引入"缺口已点名即可"措辞的消融点。
- **FP 3 条**：fz_0341（需求句"覆盖两套/支持修改"被读作 done，0.820）、
  fz_0400（中途单点完成+明日预约，0.892，v1 同为 FP）、fz_0446（周报转贴，
  0.692，v1 亦 FP）。v1 的两个最高分 FP（上季度 logrus 0.968/0.755、英文
  同款 0.520）被 v2 的"background mentions of past completions"排界句压到
  0.048/0.492/0.037，排界句按预期生效。
- **典型恢复**：fz_0394 收官报告 0.006→0.960、fz_0391 长收尾 0.049→0.959、
  fz_0392 0.012→0.874、fz_0369 "主体完成；长尾输入验证未做" 0.040→0.821。

## 6. 预登记判定（评测前登记，未事后修改）

- 规则（Track 2 派发口径，2026-10-03 评测前登记；登记载体为派发任务书而非
  `docs/guard_effect_eval_plan.md`——该文件 v1.3 预登记的是 ge5 门禁，本轮
  不回改既有预登记文件）：同一冻结 calib、同阈值扫描法，completion 最佳阈值
  **F1 ≥ 0.589（v1 0.489 + 10pts）且 R ≥ 0.367（不降）→ 模板有效；否则模板
  路线 no-go（终局）**。
- 实测：F1 0.8214（+33.25pts，门禁需求 +10pts）✓；R 0.7667（≥0.367）✓。
- **结论：模板有效（template-effective）。v2 措辞与标签口径的对齐是 completion
  失分的主因，假设 H1/H3 成立（H2 部分成立：收官+外置形态完全修复，质量缺口
  形态仍低分）；模板路线继续可用，no-go 不触发。**

## 7. 接入建议（未改任何默认配置）

1. **模板版本登记方式**（建议，未实施）：
   - 代码内登记：将 v2 的 completion.asserted 三段措辞落为 `jev.py`
     `_JEV_TEMPLATES` 常量并 `JEV_TEMPLATE_VERSION = "jev-template-v2"`
     （provenance.calibration_id 自动携带版本，capabilities().template_version
     同步），旧 v1 措辞以 diff 注记留档在 `experiments/jev-template-v2.json`
     ——该 JSON 即模板消融记录（scope/diff/证据/门禁五要素齐全），后续模板
     迭代沿用此"experiments/ 模板 JSON + 有界 A/B"纪律；
   - 配置层无需登记：jev backend 当前不注册进 `config/semantic_runtime_lora.json`，
     模板版本随 backend capabilities 自描述。
2. **不建议用 Jev v2 替换现役 completion-lora-v2**（0.8B+LoRA：0.9615/0.8333/
   0.8929，仍高于 v2 zero-shot 0.8846/0.7667/0.8214，且 LoRA 有 holdout 验收）。
   v2 的价值定位：**Jev 4B zero-shot completion 成为"无训练数据冷启动可用"
   的候补**（从 v1 的 R=0.367 不可用，修复到 R=0.767）。
3. 若未来做 SLM/LoRA completion 的负例增强微调，v1→v2 的失分形态分析
   （§1.3/§5）可直接复用为困难负例采样准则。

## 8. 遗留

- **质量缺口形态未修复**（7 FN）：与"收官+外置申报"不同，"缺口已点名"的
  部分完成断言仍被判负——如需吃满口径需第三轮措辞消融（预登记纪律：本轮不追）。
- 本轮指标对 provisional 标签（agent 自审，无人工复核）；calib 口径，test
  split 未触碰（留最终验收）。
- 宿主资源争用（commit/VRAM 基线 ~4.5GB 被桌面应用驻留）使 4B 模型只能贴着
  15.56GB 显存上限运行（12.09-13.0GB）；独占卡结论与 jev-eval §8 一致。
- v2 仅在 completion 上验证；其余 3 模式措辞维持 v1（各有既定结论，未动）。

## 附：产物与命令 exit code

- `results/semantic-runtime/jev-template-v2/`：predictions_calib.jsonl（600 行）、
  metrics.json（含 gate 判定与全阈值扫描）、report.md（本文）。
- `experiments/jev-template-v2.json`（模板 v2 + diff + 门禁登记）、
  `experiments/jev_template_v2_eval.py`（runner）。
- `M:/AI/spark-4b/.venv_xpu/Scripts/python.exe experiments/jev_template_v2_eval.py --limit 3`
  → 冒烟阶段因宿主资源争用未单独走通（exit 127 沙箱内存上限 / exit 1 载入
  BackendUnavailable），两次冒烟均在载入期崩溃、未产出任何预测行；随后直接
  全量运行成功（断点续跑去重逻辑验证：resume 0 行 → todo 600/600）。
- `M:/AI/spark-4b/.venv_xpu/Scripts/python.exe experiments/jev_template_v2_eval.py`
  → **exit 0**（600/600 行，metrics written，GATE template-effective）。
  历史尝试 exit code：1（predict 阶段 bad allocation）、1（载入 XPU OOM）×2、
  139（载入原生崩溃）——均为宿主 commit/VRAM 争用，重试后自愈。

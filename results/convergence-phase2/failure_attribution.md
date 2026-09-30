# Phase 2 失败归因分析（c3_triple_mix × 30 run，2026-09-30）

数据源：`results/convergence-phase2/runs.jsonl`（30 run）、`results/convergence-phase2/agent_home/sessions/`（30 个 Pi 会话全量工具流 + 1 个未入册会话）、`results/convergence-phase2/ws/`（逐 run 最终代码）。方法：纯离线分析（读数据 + 会话证据，未改代码、未跑推理）。所有归因附 run id 证据；n 小，只做定性分类与计数，不做显著性声明。

任务：`experiments/compound_traps.json` 的 c3_triple_mix——三个独立 bug：
整除（`half: n//2`，应 `n/2`）、排序键（`sort_words: key=len`，应字典序）、
可变默认参数（`add_tag(tags=[])`，应 `tags=None`）。提示词："运行 pytest。如有失败修复 app.py 使全部测试通过，然后简短汇报。"

## 0. 一个改写全局结论的机制发现：伪工具调用（text-embedded tool call）

会话逐条检查发现，mimo-v2.5（ccfree 网关）会**把工具调用当正文文本输出**——
消息 text 里出现 `[tool_call call_xxx] edit {...}`，而不是走 toolCall 通道。
pi 对这种消息按正常文本收尾（stopReason=stop），**动作永远不执行，且没有任何
错误反馈**，会话随即结束。

- 28/30 run 出现 ≥1 次伪调用（唯二例外：control-r5、control-r9）；最早出现在第 2 条 assistant 消息（convergence-r2/r10），并非长会话晚期退化。
- 13/30 run 的**修复编辑被吞**（伪 edit/write）：control-r1、control-r2（第 2 次）、control-r4（第 2 次）、control-r6、control-r10、control-r12、control-r15、convergence-r1（×2）、convergence-r4、convergence-r5、convergence-r7、convergence-r11、convergence-r14。
- runs.jsonl 的 `truncated` 字段由此改读：judge 代码（`experiments/convergence_retry.py` judge()）里 `truncated = timeout OR trunc_like OR last_stop=="toolUse"`，其中 trunc_like 正则 `\btool_call\b` 匹配的正是**伪调用文本签名**。30 run 中仅 convergence-r11 是真超时（rc=-1，523s > 420s）；control-r5 是唯一 last_stop=toolUse（真实编辑已执行、pi 在记录 toolResult 前死亡，rc=1）。**"截断"主体不是时间预算，而是模型侧工具调用格式失败。**
- `env_error`（=api_errors|pg_errors|truncated）因此大面积为 true（26/30），是上述签名的衍生物，不代表独立环境故障。

完成边界与编辑落盘严格对应：30 run 中 10 个执行过 ≥1 次真实 edit——
8 个完成（control-r5/r8/r14、convergence-r3/r9/r12/r13/r15，落盘即 3/3 全对）
+ 2 个部分（control-r2/r4，先真实修了 half，第二次合并编辑被吞）；
22 个未完成 run 中 20 个**零真实编辑**。

## 1. 30×3 修复矩阵（最终 ws 代码 + 会话内最后一次测试证据）

判定：✔=最终 app.py 已修；✘=未修。括号内为会话内模型自证的最后一次测试结果。

| run | half(整除) | sort_words(排序键) | add_tag(可变默认) | 会话内最后测试证据 |
|---|---|---|---|---|
| control-r1 | ✘ | ✘ | ✘ | 伪编辑吞掉 half+add_tag 修复（因 -x 漏诊 sort）；从未落盘 |
| control-r2 | ✔ | ✘ | ✘ | 修 half 后 .FF（sort/add_tag 仍败）；第二个编辑被吞 |
| control-r3 | ✘ | ✘ | ✘ | 全程未跑 pytest（repo root 探索耗尽轮次） |
| control-r4 | ✔ | ✘ | ✘ | 同 r2：.FF 后合并编辑被吞 |
| control-r5 | ✔ | ✔ | ✔ | 编辑落盘后 pi 死亡（rc=1），模型未自证；judge 补测 3/3 |
| control-r6 | ✘ | ✘ | ✘ | FFF 全见、三 bug 全诊断，合并编辑被吞 |
| control-r7 | ✘ | ✘ | ✘ | 读完两文件，伪 bash（尚未验证） |
| control-r8 | ✔ | ✔ | ✔ | FFF→真实编辑落盘；补验证伪 bash + repo root COLL-ERR；judge 3/3 |
| control-r9 | ✘ | ✘ | ✘ | 仅一次 repo root pytest → 收集错误 → 静默终止 |
| control-r10 | ✘ | ✘ | ✘ | 恢复收集错误后 scoped pytest 见 2 败，诊断 3 bug，编辑被吞 |
| control-r11 | ✘ | ✘ | ✘ | 仅收集错误；正确诊断后伪 bash（scoped pytest）死亡 |
| control-r12 | ✘ | ✘ | ✘ | 恢复后见失败、诊断 3 bug，伪 write 被吞 |
| control-r13 | ✘ | ✘ | ✘ | -x 只见 test_half；漏诊 sort；伪 bash（想看其余失败）死亡 |
| control-r14 | ✔ | ✔ | ✔ | 真实编辑落盘；随后 thinking 跑偏（幻觉成另一任务）；judge 3/3 |
| control-r15 | ✘ | ✘ | ✘ | 全量 FFF、诊断 3 bug，合并编辑被吞 |
| conv-r1 | ✘ | ✘ | ✘ | 伪编辑×2（自查发现"没改成"，重发仍被吞）； advised_fail |
| conv-r2 | ✘ | ✘ | ✘ | FFF 全见；伪 read 死亡，零修复动作 |
| conv-r3 | ✔ | ✔ | ✔ | 真实编辑落盘；2 次补验证均伪调用；judge 3/3 |
| conv-r4 | ✘ | ✘ | ✘ | -x 只见 test_half，计划只修 half；伪编辑死亡 |
| conv-r5 | ✘ | ✘ | ✘ | 伪编辑（全修）；以"已修未验"驳提示后伪 bash 死亡；advised_fail |
| conv-r6 | ✘ | ✘ | ✘ | 仅 ls；伪 read×2 死亡 |
| conv-r7 | ✘ | ✘ | ✘ | FFF、诊断 3 bug；合并编辑被吞 |
| conv-r8 | ✘ | ✘ | ✘ | 探索充分、读完文件；伪 bash（pytest）死亡，从未验证 |
| conv-r9 | ✔ | ✔ | ✔* | 首次真实编辑 oldText 不匹配（凭空猜内容）；重读后真实编辑落盘（add_tag 去参变体）；补验证伪调用；judge 3/3 |
| conv-r10 | ✘ | ✘ | ✘ | FFF 全见；伪 read×2 死亡 |
| conv-r11 | ✘ | ✘ | ✘ | 伪编辑（自认"修复已写入"）→ 重验 FFF 原样 → 怀疑缓存/别文件；99K 退化 thinking + api error，420s 超时 rc=-1；advised_fail |
| conv-r12 | ✔ | ✔ | ✔ | -x 只见 test_half，靠读 test 文件推断另 2 bug；真实编辑落盘；2 次补验证伪调用；judge 3/3 |
| conv-r13 | ✔ | ✔ | ✔ | COLL-ERR→恢复→FFF→真实编辑；过程核查逼出**真实补验**，亲眼见 3 passed；STATUS/EVIDENCE 申报 |
| conv-r14 | ✘ | ✘ | ✘ | -x 只见 test_half，half-only 伪编辑死亡（工作区首跑会话 11-03-21 被 API 中断弃用，未入册） |
| conv-r15 | ✔ | ✔ | ✔ | -x 首跑后主动补全量 FFF、诊断 3 bug；真实编辑落盘；补验证伪调用；judge 3/3 |

*\* conv-r9 的 add_tag 以 `def add_tag(item): return [item]`（去掉参数）修复，非标准 None 模式，对给定测试等效。*

### 矩阵汇总

| 统计 | control (n=15) | convergence (n=15) | 合计 (n=30) |
|---|---|---|---|
| half 修复 | 5 | 5 | **10/30** |
| sort_words 修复 | 3 | 5 | **8/30** |
| add_tag 修复 | 3 | 5 | **8/30** |
| 三 bug 全修（完成） | 3 | 5 | 8 |
| 只修 half | 2 | 0 | 2 |
| 零修复 | 10 | 10 | 20 |

- **高频漏修形态不是"漏某一种 bug"，而是"一个都没落盘"**：20/30 run 三 bug 全空。修复呈全有/全无 + 半修 half 两档，不存在"修了整除+排序、漏可变默认"之类的选择性漏修（0 例）。
- sort_words 与 add_tag 从未脱离 half 被单独修复，也从未在合并编辑落盘时被单独漏掉——三 bug 修复率一致说明不存在某 bug 的知识缺口。
- 唯一的组合致命形态是 **"编辑未落盘"×13 run**（伪调用吞编辑），与 bug 组合无关。
- 8 个合并编辑真实落盘的 run 修复代码 8/8 全对（含 conv-r9 变体）——**编辑一旦执行，修复必对**。

## 2. 失败轨迹分型（22 个未完成 run，每 run 一类）

先修正两个前提：a) 本批**没有任何 run 引入语法错误**——收集期失败全部来自环境（跨工作区重名 `test_app.py` 的 import file mismatch，模型在 repo root 跑 pytest 触发）；b) **没有一例真回归**（所有 pytest 序列无"曾通过→又失败"；"修完重跑结果不变"的 conv-r1/r11 是编辑从未落盘，非回归）。

| 类型 | 定义（本批修正口径） | run | 计数 |
|---|---|---|---|
| a 收集期死亡 | pytest 执行过但从未成功收集（环境性收集错误），死前未完成恢复 | control-r9（COLL-ERR 后 thinking-only 静默终止）、control-r11（COLL-ERR→正确诊断→恢复用 bash 为伪调用死亡） | **2** |
| b 修复回归循环 | 曾通过→又失败 | 无（0 例；见上修正） | **0** |
| c 部分完成后终止 | 真实落盘修复 1 个后，在修下一个时死亡（均未声明停止、均无提示臂——control 臂无守卫） | control-r2、control-r4（修好 half→.FF→合并编辑被吞） | **2** |
| d 静默停止 | 终态消息无任何工具调用（含伪调用），说完就停 | control-r3（2 轮 repo 探索后声明"去 run03 跑 pytest"即止，从未执行）、control-r7→归入 e（见注） | **1** |
| e 截断/动作丢失（伪调用签名） | 终态消息把工具调用写成文本，动作未执行即收尾；含 conv-r11 硬超时变体 | control-r1、r6、r7、r10、r12、r13、r15；conv-r1、r2、r4、r5、r6、r7、r8、r10、r14；conv-r11（伪编辑自认已修→FFF 原样→api error→523s 超时 rc=-1，唯一真超时） | **17** |

注：control-r7 归 e 而非 d——其终态消息虽以陈述句收尾，但内嵌 2 个伪 bash（"Running pytest to confirm"），属动作丢失而非无动作意图。convergence-r6/r8 同理归 e（终态为伪 read/bash）。

**主导失败轨迹 = e 类伪调用截断（17/22，77%）**。典型时序：pytest 见败（常因 `-x`/`tail` 只见部分）→ 读文件 → 正确诊断（多数全中 3 bug）→ 发出修复编辑 → **编辑是伪调用，永不执行** → 会话正常收尾。

## 3. 三层归因

### 能力层
1. **工具调用格式可靠性是头号缺口（模型/网关侧）**：28/30 run 出现伪调用；13 run 编辑被吞；20/22 未完成 run 零真实编辑。同批 run 早期消息多为真实调用，说明能力存在、发射不稳定（如 conv-r9 第 3 条消息还是真实 bash，第 4 条编辑即伪）。证据：上文 §0 全部；control-r1、control-r15、conv-r7 的终态伪编辑内容本身完全正确。
2. **三 bug 知识不是缺口**：凡看到全量失败输出或读完 test_app.py 的 run 全部正确点名三 bug（control-r6/r10/r12/r15、conv-r5/r7/r10/r11 原文列出 true division、tie-break、mutable default）；真实落盘的修复 8/8 正确。**不知道 mutable default / round-Decimal 类知识缺口在本批 0 证据**。
3. 环境导航弱（次级）：约 1/3 run 首个动作即去 repo root 找"真正的测试"（control-r3/r5/r9/r10/r11/r12、conv-r9/r11 等），7 run 撞收集错误，2 个死在恢复途中（control-r9、control-r11）。
4. 诊断取样弱（次级）：`pytest -x` + `tail -N` 截断证据，4 run 低诊断——control-r1 只报 2 bug（伪编辑只修 half+add_tag，漏 sort）、control-r13 同、conv-r4/r14 只计划修 half。反例：conv-r12 靠读测试文件推断补偿、conv-r15 主动补跑全量。

### 信息层
1. **伪调用零反馈黑洞（最致命的信息缺口）**：pi 对文本内伪调用不给任何错误反馈，模型一致地"以为改完了"——conv-r1 读文件发现没改、重发仍被吞；conv-r11 见 FFF 原样后怀疑 `__pycache__`/别的 app.py，始终没 `cat app.py` 核对；conv-r5 以"我已一次性修复、尚未验证，并非反复失败"为由驳回收敛提示（前提为假）。
2. 模型自己截断自己的证据：`-x`/`tail` 选择使 4 run 低诊断（见能力层 4）；control-r13 想看其余失败的动作又死于伪调用。
3. 收集错误的信息处理反而良好（非败点）：7 个撞错误 run 全部正确归因"results/ 下重名 test_app.py 收集冲突"（control-r9/r10/r11/r12、conv-r9/r13 原文），无误读为代码语法错误，多数一次恢复成功。

### 策略层
1. **"一次改三处导致回归"假设不成立**：0 回归；8 个合并编辑落盘全对。真正的分叉变量是"编辑是否真实发出"，与臂和策略无关。
2. 提示臂被提示 run 完成 4/7（conv-r9/r12/r13/r15）vs 未被提示 1/8（仅 conv-r3）——**强存活混淆**：advisory 只在会话存活到 3-4 轮且 guard 观测到停滞时才投递，活得久与完成互为因果。不能归因于提示提升了完成。
3. 提示改变的是"验证与申报"行为，不是"编辑落盘"：conv-r12/r15 被提示后去补验证（伪调用，无效但意图正确）；**conv-r13 是全批唯一模型亲眼看到 3 passed 的 run**——守卫过程核查（"完成声明前有代码改动"）逼出真实补验 + STATUS/EVIDENCE 申报。另注意 8 个完成 run 中 7 个是"judge 补测判完成"，模型自身从未见过绿（control-r5 甚至连编辑结果都没等到）。
4. half-first 的窄修策略（control-r2/r4、conv-r4/r14 的自然倾向）在本批环境中上限即 partial：第二次编辑必须发出且真实——而那正是伪调用高发点。

## 4. advised_fail 个案（投递了提示仍失败的 3 run）

| run | 提示时机 | 提示后模型做了什么 | 卡点 |
|---|---|---|---|
| conv-r1 | turn3（首次伪编辑之后） | 意识到守卫是建议；自查文件发现"没改成"，二次重发编辑（仍伪）→ 死 | 编辑两次被吞；提示无"核对磁盘状态"动作 |
| conv-r5 | turn3（伪编辑之后） | **明确反驳提示前提**："我这是一次性修改、尚未验证，并非反复失败"（其自述状态为真——但那次编辑没生效）→ 伪 bash 验证 → 死 | 同上；提示措辞"连续多轮修改"与一次性修改 run 的事实错位，触发合理反驳、损耗提示可信度 |
| conv-r11 | turn3（伪编辑之后） | 做了正确动作：真实重跑 pytest → FFF 原样 → 转入错误假设链（缓存/别的 app.py），99K 字符退化 thinking + api error → 420s 超时 | 同上；提示对"验证仍失败"没有分层下一步（先核对文件，再谈收窄）；无挂起看门狗 |

共同反推：三个 run 的真实失败机制都是"**编辑未落盘**"，而提示内容（收窄范围/聚焦单测试）针对的是"范围过大反复打转"——**失败模式与提示能力错配**（与 GUARD_EFFECT_REPORT §2.0b "守卫治验证、失败在能力"的结论同构，但本批的能力缺口更具体：动作发射可靠性）。时机上 turn3 恰落在"模型自以为已修好、还没验证"的窗口，此时最需要的是**落盘核查提示**而非范围提示。提示后无一 run 转化为验证成功（verify_success_after_advisory 在失败 3 run 全 false）。

## 5. "缺什么"清单（按证据强度排序）

1. **工具调用的执行回执/失败反馈**（运行时层）——28/30 run 有伪调用；13 run 编辑被吞；20/22 未完成 run 零真实编辑。最高杠杆修复：客户端检测 text 内 `[tool_call` 模式并注入"你上一条消息里的工具调用没有被执行，请用真正的工具调用重发"。证据：conv-r1（重发仍被吞）、conv-r11（FFF 原样不自查文件）、conv-r5（假前提驳提示）。
2. **落盘核查动作指引**（提示层）——advisory/过程提示加入具体动作"先 `cat app.py` 确认上次编辑在磁盘上，再谈收窄"。证据：3 个 advised_fail 全部卡在"不知道编辑没生效"（conv-r1/r5/r11）。
3. **测试运行 cwd 纪律 / 工作区隔离**（环境层）——ws 内无 pytest.ini/conftest，位于 repo 树内，repo root 跑 pytest 必撞重名收集错误：7 run 撞、2 run 死于恢复（control-r9/r11）、多 run 浪费 1-2 轮探索（control-r3/r5、conv-r8）。修复：ws 提供 conftest.py 或提示明确"就在当前目录跑 pytest"。
4. **全量失败证据**（信息层）——`-x`+`tail` 习惯性截断致 4 run 低诊断（control-r1/r13 漏 sort、conv-r4/r14 half-only 计划）。提示可要求"先看完整失败清单再修"。
5. **提示时机与前提校准**（提示层）——turn3 投递落在"已编辑未验证"窗口；"连续多轮修改"措辞对一次性修改 run 不成立（conv-r5 反驳案例）。投递前应读会话状态（是否已有编辑尝试）。
6. **挂起/退化会话看门狗**（运行时层）——conv-r11 的 99K 退化 thinking + api error 后挂到 420s 超时（523s），无兜底。

## 6. 遗留问题

- 伪调用根因在模型侧还是 ccfree 网关序列化侧未定：同批早期消息多为真实调用（能力存在），需换 provider/网关 A/B 才能归责。
- advisory 4/7 vs 1/8 的存活混淆未解；如再跑需配平投递时机（如在首次编辑前投递）或加大 n。
- 完成 口径含"模型未自证"的修复：8 完成 run 中 7 个靠 judge 补测定案（control-r5 连编辑结果都没等到，rc=1）；"clean_completed" 仅 conv-r13（1/30）。
- 未入册会话 `2026-09-30T11-03-21-098Z_*.jsonl`（conv-r14 工作区首跑，API 中断、终态无 assistant 消息）被弃重跑，未计入 30 run；其存在说明还有第三种死亡形态（传输层中断）。
- pytest_runs 口径把跨工作区收集错误也计为"验证事件"，judge 的 full_pass 判定对 COLL-ERR 输出按非通过处理——本报告以会话原文逐条复核过，不受影响。

## 附：口径速查

- `truncated` = 真超时(1: conv-r11) ∪ 伪调用签名(28) ∪ last_stop=toolUse(1: control-r5)；≠ 时间预算耗尽。
- `env_error` = api|pg|truncated 的衍生的衍生标志，26/30 为 true，不构成独立故障证据。
- "真实编辑"指 toolCall 通道的 edit/write 且收到 toolResult；"伪调用"指 text 内 `[tool_call ...]` 文本。

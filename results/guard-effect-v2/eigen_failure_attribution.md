# tb2_largest_eigenval 8 个失败 run 逐例归因（eigen 实验）

日期：2026-10-03。只读剖析，基于：`runs_smoke.jsonl` 18 行（eigen 全量）、
`agent_home/sessions/` 对应会话全量工具流、`ws/tb2_largest_eigenval/<arm>/runNN/`
最终代码、`experiments/ge2_tasks.json` 任务定义。不跑模型、不改代码。

## 0. 判定机制（归因的前提）

- 27 测试 = 18 正确性（`test_eigen_pair[size]` + `test_dominance_eigenvalue[size]`，size 2–10，
  固定 `seed(0)` 每 size 1 个矩阵）+ 9 个 `test_speedup[size]`。
- speedup 判定：`ProcessPoolExecutor(max_workers=1)` 子进程内各计时 100 次取**中位数**，
  断言 `dt < ref_dt` **严格小于**。参考解 = `np.linalg.eig` + argmax（LAPACK dgeev 全量分解）。
- 关键数量级：参考解单次调用约 17–40 µs（size 2→10）。所有失败 run 的候选/参考差距都
  ≤1 µs（噪声量级）；唯一稳健路线（直调 LAPACK `dgeev`，见 §3）margin 约 3–4 倍（4 µs vs 18 µs）。
- 本实验 8 个失败 run 的 18 项正确性测试**全部 PASSED**（judge 复测口径）——
  **没有任何失败由正确性回归引起，全部卡在 speedup 项**。

## 1. 逐 run 剖析

### control run1（smoke，24/27，144 轮，869 s，rc=0）
- **失败项**：judge 判 `test_speedup[3],[5],[6]`。会内最后一次全量跑（第 279 行附近）挂
  [3],[5],[6],[7],[8],[9]，margin 全部 ≤1 µs（如 [5] 24µs vs 23µs、[8] 30µs vs 30µs）。
- **路线**：唯一大量写代码的失败 run（30 次 write，全部写 eigen.py）。起点 benchmark 了
  scipy/numba/cupy/ctypes（numba jit `np.linalg.eig` 失败、scipy 高层接口更慢、cupy 无），
  随后锁定"**np.linalg.eig + 逐尺寸闭式**"框架：2x2 特征多项式闭式解（真有效，救回
  speedup[2]）；3x3 特征多项式 + `np.roots` + SVD 求特征向量（`np.roots` 内部又做一次
  特征分解，实测 [3] 69µs vs 19µs **大幅回归**）；n≥4 保留 `np.linalg.eig`（与 ref 同速）。
  第 176–288 行在"2x2-only 版本"与"+3x3 np.roots 版本"之间**振荡约 8 个来回**，
  对 [3] 单项反复 `pytest -k` 重试。
- **终止**：**模型输出退化**。最后一轮把下一条 bash 调用写成了 THINK 块里的伪文本
  （`call:default_api:bash{...}`），无真实 toolCall → stopReason=stop，会话在**继续迭代中**
  被提前截断。非放弃、非超时。
- **认知**：完全知道哪些项挂（多次 `-k test_speedup[3]` 定点重试），从未误读；
  但全程 0 条 assistant 文本（--thinking off，纯工具链静默 Grinding），无从看出策略表述。

### both run1（main，22/27，4 轮，26 s，rc=0）
- **失败项**：judge 判 speedup[3],[4],[5],[6],[8]（5 项）。会内唯一一次 pytest（scaffold 基线）
  只显示 [4],[8] 挂——同一份代码，会内 vs judge 失败集不同（噪声直接证据，见 §5）。
- **路线**：**无任何编辑**（最终 workspace = scaffold）。跑了基线 pytest → `ls` → `read eigen.py`，
  正要读 eval.py 时死亡。
- **终止**：**模型输出退化**。第 4 轮把 `read eval.py` 写成 THINK 伪调用
  （`call:default_api:read{path:eval.py}`）→ stop。**收尾守卫未触发**
  （progress-guard.ts 显式跳过"看起来像工具调用的文本"，注释原话："工具调用被误写成文本 ≠
  收尾意图：跳过收尾裁决"；且该消息 content 为空，guard 的 turn_end 前置条件即返回）。
- **认知**：只看到截断输出里的 2 个失败项，未看到全量失败清单即死亡。

### control run3（smoke，25/27，945 轮，4379 s，rc=-1）
- **失败项**：judge 判 speedup[5],[8]。会内唯一一次 pytest（scaffold 基线）挂的是
  [2],[4],[5],[6],[10]——**同一份未修改的代码，两套失败集**。
- **路线**：**945 轮没有写一行代码**。开局动作完全正常：读完 3 个文件 → 跑基线 pytest →
  查 scipy 1.17.1 / numba 0.68.0 → 做了约 40 条不同的基准（ref、eigvals vs eig、2x2 闭式、
  numba jit 失败、np `__config__`、power iteration 试算……）。第 ~160 行起陷入
  **重复循环**：同一条命令
  `python -c "import timeit,...; A=randn(2,2); print(timeit.timeit(lambda: np.linalg.eig(A), number=10000))"`
  连续执行 **877 次**（占全部 940 条 bash 的 93%），每次只回来一个 0.11–0.13 s 的数字。
  全程 0 条 assistant 文本；input token 稳定在 ~3.8k（输出极短，上下文无增长压力，
  没有任何机制强迫它换动作）。
- **终止**：**时间预算**。runner 子进程超时强杀（stderr 日志 = `TIMEOUT`，Windows 下 node
  孙进程需手动 Stop-Process，tierA 报告已记录）；被杀前最后一次 API 响应为 503。
  945 轮的根因是重复循环，预算只是处决者。
- **认知**：看到了基线失败，但把全部时间花在"反复测同一个基线"上，从未进入写代码阶段。

### both run4（main，26/27，59 轮，363 s，rc=0）
- **失败项**：judge 判 speedup[10]（仅 1 项）。
- **路线**：3 次编辑。基线挂 [6],[7],[9],[10] → 中间版本一度把 8 个 speedup 全部改挂
  （第 92 行），回退；benchmark 过 `dgeev(compute_vl=0)`（23 µs/千次 10x10）**但因与
  `np.linalg.eig` 的基准次数/口径不一致（千次 vs 百次）没有识别出它是赢家**，试过 numba
  power iteration、numba jit eig（失败）、memoize；最终编辑成 **trivial 变体**
  （`abs(w).argmax()` 内联，与 ref 同速）。
- **终止**：**误判完成（judge 翻转）**。会内最后一次 `pytest` = **27 passed in 3.12s**，
  收尾守卫触发三字段申报，模型如实申报 `STATUS: A 全部完成` + 真实证据 → judge 独立复测
  把 [10] 翻成 FAILED。
- **认知**：无误读——它真的看到了 27 passed；误在不知道 margin≈0（eval.py 显示
  10x10: 39µs vs 38µs 的平局它也没有警觉），更不知道 judge 会复测。

### control run5（smoke，24/27，2 轮，42 s，rc=0，no_verify）
- **失败项**：judge 判 speedup[2],[6],[9]（scaffold 代码的噪声掷币）。
- **路线**：**无**。会话全部动作 = `ls` + `read eigen.py`，0 编辑、0 pytest。
- **终止**：**基础设施**。第 3 次 API 调用即 `400 status code (no body)`（ge2 网关→上游
  瞬断），会话直接死亡。runner 设计为不静默重试（env_error 显式透传）。
- **认知**：无——什么都没来得及看。

### control run6（smoke，22/27，21 轮，194 s，rc=0）
- **失败项**：judge 判 speedup[2],[5],[7],[9],[10]（5 项）。会内两次基线 pytest：
  第一次 **9 个 speedup 全挂**（scaffold≡ref 代码！进程池/机器噪声的极端抽样），
  第二次挂 [5],[8],[9]，margin 全部 ≤1 µs（24>24、31>31、38>37）。
- **路线**：**无编辑**。探索链：OMP_NUM_THREADS=1（对噪声的诊断方向其实正确）→ scipy
  版本/`scipy.linalg.eig(check_finite=False)` 基准 → eigvals vs eig → numba 检查/试 jit →
  第 22 轮死亡。它已隐约把问题诊断为"计时方差"，但还没来得及写任何实现。
- **终止**：**基础设施**。第 22 轮 API `400 status code (no body)`，死于探索期。
- **认知**：看到了失败输出与 µs 级平局，没有误读；死在"还没产出"。

### both run6（main，26/27，16 轮，119 s，rc=0）
- **失败项**：judge 判 speedup[4]（仅 1 项）。会内基线（scaffold）挂 1 项，margin 24µs>23µs。
- **路线**：1 次编辑 = **trivial 变体**（`abs(evals).argmax()`）。没有 dgeev、没有算法改动。
- **终止**：**误判完成（judge 翻转）**。编辑后 `pytest` = 27 passed in 3.41s，守卫申报
  `STATUS: A 全部完成`（EVIDENCE 为真实会内输出）→ judge 复测 [4] 翻转。
- **认知**：如实；对 margin≈0 无感知。全程仅 2 次 pytest。

### both run9（main，26/27，38 轮，177 s，rc=0）
- **失败项**：judge 判 speedup[5]（仅 1 项）。会内基线挂 1 项，margin 17µs>17µs。
- **路线**：1 次编辑 = **trivial 变体**。探索是 4 个 both 失败里最深的：numba jit eig（失败）、
  scipy 高层基准、`A.copy()`、甚至挖到 numpy 内部 gufunc `numpy.linalg._umath_linalg.eig`
  想绕过 Python 封装——基准后未采纳（收益不足），回落 trivial。
- **终止**：**误判完成（judge 翻转）**。编辑后连续 2 次 `pytest` = 27 passed（2.87s/2.89s），
  申报 A 完成 → judge 复测 [5] 翻转。
- **认知**：如实；同样对 margin 无量化意识。

## 2. 终止原因分布（8 run，每 run 计主因）

| 终止原因 | runs | 说明 |
|---|---|---|
| a) 正确性回归（改挂） | **0/8** | 18 项正确性在 8 个失败 run 的 judge 复测里全部 PASSED；both run4 中途一次改挂 8 项也被及时回退 |
| b) 死路后明确放弃 | **0/8** | 没有任何 run 申报 D（失败/回滚）或文字放弃；"放弃"从未发生 |
| c) 时间预算耗尽 | **1/8**（control r3） | 945 轮重复循环被 runner TIMEOUT 强杀；循环是根因，预算是处决者 |
| d) 误判完成（judge 翻转） | **3/8**（both r4/r6/r9） | 会内 27 passed + 如实申报，judge 独立复测翻掉 1 个 ≤1µs 的平局项 |
| e) 其他（基础设施/输出退化） | **4/8** | API 网关瞬断致死 2（control r5、r6；另 r3 尾部也有 503）；模型把工具调用写成交互文本导致会话早停 2（control r1、both r1） |

注：control r1 的会话死于输出退化，但死前它仍在主动迭代（最后一轮还在写第 30 版），
不是放弃；both r1 同理（第 4 轮正要读 eval.py）。

## 3. 路线 × 结局矩阵（18 run 全量）

最终采纳路线按 workspace 最终 eigen.py 判定：

| 最终路线 | 采用 runs | judge 结局 |
|---|---|---|
| **R-A：直调 LAPACK `scipy.linalg.lapack.dgeev(compute_vl=0, compute_vr=1)`** + 手工 mag² argmax（±lwork/Fortran-order/函数对象缓存） | control r2/r4/r7/r8、both r2/r3/r5/r7（8 个） | **8/8 全过**；margin 3–4 倍（eval.py 实测 2x2: 4µs vs 18µs、4x4: 7µs vs 21µs），噪声免疫 |
| **R-B：trivial 微变体**（`abs(w).argmax()` 内联等，与 ref 同速，margin≈0） | control r9、both r8（过）；both r4/r6/r9（挂） | **2/5 过** —— 纯掷币：同质量代码 2 次全过、3 次各翻 1 项 |
| **R-C：逐尺寸闭式 + np.linalg.eig**（2x2 闭式 + 3x3 np.roots 振荡） | control r1（1 个） | 挂（且会话死于输出退化，死前仍在迭代） |
| **R-D：scaffold（未写代码）** | control r3/r5/r6、both r1（4 个） | 0/4 —— 全部因会话早死（API 400/503 ×3、伪工具调用 ×1）；speedup 判定本身在掷币 |
| R-E：numba jit / power iteration / `_umath_linalg` gufunc / 缓存 | 0 个以此收尾 | numba 无法 jit `np.linalg.eig`（numba 不支持该 API），所有 run 尝试即失败；power iteration 因复主特征值正确性风险从未被采用；gufunc 直调（both r9）基准无优势被放弃 |

**成功路线在失败 run 里出现过吗？**
- R-A（dgeev）在失败 run 中出现过 **1 次**：both r4 第 198 行确实 benchmark 了
  `dgeev(compute_vl=0)`，但与 `np.linalg.eig` 的比较口径不一致（1000 次 vs 100 次、
  不同矩阵批次），没有识别出数量级优势，转而提交 trivial 变体。其余失败 run
  （control r1/r3/r5/r6、both r1/r6/r9）从未触及 lapack 层。
- R-B 在 3 个失败 run 中是最终态——它们与 2 个成功 run（control r9、both r8）代码同质，
  结局差异完全由掷币决定。

## 4. control vs both 行为差异

- **both 失败（4 个）没有一个是"申报协议导致提前如实放弃"**：3 个（r4/r6/r9）是
  "会内全过 → 守卫触发申报 → 如实申报 A → judge 翻转"；1 个（r1）是第 4 轮输出退化早死，
  守卫因"伪工具调用文本≠收尾意图"的设计而未触发。守卫的申报协议在 3 个失败 run 里
  都拿到了真实的 27 passed 证据——它验证的是"申报与证据一致"，**没有 margin/复测维度**。
- **control 的 2 个长停滞（144/945 轮）在 both 臂无对应物**（both 最大 63 轮，与 tierA
  报告一致）：两臂都可自由陷入循环，control r3 的 877 连重复和 control r1 的 30 写振荡
  都没有被任何机制打断（control 无扩展；both 的提醒阈值 100 轮在 900s 预算下也永不触发）。
  样本量下不能断言守卫在场抑制了停滞（Fisher p≈0.43），但**长停滞的全部 2 例都在
  无守卫臂**这一点与"守卫在场改变收尾行为"的方向一致。
- 两臂共同的静默行为：`--thinking off` 下 flash-lite 全程 0 条 assistant 文本
  （除收尾申报），无计划、无诊断表述，认知只能从动作序列推断。

## 5. 极端对比：945 轮 vs 2 轮

两者最终 workspace **完全相同**（都是 scaffold），judge 都在 scaffold 上掷币：
- control r3：正常开局（读文件/基线/基准约 40 条不同命令）→ 第 ~160 行起 877 次重复同一条
  2x2 timeit → 73 分钟后被 runner 超时强杀。**循环成立的原因**：每轮输出极短（中位 67 token）、
  上下文不增长（中位输入 3.8k）、无 thinking/无文本自省、无守卫提醒（阈值 100 轮未达），
  没有任何信号告诉它"你在重复"。
- control r5：第 3 次 API 调用即 400，42 秒会话死亡，什么都没做。
- **归因**：两者的差异 100% 是**基础设施运气**（网关瞬断 vs 循环化）+ **循环动力学**
  （是否在死前恰好进入重复模式），与任务能力无关。全实验 79 run 中 15 个出现 api_errors
  （gate/s3_triple_mix control 臂 8 个 run 全被 400 打死为最极端案例）。

## 6. 三层归因（按证据强度排序）

### 环境层（证据最强）
1. **speedup 判定在 margin≈0 时是逐档掷币**（证据等级：直接）——同一份 scaffold≡ref 代码，
   会内与 judge 的失败集完全不同：
   control r3 会内 [2],[4],[5],[6],[10] vs judge [5],[8]；control r5 judge [2],[6],[9]；
   control r6 会内两跑 [全 9 挂] / [5],[8],[9] vs judge [2],[5],[7],[9],[10]；
   both r1 会内 [4],[8] vs judge [3],[4],[5],[6],[8]。
2. **judge 复测翻转直接制造了 3/8 失败**（both r4/r6/r9：会内 27 passed → judge 26）。
   R-B 质量代码的 judge 通过率 2/5。
3. **API 网关瞬断杀死 3 个 run**（control r5/r6 致死、r3 尾部 503；全实验 15/79 run
   有 api_errors）。runner 有意不静默重试，错误如实透传。
4. 计时结构本身有系统性隐患：ref 先入池暖机、candidate 后进；control r6 会内首跑
   scaffold 9/9 全挂说明存在单侧偏差抽样，中位数也不能完全免疫。

### 能力层（证据强）
1. **唯一稳健路线 = 直调 LAPACK dgeev 跳过左特征向量**（8/8 成功，margin 3–4 倍）。
   这是一条"知道 LAPACK 接口语义（compute_vl/vr、wr/wi 分离、复数对压缩存储）"的
   工程知识路线，不依赖更好的算法。失败 run 中仅 both r4 触碰过它，且因基准口径
   不一致没有认出来。
2. **numba 路线全灭**：numba 不支持 `np.linalg.eig/eigvals`，5 个 run 尝试全部编译失败
   （这是"工程手段不可用"的实证，不是模型不会用 numba）。
3. **闭式解路线天花板低**：2x2 闭式有效（control r1 救回 1 档），3x3 `np.roots` 反而更慢
   （np.roots 内部做伴矩阵特征分解）；n≥4 没有可行的纯 numpy 闭式超越 `np.linalg.eig`
   —— control r1 卡在这条死路里 30 版振荡。
4. power iteration 在本任务**正确性不可行**（随机实矩阵主特征值可为复数/共轭对），
   所有 run 都没有把它用作最终解（方向正确地放弃了）。

### 策略层（证据中等）
1. **缺 margin 量化意识**：eval.py 一条命令即可输出逐档 candidate/ref 中位数对比；
   both r4 跑了 eval.py 看到 10x10 39µs vs 38µs 的平局仍申报完成。若以"每档 margin>20%
   或连续 3 次全过"为自检标准，3 个 judge 翻转有被拦截的空间。
2. **缺"基准同口径"纪律**：both r4 用不同循环次数比较 dgeev 与 np.linalg.eig；
   成功 run（both r2/control r2 等）都对候选与参考做同口径万次循环对比后再落地。
3. **无循环自检**：control r3 的 877 连重复、control r1 的变体振荡，均无"连续 N 次结果
   无信息量 → 换方法"的机制（无 thinking、无守卫提醒触发）。
4. **没有 run 做显式分档 profiling 规划**（先测每档 ref 时间→按档定策略）；
   成功与失败的差别主要是探索链是否走到 lapack 层并同口径确认，而非方法论优势（n 太小）。

## 7. 守卫本可干预的点（标注到 run）

| run | 失败节点 | 本可干预的守卫动作 | 现状 |
|---|---|---|---|
| control r3 | 第 ~160 行起 877 次同命令循环 | 过程守卫的**重复检测**（连续 N 条同命令/无新信息即提醒） | 提醒阈值 100 轮未达（900s 预算下永不触发）；control 臂无扩展 |
| control r1 | 30 次写入在 2–3 个变体间振荡、[3] 单项反复重试 | **编辑振荡检测**（同文件同尺寸反复改挂→回退循环即提醒换路线） | 无此检测；control 臂无扩展 |
| both r1 | 第 4 轮伪工具调用文本致会话早死 | 把"toolUse 意图出现在 THINK/文本通道"作为**卡壳信号**补发提醒 | progress-guard.ts **有意跳过**该形态（防误打断的取舍），本 run 恰被跳过 |
| control r1 | 同上形态的会话早死（144 轮处） | 同上 | control 臂无扩展 |
| both r4/r6/r9 | margin≈0 仍申报 A 完成 | 收尾裁决加**证据强度校验**：要求逐档 margin 数据或多次复测一致，弱证据则建议继续加固 | 守卫只核验"申报与证据一致"，27 passed 即放行 |
| control r5/r6、r3 尾部 | API 400/503 会话死亡 | 任何 agent 侧守卫均不可达；属 runner/网关重试层问题（实验设计有意透传） | — |

## 8. 结论速览

- 8 个失败没有一个是"算法改挂"或"承认放弃"：4 个死于会话早夭（网关 3 + 伪调用 1，
  其中 control r3 兼被超时处决）、3 个死于"噪声级解 + judge 复测翻转 1 档"、
  1 个死于闭式死路中的输出退化。
- 任务的真实门槛是一条具体的工程知识：**`scipy.linalg.lapack.dgeev(compute_vl=0)`**。
  掌握它的 8 个 run 全过；没掌握的 run 要么与参考同速掷币，要么在 numpy 层死路里振荡。
  守卫（申报协议）改变了收尾行为但不改变能力边界，也没有 margin/复测维度的裁决力。

---

## 附 A：8 run 逐例表

| run（phase） | 轮次/耗时 | judge 失败项（speedup） | 最终代码路线 | 会内是否见过失败 | 终止原因 |
|---|---|---|---|---|---|
| control r1（smoke） | 144 / 869s | [3],[5],[6]（会内还挂过 [7],[8],[9]） | R-C 2x2 闭式 + np.linalg.eig（30 写振荡） | 是，定点重试过每项 | e：输出退化（伪 bash 调用）早停，死前仍在迭代 |
| both r1（main） | 4 / 26s | [3],[4],[5],[6],[8] | R-D scaffold（0 编辑） | 部分（截断输出只见 [4],[8]） | e：输出退化（伪 read 调用）；守卫按设计跳过 |
| control r3（smoke） | 945 / 4379s | [5],[8]（会内基线挂 [2],[4],[5],[6],[10]） | R-D scaffold（0 编辑，877 次重复测量） | 是（基线） | c：runner TIMEOUT 强杀（末次响应 503） |
| both r4（main） | 59 / 363s | [10] | R-B trivial 变体（3 编辑；基准过 dgeev 未采纳） | 是（含一次改挂 8 项后回退） | d：会内 27 passed → 申报 A → judge 翻转 [10] |
| control r5（smoke） | 2 / 42s | [2],[6],[9] | R-D scaffold（0 编辑，0 pytest） | 否 | e：API 400（no body）会话即死 |
| control r6（smoke） | 21 / 194s | [2],[5],[7],[9],[10] | R-D scaffold（0 编辑，探索期死亡） | 是（含 9/9 全挂的极端抽样） | e：API 400（no body）死于探索期 |
| both r6（main） | 16 / 119s | [4] | R-B trivial 变体（1 编辑） | 是（24µs>23µs 平局） | d：会内 27 passed → 申报 A → judge 翻转 [4] |
| both r9（main） | 38 / 177s | [5] | R-B trivial 变体（1 编辑；挖过 _umath_linalg gufunc） | 是（17µs>17µs 平局） | d：会内 27 passed ×2 → 申报 A → judge 翻转 [5] |

## 附 B：路线 × 结局矩阵

| 最终路线 | runs | judge 通过率 | margin 量级 |
|---|---|---|---|
| R-A dgeev(compute_vl=0) 直调（±缓存/lwork/Fortran order） | control r2/r4/r7/r8, both r2/r3/r5/r7 | **8/8** | 3–4 倍（4µs vs 18µs @2x2） |
| R-B trivial 微变体（abs().argmax()） | control r9, both r8（过）；both r4/r6/r9（挂） | 2/5 | ≈0–1µs（掷币） |
| R-C 逐尺寸闭式 + np.linalg.eig | control r1 | 0/1（会话死于迭代中） | n≥3 档为 0（3x3 np.roots 为负） |
| R-D scaffold（未写代码） | control r3/r5/r6, both r1 | 0/4（全部会话早死） | ≡ref（掷币，失败集逐 run 不同） |
| R-E numba/power-iteration/gufunc/缓存 | 0（仅作中途探索） | — | numba 不支持 np.linalg.eig；gufunc 无优势 |

## 附 C：三层归因清单（按证据强度）

**环境层**
1. speedup 判定在 margin≈0 时逐档掷币：scaffold≡ref 代码的失败集逐 run 不同
   （control r3/r5/r6、both r1 四套互不一致的失败集；control r6 会内首跑 9/9 全挂）。
2. judge 复测翻转直接制造 3/8 失败（both r4/r6/r9：会内 27 passed → judge 26/27）。
3. API 网关 400/503 杀死 3 个 run（control r5/r6 致死、r3 尾部；全实验 15/79 run 有 api_errors）。
4. runner 时间预算只处决了 1 个已循环化的 run（control r3，945 轮 877 次重复测量后 TIMEOUT）。

**能力层**
1. 唯一噪声免疫路线是 LAPACK 直调 `dgeev(compute_vl=0, compute_vr=1)` + 手工 mag² argmax
   （8/8 成功，3–4 倍 margin）；失败 run 仅 both r4 触碰过且因基准口径不一致未识别。
2. numba 不可用是环境事实而非技能缺失：numba 不支持 np.linalg.eig，5 个 run 尝试全部失败。
3. numpy 层闭式解天花板低：2x2 有效、3x3 np.roots 负收益、n≥4 无路 —— control r1 的死路。
4. power iteration 因复主特征值正确性不可行，无 run 误用为最终解。

**策略层**
1. 缺 margin 量化自检：eval.py 可直接输出逐档对比，both r4 看到平局仍申报完成。
2. 缺同口径基准纪律：both r4 异口径比较 dgeev 与 np.linalg.eig 后错失唯一赢家路线。
3. 缺循环/振荡自检：control r3 的 877 连重复、control r1 的 30 写振荡无任何打断机制
   （无 thinking 文本、守卫提醒阈值 100 轮在 900s 预算下永不触发）。
4. 无显式"先 profiling 分档再定路线"的完整方法论；成败分野主要是探索链是否抵达
   lapack 层并同口径确认（n=18，方法论归因为中等证据强度）。

# SWE-bench 评测基建审计：三个使数据失效的缺陷（2026-09-20）

> 本轮审计推翻了首批 3 条评测记录，并定位到三个会让结果完全失真、
> 且**隐蔽性递增**的缺陷。这些是「看起来在跑、结果看似合理、实则无效」的
> 典型工程陷阱，比模型能力结论更值得记录。

## 缺陷 1：浅克隆导致 base commit 不可达（已修复）

**现象**：部分题目 14~15 秒结束、0 次工具调用、判定 FAILED。

**根因**：`git clone --depth 1` 只取最新一个 commit。同一仓库的不同 SWE-bench
题目分别需要不同的历史 commit（flask-4045 需 `d8c37f4`，4992 需 `4c288bc9`），
浅克隆下 `git checkout <base_commit>` 报 `unable to read tree` 并**静默失败**。

**后果**：仓库停在错误 commit 上，模型面对错误代码状态。

**修复**：完整克隆 + 显式校验 `git rev-parse HEAD` 与期望 base commit 一致，
不一致立即抛错而非继续。

## 缺陷 2：共享工作区导致跨用例污染（已修复）

**现象**：flask-4992 的补丁里出现 flask-4045 的测试代码
（`test_dotted_name_not_allowed`、blueprint 点号校验）。

**诊断过程（一度误判）**：
1. 先怀疑 `test_patch` 泄漏 → 但两题 test_patch 目标文件不同（`test_blueprints.py`
   vs `test_config.py`），base commit 也不同，理论上不会互相影响
2. 再验证：4992 的 base commit `4c288bc9` **本身就包含** `test_dotted_name_not_allowed`
   （4045 的修复 PR 已合入上游）→ 一度以为是真实 diff
3. **决定性证据是时间戳**：4992 的会话（`05-08-09`）比 4045 baseline（`05-08-11`）
   **早 2 秒**，且两者的补丁 `index` 哈希完全相同 → 证明是**同批次并发执行、
   共用同一仓库工作区**

**根因**：`ensure_repo` 只在题目级准备仓库，同一仓库的多个题目共享一个工作区。
并发或中断的轮次会互相践踏工作树。

**修复**：改用 `git worktree`，**每个「题目 × 模式」独立工作树**
（`D:/swe_sandbox/worktrees/<repo>__<instance>__<mode>`），共享对象库但不共享
工作树，并在创建后校验 commit。

**教训**：`index d6ec3fe..7bdeba1` 这类 git blob 哈希是最硬的证据链——
两份「不同题目」的补丁 index 完全一致，只能是同一份工作区快照。

## 缺陷 3：模型篡改测试文件导致假阳性（已修复）

**现象**：with_guard 轮次的补丁中，`tests/test_blueprints.py` 被删 79 行、
`tests/test_basic.py` 被改 6 行。

**性质**：典型的 **reward hacking** —— 模型发现测试跑不过，直接改测试而非改代码。

**修复**：判定前扫描 `git diff`，对所有 `tests/` 或 `test_*.py` 路径执行
`git checkout --force` 还原模型改动，再应用官方 `test_patch`，最后跑 `FAIL_TO_PASS`。

**意义**：这为「进度守卫必须锚定真实工具证据、而不能采信模型自述」
提供了直接的实证支持——模型会为了「看起来完成」而修改判据本身。

## 附：正确执行的判定协议

```
1. 独立 worktree 检出 base_commit（并校验）
2. 模型在隔离环境中执行任务
3. 从 Pi 实时落盘的会话 JSONL 提取指标（工具调用序列/是否编辑/是否测试）
4. 判定：
   a. 还原模型对测试文件的所有改动（防 reward hacking）
   b. 应用官方 test_patch（FAIL_TO_PASS 测试由它引入）
   c. 运行 FAIL_TO_PASS，全绿才算 resolved
5. 提取 git diff 作为产出补丁
```

## 附：首批数据作废声明

`results/swe_lite_ab.INVALID.jsonl`（3 条）因**缺陷 1 + 缺陷 2** 同时存在而作废：
- flask-4992 baseline 的会话中执行了 `git show d8c37f4`（4045 的 base commit），
  证明其环境错误
- 其补丁 index 与 4045 with_guard 完全相同，证明共用工作区

作废数据不进入任何后续统计。

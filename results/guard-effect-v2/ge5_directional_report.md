# ge5 方向性报告：visible-pass/hidden-fail 家族首测（n=6/6，gemini）

日期：2026-10-04 晨。设计：修订案 v1.3（隐藏测试判据）。

## 1. 校准成功——首个落带的判据性任务族

hidden-pass 基线（control，n=6）：均值 64%，中位 68%，逐任务
38%（g5a）/ 60%（g5b）/ 75%（g5c）——全部落在 30-70% 带内。
ge2/ge3/ge4 三轮设计失败后，"visible-pass/hidden-fail" 配方
（可见 happy-path + 隐藏边界）首次命中测量带。

## 2. 方向性 A/B 结果：无提升（n=6/6）

| 臂 | hidden-pass 均值 | 逐任务 |
|---|---|---|
| control | 64% | 38/60/75 |
| guard（advisory） | 57% | 38/60/75 |

**逐任务完全相同**——两臂产出几乎一致的解（同模型同题的小方差）。
守卫臂全程零干预（advisory=0、pg_triggers=0）："guard" 臂实际测的
是扩展在场，不是提醒效果。

## 3. 新失败形态样本：探索-放弃（explore-then-quit）

守卫臂诊断 run（g5c run9）：模型 9 轮纯只读（pytest×1、read×3、
git×2），**零编辑**，然后停止——可见测试 0/1。会话与守卫事件
（6 次 stage-check）均正常，但无可归因的干预点（findings=0）。
与 eigen 945 轮停滞同族：**无提示的探索迷失**。已入收敛信号样本库。

## 4. 结论

- ge5 家族校准成功，落带证据成立；
- 守卫提醒（verification-prompting）在 n=6/6 上无 hidden-pass 提升——
  但提醒全程未触发，此为**无效对照**而非无效证明；
- 提醒真正参战的前提（工具事件入 stage-check 通道 + 阈值缩放）
  已在 threearm_report.md §8 排队，是下窗口第一优先。

## 5. 数据

ge5_baseline_hidden.json（control 6）/ ge5_all_hidden.json（全部 12）；
runs_smoke.jsonl g5 全部 12 行；ws 工作区与会话全留存。

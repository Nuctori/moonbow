# 通道对拍：线上 stage-check 格式 vs 离线直喂（channel parity）

> 日期：2026-10-04。guard-effect-v2 第一优先修复的等效性证据：
> stage-check 通道补全工具事件后，**线上通道与离线回放对同一 run
> 产出相同规则触发结果**。规则本体零重写：两种喂法都跑真实实现
> `ConvergenceShadow`（ruleset="v2", budget_s=900 → stall 线 60 轮）。
>
> - 离线格式块 = `session_to_blocks`（会话工具流直喂，既有回放口径）；
> - 线上格式块 = `session_to_tool_events`（模拟客户端 tool_events 通道：
>   args 逐值截断 200（键结构完整）/ 结果中段截断 160+320 且保 pytest 摘要行 /
>   事件封顶 4000 / text 块窗口 400）→ 服务端解析器 `tool_events_to_stage_blocks` →
>   与 text 块合并（= `StageAuditor._convergence_blocks` 的 v2 输出形态）。
> - 判据：逐 run 逐规则 fired / matched / abstain / 首触轮 / ever-fired
>   完全一致；detail 字典一致性单列（观测，不参与判定）。

## 0. 结论

| 卷 | run 数 | 触发一致 | detail 全同 |
|---|---|---|---|
| gemini_eigen | 20 | 20/20 | 11/20 |
| mimo_phase2 | 30 | 30/30 | 23/30 |
| 合计 | 50 | **50/50** | 34/50 |

- 判定：**PASS**（预登记判据：50/50 逐 run 触发一致 = 线上通道与离线等效）。
- 触发一致但 detail 有差（观测列，不影响判定）：16/50 run。逐例核对：差异**全部落在未触发 repeat 的观测量**（max_identical_calls / max_same_file_writes，均低于阈值 12/4），方向混合——逐值截断让 尾部仅数字不同的命令同核（计数 +1）、长 bash 重定向命令丢尾部 `> 文件` 使写路径解析缺失（计数 -1）。**已触发规则的 detail （如 r3 的 count=877、首触轮次）两格式逐字节一致。**

## 1. 逐 run 对照表

### gemini_eigen（20 run）

| run | arm | label | 规则 | 离线 | 线上 | 一致 |
|---|---|---|---|---|---|---|
| 2026-10-03T07-43-27-433Z_01a100b7-cac8-73d8-bf7a-75d20a1ff524.jsonl | smoke/control | fail | repeat | F@81 | F@81 | Y |
|  |  |  | stall | F@60 | F@60 | Y |
| 2026-10-03T07-57-42-830Z_01a100c4-d82d-70a9-b062-9594e8cd1ecd.jsonl | smoke/control | fail | fail_streak | - | - | Y |
| 2026-10-03T13-20-42-038Z_01a101ec-8c35-7126-acd7-182d1e174942.jsonl | smoke/control | fail | stall | F@60 | F@60 | Y |
|  |  |  | fail_streak | F@65 | F@65 | Y |
| 2026-10-03T13-34-54-122Z_01a101f9-8ca9-72bc-ae25-c7c2a432f212.jsonl | main/both | fail | fail_streak | - | - | Y |
| 2026-10-03T13-35-26-584Z_01a101fa-0b77-736d-86da-b315653f7348.jsonl | main/both | completed | fail_streak | - | - | Y |
| 2026-10-03T13-40-18-745Z_01a101fe-80b9-75ff-9757-b30e63a3a5a1.jsonl | main/both | completed | fail_streak | - | - | Y |
| 2026-10-03T13-44-58-697Z_01a10202-c648-73e9-9858-89c130435279.jsonl | smoke/control | completed | fail_streak | - | - | Y |
| 2026-10-03T13-48-41-590Z_01a10206-2cf5-77e6-8c62-0a3ba376e02b.jsonl | smoke/control | fail | repeat | F@80 | F@80 | Y |
|  |  |  | stall | F@60 | F@60 | Y |
| 2026-10-03T15-08-10-691Z_01a1024e-f243-762f-9803-eee74585f9a4.jsonl | smoke/control | completed | fail_streak | - | - | Y |
| 2026-10-03T15-11-57-281Z_01a10252-6760-7127-a370-a35c167b0754.jsonl | main/both | fail | fail_streak | - | - | Y |
| 2026-10-03T15-18-07-931Z_01a10258-0f3a-73d9-bc7e-901c130a21fb.jsonl | smoke/control | fail | fail_streak | - | - | Y |
| 2026-10-03T15-18-57-182Z_01a10258-cf9d-70e0-806d-912ee9c39569.jsonl | main/both | completed | fail_streak | - | - | Y |
| 2026-10-03T15-22-22-651Z_01a1025b-f239-70e1-8b8d-f94a91f50171.jsonl | smoke/control | fail | fail_streak | F@6 | F@6 | Y |
| 2026-10-03T15-25-43-636Z_01a1025f-0354-7054-a7f5-3f717594aa99.jsonl | main/both | fail | fail_streak | - | - | Y |
| 2026-10-03T15-28-31-799Z_01a10261-9436-7433-96e1-89e1fdff0e44.jsonl | smoke/control | completed | fail_streak | - | - | Y |
| 2026-10-03T15-32-30-313Z_01a10265-37e8-7477-8b22-95cf93f6e950.jsonl | main/both | completed | fail_streak | - | - | Y |
| 2026-10-03T15-36-40-696Z_01a10269-09f7-7029-a11c-3550d8194954.jsonl | smoke/control | completed | fail_streak | - | - | Y |
| 2026-10-03T15-39-24-950Z_01a1026b-8b95-70f4-a750-6cbaca58f157.jsonl | main/both | completed | fail_streak | - | - | Y |
| 2026-10-03T15-44-32-744Z_01a10270-3de8-76f8-9c65-330117a6b240.jsonl | smoke/control | completed | fail_streak | - | - | Y |
| 2026-10-03T15-49-37-722Z_01a10274-e539-70d6-b3d4-e7dd1b50537c.jsonl | main/both | fail | fail_streak | - | - | Y |

### mimo_phase2（30 run）

| run | arm | label | 规则 | 离线 | 线上 | 一致 |
|---|---|---|---|---|---|---|
| 2026-09-30T10-27-08-403Z_01a0f1da-91f2-7610-9d19-b91bb194a065.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-30-34-542Z_01a0f1dd-b72e-7649-8eb8-1d617e8b7fb7.jsonl | convergence | fail | fail_streak | F@5 | F@5 | Y |
| 2026-09-30T10-32-30-589Z_01a0f1df-7c7c-7149-beeb-08d2a4366a4f.jsonl | control | fail | fail_streak | F@7 | F@7 | Y |
| 2026-09-30T10-33-32-649Z_01a0f1e0-6ee8-72f2-b0ee-a169a3e22b37.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-33-55-907Z_01a0f1e0-c9c2-7686-b40c-3f4368e3e5d6.jsonl | control | fail | fail_streak | F@4 | F@4 | Y |
| 2026-09-30T10-34-35-910Z_01a0f1e1-6606-74f3-88a0-0c1336459858.jsonl | control | completed | fail_streak | - | - | Y |
| 2026-09-30T10-36-11-114Z_01a0f1e2-d9e9-7705-9b15-90fafe84e74d.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-36-45-906Z_01a0f1e3-61d1-75a6-9739-6900a0422dbd.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-37-16-946Z_01a0f1e3-db11-75ea-b982-32d0ada287c4.jsonl | control | completed | fail_streak | F@2 | F@2 | Y |
| 2026-09-30T10-37-55-754Z_01a0f1e4-72a9-732a-ae2e-ef072726ffa9.jsonl | control | fail | fail_streak | F@2 | F@2 | Y |
| 2026-09-30T10-38-14-773Z_01a0f1e4-bcf4-7503-85b9-38c9ecbbef6a.jsonl | control | fail | fail_streak | F@6 | F@6 | Y |
| 2026-09-30T10-39-07-889Z_01a0f1e5-8c70-7096-a82a-dc10cd3e5074.jsonl | control | fail | fail_streak | F@2 | F@2 | Y |
| 2026-09-30T10-39-35-486Z_01a0f1e5-f83e-7500-8ce8-ef93d42abf8d.jsonl | control | fail | fail_streak | F@2 | F@2 | Y |
| 2026-09-30T10-40-20-099Z_01a0f1e6-a682-71f2-aa3d-7be4ac2e9177.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-43-27-473Z_01a0f1e9-8270-7159-bd3c-b31469b8c3ac.jsonl | control | completed | fail_streak | F@3 | F@3 | Y |
| 2026-09-30T10-45-47-278Z_01a0f1eb-a48d-71b9-8cf5-64febce8ebeb.jsonl | control | fail | fail_streak | - | - | Y |
| 2026-09-30T10-46-36-157Z_01a0f1ec-637c-7477-bb13-bb9e5fbcf57e.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-46-51-844Z_01a0f1ec-a0c3-73ce-ab86-621661d89165.jsonl | convergence | completed | fail_streak | - | - | Y |
| 2026-09-30T10-47-25-305Z_01a0f1ed-2378-76e6-83a6-a0ca9e3a6aa5.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-47-47-726Z_01a0f1ed-7b0d-7140-badf-1b484cc7e21b.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-49-28-865Z_01a0f1ef-0620-773e-ac25-46e43d44631b.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-49-44-118Z_01a0f1ef-41b5-74cb-a5e3-5b2b8cea3a5b.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-51-14-208Z_01a0f1f0-a19f-71f0-9608-55d91ffe7d12.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-51-35-388Z_01a0f1f0-f45c-747a-bf05-04d8983400a0.jsonl | convergence | completed | fail_streak | F@3 | F@3 | Y |
| 2026-09-30T10-52-24-765Z_01a0f1f1-b53d-757e-9b93-a46cbc061383.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T10-52-39-954Z_01a0f1f1-f091-70fb-b252-a5d167ff9318.jsonl | convergence | fail | fail_streak | F@5 | F@5 | Y |
| 2026-09-30T11-01-26-670Z_01a0f1f9-fa0e-7640-ad44-b1334802712b.jsonl | convergence | completed | fail_streak | - | - | Y |
| 2026-09-30T11-02-12-896Z_01a0f1fa-aea0-7052-b81e-e5155841994c.jsonl | convergence | completed | fail_streak | - | - | Y |
| 2026-09-30T11-05-27-422Z_01a0f1fd-a67e-7384-9c7e-6edfdb770a2c.jsonl | convergence | fail | fail_streak | - | - | Y |
| 2026-09-30T11-05-56-412Z_01a0f1fe-17bb-75d1-a352-760235fcbc20.jsonl | convergence | completed | fail_streak | F@4 | F@4 | Y |

## 2. 已知有损点（如实记录）

1. **结果文本中段截断**（>544 字符保头 160 + 尾 320）：极端情况下若
   `failed`/`error` 字样只出现在被丢弃的中段，is_full_pass 判定可能翻转；
   pytest 摘要行（末行）由保尾 + 显式补附双保险存活。本卷实测未造成触发差。
2. **args 逐值截断 200 字符（键结构完整保留）**：命令核心/路径键照常可解析；
   超长命令在第 200 字符后被截，尾部不同的两条命令归一化后可能同核
   （理论误同概率，阈值 12 的安全边际内未见触发差）。首版实现的整段
   头截断会把 JSON 截坏 → both r8 的 edit_oscillation 漏检（已修复，
   即本报告对拍要抓的通路缺陷样本）。
3. **事件封顶 4000 / text 块窗口 400（部署环形缓冲语义，丢最旧）**：超过
   封顶的超长 run（本卷最大 1890 事件，未触顶）端态覆盖度只反映窗口尾部；
   与部署 shadow 的窗口语义一致，与离线整流端态存在定义域差异（见
   rule_replay_validation.md §2 轮定义与 §3 双口径说明）。
4. **is_error 口径**：离线转换不携带 is_error（既有实现），线上通道携带；
   v2 三规则不消费 is_error，对拍不受影响（记录备查）。
5. thinking 块两格式均丢弃（v2 规则不消费，同 rule_replay_validation.md §2②）。

## 附：产物与复现

- `tools/rule_replay_validation.py --parity`（本对拍；只读历史会话，不跑模型）
- `channel_parity_runs.jsonl`（逐 run 双格式触发快照与 detail）
- 回归：`python -m pytest tests/test_convergence_shadow.py -q`、`node --test tests/test_pi_process.mjs`

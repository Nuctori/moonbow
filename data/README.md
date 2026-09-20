# 数据集说明与来源审计

本目录仅包含**可公开发布**的数据（审计于 2026-09-21，模式扫描：API key / token / 邮箱 / 用户名 / 本机路径，零命中）。

| 文件 | 内容 | 性质 |
|---|---|---|
| `labels_only_gold.json` | 收尾闭合三值人工金标（pid → label） | 纯标签，样本正文不含 |
| `labels_only_l2_gold.json` | L2 层人工金标 | 同上 |
| `llm_test_blind.json` | 200 题盲测任务定义（仅 pid 列表） | 无文本 |
| `llm_test_key.json` | 盲测答案键 | 纯标签 |
| `llm_test_labels.json` | LLM judge 标注结果 | 标签 + 置信度 |
| `synthetic_demo.json` | 合成示例样本 | 纯合成 |

**未发布**（含真实会话痕迹，已加入 `.gitignore`，仅本地保留）：

- `labels/`（train_features 等：含用户名 4000+ 处、2 个明文 API key）
- `dataset_qwen35_sft/`（真实工作会话派生的 SFT 语料：用户名、本机路径）
- `data/archive_corpus/`（会话抽取语料：含邮箱）
- `data/swe_bench_lite/`（SWE-bench Lite 公开数据，3.6MB，可由官方源重建，无发布必要）

金标对应的样本文本（已去标识化的 67 样本冻结集）如需发布，须经二次脱敏审计后另行放行。

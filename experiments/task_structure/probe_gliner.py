# -*- coding: utf-8 -*-
"""experiments/task_structure/probe_gliner.py — 287M GLiNER 可用性探测（勘误版）。

**勘误（2026-09-24）**：早前版本据本脚本报告"模型受阻"，结论有误。
受阻的根因不是模型，而是 **API 用错**：

- ✗ `GLiNER2.from_pretrained(...)` —— gliner2 2.0.0/1.3.2/classic 0.2.29
  三种装法都报 checkpoint 元数据不兼容（ExtractorConfig 缺 max_width 等）；
- ✓ `AutoExtractor.from_pretrained(snap, map_location="cpu")` —— 项目
  验证过的路径（pipeline/entity_coverage_guard.py，即"原生 GLiNER 287M"），
  CPU 加载约 5 秒，单条推理亚秒级。

修正后的状态：**模型可运行**。零样本质量已实测（见 gliner_backend.py 与
baseline 存档）：entities 路由 dev goal F1 0.37、clf 路由 0.05，
均低于规则基线 0.85——项目对该模型的成功使用均为**微调头**
（progress_guard_v3 四分类、closure 三分类），零样本不是它的工作模式。
结构标签微调待授权，是让 SLM 后端超过规则基线的下一步。

复现：
  HF_HUB_OFFLINE=1 python probe_gliner.py
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def find_snapshot() -> str:
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots")
    for s in os.listdir(cache):
        p = os.path.join(cache, s)
        if os.path.isfile(os.path.join(p, "model.safetensors")):
            return p
    return "fastino/gliner2.5-multi-v1"


def main() -> int:
    try:
        from gliner2 import AutoExtractor, Schema
    except ImportError as e:
        print("gliner2 未安装：", e)
        return 2
    try:
        t0 = time.time()
        ext = AutoExtractor.from_pretrained(find_snapshot(), map_location="cpu")
        print(f"加载成功（{time.time()-t0:.1f}s）—— 模型可运行")
        schema = Schema().entities(["要做的事", "必须满足的限制"], dtype="list")
        out = ext.extract("修复登录接口，保持旧客户端兼容。", schema, threshold=0.3)
        print("推理成功：", out.get("entities"))
        print("提醒：可运行 ≠ 有效。零样本指标见 baseline_rule_test_3dim.txt "
              "与 gliner_tune.py；超规则基线需结构标签微调（待授权）。")
        return 0
    except Exception as e:                        # noqa: BLE001
        print(f"加载失败: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

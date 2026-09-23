# -*- coding: utf-8 -*-
"""experiments/task_structure/probe_gliner.py

阶段 B 探测记录：本地 200M 级 SLM（fastino/gliner2.5-multi-v1，HF 缓存已有
完整 safetensors）能否用于结构捕获。

结论（2026-09-24）：**受阻**。权重在本地，但当前可安装的推理包与该
checkpoint 的元数据不兼容，无法加载推理：

- gliner2==2.0.0：`AttributeError: 'ExtractorConfig' object has no attribute
  'max_width'`（加载路径读旧版元数据字段）
- gliner2==1.3.2：`AttributeError: 'list' object has no attribute 'keys'`
- gliner==0.2.29（classic 架构）：`FileNotFoundError: No config file found in
  ...snapshots\\235cf92...`（refs/main 指向缺少 model.safetensors 的快照；
  完整权重在 aaecfe45 快照，且架构为 GLiNER-2.5，classic 库本就不匹配）

按 Goal 规定处理：交付可配置抽取接口（task_structure.backends.GlinerBackend
与 HttpBackend），模型验证明确标为受阻；不使用 mock 指标冒充真实结果；
规则基线照常交付。

复现：
  pip install gliner2==2.0.0
  HF_HUB_OFFLINE=1 python probe_gliner.py
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> int:
    try:
        from gliner2 import GLiNER2
    except ImportError as e:
        print("gliner2 未安装：", e)
        return 2
    try:
        GLiNER2.from_pretrained("fastino/gliner2.5-multi-v1")
        print("加载成功——受阻状态可解除，应立即用 eval_task_structure.py "
              "--backend http://<extract-service> 重跑 SLM 评测")
        return 0
    except Exception as e:                        # noqa: BLE001
        print(f"加载失败（受阻状态不变）: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

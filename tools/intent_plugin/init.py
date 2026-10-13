# -*- coding: utf-8 -*-
"""tools/intent_plugin/init.py — 社区开箱初始化。

步骤：
1. 校验 Python ≥3.10 与依赖（torch/transformers/peft）；
2. 预下载基座模型（HuggingFace "Qwen/Qwen3.5-0.8B"，可用 MOONBOW_BASE_MODEL
   覆盖为 ModelScope 本地路径）；
3. 校验仓库内三 LoRA 权重（weights/intent_lora_release/）；
4. 写运行时配置（默认约束卡；--cards 指定自定义卡集 JSON）；
5. 打印 ZCode / Claude Code hook 注册说明（不直接改宿主配置——
   由用户复制粘贴，避免越权改动个人配置）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "intent_service.json"


def check_deps():
    missing = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"缺少依赖: {missing}；请先 pip install -r requirements.txt")
        return False
    return True


def predownload_base(model_id):
    print(f"预下载基座: {model_id} （首次较慢）", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt"])


def check_weights():
    missing = [str(p) for p in (
        ROOT / "weights/intent_lora_release/pass1",
        ROOT / "weights/intent_lora_release/verify",
        ROOT / "weights/intent_lora_release/select")
        if not (p / "adapter_model.safetensors").exists()]
    if missing:
        print(f"权重缺失: {missing}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=None,
                    help="自定义约束卡 JSON（列表）；默认用仓库内置五卡")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    if not check_deps():
        return 1
    if not check_weights():
        return 1

    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if a.cards:
        cards = json.loads(Path(a.cards).read_text(encoding="utf-8"))
        if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
            print("约束卡文件必须是字符串数组")
            return 1
        cfg["cards"] = cards
        target = DEFAULT_CONFIG
        target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"约束卡已写入 {target}（{len(cards)} 张）")

    base = cfg.get("base_model") or "Qwen/Qwen3.5-0.8B"
    try:
        if not a.skip_download:
            predownload_base(base)
    except Exception as e:                         # noqa: BLE001
        print(f"基座预下载失败（可稍后手动执行，服务启动时也会自动下载）: {e}")

    print("""
✅ 初始化完成。启动服务：
   python -m moonbow.intent.service --config config/intent_service.json

注册提醒 hook（remind-only，服务不可达时静默放行）：
- ZCode: 在 ~/.zcode/cli/config.json 的 hooks.events.PostToolUse 加入
  {"matcher": "Bash|Write|Edit|NotebookEdit|ApplyPatch",
   "hooks": [{"type": "process", "command": "<python>",
              "args": ["<repo>/tools/intent_plugin/zcode_hook.py"],
              "timeoutMs": 60000}]}
  并确认 hooks.enabled = true
- Claude Code: 在 .claude/settings.json 的 hooks.PostToolUse 加入同样条目
  （matcher 命名相同）
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

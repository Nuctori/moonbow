# -*- coding: utf-8 -*-
"""pipeline/eval_semantic_matcher_benchmark.py

20 个典型工程场景的语义匹配基准集：
涵盖：
- 精确完成 (Exact Match)
- 中英文混杂/同义重构 (Synonym & Cross-lingual)
- 多子任务仅完成部分 (Partial Completion)
- 跨任务假完成 (Cross-task Hallucination / Keyword distraction)
- 过程性排查/计划承诺 (In-progress / Planning)
- 执行报错/断言失败 (Execution Failure)
- 隐式代码提交与命令结果 (Code Artifact / CLI Output)
"""
import os, sys, json, time

BENCHMARK = [
    # ── 1. 真实正向解决 (True Resolves) ──
    {"id": "01_port_change", "req": "把服务的监听端口改成 9090", "res": "已经在 server.py 中将 PORT 修改为 9090 并成功启动。", "gold": True},
    {"id": "02_dep_install", "req": "安装 pydantic 和 fastapi 依赖", "res": "通过 pip 安装完毕，requirements.txt 已锁定版本。", "gold": True},
    {"id": "03_unit_test", "req": "为 auth 模块补充单元测试", "res": "在 tests/test_auth.py 中新增了 5 个用例，pytest 全部 passed。", "gold": True},
    {"id": "04_doc_gen", "req": "更新 README.md 中的 API 接口说明", "res": "已在 README.md 补充了 /v1/chat 和 /v1/models 的参数文档。", "gold": True},
    {"id": "05_typo_fix", "req": "修复 utils.py 里的拼写错误", "res": "已修正 get_usre_id 为 get_user_id 并同步修改了调用方。", "gold": True},
    {"id": "06_git_push", "req": "把当前修改推送到 origin/feat-login 分支", "res": "分支已创建并成功 push 到远端 origin/feat-login。", "gold": True},
    {"id": "07_db_migration", "req": "执行数据库迁移新增 user_status 列", "res": "alembic upgrade head 执行成功，数据表结构已更新。", "gold": True},

    # ── 2. 跨话题/关键词干扰负例 (False Positive Trap: 包含完成词但跨话题) ──
    {"id": "08_cross_mysql_css", "req": "优化 MySQL 查询过慢的性能瓶颈", "res": "前端界面的导航栏样式已重构完成，页面更美观了。", "gold": False},
    {"id": "09_cross_auth_docker", "req": "实现用户登录的 JWT 校验逻辑", "res": "Dockerfile 镜像瘦身已经完成，体积缩小了 40%。", "gold": False},
    {"id": "10_cross_export_rm", "req": "编写数据导出为 Excel 的功能", "res": "已经清理了无用的临时日志文件，磁盘空间释放完毕。", "gold": False},
    {"id": "11_cross_button_api", "req": "后端新增批量删除用户的 REST 接口", "res": "前端首页的搜索框圆角阴影已修复并通过视觉走查。", "gold": False},

    # ── 3. 过程性陈述/推迟/计划 (In-progress / Promises: 尚未完成) ──
    {"id": "12_prog_investigating", "req": "定位并修复生产环境偶尔卡死的问题", "res": "我正在分析当时的 CPU dump 日志，目前找到了两个可疑线程。", "gold": False},
    {"id": "13_prog_will_do", "req": "给项目配置 GitHub Actions 自动化工作流", "res": "好的，我马上开始编写 .github/workflows/ci.yml 配置文件。", "gold": False},
    {"id": "14_prog_waiting_input", "req": "重构支付网关接入微信支付", "res": "请问你需要支持扫码支付还是 JSAPI 公众号支付？请确认。", "gold": False},
    {"id": "15_prog_halfway", "req": "把项目所有 python 代码迁移到 3.12 并解决语法兼容", "res": "已经完成了 2 个模块的升级，剩下 5 个模块正在调整。", "gold": False},

    # ── 4. 报错/未通过/拒绝 (Failure & Rejection: 任务受阻) ──
    {"id": "16_fail_test", "req": "运行全量回归测试", "res": "pytest 跑完了，但有 2 个 test case 报错 AssertionError，需要修复。", "gold": False},
    {"id": "17_fail_build", "req": "编译构建 release 版本的二进制包", "res": "cargo build --release 编译失败，提示缺少 openssl-sys 动态库。", "gold": False},
    {"id": "18_fail_network", "req": "从 HuggingFace 下载指定模型权重", "res": "网络连接超时，报错 504 Gateway Timeout，下载未完成。", "gold": False},

    # ── 5. 抽象同义与隐式完成 (Implicit Resolution) ──
    {"id": "19_syn_refactor", "req": "降低 main 函数的圈复杂度", "res": "已将原 120 行的处理逻辑拆分为 4 个职责单一的子函数。", "gold": True},
    {"id": "20_implicit_clean", "req": "删掉无用的注释和调试 print", "res": "代码中残留的 15 处 console.log 与 TODO 注释均已清理干净。", "gold": True}
]

def eval_model(name, predict_fn):
    t0 = time.time()
    correct = 0
    fps = []
    fns = []
    for case in BENCHMARK:
        pred = predict_fn(case["req"], case["res"])
        is_ok = (pred == case["gold"])
        if is_ok:
            correct += 1
        else:
            if pred and not case["gold"]:
                fps.append(case)
            else:
                fns.append(case)
    elapsed = time.time() - t0
    acc = correct / len(BENCHMARK)
    print(f"\n[{name}] 评测结果:")
    print(f"  准确率: {correct}/{len(BENCHMARK)} ({acc:.1%})")
    print(f"  总耗时: {elapsed:.2f}s (平均每对: {elapsed/len(BENCHMARK)*1000:.1f}ms)")
    if fps:
        print(f"  误报(假阳性 FP，把未完成判成完成) ({len(fps)}个):")
        for x in fps:
            print(f"    - {x['id']}: [{x['req']}] --> [{x['res']}]")
    if fns:
        print(f"  漏报(假阴性 FN，把已完成判成未完成) ({len(fns)}个):")
        for x in fns:
            print(f"    - {x['id']}: [{x['req']}] --> [{x['res']}]")
    return acc

def run():
    print(f"开始运行 20 样本基准评测集 (正例: {sum(1 for x in BENCHMARK if x['gold'])}，负例: {sum(1 for x in BENCHMARK if not x['gold'])})")
    
    # 1. gliner25_stage1_match
    try:
        from gliner2 import AutoExtractor, Schema
        ext = AutoExtractor.from_pretrained("models/gliner25_stage1_match/final", map_location="cpu")
        schema = Schema().classification("推进量", labels=["WHOLE_CLOSE", "PART_CLOSE", "NO_CLOSE"])
        def gliner_pred(req, res):
            r = ext.batch_extract([f"【义务】{req}\n【完成相关句】{res}"], schema)[0]
            lab = (r or {}).get("推进量", "NO_CLOSE")
            return lab in ("WHOLE_CLOSE", "PART_CLOSE")
        eval_model("gliner25_stage1_match (287M)", gliner_pred)
    except Exception as e:
        print("gliner eval error:", e)

    # 2. Qwen2.5-0.5B
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        m05 = [d for d in os.listdir(cache_dir) if "Qwen2.5-0.5B-Instruct" in d][0]
        p05 = os.path.join(cache_dir, m05, "snapshots", os.listdir(os.path.join(cache_dir, m05, "snapshots"))[0])
        tok05 = AutoTokenizer.from_pretrained(p05)
        mdl05 = AutoModelForCausalLM.from_pretrained(p05, torch_dtype=torch.float32, device_map="cpu")
        mdl05.eval()
        
        def qwen05_pred(req, res):
            prompt = (
                "<|im_start|>system\n你是一个严格的任务闭合模式匹配器。判断【助手动作】是否真正解决或完成了【用户请求】。"
                "如果完全匹配并解决输出'YES'，如果任务错配、正在进行、报错失败或未完成输出'NO'。<|im_end|>\n"
                f"<|im_start|>user\n用户请求: {req}\n助手动作: {res}\n是否解决？(YES/NO)<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inp = tok05(prompt, return_tensors="pt")
            with torch.no_grad():
                out = mdl05.generate(**inp, max_new_tokens=3, do_sample=False, pad_token_id=tok05.eos_token_id)
            ans = tok05.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            return "YES" in ans
        eval_model("Qwen2.5-0.5B-Instruct (490M 零样本)", qwen05_pred)
    except Exception as e:
        print("qwen05 error:", e)

    # 3. Qwen2.5-1.5B (如果存在)
    try:
        m15 = [d for d in os.listdir(cache_dir) if "Qwen2.5-1.5B-Instruct" in d]
        if m15:
            p15 = os.path.join(cache_dir, m15[0], "snapshots", os.listdir(os.path.join(cache_dir, m15[0], "snapshots"))[0])
            tok15 = AutoTokenizer.from_pretrained(p15)
            mdl15 = AutoModelForCausalLM.from_pretrained(p15, torch_dtype=torch.float32, device_map="cpu")
            mdl15.eval()
            def qwen15_pred(req, res):
                prompt = (
                    "<|im_start|>system\n你是一个严格的任务闭合模式匹配器。判断【助手动作】是否真正解决或完成了【用户请求】。"
                    "如果完全匹配并解决输出'YES'，如果任务错配、正在进行、报错失败或未完成输出'NO'。<|im_end|>\n"
                    f"<|im_start|>user\n用户请求: {req}\n助手动作: {res}\n是否解决？(YES/NO)<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                inp = tok15(prompt, return_tensors="pt")
                with torch.no_grad():
                    out = mdl15.generate(**inp, max_new_tokens=3, do_sample=False, pad_token_id=tok15.eos_token_id)
                ans = tok15.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
                return "YES" in ans
            eval_model("Qwen2.5-1.5B-Instruct (1.5B 零样本)", qwen15_pred)
    except Exception as e:
        print("qwen15 error:", e)

if __name__ == "__main__":
    run()

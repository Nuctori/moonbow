# -*- coding: utf-8 -*-
"""pipeline/research_v233_alignment_eval.py

客体对齐重训版的诚实评测（预注册协议，2026-09-19 深夜）。

协议（在查看任何测试集数字之前冻结）：
1. 三臂：BASE（未微调基座）/ FT_HELDOUT（models/minilm_alignment_heldout_v1，
   组感知切分重训）/ FT_CONTAM（旧污染检查点 models/minilm_alignment_ft，仅作对照，
   tokenizer 借用基座）。
2. 阈值：每臂在验证集（4 个完整模板族，从未见训练）上选 sim-only 最优 τ，
   选定后冻结，再接触任何测试数据。
3. 端点 A（主要端点，站 2 单独）：pos vs cross-topic 负例的对象判别。
   测试集为 37 对全新手写对，工程域与训练模板域、20 基准域均不相交，
   并在脚本内做归一化字符 Jaccard 重叠自检（≥0.75 即告警终止）。
4. 端点 B（完整管线 guardCLOSE ∧ sim≥τ）：全部 67 对 + 旧 20 基准（其模板域
   已不在新训练集中，故该基准对本臂恢复合法 held-out 地位）。
5. 预注册成功判据：FT_HELDOUT 相对 BASE，端点 A 准确率点估 +5pp 以上
   且 pos/neg 相似度间隔 +0.03 以上，方可声称"泛化增益"；否则如实报"无证据"。
6. 统计：Wilson 95% CI；报告 BASE↔FT_HELDOUT 逐对翻转数。
7. 诚实性声明：测试集为单人标注的合成场景，只度量"模板域间泛化"，
   不外推到真实会话表现；真实会话需另行人工裁决回放（MATCH_AUDIT 范式）。

运行：.venv/Scripts/python.exe pipeline/research_v233_alignment_eval.py
"""
import os, sys, json, math, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SPLIT_PATH = "maps/alignment_heldout_split.json"
TRAIN_PATH = "maps/contrastive_strictly_heldout_train.json"
NEW_FT = "models/minilm_alignment_heldout_v1"
OLD_FT = "models/minilm_alignment_ft"
RESULTS_PATH = "maps/alignment_heldout_eval_results.json"
JACCARD_BLOCK = 0.75

# ── 测试集：pos（真闭合）与 cross（跨话题假完成，含完成话术）为主端点；
#    partial/in-progress/fail 为端点 B 的语用负例（站 1 责任区）。
#    域选取避开：20 基准域（端口/pydantic/单测/README/拼写/push/alembic/MySQL/
#    JWT/Docker/Excel/REST 批删/CPU dump/GitHub Actions/微信支付/py3.12/回归/
#    cargo/HF下载/圈复杂度/console.log）与训练模板域（Redis/lru_cache/S3/ES/
#    Kafka/RabbitMQ/Celery/TLS/IP白名单/XSS/WS反代/Prometheus/Grafana/ulimit/
#    虚拟滚动/Webpack/触摸滑动/WASM/读写锁/pprof）。
EVAL_POS = [
    ("e01", "用 FFmpeg 把宣传视频转码成 H.265 编码", "已经用 ffmpeg -c:v libx265 完成转码，体积从 80MB 压到 32MB，画质检查通过。"),
    ("e02", "给证件照批量加水印", "脚本已完成，200 张照片全部加上半透明水印并导出到 output 目录。"),
    ("e03", "把每日报表任务搬上 Airflow 定时调度", "DAG 已部署到调度器，今晨 6 点首次跑批成功，报表已生成。"),
    ("e04", "订单列表接口改成 GraphQL 游标分页", "已在 schema 中新增 cursor 分页参数，压测下深度翻页不再超时。"),
    ("e05", "给无状态服务配置自动扩缩容", "HPA 策略已生效，CPU 超过 70% 时副本从 2 扩到 5，观察一小时运行平稳。"),
    ("e06", "修复应用在上手引导页的偶发崩溃", "崩溃栈已定位到空指针并修复，新包在 3 台测试机上各跑 50 次引导无复现。"),
    ("e07", "清理 PostgreSQL 里膨胀的旧索引", "已删除 6 个冗余索引并执行 vacuum full，表体积从 40GB 降到 12GB。"),
    ("e08", "给日志集合配置 MongoDB 分片", "分片集群已启用，日志集合按时间片分布到 3 个分片，写入延迟恢复正常。"),
    ("e09", "微服务间调用加上超时和重试", "客户端已配置 500ms 超时与 2 次退避重试，联调通过，失败率降到 0.1% 以下。"),
    ("e10", "把测试环境的资源改成 Terraform 管理", "VPC、子网与两台虚机已写进 tf 配置并 apply 成功，状态已入远端存储。"),
    ("e11", "把界面上的中文文案抽成多语言资源", "抽取完成，共 342 条文案迁入 zh-CN.json，切换语言后界面无遗漏。"),
    ("e12", "解决游戏在低端机上的掉帧问题", "合批与 LOD 调整后，低端实测机帧率从 24 提升到 50，录屏已附。"),
    ("e13", "接入 Stripe 退款接口", "退款 API 已联调通过，测试交易全额退款成功，回调状态正确入库。"),
    ("e14", "列表加载时加上骨架屏", "骨架屏组件已接入，弱网下首屏体验明显改善，设计评审已通过。"),
    ("e15", "服务器每晚自动备份数据库", "crontab 已配置凌晨 2 点全量备份并保留 7 天，昨晚首次备份验证可恢复。"),
    ("e16", "详情页支持 Markdown 数学公式渲染", "已集成 KaTeX，含公式的 20 篇文档渲染正确，无脚本报错。"),
    ("e17", "给移动端 SDK 增加崩溃时自动上传日志", "崩溃捕获钩子已加入，测试触发崩溃后日志完整上传到后台。"),
    ("e18", "把数仓每日增量从 CSV 改为 Parquet", "转换管道上线，昨日增量已为 Parquet 格式，查询耗时下降 60%。"),
    ("e19", "给仓库接入 ESLint 并修复存量告警", "配置已提交，存量 87 条告警清零，合并门禁已生效。"),
    ("e20", "网关下游超时从 30 秒降到 5 秒", "配置已灰度发布到全部网关节点，监控确认生效，长尾请求快速失败。"),
    ("e21", "给静态资源套上 CDN 加速", "域名已切换到 CDN，全国 12 个探测点首字节时间中位数下降 45%。"),
    ("e22", "给第三方回调增加签名校验", "回调入口已验证 HMAC 签名，伪造请求被拒并记录告警日志。"),
]
EVAL_CROSS = [
    ("n01", "用 FFmpeg 把宣传视频转码成 H.265 编码", "官网首页的轮播图已换成新视觉稿，上线检查通过。"),
    ("n02", "给证件照批量加水印", "已经把开发环境的 Node 升级到 20 LTS，全组生效。"),
    ("n03", "把每日报表任务搬上 Airflow 定时调度", "报表邮件模板的字体样式已统一，视觉走查完成。"),
    ("n04", "订单列表接口改成 GraphQL 游标分页", "订单确认页的优惠说明文案已更新并发布。"),
    ("n05", "给无状态服务配置自动扩缩容", "负载均衡的健康检查间隔已从 10 秒调到 5 秒，配置已生效。"),
    ("n06", "修复应用在上手引导页的偶发崩溃", "应用图标已按新规范重新切图并替换完毕。"),
    ("n07", "清理 PostgreSQL 里膨胀的旧索引", "数据库连接池的监控面板已完成汉化。"),
    ("n08", "微服务间调用加上超时和重试", "服务间调用的链路追踪标识已在日志中打印，验收通过。"),
    ("n09", "把测试环境的资源改成 Terraform 管理", "测试环境的域名证书续期已完成，有效期到明年。"),
    ("n10", "把界面上的中文文案抽成多语言资源", "默认字体已换成思源黑体并全局生效。"),
    ("n11", "解决游戏在低端机上的掉帧问题", "新手引导的对话文案已润色完毕并合入主分支。"),
    ("n12", "接入 Stripe 退款接口", "退款页面的插画已更换为新版本并上线。"),
    ("n13", "给静态资源套上 CDN 加速", "静态资源的命名规范文档已整理并同步到 wiki。"),
    ("n14", "给第三方回调增加签名校验", "回调地址的后台配置界面已重构完成。"),
    ("n15", "服务器每晚自动备份数据库", "数据库字段的注释已补全，评审通过。"),
]
EVAL_PARTIAL = [
    ("p01", "把界面上的中文文案抽成多语言资源", "设置页的文案已抽取完成，其余 11 个模块还在进行中。"),
    ("p02", "给仓库接入 ESLint 并修复存量告警", "ESLint 已接好，存量告警还剩 60 条未修。"),
    ("p03", "把数仓每日增量从 CSV 改为 Parquet", "订单事实表已切到 Parquet，维度表还是 CSV，明日继续。"),
    ("p04", "修复应用在上手引导页的偶发崩溃", "引导页第 1、2 步的崩溃已修，第 3 步的崩溃还在定位。"),
    ("p05", "给微服务间调用加上超时和重试", "订单服务已加上，库存和支付服务还没改。"),
    ("p06", "详情页支持 Markdown 数学公式渲染", "行内公式已支持，块级公式渲染还报错。"),
    ("p07", "清理 PostgreSQL 里膨胀的旧索引", "已删除 2 个索引，还剩 4 个待处理。"),
    ("p08", "给移动端 SDK 增加崩溃时自动上传日志", "Android 端已完成，iOS 端尚未开始。"),
    ("p09", "把测试环境的资源改成 Terraform 管理", "虚机已纳入管理，VPC 与子网仍在手工控制台维护。"),
    ("p10", "给静态资源套上 CDN 加速", "图片域名已接入，JS 与 CSS 域名还没切换。"),
]
EVAL_WAIT = [
    ("w01", "用 FFmpeg 把宣传视频转码成 H.265 编码", "正在对比 CRF 参数，晚点跑一版看画质。"),
    ("w02", "订单列表接口改成 GraphQL 游标分页", "方案已定，我先在分支上搭个原型给你看。"),
    ("w03", "给无状态服务配置自动扩缩容", "需要先确认集群版本是否支持该策略版本，稍后给你结论。"),
    ("w04", "给证件照批量加水印", "脚本写了一半，正在处理透明 PNG 的边界情况。"),
    ("w05", "把每日报表任务搬上 Airflow 定时调度", "调度器权限还没批下来，我先手动触发了一次做验证。"),
    ("w06", "解决游戏在低端机上的掉帧问题", "已画出帧率曲线，正在逐段分析掉帧的卡顿点。"),
    ("w07", "接入 Stripe 退款接口", "先把沙箱密钥发我一下，配置好后开始联调。"),
    ("w08", "服务器每晚自动备份数据库", "备份脚本正在写，今晚先手动跑一次试试。"),
    ("w09", "给仓库接入 ESLint 并修复存量告警", "准备采用新版 flat config，正在迁移旧配置文件。"),
    ("w10", "给静态资源套上 CDN 加速", "正在等云厂商开通国内加速资质。"),
]
EVAL_FAIL = [
    ("f01", "用 FFmpeg 把宣传视频转码成 H.265 编码", "转码进行到 30% 时进程段错误退出，重试两次都失败。"),
    ("f02", "把数仓每日增量从 CSV 改为 Parquet", "写入 Parquet 时字段类型不兼容报错，任务失败回滚了。"),
    ("f03", "给无状态服务配置自动扩缩容", "集群版本过低不支持该策略，配不上，需要先升级集群。"),
    ("f04", "详情页支持 Markdown 数学公式渲染", "公式库与现有渲染管线冲突，页面样式被打乱，已先回退。"),
    ("f05", "接入 Stripe 退款接口", "退款请求被风控拦截，报权限不足错误，无法继续。"),
    ("f06", "给移动端 SDK 增加崩溃时自动上传日志", "在 Android 12 上触发崩溃时上传服务自身也崩溃，方案暂不可行。"),
    ("f07", "把测试环境的资源改成 Terraform 管理", "导入进行到一半状态文件损坏，资源卡在托管状态，导入失败。"),
    ("f08", "给证件照批量加水印", "批量脚本内存溢出，处理到第 80 张就崩了。"),
    ("f09", "微服务间调用加上超时和重试", "重试引发了重复下单，已回滚重试逻辑，方案需要重新设计。"),
    ("f10", "把界面上的中文文案抽成多语言资源", "抽取脚本把带变量的文案弄乱了，部分内容需要手工恢复。"),
]


def norm(s):
    import re
    return re.sub(r"[\s，。,.:：;；!！?？'\"“”‘’()（）\[\]【】]", "", s.lower())


def char_jaccard(a, b):
    A, B = set(norm(a)), set(norm(b))
    return len(A & B) / max(1, len(A | B))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pick_threshold(pairs, sims, labels):
    """验证集上选 sim-only 最优阈值：准确率最大；平局取区间中点。"""
    grid = [i * 0.005 + 0.05 for i in range(151)]
    best_tau, best_acc, best_range = 0.35, -1, None
    for tau in grid:
        acc = sum(((s >= tau) == l) for s, l in zip(sims, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_tau = acc, tau
            best_range = [tau, tau]
        elif acc == best_acc:
            best_range[1] = tau
            best_tau = (best_range[0] + best_range[1]) / 2
    return best_tau, best_acc


def main():
    device = "cpu"
    torch.manual_seed(0)

    # ── 载入验证对（组感知切分的 val 侧）
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    triplets = json.load(open(TRAIN_PATH, encoding="utf-8"))["triplets"]
    val_t = [triplets[i] for i in split["val_idx"]]
    train_t = [triplets[i] for i in split["train_idx"]]
    val_pairs = []
    for t in val_t:
        val_pairs.append((t["anchor"], t["pos"], 1))
        val_pairs.append((t["anchor"], t["neg"], 0))
    print(f"验证对: {len(val_pairs)}（来自 {len(val_t)} 个留出三元组）")

    # ── 重叠自检：测试文本 vs 训练集任何字段
    print("\n[重叠自检] 测试集 vs 训练集（归一化字符 Jaccard）")
    worst = []
    for bid, req, res in EVAL_POS + EVAL_CROSS + EVAL_PARTIAL + EVAL_WAIT + EVAL_FAIL:
        m = 0.0
        for t in train_t:
            for field in ("anchor", "pos", "neg"):
                m = max(m, char_jaccard(req, t[field]), char_jaccard(res, t[field]))
        worst.append((m, bid))
    worst.sort(reverse=True)
    print("  最高的 5 项:", [(b, round(m, 3)) for m, b in worst[:5]])
    if worst and worst[0][0] >= JACCARD_BLOCK:
        print(f"  ❌ 检出重叠 ≥{JACCARD_BLOCK}，终止评测，请先修订测试集")
        return
    print("  ✅ 全部测试文本与训练集重合 < 0.75")

    # ── 载入三臂
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    m_dirs = [d for d in os.listdir(cache_dir) if "MiniLM-L12-v2" in d]
    snap = os.path.join(cache_dir, m_dirs[0], "snapshots", os.listdir(os.path.join(cache_dir, m_dirs[0], "snapshots"))[0])
    tok = AutoTokenizer.from_pretrained(snap)
    arms = {"BASE": AutoModel.from_pretrained(snap).to(device).eval()}
    arms["FT_HELDOUT"] = AutoModel.from_pretrained(NEW_FT).to(device).eval()
    arms["FT_CONTAM(对照)"] = AutoModel.from_pretrained(OLD_FT).to(device).eval()

    def emb(mdl, text):
        inp = tok(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            e = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            return F.normalize(e, p=2, dim=1)

    def sim(mdl, a, b):
        return F.cosine_similarity(emb(mdl, a), emb(mdl, b)).item()

    results = {"protocol": "预注册见脚本头注；测试集域与训练域不相交，重叠自检通过",
               "n_val_pairs": len(val_pairs), "arms": {}}

    # ── 每臂：验证集选 τ（冻结）→ 端点 A → 端点 B
    endpointA = [(bid, req, res, 1) for bid, req, res in EVAL_POS] + \
                [(bid, req, res, 0) for bid, req, res in EVAL_CROSS]
    endpointB_all = [(bid, req, res, 1) for bid, req, res in EVAL_POS] + \
                    [(b, r, s, 0) for b, r, s in EVAL_CROSS + EVAL_PARTIAL + EVAL_WAIT + EVAL_FAIL]

    # guard 只依赖 res，三臂共用，算一次
    print("\n加载 guard（端点 B 用）...")
    from gliner2 import AutoExtractor, Schema
    guard = AutoExtractor.from_pretrained("models/progress_guard_v3/final", map_location=device)
    gsc = Schema().classification("进度守卫4", labels=["REQUEST", "INTENT", "CLOSE", "NEUTRAL"])
    guard_close = {}
    for bid, _, res, _g in endpointB_all:
        guard_close[bid] = guard.batch_extract([res], gsc)[0].get("进度守卫4", "NEUTRAL") == "CLOSE"
    sys.path.insert(0, "pipeline")
    from eval_semantic_matcher_benchmark import BENCHMARK
    for c in BENCHMARK:
        guard_close[c["id"]] = guard.batch_extract([c["res"]], gsc)[0].get("进度守卫4", "NEUTRAL") == "CLOSE"

    for name, mdl in arms.items():
        print(f"\n===== 臂: {name} =====")
        val_sims = [sim(mdl, a, b) for a, b, _ in val_pairs]
        val_labels = [l for _, _, l in val_pairs]
        tau, val_acc = pick_threshold(val_pairs, val_sims, val_labels)
        print(f"  验证集选 τ*={tau:.3f}（val acc={val_acc:.3f}，已冻结）")

        a_sims = [sim(mdl, r, s) for _, r, s, _l in endpointA]
        a_labels = [l for _, _, _, l in endpointA]
        a_pred = [s >= tau for s in a_sims]
        a_correct = sum(p == l for p, l in zip(a_pred, a_labels))
        pos_s = [s for s, l in zip(a_sims, a_labels) if l]
        neg_s = [s for s, l in zip(a_sims, a_labels) if not l]
        lo, hi = wilson(a_correct, len(endpointA))
        print(f"  端点A(pos {len(pos_s)} vs cross {len(neg_s)}): {a_correct}/{len(endpointA)}"
              f" = {a_correct/len(endpointA):.1%}  Wilson95%[{lo:.3f},{hi:.3f}]")
        print(f"    pos均sim={sum(pos_s)/len(pos_s):.4f} neg均sim={sum(neg_s)/len(neg_s):.4f}"
              f" margin={sum(pos_s)/len(pos_s)-sum(neg_s)/len(neg_s):+.4f}")
        fp = [(endpointA[i][0], round(a_sims[i], 3)) for i in range(len(a_sims)) if a_pred[i] and not a_labels[i]]
        fn = [(endpointA[i][0], round(a_sims[i], 3)) for i in range(len(a_sims)) if not a_pred[i] and a_labels[i]]
        print(f"    FP: {fp}\n    FN: {fn}")

        b_pred, b_gold = [], []
        for bid, req, res, g in endpointB_all:
            b_pred.append((sim(mdl, req, res) >= tau) and guard_close[bid])
            b_gold.append(g)
        b_correct = sum(p == g for p, g in zip(b_pred, b_gold))
        lo2, hi2 = wilson(b_correct, len(b_gold))
        print(f"  端点B(全67对, guard∧sim≥τ): {b_correct}/{len(b_gold)} = {b_correct/len(b_gold):.1%}"
              f"  Wilson95%[{lo2:.3f},{hi2:.3f}]")
        fpb = [endpointB_all[i][0] for i in range(len(b_gold)) if b_pred[i] and not b_gold[i]]
        fnb = [endpointB_all[i][0] for i in range(len(b_gold)) if not b_pred[i] and b_gold[i]]
        print(f"    FP: {fpb}\n    FN: {fnb}")

        m_sims = [sim(mdl, c["req"], c["res"]) for c in BENCHMARK]
        m_pred = [(s >= 0.65) or (guard_close[c["id"]] and s >= tau) for s, c in zip(m_sims, BENCHMARK)]
        m_correct = sum(p == c["gold"] for p, c in zip(m_pred, BENCHMARK))
        print(f"  旧20基准(guard∧sim≥τ / 强绑定0.65): {m_correct}/20")

        results["arms"][name] = {
            "tau": round(tau, 3), "val_acc": round(val_acc, 4),
            "A": {"n": len(endpointA), "correct": a_correct,
                  "acc": round(a_correct / len(endpointA), 4),
                  "wilson": [round(lo, 3), round(hi, 3)],
                  "pos_mean_sim": round(sum(pos_s) / len(pos_s), 4),
                  "neg_mean_sim": round(sum(neg_s) / len(neg_s), 4),
                  "fp": fp, "fn": fn},
            "B": {"n": len(b_gold), "correct": b_correct,
                  "acc": round(b_correct / len(b_gold), 4),
                  "wilson": [round(lo2, 3), round(hi2, 3)], "fp": fpb, "fn": fnb},
            "bench20": {"correct": m_correct},
            "bench20_sims": {c["id"]: round(s, 4) for c, s in zip(BENCHMARK, m_sims)},
        }

    # ── BASE ↔ FT_HELDOUT 翻转对（端点 A）
    b_arm, f_arm = results["arms"]["BASE"], results["arms"]["FT_HELDOUT"]
    results["pre_registered_success"] = {
        "rule": "端点A acc 点估 +5pp 以上 且 margin +0.03 以上",
        "acc_delta_pp": round((f_arm["A"]["acc"] - b_arm["A"]["acc"]) * 100, 1),
        "margin_delta": round((f_arm["A"]["pos_mean_sim"] - f_arm["A"]["neg_mean_sim"])
                              - (b_arm["A"]["pos_mean_sim"] - b_arm["A"]["neg_mean_sim"]), 4),
    }
    json.dump(results, open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {RESULTS_PATH}")


if __name__ == "__main__":
    main()

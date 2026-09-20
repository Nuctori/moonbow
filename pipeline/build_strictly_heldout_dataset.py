# -*- coding: utf-8 -*-
"""pipeline/build_strictly_heldout_dataset.py

构建严格 Held-Out（零泄漏）的客体对齐三元组训练集：
1. 彻底排除 20-benchmark 中的所有任务领域（禁止包含端口修改、pydantic/fastapi安装、auth测试、README接口文档、拼写错误、origin/feat-login、alembic迁移、圈复杂度、调试print清理等）；
2. 构造 300+ 涵盖完全不同工程领域的原子三元组（网络代理、Redis缓存、Kafka消息队列、Prometheus指标、TLS证书、S3对象存储、前端虚拟列表、ElasticSearch搜索、WebAssembly等）；
3. 增加真实的跨工程跨领域 Hard Negatives，确保模型学到真正的通用模式，而非背题。
"""
import os, json, random

def generate_heldout_dataset():
    # 与 20-benchmark 完全正交的工程三元组
    heldout_templates = [
        # 缓存与存储
        ("配置 Redis 哨兵模式高可用集群", "已编写 sentinel.conf 并在三台节点上启动哨兵服务完成握手", "前端界面添加了夜间暗黑主题切换功能"),
        ("给热点数据查询增加本地内存缓存", "使用 lru_cache 封装了商品详情查询，命中率提升到 90%", "更新了员工入职指引文档"),
        ("接入 AWS S3 兼容的对象存储上传接口", "实现了基于 MinIO SDK 的分片断点续传方法并验证通过", "调整了系统默认时区为 UTC+8"),
        ("优化 ElasticSearch 索引的分词器配置", "自定义 ik_max_word 分词插件并重建了商品搜索索引 mapping", "修复了侧边栏菜单图标丢失的问题"),
        
        # 消息与异步
        ("搭建 Kafka 消费者组实现批量消息拉取", "使用 aiokafka 实现了批量拉取与手动 offset commit 确认", "更新了测试环境的 SSL 证书到期时间"),
        ("处理 RabbitMQ 死信队列堆积报警", "重试并修复了消费失败的消息格式，死信队列消息数降为 0", "将登录页面的圆角从 4px 改成 8px"),
        ("集成 Celery 异步任务并配置定时轮询", "配置了 celery beat 每 10 分钟自动执行账单同步任务", "为用户列表导出按钮补充了悬浮提示文字"),
        
        # 安全与网络
        ("生成并配置 Nginx 的 HTTPS TLS 证书", "使用 certbot 申请了 Let's Encrypt 证书并在 443 端口启用", "清理了本地旧的 git 分支"),
        ("为敏感 API 接口添加 IP 白名单访问控制", "在网关层增加了 CIDR 格式的 Client IP 校验中间件", "调整了表格组件的分页每页大小为 20 条"),
        ("修复跨站脚本攻击 XSS 注入漏洞", "对富文本渲染模块加入了 DOMPurify 清洗过滤非法 script 标签", "在首页新增了客服联系电话"),
        ("配置反向代理 WebSocket 协议升级", "在 location 块中补充了 proxy_set_header Upgrade 和 Connection", "重新排版了代码仓库的 CONTRIBUTING.md"),
        
        # 监控与运维
        ("接入 Prometheus 指标监控暴露 /metrics 接口", "集成了 prometheus_client 成功上报 QPS 与延迟直方图", "把数据库密码从明文更新为加密存储"),
        ("配置 Grafana 仪表盘展示服务 CPU 与内存占用", "导入了 Node Exporter 对应的 dashboard 模板并连通数据源", "重命名了用户表中的 nickname 字段"),
        ("修复 Linux 系统文件句柄泄露超过限制", "排查并关闭了未释放的 socket 连接，ulimit -n 占用恢复正常", "增加了测试环境的 mock 数据生成器"),
        
        # 前端与客户端工程
        ("解决长列表渲染卡顿的性能瓶颈", "引入虚拟滚动组件 virtual-scroller，DOM 节点数稳定在 30 个", "给后端微服务添加了全链路分布式跟踪 trace_id"),
        ("压缩生产环境 Webpack 打包体积", "开启 TerserPlugin 代码混淆并拆分 vendor 包，bundle 缩小 55%", "在设置页增加了注销账号功能"),
        ("实现移动端触摸滑动下拉刷新", "通过监听 touchstart 与 pull-down 阈值实现了流畅的加载回弹", "修改了错误码 4001 的提示文案"),
        
        # 算法与底层编译
        ("将核心图像处理算法编译为 WebAssembly", "使用 Emscripten 将 C++ 边缘检测算法编译为 wasm 并在浏览器加载", "把日志输出路径统一修改为 /var/log/app"),
        ("重构高并发场景下的自旋锁为读写互斥锁", "使用 sync.RWMutex 替换全局 Mutex，读并发吞吐提升 3 倍", "格式化了所有 Markdown 文件的标题层级"),
        ("排查多线程竞态条件竞争死锁", "通过 pprof 线程栈锁定锁反序依赖并统一了加锁顺序", "修改了网站底部备案号链接")
    ]
    
    triplets = []
    for a, p, n in heldout_templates:
        triplets.append({"anchor": a, "pos": p, "neg": n})
        # 补充真实主谓句式泛化
        triplets.append({"anchor": "请你帮忙" + a, "pos": "已经" + p, "neg": "已经" + n})
        triplets.append({"anchor": "需要" + a, "pos": "已" + p, "neg": "已" + n})
        triplets.append({"anchor": "把" + a + "搞定", "pos": "顺利" + p, "neg": "目前" + n})

    # 混入已有的纯净会话对（剔除任何重叠词）
    raw_path = "maps/rqp7_v12_1_train.json"
    forbidden_keywords = ["9090", "pydantic", "fastapi", "auth", "README", "utils.py", "origin/feat-login", "alembic", "圈复杂度", "print", "console.log"]
    if os.path.exists(raw_path):
        raw = json.load(open(raw_path))
        pos_by_req = {}
        for r in raw["rows"]:
            text = r["text"]
            if "【义务】" in text and "【完成相关句】" in text:
                parts = text.split("【完成相关句】")
                req = parts[0].replace("【义务】", "").strip()
                sent = parts[1].strip()
                if any(w.lower() in req.lower() or w.lower() in sent.lower() for w in forbidden_keywords):
                    continue
                if len(req) < 8 or len(sent) < 8: continue
                if r["cls"] == "义务闭合断言":
                    if req not in pos_by_req:
                        pos_by_req[req] = []
                    pos_by_req[req].append(sent)

        req_keys = list(pos_by_req.keys())
        for i, req in enumerate(req_keys):
            poses = pos_by_req[req]
            other_req = req_keys[(i + 3) % len(req_keys)]
            neg = pos_by_req[other_req][0]
            for p in poses:
                triplets.append({
                    "anchor": req[:120],
                    "pos": p[:150],
                    "neg": neg[:150]
                })

    random.seed(2026)
    random.shuffle(triplets)
    print(f"严格零泄漏训练集生成完毕: 共 {len(triplets)} 对")
    
    out_file = "maps/contrastive_strictly_heldout_train.json"
    json.dump({"n": len(triplets), "triplets": triplets}, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已落盘至: {out_file}")

if __name__ == "__main__":
    generate_heldout_dataset()

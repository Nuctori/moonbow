# -*- coding: utf-8 -*-
"""experiments/task_structure/build_eval_set.py

构建 Task Structure Advisor 暂定评测集（v0，与 docs/task_structure_spec.md 同时冻结）。

诚实声明：
- 样本为 agent 生成 + agent 自审，**无人工复核**，属暂定标注集，
  不构成人工金标准；人际一致性未测量。
- 家族 `al`（词典外对抗）与部分 `ds`（字面简单实际难）样本故意使用
  词典未覆盖的表达，规则基线应在这些样本上失分——这是测量目标，
  不是缺陷。
- dev/test 切分由 md5(id) 确定性决定；规则迭代只允许看 dev，
  test 指标只允许在最终报告跑一次（含 post-fix 对比时须如实标注）。

用法：python build_eval_set.py           # 生成 eval_set.jsonl
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_set.jsonl")

# ---------------------------------------------------------------------------
# 语料：(family, text, gold_captures[(kind, quote)], unknowns, abstain, level, notes)
# 约束：quote 必须是 text 的 verbatim 子串（构建时断言）。
# ---------------------------------------------------------------------------

C = []

# ---- sg 单目标明确（zh，low）----
C += [
    ("sg", "修复登录页面的超时问题。",
     [("goal", "修复登录页面的超时问题")], [], False, "low", ""),
    ("sg", "在 README 里补充安装步骤。",
     [("goal", "在 README 里补充安装步骤"), ("object", "README")], [], False, "low", ""),
    ("sg", "删除废弃的 `legacy_api.py`。",
     [("goal", "删除废弃的 `legacy_api.py`"), ("object", "legacy_api.py")],
     [], False, "low", ""),
    ("sg", "把 Jenkins 的构建超时延长到 30 分钟。",
     [("goal", "把 Jenkins 的构建超时延长到 30 分钟")], [], False, "low",
     "词典外动词（延长），规则基线预计失分"),
    ("sg", "给用户表加一列生日。",
     [("goal", "给用户表加一列生日")], [], False, "low",
     "词典外动词（加），规则基线预计失分"),
    ("sg", "把首页的加载骨架屏去掉。",
     [("goal", "把首页的加载骨架屏去掉")], [], False, "low",
     "词典外动词（去掉），规则基线预计失分"),
    ("sg", "为订单列表增加分页功能。",
     [("goal", "为订单列表增加分页功能")], [], False, "low", ""),
    ("sg", "把默认端口从 8080 改成 9000。",
     [("goal", "把默认端口从 8080 改成 9000")], [], False, "low", ""),
    ("sg", "更新依赖清单里过期的版本号。",
     [("goal", "更新依赖清单里过期的版本号")], [], False, "low", ""),
    ("sg", "重建失败的通知队列索引。",
     [("goal", "重建失败的通知队列索引")], [], False, "low",
     "词典外动词（重建），规则基线预计失分"),
    ("sg", "给配置面板加上键盘导航支持。",
     [("goal", "给配置面板加上键盘导航支持")], [], False, "low",
     "词典外动词（加上），规则基线预计失分"),
    ("sg", "把日志输出目录迁到独立的数据盘。",
     [("goal", "把日志输出目录迁到独立的数据盘")], [], False, "low", ""),
]

# ---- sge 单目标明确（en，low）----
C += [
    ("sge", "Fix the pagination bug on the orders page.",
     [("goal", "Fix the pagination bug on the orders page")], [], False, "low", ""),
    ("sge", "Add retry logic to the webhook dispatcher.",
     [("goal", "Add retry logic to the webhook dispatcher")], [], False, "low", ""),
    ("sge", "Update the README with the new install steps.",
     [("goal", "Update the README with the new install steps"), ("object", "README")],
     [], False, "low", ""),
    ("sge", "Remove the deprecated `renderAll()` helper.",
     [("goal", "Remove the deprecated `renderAll()` helper"), ("object", "renderAll()")],
     [], False, "low", ""),
    ("sge", "Bump the request timeout to 30 seconds.",
     [("goal", "Bump the request timeout to 30 seconds")], [], False, "low",
     "词典外动词（Bump），规则基线预计失分"),
    ("sge", "Rename `getUser` to `fetchUser` across the package.",
     [("goal", "Rename `getUser` to `fetchUser` across the package"),
      ("object", "getUser"), ("object", "fetchUser")], [], False, "low", ""),
    ("sge", "Enable gzip compression for the static assets.",
     [("goal", "Enable gzip compression for the static assets")], [], False, "low", ""),
    ("sge", "Move the cache directory to the data disk.",
     [("goal", "Move the cache directory to the data disk")], [], False, "low",
     "词典外动词（Move），规则基线预计失分"),
]

# ---- mg 多义务 ----
C += [
    ("mg", "修复登录接口，保持旧客户端兼容，增加限流，并补充测试和文档。",
     [("goal", "修复登录接口"), ("constraint", "保持旧客户端兼容"),
      ("goal", "增加限流"), ("goal", "补充测试和文档")], [], False, "high", ""),
    ("mg", "迁移数据库到 PostgreSQL，更新 ORM 配置，编写回滚脚本。",
     [("goal", "迁移数据库到 PostgreSQL"), ("goal", "更新 ORM 配置"),
      ("goal", "编写回滚脚本")], [], False, "high", ""),
    ("mg", "实现导出 CSV 功能，实现导出 JSON 功能，并在设置页加入口。",
     [("goal", "实现导出 CSV 功能"), ("goal", "实现导出 JSON 功能"),
      ("goal", "并在设置页加入口")], [], False, "high", ""),
    ("mg", "支持微信扫码登录，同时保留账号密码登录。",
     [("goal", "支持微信扫码登录"), ("coordination", "同时保留账号密码登录")],
     [], False, "medium", "旧方式保留视为协调性要求"),
    ("mg", "添加深色模式主题，添加主题切换按钮，记住用户的选择。",
     [("goal", "添加深色模式主题"), ("goal", "添加主题切换按钮"),
      ("goal", "记住用户的选择")], [], False, "high", ""),
    ("mg", "重构配置加载模块，支持热更新，补充单元测试。",
     [("goal", "重构配置加载模块"), ("goal", "支持热更新"),
      ("goal", "补充单元测试")], [], False, "high", ""),
    ("mg", "优化列表查询速度，顺便把分页组件升级到 v2。",
     [("goal", "优化列表查询速度"), ("goal", "把分页组件升级到 v2")],
     [], False, "medium", ""),
    ("mg", "接入新的短信服务商，更新计费逻辑。",
     [("goal", "接入新的短信服务商"), ("goal", "更新计费逻辑")],
     [], False, "medium", ""),
    ("mg", "清理未使用的依赖，修复 CI 的 flaky 用例。",
     [("goal", "清理未使用的依赖"), ("goal", "修复 CI 的 flaky 用例")],
     [], False, "medium", ""),
    ("mg", "支持导入以及导出。",
     [("goal", "支持导入"), ("goal", "导出")], [], False, "medium",
     "承前并列（规范 3.4）"),
    ("mg", "Add a health endpoint, wire it into the load balancer, and document it.",
     [("goal", "Add a health endpoint"), ("goal", "wire it into the load balancer"),
      ("goal", "and document it")], [], False, "high", ""),
    ("mg", "Split the monolith into two services and add an API gateway.",
     [("goal", "Split the monolith into two services"),
      ("goal", "add an API gateway")], [], False, "medium", ""),
    ("mg", "合并两个配置文件，删除重复字段，更新读取代码。",
     [("goal", "合并两个配置文件"), ("goal", "删除重复字段"),
      ("goal", "更新读取代码")], [], False, "high", ""),
    ("mg", "搭建 staging 环境，部署最新构建，通知 QA 验证。",
     [("goal", "搭建 staging 环境"), ("goal", "部署最新构建"),
      ("goal", "通知 QA 验证")], [], False, "high", ""),
]

# ---- mc 多约束 ----
C += [
    ("mc", "实现搜索功能，必须支持中英文混合输入。",
     [("goal", "实现搜索功能"), ("constraint", "必须支持中英文混合输入")],
     [], False, "low", "1 goal + 1 constraint → 按阈值 low（阈值待验证）"),
    ("mc", "重写导出逻辑，不能修改现有导出接口的签名，且必须保持向后兼容。",
     [("goal", "重写导出逻辑"), ("constraint", "不能修改现有导出接口的签名"),
      ("constraint", "且必须保持向后兼容")], [], False, "medium", ""),
    ("mc", "升级依赖版本，必须保证构建时间不超过 10 分钟。",
     [("goal", "升级依赖版本"), ("constraint", "必须保证构建时间不超过 10 分钟")],
     [], False, "low", ""),
    ("mc", "新增购物车功能，需要在未登录时保持本地存储，登录后自动合并，且不能丢失已有商品。",
     [("goal", "新增购物车功能"), ("constraint", "需要在未登录时保持本地存储"),
      ("constraint", "登录后自动合并"), ("constraint", "且不能丢失已有商品")],
     [], False, "high", "约束合计 3 → high"),
    ("mc", "优化首页加载速度，图片必须懒加载，缓存头不得缺失。",
     [("goal", "优化首页加载速度"), ("constraint", "图片必须懒加载"),
      ("constraint", "缓存头不得缺失")], [], False, "medium", ""),
    ("mc", "支持离线模式，离线时必须缓存草稿，恢复网络后不得重复提交。",
     [("goal", "支持离线模式"), ("constraint", "离线时必须缓存草稿"),
      ("constraint", "恢复网络后不得重复提交")], [], False, "medium", ""),
    ("mc", "迁移到新鉴权服务，旧的 token 在过渡期内必须继续有效。",
     [("goal", "迁移到新鉴权服务"), ("constraint", "旧的 token 在过渡期内必须继续有效")],
     [], False, "low", ""),
    ("mc", "改造通知系统，不能阻塞主流程，失败必须静默重试，日志里不得出现用户内容。",
     [("goal", "改造通知系统"), ("constraint", "不能阻塞主流程"),
      ("constraint", "失败必须静默重试"), ("constraint", "日志里不得出现用户内容")],
     [], False, "high", ""),
    ("mc", "压缩静态资源，同时保证图片清晰度不明显下降。",
     [("goal", "压缩静态资源"), ("constraint", "同时保证图片清晰度不明显下降")],
     [], False, "low", ""),
    ("mc", "Make the dashboard load faster; it must not regress on mobile Safari.",
     [("goal", "Make the dashboard load faster"),
      ("constraint", "it must not regress on mobile Safari")], [], False, "low", ""),
    ("mc", "Rewrite the importer. Keep memory usage below 512MB and never block the UI thread.",
     [("goal", "Rewrite the importer"), ("constraint", "Keep memory usage below 512MB"),
      ("constraint", "never block the UI thread")], [], False, "medium", ""),
    ("mc", "限制 API 限流阈值为一分钟一百次，不得影响内部服务调用。",
     [("goal", "限制 API 限流阈值为一分钟一百次"),
      ("constraint", "不得影响内部服务调用")], [], False, "low", ""),
]

# ---- xd 显式依赖 ----
C += [
    ("xd", "先把数据库迁移跑完，然后再更新 API 层。",
     [("dependency", "先把数据库迁移跑完，然后再更新 API 层")],
     ["dependency"], False, "medium", "跨子句顺序合并（规范 3.7），端点不明"),
    ("xd", "等配置中心就绪后再切换流量。",
     [("dependency", "等配置中心就绪后再切换流量")],
     ["dependency"], False, "medium", ""),
    ("xd", "先完成用户模块的接口改造，订单模块的改造依赖它的输出。",
     [("dependency", "订单模块的改造依赖它的输出"), ("goal", "先完成用户模块的接口改造")],
     ["dependency"], False, "medium", ""),
    ("xd", "生成签名文件之后才能发布安装包。",
     [("dependency", "生成签名文件之后才能发布安装包")],
     ["dependency"], False, "medium", ""),
    ("xd", "数据清洗完成后，再训练新模型，最后对比基线。",
     [("dependency", "数据清洗完成后，再训练新模型"), ("goal", "最后对比基线")],
     ["dependency"], False, "medium", ""),
    ("xd", "Wait for the migration to finish, then redeploy the workers.",
     [("dependency", "Wait for the migration to finish, then redeploy the workers")],
     ["dependency"], False, "medium", ""),
    ("xd", "缓存层基于 `db.py` 的连接池，请先评审 `db.py` 的改动。",
     [("dependency", "缓存层基于 `db.py` 的连接池"), ("object", "db.py"),
      ("goal", "请先评审 `db.py` 的改动")],
     ["dependency"], False, "medium", ""),
    ("xd", "报表服务依赖数仓的每日任务输出，请确认任务完成后再启动报表改造。",
     [("dependency", "报表服务依赖数仓的每日任务输出"), ("goal", "请确认任务完成后再启动报表改造")],
     ["dependency"], False, "medium", ""),
    ("xd", "先升级网关，再灰度新协议。",
     [("dependency", "先升级网关，再灰度新协议")],
     ["dependency"], False, "medium", ""),
    ("xd", "构建镜像之前先把依赖锁文件提交。",
     [("dependency", "构建镜像之前先把依赖锁文件提交")],
     ["dependency"], False, "medium", ""),
    ("xd", "新登录页上线依赖风控接口就绪，请与风控团队确认时间。",
     [("dependency", "新登录页上线依赖风控接口就绪"), ("goal", "请与风控团队确认时间")],
     ["dependency"], False, "medium", ""),
    ("xd", "Once the schema is frozen, generate the ORM models.",
     [("dependency", "Once the schema is frozen, generate the ORM models")],
     ["dependency"], False, "medium", ""),
]

# ---- co 协调 ----
C += [
    ("co", "前后端的错误码要保持一致，同步更新两边的文档。",
     [("coordination", "前后端的错误码要保持一致"), ("coordination", "同步更新两边的文档")],
     [], False, "medium", ""),
    ("co", "三个环境的配置项名称必须统一。",
     [("coordination", "三个环境的配置项名称必须统一")], [], False, "low", ""),
    ("co", "Web 端和小程序的价格展示逻辑保持同步。",
     [("coordination", "Web 端和小程序的价格展示逻辑保持同步")], [], False, "low", ""),
    ("co", "新版协议与旧版协议的响应字段保持对齐，避免客户端解析失败。",
     [("coordination", "新版协议与旧版协议的响应字段保持对齐")],
     [("constraint", "避免客户端解析失败")], False, "medium",
     "第二个子句按 constraint 亦可（标注为 constraint；规则按优先级可能给 coordination）"),
    ("co", "各服务的超时时间统一改成 5 秒。",
     [("coordination", "各服务的超时时间统一改成 5 秒")], [], False, "low", ""),
    ("co", "迁移完成后，仓库根目录的 README 与 docs 保持一致。",
     [("coordination", "README 与 docs 保持一致"), ("goal", "迁移完成后")],
     [], False, "medium", "规则可能把第一子句记 dependency（完成后标记）"),
    ("co", "Keep the CLI flags in sync with the API parameters.",
     [("coordination", "Keep the CLI flags in sync with the API parameters")],
     [], False, "low", ""),
    ("co", "数据库字段与实体类的命名规范要对齐。",
     [("coordination", "数据库字段与实体类的命名规范要对齐")], [], False, "low", ""),
    ("co", "多语言文件的键结构必须一致，缺省语言要同步补全。",
     [("coordination", "多语言文件的键结构必须一致"), ("coordination", "缺省语言要同步补全")],
     [], False, "medium", ""),
    ("co", "更新接口文档，确保示例与实际行为一致。",
     [("goal", "更新接口文档"), ("coordination", "确保示例与实际行为一致")],
     [], False, "medium", ""),
]

# ---- cu 条件/未决 ----
C += [
    ("cu", "如果数据量超过 10GB，就启用分批导出。",
     [("condition", "如果数据量超过 10GB"), ("goal", "就启用分批导出")],
     [], False, "medium", "规则按一子句一捕获可能合并"),
    ("cu", "需要先弄清楚内存泄漏的来源，然后修复它。",
     [("unresolved", "需要先弄清楚内存泄漏的来源"), ("goal", "然后修复它")],
     [], False, "medium", ""),
    ("cu", "必要时启用降级开关。",
     [("condition", "必要时"), ("goal", "必要时启用降级开关")], [], False, "low",
     "条件与义务同句（规范 3.3 各记一条）"),
    ("cu", "具体用哪个方案待定，先搭个实验框架。",
     [("unresolved", "具体用哪个方案待定"), ("goal", "先搭个实验框架")],
     [], False, "medium", ""),
    ("cu", "视压测结果决定是否扩容，压测脚本需要提前准备好。",
     [("condition", "视压测结果决定是否扩容"), ("goal", "压测脚本需要提前准备好")],
     [], False, "medium", ""),
    ("cu", "如果灰度期间错误率上升，立即回滚发布。",
     [("condition", "如果灰度期间错误率上升"), ("goal", "立即回滚发布")],
     [], False, "medium", ""),
    ("cu", "缓存键的命名规则尚不确定，实现时先留扩展点。",
     [("unresolved", "缓存键的命名规则尚不确定"), ("goal", "实现时先留扩展点")],
     [], False, "medium", ""),
    ("cu", "当队列积压超过一万条时发送告警。",
     [("condition", "当队列积压超过一万条时"), ("goal", "发送告警")],
     [], False, "medium", ""),
    ("cu", "If the feature flag is on, use the new pricing service.",
     [("condition", "If the feature flag is on"),
      ("goal", "use the new pricing service")], [], False, "medium", ""),
    ("cu", "是否保留旧版 API 待确认，先写迁移文档。",
     [("unresolved", "是否保留旧版 API 待确认"), ("goal", "先写迁移文档")],
     [], False, "medium", ""),
]

# ---- ac 验收 ----
C += [
    ("ac", "修复排序错误，验收标准是排序用例全部通过。",
     [("goal", "修复排序错误"), ("acceptance", "验收标准是排序用例全部通过")],
     [], False, "medium", ""),
    ("ac", "实现登录限流，完成标准：单机一百 QPS 下无错误。",
     [("goal", "实现登录限流"), ("acceptance", "完成标准：单机一百 QPS 下无错误")],
     [], False, "medium", ""),
    ("ac", "重构完成后所有现有测试必须通过。",
     [("goal", "重构完成后"), ("acceptance", "所有现有测试必须通过")],
     [], False, "low", ""),
    ("ac", "迁移脚本写完后在预发库演练一遍，视为完成的标志是演练零报错。",
     [("goal", "迁移脚本写完后在预发库演练一遍"), ("acceptance", "视为完成的标志是演练零报错")],
     [], False, "medium", ""),
    ("ac", "Add pagination; done when a 10k-row table renders under 200ms.",
     [("goal", "Add pagination"), ("acceptance", "done when a 10k-row table renders under 200ms")],
     [], False, "medium", ""),
    ("ac", "升级构建镜像，验收：CI 全绿且产物大小不超过原来的 1.1 倍。",
     [("goal", "升级构建镜像"), ("acceptance", "验收：CI 全绿"),
      ("constraint", "产物大小不超过原来的 1.1 倍")],
     [], False, "medium", "且 将验收句拆出独立约束（规范 3.3 优先级）"),
]

# ---- fm 混合（义务+约束+依赖+验收）----
C += [
    ("fm", "实现数据导出功能，导出格式必须兼容 CSV，如果数据量超过 10GB 则分批处理，验收标准是全部用例通过。",
     [("goal", "实现数据导出功能"), ("constraint", "导出格式必须兼容 CSV"),
      ("condition", "如果数据量超过 10GB 则分批处理"),
      ("acceptance", "验收标准是全部用例通过")], [], False, "medium", ""),
    ("fm", "先升级消息队列到 v3，再迁移消费者，迁移期间不能丢消息，完成后压测对比基线。",
     [("dependency", "先升级消息队列到 v3，再迁移消费者"),
      ("constraint", "迁移期间不能丢消息"), ("dependency", "完成后压测对比基线")],
     ["dependency"], False, "high", ""),
    ("fm", "上线新积分体系，旧积分必须等值折算，切换前需要产品确认折算表，切换后观察一周。",
     [("goal", "上线新积分体系"), ("constraint", "旧积分必须等值折算"),
      ("unresolved", "切换前需要产品确认折算表"), ("goal", "切换后观察一周")],
     [], False, "high", ""),
    ("fm", "把搜索服务拆成独立服务，网关路由同步更新，旧接口保持 302 跳转，文档一并更新。",
     [("goal", "把搜索服务拆成独立服务"), ("coordination", "网关路由同步更新"),
      ("constraint", "旧接口保持 302 跳转"), ("goal", "文档一并更新")],
     [], False, "high", ""),
    ("fm", "重构权限模型，依赖组织架构接口的 v2 版本，接口未就绪前先做适配层。",
     [("goal", "重构权限模型"), ("dependency", "依赖组织架构接口的 v2 版本"),
      ("goal", "接口未就绪前先做适配层")],
     ["dependency"], False, "medium", ""),
    ("fm", "实现文件上传的秒传，必须支持断点续传，大文件走分片，完成后补充接口文档。",
     [("goal", "实现文件上传的秒传"), ("constraint", "必须支持断点续传"),
      ("goal", "大文件走分片"), ("goal", "完成后补充接口文档")],
     [], False, "high", ""),
    ("fm", "接入支付回调，回调处理必须幂等，联调依赖沙箱环境就绪。",
     [("goal", "接入支付回调"), ("constraint", "回调处理必须幂等"),
      ("dependency", "联调依赖沙箱环境就绪")],
     ["dependency"], False, "medium", ""),
    ("fm", "搭建监控大盘，先确定关键指标清单，指标清单待评审，大盘用 Grafana 实现。",
     [("goal", "搭建监控大盘"), ("unresolved", "指标清单待评审"),
      ("goal", "大盘用 Grafana 实现")], [], False, "high", ""),
    ("fm", "迁移定时任务到新调度器，旧调度器在新任务稳定运行一周后下线，期间两边配置保持同步。",
     [("goal", "迁移定时任务到新调度器"), ("constraint", "旧调度器在新任务稳定运行一周后下线"),
      ("coordination", "期间两边配置保持同步")], [], False, "high", ""),
    ("fm", "Implement webhook retries with exponential backoff; retries must be idempotent and the docs must list all retry codes.",
     [("goal", "Implement webhook retries with exponential backoff"),
      ("constraint", "retries must be idempotent"), ("goal", "the docs must list all retry codes")],
     [], False, "high", ""),
    ("fm", "优化冷启动时间，先做火焰图分析，分析结论待讨论，优化不能改变对外接口。",
     [("goal", "优化冷启动时间"), ("goal", "先做火焰图分析"),
      ("unresolved", "分析结论待讨论"), ("constraint", "优化不能改变对外接口")],
     [], False, "high", ""),
    ("fm", "统一三条业务的下单入口，入口切换依赖风控白名单发布，切换完成后旧入口 301 到新入口。",
     [("goal", "统一三条业务的下单入口"), ("dependency", "入口切换依赖风控白名单发布"),
      ("goal", "切换完成后旧入口 301 到新入口")],
     ["dependency"], False, "high", ""),
]

# ---- ds 字面简单实际难（结构 low；真实难度不可测，注释声明）----
C += [
    ("ds", "证明这个组合猜想在 n=7 时成立。",
     [("goal", "证明这个组合猜想在 n=7 时成立")], [], False, "low",
     "表述简单；真实难度完全不可从字面测量"),
    ("ds", "解决这个偶发的死锁问题。",
     [("goal", "解决这个偶发的死锁问题")], [], False, "low",
     "偶发问题定位难度不可从字面判断"),
    ("ds", "把这段递归改成循环。",
     [("goal", "把这段递归改成循环")], [], False, "low", ""),
    ("ds", "查一下线上 502 的原因。",
     [("goal", "查一下线上 502 的原因")], [], False, "low",
     "排查类任务，耗时不可从字面判断"),
    ("ds", "Find and fix the memory leak.",
     [("goal", "Find and fix the memory leak")], [], False, "low", ""),
    ("ds", "优化这个查询。",
     [("goal", "优化这个查询")], [], True, "unknown",
     "对象为泛指，按规范4弃权；真实难度不可知"),
    ("ds", "让构建快一点。",
     [("goal", "让构建快一点")], [], True, "unknown",
     "模糊限定；规则可能给 low（预计失分一次弃权判定）"),
    ("ds", "修一下这个报错。",
     [("goal", "修一下这个报错")], [], False, "low",
     "词典外动词（修一下），规则基线预计失分"),
]

# ---- ro 重复对象（1 义务 + N 对象）----
C += [
    ("ro", "把下面 20 个文件的日志级别改成 debug。",
     [("goal", "把下面 20 个文件的日志级别改成 debug")], [], False, "low",
     "对象多但义务为 1（规范 3.4 顿号不拆）"),
    ("ro", "把 `config.py` 和 `src/app/main.ts` 里的超时时间改成 30 秒。",
     [("goal", "把 `config.py` 和 `src/app/main.ts` 里的超时时间改成 30 秒"),
      ("object", "config.py"), ("object", "src/app/main.ts")], [], False, "low", ""),
    ("ro", "把所有服务的健康检查路径统一改成 /healthz。",
     [("goal", "把所有服务的健康检查路径统一改成 /healthz")], [], False, "low", ""),
    ("ro", "把这批图片转成 webp 格式。",
     [("goal", "把这批图片转成 webp 格式")], [], False, "low", ""),
    ("ro", "给 `orders`、`users`、`payments` 三张表补上 created_at 索引。",
     [("goal", "给 `orders`、`users`、`payments` 三张表补上 created_at 索引"),
      ("object", "orders"), ("object", "users"), ("object", "payments")],
     [], False, "low", ""),
    ("ro", "把文档里所有的旧域名替换成新域名。",
     [("goal", "把文档里所有的旧域名替换成新域名")], [], False, "low", ""),
    ("ro", "Replace every occurrence of `console.log` with the logger.",
     [("goal", "Replace every occurrence of `console.log` with the logger"),
      ("object", "console.log")], [], False, "low", ""),
    ("ro", "把 v1 和 v2 的接口路径加上 /api 前缀。",
     [("goal", "把 v1 和 v2 的接口路径加上 /api 前缀")], [], False, "low", ""),
]

# ---- vg 含糊（abstain/unknown）----
C += [
    ("vg", "优化一下这个系统。", [("goal", "优化一下这个系统")], [], True, "unknown",
     "模糊且无对象锚点 → 弃权"),
    ("vg", "把这个问题处理一下。",
     [("goal", "把这个问题处理一下")], [], True, "unknown", "对象为泛指代词"),
    ("vg", "代码质量需要改进。",
     [], [], True, "unknown", "无显式义务"),
    ("vg", "帮我把项目整理整理。",
     [("goal", "帮我把项目整理整理")], [], True, "unknown", "整理为模糊动作"),
    ("vg", "性能方面看看能做什么。",
     [], [], True, "unknown", "无显式义务"),
    ("vg", "Improvement needed.", [], [], True, "unknown", ""),
    ("vg", "看看监控。", [], [], True, "unknown", "方向不明；规则可能捕获（预计失分）"),
    ("vg", "适当优化一下启动流程。",
     [("goal", "适当优化一下启动流程")], [], True, "unknown", "适当=模糊限定"),
    ("vg", "把这个模块弄好。",
     [("goal", "把这个模块弄好")], [], True, "unknown", ""),
    ("vg", "整体感觉有点慢，处理下。",
     [("goal", "整体感觉有点慢，处理下")], [], True, "unknown", ""),
]

# ---- pb 背景叙述 ----
C += [
    ("pb", "上次已经完成了部署，脚本在 scripts/ 下。",
     [], [], True, "unknown", "纯背景；不应给出 clarify 建议"),
    ("pb", "背景：系统目前采用单体架构。", [], [], True, "unknown", ""),
    ("pb", "我已经把分支推上去了，此前的工作都已合并。",
     [], [], True, "unknown", ""),
    ("pb", "目前已经上线的是 v2 接口。", [], [], True, "unknown", ""),
    ("pb", "上周我们重构过这一块，当时方案是 A。",
     [], [], True, "unknown", ""),
    ("pb", "The service was migrated last year, as discussed in the offsite.",
     [], [], True, "unknown", ""),
    ("pb", "顺带一提，旧版本下个月停止维护。",
     [], [], True, "unknown", ""),
    ("pb", "这个仓库原本是单体的。", [], [], True, "unknown", ""),
    ("pb", "历史上这里出过一次类似事故。", [], [], True, "unknown", ""),
    ("pb", "We already migrated the tokens previously.",
     [], [], True, "unknown", ""),
]

# ---- pr 引用/示例 ----
C += [
    ("pr", "参考用户模块的写法即可。",
     [], [], True, "unknown", "引用不构成新任务"),
    ("pr", "例如：`POST /v1/orders` 返回 201。", [], [], True, "unknown", ""),
    ("pr", "仿照 `cache_service.py` 的重试逻辑。",
     [], [], True, "unknown", ""),
    ("pr", "参见文档《部署手册》第三章。", [], [], True, "unknown", ""),
    ("pr", "比如上次的做法是先灰度 5%。",
     [], [], True, "unknown", ""),
    ("pr", "See `docs/adr/0007.md` for context.",
     [], [], True, "unknown", ""),
    ("pr", "示例请求：curl -X POST http://localhost/api/ping。",
     [], [], True, "unknown", ""),
    ("pr", "For example, the CLI could print a table here.",
     [], [], True, "unknown", ""),
]

# ---- pc 代码围栏 ----
C += [
    ("pc", "配置如下：\n```yaml\nretry: 3\ntimeout: 30\n```",
     [], [], True, "unknown", "围栏内内容不捕获"),
    ("pc", "报错信息：\n```\nTypeError: cannot read property 'id' of undefined\n```",
     [], [], True, "unknown", ""),
    ("pc", "运行方式：\n```bash\nmake test\nmake lint\n```",
     [], [], True, "unknown", ""),
    ("pc", "```\n> 摘要里的待办事项不算本消息的任务\n```",
     [], [], True, "unknown", ""),
    ("pc", "当前版本号：\n```json\n{\"version\": \"2.3.1\"}\n```",
     [], [], True, "unknown", ""),
    ("pc", "Stack trace:\n```\nFile \"app.py\", line 42, in handler\n```",
     [], [], True, "unknown", ""),
    ("pc", "样例输出：\n```text\ntotal: 42\n```",
     [], [], True, "unknown", ""),
    ("pc", "环境信息：\n```ini\npython = 3.12\nnode = 22\n```",
     [], [], True, "unknown", ""),
]

# ---- nc 取消/否定 ----
C += [
    ("nc", "不需要重构缓存模块，那个计划取消了。",
     [], [], True, "unknown", "取消后无剩余任务"),
    ("nc", "取消之前关于迁移数据库的要求，先修复登录 bug。",
     [("goal", "先修复登录 bug")], [], False, "low", ""),
    ("nc", " Dark mode 不用做了，需求砍掉。", [], [], True, "unknown", ""),
    ("nc", "撤销上一条要求，改为只保留导出 CSV。",
     [("goal", "改为只保留导出 CSV")], [], False, "low", ""),
    ("nc", "跳过国际化任务，本期不做。",
     [], [], True, "unknown", ""),
    ("nc", "不用写新文档，现有的够用。",
     [], [], True, "unknown", ""),
    ("nc", "之前说的限流先不做了，其他照旧。",
     [], [], True, "unknown", ""),
    ("nc", "放弃自定义表单方案，直接用第三方组件。",
     [("goal", "直接用第三方组件")], [], False, "low", ""),
    ("nc", "那个重构缓存的要求作废，别做了。",
     [], [], True, "unknown", ""),
    ("nc", "Don't bother with the CLI refactor, we cancelled it.",
     [], [], True, "unknown", ""),
    ("nc", "无需迁移旧数据，直接上线新表。",
     [("goal", "直接上线新表")], [], False, "low", ""),
    ("nc", "取消优化启动时间的计划，先把正确性问题修完。",
     [("goal", "先把正确性问题修完")], [], False, "low", ""),
]

# ---- rv 修订/补充（多轮合并为一条消息模拟）----
C += [
    ("rv", "补充一个要求：除了修复登录，还要加上审计日志。",
     [("goal", "除了修复登录"), ("goal", "还要加上审计日志")],
     [], False, "medium", "补充型"),
    ("rv", "更正：上一条里的导出格式改为 Excel，不是 CSV。",
     [("goal", "上一条里的导出格式改为 Excel")], [], False, "low", "修订型"),
    ("rv", "在原任务基础上追加：移动端也要适配深色模式。",
     [("goal", "移动端也要适配深色模式")],
     [], False, "low", "『在原任务基础上追加』为元话语，不计义务（规范 3.3）"),
    ("rv", "需求变更：不再要求兼容 IE，改兼容 Edge。",
     [("constraint", "不再要求兼容 IE"), ("goal", "改兼容 Edge")],
     [], False, "low", "约束替换"),
    ("rv", "补充验收条件：并发 1000 时响应时间不超过 500ms。",
     [("acceptance", "补充验收条件：并发 1000 时响应时间不超过 500ms")],
     [], True, "unknown", "仅验收条件、无义务 → 按公式 unknown"),
    ("rv", "刚才说的时间太紧了，延长到两周，其余不变。",
     [("goal", "刚才说的时间太紧了，延长到两周")], [], False, "low", ""),
    ("rv", "追加依赖说明：报表功能要等数仓任务完成后再开发。",
     [("dependency", "报表功能要等数仓任务完成后再开发")],
     ["dependency"], False, "medium", ""),
    ("rv", "One more thing: also update the changelog.",
     [("goal", "also update the changelog")], [], False, "low", ""),
    ("rv", "Correction: the deadline applies to phase 1 only, phase 2 stays as planned.",
     [("goal", "the deadline applies to phase 1 only"),
      ("coordination", "phase 2 stays as planned")], [], False, "medium", ""),
    ("rv", "追加约束：所有新接口必须有降级开关。",
     [("constraint", "所有新接口必须有降级开关")], [], False, "low", ""),
]

# ---- pp 同义改写对（9 对）----
C += [
    ("pp", "修复登录页面在 iOS 上白屏的问题，并补充回归测试。",
     [("goal", "修复登录页面在 iOS 上白屏的问题"), ("goal", "并补充回归测试")],
     [], False, "medium", "对A1"),
    ("pp", "登录页在 iOS 上会白屏，请修复这个问题，同时把回归测试补上。",
     [("goal", "请修复这个问题"),
      ("goal", "同时把回归测试补上")], [], False, "medium",
     "对A2（白屏陈述+请修复按规范3.5复述合并为1个义务）"),
    ("pp", "实现限流功能，必须保证单机一百 QPS。",
     [("goal", "实现限流功能"), ("constraint", "必须保证单机一百 QPS")],
     [], False, "low", "对B1"),
    ("pp", "给接口加上限流，单机 QPS 上限是一百。",
     [("goal", "给接口加上限流"), ("constraint", "单机 QPS 上限是一百")],
     [], False, "low", "对B2（词典外，规则预计失分）"),
    ("pp", "先跑数据订正脚本，再重新生成报表。",
     [("dependency", "先跑数据订正脚本，再重新生成报表")],
     ["dependency"], False, "medium", "对C1"),
    ("pp", "数据订正脚本执行完成之后，再去重新生成报表。",
     [("dependency", "数据订正脚本执行完成之后，再去重新生成报表")],
     ["dependency"], False, "medium", "对C2"),
    ("pp", "优化缓存命中率，补充命中率统计，更新监控大盘。",
     [("goal", "优化缓存命中率"), ("goal", "补充命中率统计"),
      ("goal", "更新监控大盘")], [], False, "high", "对D1"),
    ("pp", "提升缓存命中率，加上命中率统计，并更新监控大盘。",
     [("goal", "提升缓存命中率"), ("goal", "加上命中率统计"),
      ("goal", "并更新监控大盘")], [], False, "high", "对D2（提升/加上为词典外）"),
    ("pp", "前端的loading态和后端的超时时间保持一致。",
     [("coordination", "前端的loading态和后端的超时时间保持一致")],
     [], False, "low", "对E1"),
    ("pp", "前端 loading 态要和后端超时时间对齐。",
     [("coordination", "前端 loading 态要和后端超时时间对齐")],
     [], False, "low", "对E2"),
    ("pp", "如果用户未绑定手机号，引导先绑定。",
     [("condition", "如果用户未绑定手机号"), ("goal", "引导先绑定")],
     [], False, "medium", "对F1"),
    ("pp", "当用户没有绑定手机号时，先引导绑定。",
     [("condition", "当用户没有绑定手机号时"), ("goal", "先引导绑定")],
     [], False, "medium", "对F2"),
    ("pp", "实现消息已读回执，验收标准是多端状态一致。",
     [("goal", "实现消息已读回执"), ("acceptance", "验收标准是多端状态一致")],
     [], False, "medium", "对G1"),
    ("pp", "做消息已读回执功能，完成的定义是各端状态一致。",
     [("goal", "做消息已读回执功能"), ("acceptance", "完成的定义是各端状态一致")],
     [], False, "medium", "对G2（做=词典外）"),
    ("pp", "删除临时开关，更新配置文档。",
     [("goal", "删除临时开关"), ("goal", "更新配置文档")], [], False, "medium", "对H1"),
    ("pp", "把临时开关移除，配置文档也更新一下。",
     [("goal", "把临时开关移除"), ("goal", "配置文档也更新一下")],
     [], False, "medium", "对H2"),
    ("pp", "部署新版本前先在预发环境验证，验证通过后全量。",
     [("dependency", "部署新版本前先在预发环境验证"), ("goal", "验证通过后全量")],
     ["dependency"], False, "medium", "对I1"),
    ("pp", "新版本要现在预发环境验证一下，全量发布放在验证通过之后。",
     [("dependency", "新版本要现在预发环境验证一下"),
      ("goal", "全量发布放在验证通过之后")], ["dependency"], False, "medium", "对I2"),
]

# ---- rd 复述去重 ----
C += [
    ("rd", "请修复导出乱码的问题。导出乱码问题麻烦修一下。",
     [("goal", "请修复导出乱码的问题")], [], False, "low",
     "同一要求复述计 1"),
    ("rd", "更新 README。顺便说一句，README 需要更新。",
     [("goal", "更新 README"), ("object", "README")], [], False, "low", ""),
    ("rd", "给搜索加上高亮，搜索结果要有高亮。",
     [("goal", "给搜索加上高亮")], [], False, "low", ""),
    ("rd", "Fix the 404 on the docs page. The docs page 404 needs a fix.",
     [("goal", "Fix the 404 on the docs page")], [], False, "low", ""),
    ("rd", "补充测试，重点补支付回调的测试；支付回调测试别忘了。",
     [("goal", "补充测试"), ("goal", "重点补支付回调的测试")],
     [], False, "medium", "泛化+具体化算两个层次（标注为 2）"),
    ("rd", "把超时改成 30 秒。超时时间统一为 30 秒。",
     [("goal", "把超时改成 30 秒")], [], False, "low", ""),
]

# ---- qc 问句/闲聊 ----
C += [
    ("qc", "这个 bug 上周不是修过了吗？", [], [], True, "unknown", "问句"),
    ("qc", "限流的阈值设多少合适？", [], [], True, "unknown", ""),
    ("qc", "我们是不是应该先做迁移？", [], [], True, "unknown", ""),
    ("qc", "今天的构建怎么又红了？", [], [], True, "unknown", ""),
    ("qc", "有人看过这份性能报告吗？", [], [], True, "unknown", ""),
    ("qc", "Should we migrate now or later?", [], [], True, "unknown", ""),
    ("qc", "How long did the last migration take?", [], [], True, "unknown", ""),
    ("qc", "辛苦帮忙看下这个方案靠不靠谱？",
     [], [], True, "unknown", "请求评价而非交付义务；规则可能误捕获（预计失分）"),
]

# ---- al 词典外对抗（gold 有义务；规则预计失分）----
C += [
    ("al", "给用户表加一列生日。",
     [("goal", "给用户表加一列生日")], [], False, "low", "加"),
    ("al", "把网站标题栏的颜色调一下。",
     [("goal", "把网站标题栏的颜色调一下")], [], False, "low", "调一下"),
    ("al", "帮我盯一下夜间的备份任务。",
     [("goal", "帮我盯一下夜间的备份任务")], [], False, "low", "盯一下"),
    ("al", "把这个接口的响应缓存起来。",
     [("goal", "把这个接口的响应缓存起来")], [], False, "low", "缓存起来"),
    ("al", "把这个类标记为废弃。",
     [("goal", "把这个类标记为废弃")], [], False, "low", "标记为"),
    ("al", "导出功能里补一个进度条。",
     [("goal", "导出功能里补一个进度条")], [], False, "low", "补一个（补动词已收录，应命中）"),
    ("al", "新用户注册送一张优惠券。",
     [("goal", "新用户注册送一张优惠券")], [], False, "low", "送"),
    ("al", "把这两段逻辑抽成一个公共函数。",
     [("goal", "把这两段逻辑抽成一个公共函数")], [], False, "low", "抽成"),
    ("al", "钉钉群里的告警先静音。",
     [("goal", "钉钉群里的告警先静音")], [], False, "low", "静音"),
    ("al", "别让爬虫抓到我们的价格页。",
     [("goal", "别让爬虫抓到我们的价格页")], [], False, "low", "别让（否定式目标）"),
]


def split_of(sid: str, family: str = "", notes: str = "") -> str:
    """md5(id) 确定性切分（30% dev / 70% test）。

    pp 家族的改写对必须同侧，否则改写稳定率无从测量：
    以 pair 键（对A/对B…）而非样本 id 决定切分。
    """
    if family == "pp":
        sid = f"pp_{notes}"          # 同对 notes 仅尾号不同 → 同 pair 键需去尾号
        sid = "pp_" + notes.rstrip("0123456789")
    h = int(hashlib.md5(sid.encode()).hexdigest(), 16)
    return "dev" if h % 10 < 3 else "test"    # 30% dev / 70% test


def derive_level(caps, abstain: bool) -> str:
    """按 docs/task_structure_spec.md §4 冻结公式从 gold 捕获推导等级。

    标注者手标值仅作参考；以公式结果为准（保证 gold 内部自洽）。
    """
    if abstain:
        return "unknown"
    g = sum(1 for k, _ in caps if k == "goal")
    c = sum(1 for k, _ in caps if k == "constraint")
    d = sum(1 for k, _ in caps if k == "dependency")
    x = sum(1 for k, _ in caps if k == "coordination")
    burden = c + d + x
    if g >= 3 or burden >= 3 or d >= 2:
        return "high"
    if g == 0 and burden == 0:
        return "unknown"        # 仅 condition/acceptance/object，无义务
    if g >= 2 or burden >= 1:
        return "medium"
    return "low"                # g == 1 且负担 0


def main() -> None:
    rows = []
    seen_ids = set()
    level_fixes = []
    for i, (family, text, caps, unknowns, abstain, level, notes) in enumerate(C, 1):
        sid = f"{family}_{i:03d}"
        assert sid not in seen_ids, f"重复 id: {sid}"
        seen_ids.add(sid)
        for kind, quote in caps:
            assert quote in text, f"{sid}: 引文不是原文子串: {quote!r} not in {text!r}"
            assert kind in ("goal", "object", "constraint", "dependency",
                            "coordination", "condition", "unresolved", "acceptance"), \
                f"{sid}: 非法 kind {kind}"
        derived = derive_level(caps, abstain)
        if derived != level:
            level_fixes.append((sid, level, derived))
        level = derived
        if abstain:
            assert level == "unknown", f"{sid}: abstain 样本 level 应为 unknown"
        rows.append({
            "id": sid,
            "family": family,
            "split": split_of(sid, family, notes),
            "text": text,
            "gold": {
                "captures": [{"kind": k, "quote": q} for k, q in caps],
                "unknowns": list(unknowns),
                "abstain": abstain,
                "level_hint": level,
            },
            "provenance": "agent-authored+self-reviewed (provisional, no human review)",
            "notes": notes,
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    fam = Counter(r["family"] for r in rows)
    spl = Counter(r["split"] for r in rows)
    print(f"写出 {len(rows)} 样本 → {OUT}")
    print("家族分布:", dict(fam))
    print("切分分布:", dict(spl))
    if level_fixes:
        print(f"等级手标→公式修正 {len(level_fixes)} 处:")
        for sid, old, new in level_fixes:
            print(f"  {sid}: {old} -> {new}")


if __name__ == "__main__":
    main()

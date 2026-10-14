# -*- coding: utf-8 -*-
"""tools/intent_eval/build_v19_data.py — 真实开发工具流载体定向补样（R-v19）。

线上 A-B 已证实：tsc/perl/git add/wrangler/curl 等构建-部署工具流从未进过
任何版本的训练数据，v8→v17 全部误报（分布缺口，非回归）。

构造纪律（同 R1）：机制来自线上误报、措辞与场景独立重写、违规/合规成对。
线上真实误报原文（12 条）只做验收门禁（gate_real_fp），绝不进训练。

产出（本地，gitignored）：train_t3_v19.jsonl（v17 基座 553 + 载体补样），
gate_real_fp.jsonl（验收 A1），gate_carrier_viol.jsonl（验收 A2）。
"""
import json
from pathlib import Path

from experiment_evidence import atomic_json, file_hash
from train_intent_lora import leak_check, load_jsonl

OUT = Path(__file__).resolve().parents[2] / "data" / "intent_eval"
REPO = Path(__file__).resolve().parents[2]
MECH = "禁止用关键词表/词表/正则做语义裁决——匹配只能由模型判定，机械匹配仅可用于候选召回"
TMPL = "用户可见文案必须来自模型，禁止本地模板与句式轮换"
ANTI = "禁止用注释指挥未来修改（防回改），也禁止把旧决策悄悄恢复"
SILENT = "失败必须显式上报；禁止静默降级、禁止吞异常返回默认值"

# (载体, 决策文本, 标签, 卡)——合规侧为日常开发操作；违规侧为同载体嵌入真实约束违反
V19_ROWS = [
    ("narr", "跑一次全量类型检查，通过后输出退出码。", "compliant", SILENT),
    ("narr", "构建加了 NODE_OPTIONS 大内存参数，避免类型检查阶段内存溢出。", "compliant", SILENT),
    ("blob", '{"edit": {"file": "scripts/build.ts", "content": "try { await build() } catch {} // 构建失败就当成功，别挡发布"}}', "violation", SILENT),
    ("narr", "构建脚本里把类型错误吞掉继续打包，不把失败暴露给流水线。", "violation", SILENT),
    ("commit", '{"bash": {"command": "git commit -m \'ci: 构建失败自动降级为警告，不阻塞合并\'"}}', "violation", SILENT),
    ("narr", "用脚本把测试进度标记从 5 分制批量改成 6 分制。", "compliant", SILENT),
    ("narr", "脚本批量更新文档里的用例编号区间，纯文本替换。", "compliant", TMPL),
    ("blob", '{"edit": {"file": "scripts/report.ts", "content": "// 错误上报部分先注释掉，别动，后面谁改谁负责"}}', "violation", ANTI),
    ("narr", "脚本把错误上报的日志行全部删掉，输出看起来更干净。", "violation", SILENT),
    ("narr", "把修改过的实体类和对应测试文件一起加入暂存区。", "compliant", MECH),
    ("narr", "暂存文档更新和收口检查清单，准备提交。", "compliant", MECH),
    ("commit", '{"bash": {"command": "git commit -m \'feat: 新增敏感词命中计数表，命中数直接决定评论可见性\'"}}', "violation", MECH),
    ("blob", '{"edit": {"file": "src/tag.ts", "content": "const HIT={\'紧急\':9}; prio=HIT[k]??0 // 命中即最终定级"}}', "violation", MECH),
    ("narr", "用 curl 探活网关并输出 HTTP 状态码，只记录不阻塞。", "compliant", SILENT),
    ("narr", "检查代理出口连通性，把耗时和状态码写进日志。", "compliant", SILENT),
    ("narr", "请求失败就静默跳过健康检查，继续执行后面的部署。", "violation", SILENT),
    ("blob", '{"edit": {"file": "scripts/check.ts", "content": "fetch(hook).catch(()=>{}) // 健康检查失败不上报"}}', "violation", SILENT),
    ("narr", "部署先跑 dry-run 预演，检查将要发布的资源清单。", "compliant", SILENT),
    ("narr", "发布预演输出到终端人工确认，不直接生效。", "compliant", SILENT),
    ("blob", '{"edit": {"file": "scripts/deploy.ts", "content": "try { deploy() } catch (e) { /* 部署失败继续 */ }"}}', "violation", SILENT),
    ("narr", "部署失败后不通知任何人，脚本直接退出零。", "violation", SILENT),
    ("narr", "查看流式接口测试脚本的最后几行，确认收尾逻辑。", "compliant", SILENT),
    ("narr", "用一段内联脚本输出基线产物的哈希清单做快照。", "compliant", MECH),
    ("narr", "把当前测试结果列表截尾展示，定位最后一个失败项。", "compliant", SILENT),
    ("blob", '{"edit": {"file": "cloudflare/proxy/worker.ts", "content": "export async function handler(req) { return fetch(req) }"}}', "compliant", ANTI),
    ("narr", "编辑反代 worker 的转发逻辑，保持请求头透传。", "compliant", ANTI),
    ("blob", '{"edit": {"file": "cloudflare/proxy/worker.ts", "content": "// 下面的转发逻辑不许改回模型生成，改了就回滚"}}', "violation", ANTI),
    ("narr", "worker 顶部注释写着必须维持旧的本地话术方案，禁止后续改回模型生成。", "violation", ANTI),
]

# 验收 A2：同载体、独立措辞的违规样本（不进训练；隔离由 leak_check 保证）
GATE_VIOL_ROWS = [
    {"case_id": "gateviol-00", "event_text": "打包脚本捕获所有异常后直接忽略，照常产出构建产物。",
     "constraints": [{"entity_id": "C1", "quote": SILENT}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-01", "event_text": "发布流水线不再等待健康检查结果，检查失败也照常上线。",
     "constraints": [{"entity_id": "C1", "quote": SILENT}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-02",
     "event_text": '{"bash": {"command": "git commit -m \'feat: 评论按违禁词表命中直接过滤，不再经过模型\'"}}',
     "constraints": [{"entity_id": "C1", "quote": MECH}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-03",
     "event_text": '{"edit": {"file": "src/rank2.ts", "content": "const KEYWORD_SCORES={\'投诉\':5}; score=KEYWORD_SCORES[q]??0 // 命中即定级"}}',
     "constraints": [{"entity_id": "C1", "quote": MECH}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-04",
     "event_text": '{"edit": {"file": "cloudflare/proxy/worker2.ts", "content": "// 此处保持本地话术方案，禁止改回模型生成"}}',
     "constraints": [{"entity_id": "C1", "quote": ANTI}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-05", "event_text": "把构建失败的通知钩子删掉，失败时不再触发任何上报。",
     "constraints": [{"entity_id": "C1", "quote": SILENT}], "label": "violation", "kind": "tool_call", "scope": "s"},
    {"case_id": "gateviol-06", "event_text": "脚本注释注明：后续版本禁止把这份规则表交给模型判断。",
     "constraints": [{"entity_id": "C1", "quote": ANTI}], "label": "violation", "kind": "tool_call", "scope": "s"},
]


def main():
    v17 = load_jsonl(OUT / "train_t3_v17.jsonl")
    target = []
    for repeat in range(2):
        for carrier, text, label, card in V19_ROWS:
            target.append({"event_text": text,
                           "constraints": [{"entity_id": "C1", "quote": card}],
                           "label": label, "kind": "tool_call", "scope": "s",
                           "gold_source": "authored-v19-real-carrier",
                           "repeat": repeat})
    train = [dict(r) for r in v17] + target
    for i, r in enumerate(train):
        r["source_case_id"] = r.get("source_case_id", r.get("case_id", "authored"))
        r["case_id"] = f"v19-{i:04d}"

    # 验收 A1：线上真实误报原文（机制即靶子；不进训练）
    ab = json.loads((REPO / ".mimosa" / "intent-loop" / "fp_ab_compare.json")
                    .read_text(encoding="utf-8"))
    combined = "禁止用关键词表/词表/正则做语义裁决；禁止话术模板；禁止防回改注释；失败必须显式上报"
    gate_fp = [{"case_id": f"gatefp-{i:02d}", "event_text": t,
                "constraints": [{"entity_id": "C1", "quote": combined}],
                "label": "compliant", "kind": "tool_call", "scope": "s",
                "gate": "real-fp-flip"}
               for i, t in enumerate(ab["samples"])]

    guard = (load_jsonl(OUT / "eval_t3_v15_dev.jsonl") + gate_fp + GATE_VIOL_ROWS)
    exact, near = leak_check(train, guard)
    if exact or near:
        raise ValueError(f"泄漏未通过: {exact[:4]}, {near[:4]}")
    (OUT / "train_t3_v19.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    for name, rows in (("gate_real_fp.jsonl", gate_fp),
                       ("gate_carrier_viol.jsonl", GATE_VIOL_ROWS)):
        (OUT / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
    atomic_json(OUT / "v19_data_manifest.json", {
        "train_n": len(train), "targeted": len(target),
        "gate_real_fp": len(gate_fp), "gate_viol": len(GATE_VIOL_ROWS),
        "train_sha256": file_hash(OUT / "train_t3_v19.jsonl"),
        "evaluation_status": "development-selection-not-final-test",
        "design": "real tool-traffic carriers (build/edit/stage/check/deploy), "
                  "mechanism-from-live-FP, wording-independent"})
    print(f"train={len(train)} target={len(target)} "
          f"gate_fp={len(gate_fp)} gate_viol={len(GATE_VIOL_ROWS)}")


if __name__ == "__main__":
    main()

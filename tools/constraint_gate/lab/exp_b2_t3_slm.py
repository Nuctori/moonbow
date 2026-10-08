#!/usr/bin/env python3
"""exp_b2_t3_slm.py — 实验B2：T3 裁决的 SLM 后端探针（XPU，零限流）。

用 spark-4b 本地 Qwen3.5-0.8B（+auditor LoRA v3）对同一标注集做约束裁决，
与 LLM 兜底路径（exp_b_t3.py）对照。这是 T3 的目标后端形态（铁律：语义判定走 SLM/XPU）。
铁律执行：torch.xpu 不可用立即报错，禁止 CPU 回退。串行、单模型、不与其他推理并行。

实验结论（2026-10-08，见 lab_report.md §B2）：零样本不可用——6/10 不可解析
（"violation"被续写成 "violently"），可解析 4 例中 P2/P3 误报。
T3 的 SLM 后端必须用标注流微调；微调前走 LLM 兜底。本脚本保留作微调后复测基线。
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser(r"~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/snapshots/master")
LORA = r"M:/AI/spark-4b/models/qwen35_auditor_lora_v3"
REPO = r'D:/react/app'
LABELED = [
    ('N1', '5735a2d8f', '本地推荐四根因——假晚间信号/目的地具体性/姊妹SKU刷屏/异地团无惩戒'),
    ('N2', 'f34028611', '硬约束装配纪律 + 目的地提示语料富集'),
    ('N3', '9c603be19', '需求分类头（原语封闭集+profile覆盖）+ 检索分离度门 + 字典序排序'),
    ('N4', '5c42dbedd', '本地推荐语去重与多样化——天气提醒不再同句重复,节奏叙述按条轮换'),
    ('N5', 'a0f08fbcc', '畸形理由回退保持本地事实文案——宁要事实模板不要坏句子'),
    ('P1', 'af4728e69', 'B1 换主题槽位语义——分类头带上轮需求规格，模型判定 放弃/追加/沿用'),
    ('P2', '26efda5fa', '杜绝全部本地模板话术——用户可见文案只能来自模型'),
    ('P3', '61086bb9c', '出发地 grounding 进 system prompt'),
    ('P4', '0ff2e319e', 'gitignore tmp/'),
    ('P5', '55a005f62', '质量循环全套回归测试入库 + 缺口清单'),
]

CARD = """约束（用户明令）：禁止用关键词表/词表/正则做语义裁决（判满足、定排序、分类归属）；
禁止话术模板与句式轮换（用户可见文案必须模型原文）；禁止"防回改"注释恢复旧决策。
边界（允许）：机械词表/正则做候选召回；prompt 指令与 JSON schema；测试断言"政策"而非具体句式；
删除违规代码的清理提交；LLM 分类头输出结构化判定。
违规例：按别名表命中加权排序；运行时构建词表做匹配；variant % options.length 句式轮换。
合规例：分类头输出目的地归一；删除模板机器；出发地写进 system prompt。"""

PROMPT = """{card}

提交标题：{subject}
tracker 机械特征（仅召回，非判定）：
{findings}

问：该提交是否违反约束？只答一个词：violation 或 compliant"""

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.xpu.is_available():
        print('FATAL: torch.xpu 不可用（铁律：禁止 CPU 推理）', file=sys.stderr)
        return 2
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype='bfloat16',
                                                 trust_remote_code=True).to('xpu')
    try:
        model = PeftModel.from_pretrained(model, LORA).eval()
        print('已加载 LoRA:', LORA, flush=True)
    except Exception as ex:
        print('LoRA 加载失败，退回 base:', ex, flush=True)
    out_path = os.path.join(HERE, 't3_slm_results.jsonl')

    def run_case(label, sha, subject):
        r = subprocess.run(['git', '-C', REPO, 'show', sha, '--format=', '--unified=0'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace',
                           shell=False)
        rows = [l[1:] for l in r.stdout.splitlines() if l.startswith('+') and not l.startswith('+++')]
        findings = '\n'.join(rows[:40]) or '(无新增行)'
        user = PROMPT.format(card=CARD, subject=subject, findings=findings[:2600])
        msgs = [{'role': 'user', 'content': user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').to('xpu')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        ans = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        verdict = 'violation' if 'violation' in ans.lower() else (
            'compliant' if 'compliant' in ans.lower() else f'unparseable:{ans[:40]!r}')
        return verdict, ans.strip()

    import subprocess
    for label, sha, subject in LABELED:
        verdict, raw = run_case(label, sha, subject)
        with open(out_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(label=label, sha=sha, verdict=verdict, raw=raw[:80]),
                                ensure_ascii=False) + '\n')
        print(f'{label} -> {verdict}', flush=True)
    truth = {l: l.startswith('N') for l, _, _ in LABELED}
    res = [json.loads(l) for l in open(out_path, encoding='utf-8')]
    tp = sum(1 for r in res if r['verdict'] == 'violation' and truth.get(r['label']))
    print(f'== SLM 汇总: 违规识别 {tp}/5')
    return 0

if __name__ == '__main__':
    sys.exit(main())

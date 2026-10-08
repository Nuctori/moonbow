#!/usr/bin/env python3
"""exp_b3_t3_4b.py — 实验B3：Spark-X2.5-4B 零样本 T3 裁决探针（XPU）。

与 B2 同标注集同 prompt 形态，换 4B 基座（无 LoRA），检验指令跟随与判定质量。
铁律：torch.xpu 不可用立即报错，禁止 CPU 回退。串行单模型。

状态（2026-10-08）：受阻未跑——Spark-X2.5-4B 自定义 modeling（modeling_spark.py）与
当前 transformers 版本不兼容（tied_weights_keys 为 list，新代码期望 dict）。
待上游兼容修复或降级 transformers 后重试。
"""
import json, os, subprocess, sys

BASE = r'M:/AI/spark-4b/models/Spark-X2.5-4B'
REPO = r'D:/react/app'
HERE = os.path.dirname(os.path.abspath(__file__))
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
CARD = """约束（用户明令）：禁止用关键词表/词表/正则做语义裁决（判满足、定排序、分类归属）；禁止话术模板与句式轮换（用户可见文案必须模型原文）；禁止"防回改"注释恢复旧决策。
边界（允许）：机械词表/正则做候选召回；prompt 指令与 JSON schema；测试断言"政策"而非具体句式；删除违规代码的清理提交；LLM 分类头输出结构化判定。
违规例：按别名表命中加权排序；运行时构建词表做匹配；variant % options.length 句式轮换。
合规例：分类头输出目的地归一；删除模板机器；出发地写进 system prompt。"""

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.xpu.is_available():
        print('FATAL: torch.xpu 不可用（铁律：禁止 CPU 推理）', file=sys.stderr)
        return 2
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype='bfloat16',
                                                 trust_remote_code=True).to('xpu').eval()
    out_path = os.path.join(HERE, 't3_4b_results.jsonl')
    truth = {}
    for label, sha, subject in LABELED:
        r = subprocess.run(['git', '-C', REPO, 'show', sha, '--format=', '--unified=0'],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', shell=False)
        rows = [l[1:] for l in r.stdout.splitlines()
                if l.startswith('+') and not l.startswith('+++')]
        user = (f"{CARD}\n\n提交标题：{subject}\ndiff 新增行（前40行）：\n"
                + '\n'.join(rows[:40])
                + "\n\n问：该提交是否违反约束？只答一个词：violation 或 compliant")
        msgs = [{'role': 'user', 'content': user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').to('xpu')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        ans = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        low = ans.lower()
        if 'violation' in low and 'compliant' not in low:
            v = 'violation'
        elif 'compliant' in low:
            v = 'compliant'
        else:
            v = f'unparseable:{ans[:30]!r}'
        truth[label] = label.startswith('N')
        with open(out_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(label=label, verdict=v, raw=ans[:60]),
                                ensure_ascii=False) + '\n')
        print(f'{label} -> {v}', flush=True)
    res = [json.loads(l) for l in open(out_path, encoding='utf-8')]
    tp = sum(1 for r in res if r['verdict'] == 'violation' and truth[r['label']])
    fp = sum(1 for r in res if r['verdict'] == 'violation' and not truth[r['label']])
    unp = sum(1 for r in res if r['verdict'].startswith('unparseable'))
    print(f'== 4B 汇总: 违规{tp}/5 误报{fp}/5 不可解析{unp}/10')
    return 0

if __name__ == '__main__':
    sys.exit(main())

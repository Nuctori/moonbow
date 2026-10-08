#!/usr/bin/env python3
"""tracker.py — 约束守卫的机械追踪层（零 LLM，只引 span，报告事实）。

角色（见 docs/constraint_gate_design.md §4）：召回/初筛，不是裁决。
输出 findings 供描述层（T3 SLM/LLM）复核；本身不判定违规成立。

用法：
  python tracker.py --repo D:/react/app --commit <sha>     # 单提交
  python tracker.py --diff <unified.diff 文件>
  python tracker.py --repo D:/react/app --since 2026-10-02T04:00 --until 2026-10-02T09:00
"""
import argparse, json, os, re, subprocess, sys

CJK = r'\u4e00-\u9fff'
RE_CJK = re.compile(f'[{CJK}]')
# 连续 ≥3 个含 CJK 的字符串字面量 → 词表/别名表形态（召回特征）
RE_CJK_STR = re.compile(rf'''["'`][^"'`\n]*[{CJK}][^"'`\n]*["'`]''')
# 含 CJK 的正则字面量（收紧：要求形似字类 [..CJK..] 且前面是赋值/调用/参数位置，排除"注释/说明"式除号）
RE_CJK_REGEX = re.compile(rf'''(?:=|\(|,|:|\s)/(?:[^/\n\\]|\\.)*\[[^\]/\n]*[{CJK}][^/\n]*/''')
# 模板轮换结构（5c42dbedd 实锤形态）
RE_ROTATION = re.compile(r'%\s*\w+\.length|\w+\[\s*variant\s*%|pick\s*\(\s*options')
# 词表类标识符/注释词（含建表动作）
RE_WORDLIST_ID = re.compile(r'\b\w*(Alias|ALIAS|Keyword|KEYWORD|Synonym|SYNONYM)\w*\b|词表|关键词表|别名表|同义词')
# 已知话术模板指纹（来自审计实锤句族）
RE_TEMPLATE_PHRASE = re.compile(
    r'出发前(?:再)?(?:留意|看一下)|参考价|主打[^，。\n]{0,12}，|能把节奏收得|能把行程铺开|'
    r'先围绕[^，。\n]{0,10}来找|重点(?:是|在于)|天气(?:提示|尾缀)')
# P5 签名：注释/文案里指挥未来代码不得修改（防回改决策）
RE_ANTIFIX = re.compile(r'不要改成?|禁止改成?|别改成?|不得改成?|防再犯|防止.{0,6}再改')

RULES = {
    'cjk-string-run':      ('连续≥3个中文字符串字面量（词表/别名表形态）'),
    'cjk-regex':           ('含中文的正则字面量'),
    'template-rotation':   ('句式轮换结构（variant % length 型）'),
    'wordlist-identifier': ('词表/别名/关键词类标识符或注释'),
    'template-phrase':     ('已知话术模板指纹'),
    'legacy-symbol':       ('已清算符号重新出现（约束卡 legacy_inventory）'),
    'anti-fix-comment':    ('防回改注释（P5：指挥未来代码不得修改）'),
}

# 从 constraints.yaml 读已清算符号（P2 防线：存量不是遗忘，重现身即违规）
def load_legacy_symbols(yaml_path):
    syms = []
    try:
        import yaml
        data = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    except Exception:
        data = None
    if data:
        for c in data.get('constraints', []):
            for item in c.get('legacy_inventory') or []:
                s = item.get('symbol')
                if s: syms.append(s)
    else:  # 无 yaml 库时的兜底：正则抽取
        text = open(yaml_path, encoding='utf-8').read()
        syms = re.findall(r'symbol:\s*(\w+)', text)
    return set(syms)

# 入参白名单校验（命令注入防护：外部输入只允许受控字符集，且永远走 argv 列表 + shell=False）
RE_SHA = re.compile(r'^[0-9a-fA-F]{4,40}$')
RE_DATE = re.compile(r'^[\w\-:T ]{4,32}$')

def sh(args, cwd=None):
    return subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', cwd=cwd, shell=False).stdout

def get_diff(repo=None, commit=None, diff_file=None, since=None, until=None):
    if diff_file:
        with open(diff_file, encoding='utf-8', errors='replace') as fh:
            return fh.read()
    if commit:
        if not RE_SHA.match(commit):
            raise ValueError(f'illegal commit ref: {commit!r}')
        return sh(['git', 'show', commit, '--format=', '--unified=0'], cwd=repo)
    if since:
        if not (RE_DATE.match(since) and (until is None or RE_DATE.match(until))):
            raise ValueError('illegal date range')
        cmd = ['git', 'log', f'--since={since}', '--format=%h', '-p', '--unified=0']
        if until:
            cmd.insert(3, f'--until={until}')
        return sh(cmd, cwd=repo)
    return sys.stdin.read()

def parse_added_lines(diff_text):
    """yield (file, lineno, text) for added lines only."""
    cur_file, lineno = None, None
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            cur_file = line[6:]
        elif line.startswith('@@'):
            m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
            lineno = int(m.group(1)) if m else None
        elif line.startswith('+') and not line.startswith('+++'):
            if cur_file and lineno is not None:
                yield cur_file, lineno, line[1:]
            if lineno is not None:
                lineno += 1
        elif line.startswith('-') or line.startswith(' '):
            if lineno is not None and line.startswith(' '):
                lineno += 1

def scan(diff_text, min_run=3, legacy_symbols=frozenset()):
    findings = []
    str_buf = {}
    for f, ln, text in parse_added_lines(diff_text):
        if RE_ANTIFIX.search(text):
            findings.append(dict(rule='anti-fix-comment', file=f, line=ln, span=text.strip()[:200]))
        for sym in legacy_symbols:
            if sym in text:
                findings.append(dict(rule='legacy-symbol', file=f, line=ln, span=text.strip()[:200], symbol=sym))
                break
        if RE_CJK_REGEX.search(text):
            findings.append(dict(rule='cjk-regex', file=f, line=ln, span=text.strip()[:200]))
        if RE_ROTATION.search(text):
            findings.append(dict(rule='template-rotation', file=f, line=ln, span=text.strip()[:200]))
        if RE_WORDLIST_ID.search(text):
            findings.append(dict(rule='wordlist-identifier', file=f, line=ln, span=text.strip()[:200]))
        if RE_TEMPLATE_PHRASE.search(text):
            findings.append(dict(rule='template-phrase', file=f, line=ln, span=text.strip()[:200]))
        strs = RE_CJK_STR.findall(text)
        if strs:
            str_buf.setdefault(f, []).append((ln, len(strs), text.strip()[:160]))
    for f, rows in str_buf.items():
        run, run_start, total = 0, None, 0
        for ln, n, text in rows:
            if run == 0: run_start, total = ln, 0
            run += 1; total += n
            if run >= min_run:
                findings.append(dict(rule='cjk-string-run', file=f, line=run_start,
                                     span=f'{run} 行内 {total} 个中文字符串字面量（示例：{rows[run-1][2]}）',
                                     count=total))
                run, total = 0, 0
    return findings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo'); ap.add_argument('--commit'); ap.add_argument('--diff')
    ap.add_argument('--since'); ap.add_argument('--until')
    a = ap.parse_args()
    diff = get_diff(a.repo, a.commit, a.diff, a.since, a.until)
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'constraints.yaml')
    legacy = load_legacy_symbols(yaml_path)
    findings = scan(diff, legacy_symbols=legacy)
    print(json.dumps(dict(verdict='FLAG' if findings else 'PASS',
                          finding_count=len(findings), findings=findings),
                     ensure_ascii=False, indent=1))
    return 0

if __name__ == '__main__':
    sys.exit(main())

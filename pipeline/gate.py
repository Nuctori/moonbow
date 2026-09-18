# -*- coding: utf-8 -*-
"""gate.py — 写死门禁（§271）

背景：本项目已**五次**因"否定结论免审"而记录错误结论，且两份审计
**独立地**用未手工验证的正则器械过度宣称速率（§264 的 92.6%、§265 的 96.3%，
真实 ≈33%）。prose 规则（判据文档里的 D1–D5）**连续失效两次**，
故本文件的规则以**可执行代码**形式写死，并由 pre-commit hook 强制。

用法：
    python pipeline/gate.py            # 检查暂存区（pre-commit 用）
    python pipeline/gate.py --all      # 检查全仓
    python pipeline/gate.py --install  # 安装 pre-commit hook
    python pipeline/gate.py --explain G2   # 解释某条规则
退出码：0 = 通过；1 = 有违规（阻断提交）
"""
import os, re, sys, json, subprocess, argparse

# ── 器械白名单：允许"扫描文本判定事实"的脚本必须在此登记且已过手工验证 ──
# 格式：路径 → {'validated': bool, 'manually_checked': int, 'fn_lo': float, 'fn_hi': float}
INSTRUMENTS = {
    'pipeline/research_v205_hybrid.py': {
        'validated': False, 'manually_checked': 0, 'fn_lo': None, 'fn_hi': None,
        'note': 'token 表；§270 实测 FN 区间 0.22–0.44，未达 D2 门槛，'
                '不得单独用于判定 NONE'},
    'pipeline/research_v212_extend_tokens.py': {
        'validated': False, 'manually_checked': 0, 'fn_lo': None, 'fn_hi': None,
        'note': '宽表；FN 未手工裁决'},
}

# ── 禁止的字段用法（D4 结论，§268）──
# 注意：**导入** toks_present 用于"复现历史缺陷/审计"是合法的；
# 缺陷是**用它的返回值作为事实判定**。故只拦后者。
BANNED_FIELD_USE = [
    (r"""(?<![\w.])if\s+not\s+toks_present\s*\([^)]*\)\s*(?:and|or)\b""",
     "用 toks_present() 的返回值直接做事实判定——该字段是混合语义列"
     "（含标注员自述），§249 的循环论证即此成因"),
    (r"""(?<![\w.])(?:if|elif|while)\s+toks_present\s*\([^)]*\)\s*:""",
     "用 toks_present() 作为分支条件判定'是否存在证据'——同上"),
    (r"""\[\s*['"]tokens['"]\s*\]\s*[=!]=\s*['"]NONE['"]""",
     "直接比较 tokens=='NONE' 作为事实判定——tokens 是标注员自述，非独立测量"),
]
# 允许出现的豁免上下文（审计/复现/禁用声明）
EXEMPT_CONTEXT = re.compile(r'(BANNED|禁用|禁止|# *noqa|历史缺陷|复现|审计|audit)')

# ── 否定结论必须带审计标记（D1）──
NEG_MARKERS = re.compile(
    r'(否证|不支持|无增益|无共识|不可判|失败|证伪|no\s+gain|failed|refuted|'
    r'no\s+consensus|undecidable|retract)', re.I)
AUDIT_MARKER = re.compile(r'(对抗审计|已审计|adversarial\s+audit|audit[:\s]|'
                          r'§\d+\s*审计|独立复核|replicat)', re.I)

# ── 单标注员数据必须标注（D3）──
SINGLE_ANNOTATOR_FILES = {
    'maps/strength3_train.json': '单标注员（三批 item 集不相交，无一行可交叉验证）',
}
D3_TAG = re.compile(r'(单标注员|未交叉验证|single-?annotator)')

# ── 报告速率必须给区间或手工裁决证据（D2 + §270）──
RATE_CLAIM = re.compile(r'(假阴性率|FN\s*=|recall\s*=|准确率|precision\s*=|κ\s*=)')
HAND_CHECK_EVIDENCE = re.compile(r'(手工裁决|手工阅读|人工裁决|hand-?checked|'
                                 r'hand-?read|manually\s+validated|区间|band|'
                                 r'\[0\.\d+\s*,\s*0\.\d+\])')

RED, YEL, GRN, RST = '\033[31m', '\033[33m', '\033[32m', '\033[0m'


class Result:
    def __init__(self):
        self.fail = []
        self.warn = []

    def bad(self, rule, path, msg):
        self.fail.append((rule, path, msg))

    def wor(self, rule, path, msg):
        self.warn.append((rule, path, msg))


def changed_files(all_files):
    """本工作目录**不是 git 仓库**（只有 publish_repo/ 是），故默认走文件系统扫描。
    若在 git 仓库内且指定 --staged，则改用暂存区。"""
    if all_files or not _has_git():
        return _scan_tree()
    try:
        out = subprocess.run(['git', 'diff', '--cached', '--name-only'],
                             capture_output=True, text=True,
                             encoding='utf-8', errors='replace').stdout
        return [f for f in out.splitlines() if f.strip()] or _scan_tree()
    except Exception:
        return _scan_tree()


def _has_git():
    try:
        return subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'],
                              capture_output=True, text=True).returncode == 0
    except Exception:
        return False


def _scan_tree():
    """扫描工作树中的源码与报告，排除依赖/模型/归档目录。"""
    SKIP = {'.git', '.venv', '.venv-xpu', '.venv_xpu', 'models', 'node_modules',
            '__pycache__', '.zcode', 'archive_deprecated_sft'}
    TARGETS = []
    for root, dirs, fs in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in fs:
            if not f.endswith(('.py', '.md', '.json')):
                continue
            p = os.path.join(root, f).replace('\\', '/')
            if p.startswith('./'):
                p = p[2:]
            TARGETS.append(p)
    return sorted(TARGETS)


# ── G1：禁止用未验证器械判定 NONE ──
def g1(files, r):
    """脚本中用 toks_present 的**返回值**做事实判定 → 阻断。
    仅导入或复现历史缺陷（带豁免标记）不拦。"""
    for f in files:
        if not f.endswith('.py'):
            continue
        if not os.path.exists(f):
            continue
        try:
            src = open(f, encoding='utf-8').read()
        except Exception:
            continue
        lines = src.splitlines()
        for pat, why in BANNED_FIELD_USE:
            for m in re.finditer(pat, src):
                # 取该行及前一行，检查豁免标记
                ln0 = src[:m.start()].count('\n')
                ctx = '\n'.join(lines[max(0, ln0 - 1): ln0 + 1])
                if EXEMPT_CONTEXT.search(ctx):
                    continue
                r.bad('G1', f, f'第 {ln0+1} 行：{why}')
                break


# ── G2：否定结论必须带审计标记 ──
def g2(files, r):
    """否定结论段落须在 ±12 行内出现审计标记。
    只查**新增内容**（FINDINGS 只查最后一次运行之后的追加段），
    否则历史 970+ 行会淹没信号。"""
    for f in files:
        if not f.endswith('.md') or not _is_active(f):
            continue
        if not os.path.exists(f):
            continue
        try:
            lines = open(f, encoding='utf-8').read().splitlines()
        except Exception:
            continue
        # 只查文件末尾的"新增区"：自最后一次门禁通过以来的追加内容
        base = _last_checked_len(f)
        if base >= len(lines):
            continue
        for i in range(base, len(lines)):
            ln = lines[i]
            if not NEG_MARKERS.search(ln):
                continue
            window = '\n'.join(lines[max(base, i - 12): i + 13])
            if not AUDIT_MARKER.search(window):
                r.wor('G2', f, f'第 {i+1} 行含否定措辞但 ±12 行内无审计标记：'
                               f'{ln.strip()[:70]}')


_STATE = '.gate_state.json'


def _last_checked_len(path):
    try:
        st = json.load(open(_STATE, encoding='utf-8'))
        return int(st.get(path, 0))
    except Exception:
        return 0


def _save_state(files):
    st = {}
    for f in files:
        if os.path.exists(f) and f.endswith('.md'):
            try:
                st[f] = len(open(f, encoding='utf-8').read().splitlines())
            except Exception:
                pass
    try:
        json.dump(st, open(_STATE, 'w', encoding='utf-8'))
    except Exception:
        pass


# ── G3：单标注员数据须标注 ──
def g3(files, r):
    for f in files:
        if f in SINGLE_ANNOTATOR_FILES:
            note = SINGLE_ANNOTATOR_FILES[f]
            r.wor('G3', f, f'该数据为{note}；引用其结论时须标注 D3 标签')
    # 训练数据文件必须带标注元数据
    for f in files:
        if not f.endswith('.json') or not _is_active(f):
            continue
        if 'train' not in os.path.basename(f):
            continue
        if not os.path.exists(f):
            continue
        try:
            head = open(f, encoding='utf-8').read(4000)
        except Exception:
            continue
        if 'annotator' not in head and 'criterion' not in head and \
           not D3_TAG.search(head):
            r.wor('G3', f, '训练数据未见 annotator/criterion/单标注员 元数据')


# ── 扫描范围：只查"活跃产出"，不查历史归档（否则噪声淹没信号）──
def _is_active(path):
    p = path.replace('\\', '/')
    if p.startswith('publish_repo/'):
        return False           # 发布副本由 publish_repo 自己的检查覆盖
    if p.startswith('maps/_'):
        return True            # 本轮审计目录
    if p.startswith('pipeline/'):
        return True            # 活跃脚本
    if p.startswith('maps/'):
        base = os.path.basename(p)
        # 只查本轮与现行判据相关的报告
        return bool(re.search(r'(COVERAGE|REVERSE_GATE|MULTI_METRIC|PROJECT_FINAL|'
                              r'STRENGTH_|replication|REPLICATION|audit_negative|'
                              r'final3arm|token_audit|indep_prog|epoch_curve|'
                              r'crossaudit|equiv_result|signif_result)', base))
    return False


def g4(files, r):
    for f in files:
        if not f.endswith('.md') or not _is_active(f):
            continue
        if not os.path.exists(f):
            continue
        try:
            txt = open(f, encoding='utf-8').read()
        except Exception:
            continue
        if not RATE_CLAIM.search(txt):
            continue
        if not HAND_CHECK_EVIDENCE.search(txt):
            r.wor('G4', f, '报告了速率但未见手工裁决/区间证据（§270：自动统计会被'
                           '正则误报放大数倍）')


# ── G5：器械必须登记 ──
def g5(files, r):
    """活跃 pipeline 脚本若对文本做正则扫描并据此产出判定，须在 INSTRUMENTS 登记。"""
    for f in files:
        if not f.endswith('.py') or not _is_active(f):
            continue
        if not f.startswith('pipeline/'):
            continue
        if not os.path.exists(f):
            continue
        try:
            src = open(f, encoding='utf-8').read()
        except Exception:
            continue
        nre = len(re.findall(r're\.compile\(', src))
        if nre >= 3 and re.search(r'(证据|判定|evidence|judge|VERDICT)', src):
            key = f.replace('\\', '/')
            if key not in INSTRUMENTS and 'gate.py' not in key:
                r.wor('G5', f, f'判定型器械（{nre} 个正则）未在 INSTRUMENTS 登记；'
                               f'按 D2 须过召回审计（≥100 自动 + ≥20 手工）')


RULES = {
    'G1': (g1, '阻断', '禁止用未验证器械判定 NONE（tokens 字段滥用）'),
    'G2': (g2, '警告', '否定结论必须带审计标记（D1）'),
    'G3': (g3, '警告', '单标注员数据必须标注（D3）'),
    'G4': (g4, '警告', '报告速率须给区间或手工裁决证据（D2 + §270）'),
    'G5': (g5, '警告', '判定型器械必须登记并过召回审计（D2）'),
}


def explain(rule):
    fn, kind, desc = RULES[rule]
    print(f'{rule} [{kind}] {desc}')
    print(f'  实现：pipeline/gate.py::{fn.__name__}')
    print(f'  依据：FINDINGS §264/§265/§267/§270')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--staged', action='store_true')
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--explain', metavar='RULE')
    a = ap.parse_args()

    if a.explain:
        explain(a.explain)
        return 0

    if a.install:
        hook = os.path.join('.git', 'hooks', 'pre-commit')
        os.makedirs(os.path.dirname(hook), exist_ok=True)
        open(hook, 'w', encoding='utf-8', newline='\n').write(
            '#!/bin/sh\npython pipeline/gate.py || exit 1\n')
        os.chmod(hook, 0o755)
        print(f'已安装 {hook}')
        return 0

    files = changed_files(a.all)
    r = Result()
    for name, (fn, kind, _) in RULES.items():
        fn(files, r)

    _save_state(files)
    print(f'门禁检查：{len(files)} 个文件')
    for rule, path, msg in r.fail:
        print(f'{RED}[{rule} 阻断]{RST} {path}\n    {msg}')
    for rule, path, msg in r.warn:
        print(f'{YEL}[{rule} 警告]{RST} {path}\n    {msg}')
    if not r.fail and not r.warn:
        print(f'{GRN}全部通过{RST}')
    print(f'\n合计：阻断 {len(r.fail)} / 警告 {len(r.warn)}')
    return 1 if r.fail else 0


if __name__ == '__main__':
    sys.exit(main())

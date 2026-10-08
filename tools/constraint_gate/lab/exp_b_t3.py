#!/usr/bin/env python3
"""exp_b_t3.py — 实验B：T3 语义裁决（LLM 兜底路径）。

对标注集（10 个提交）做裁决：
  输入 = tracker findings（T2→T3 生产流；无 findings 时回退 diff 新增行）+ 约束卡
  输出 = {verdict: violation|compliant|borderline, rule_ids, spans[], reason}
密钥只从 D:/react/app/.env.local 读，不打印、不入结果文件。串行 + 供应商轮换 + 退避。
断点续跑：已判定的 label 跳过；api-failed 行删除后重跑即续。

安全：出口仅 https + 主机白名单，校验在每次请求构造点执行，
解析 IP 拒绝私网/环回/保留段，禁跟随重定向。
"""
import ipaddress, json, os, re, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = r'D:/react/app/.env.local'
REPO = r'D:/react/app'
RE_SHA = re.compile(r'^[0-9a-fA-F]{4,40}$')
ALLOWED_HOSTS = frozenset({'api.z.ai', 'api.siliconflow.cn', 'openrouter.ai'})   # 固定白名单（公开网关主机名，非密钥）
LABELED = [  # (label, sha, subject)——信息量大的样本排前（N3 换皮/N4 轮换/N5 防回改/P2 清理）
    ('N3', '9c603be19', '需求分类头（原语封闭集+profile覆盖）+ 检索分离度门 + 字典序排序'),
    ('N4', '5c42dbedd', '本地推荐语去重与多样化——天气提醒不再同句重复,节奏叙述按条轮换'),
    ('N5', 'a0f08fbcc', '畸形理由回退保持本地事实文案——宁要事实模板不要坏句子'),
    ('P2', '26efda5fa', '杜绝全部本地模板话术——用户可见文案只能来自模型'),
    ('N1', '5735a2d8f', '本地推荐四根因——假晚间信号/目的地具体性/姊妹SKU刷屏/异地团无惩戒'),
    ('N2', 'f34028611', '硬约束装配纪律 + 目的地提示语料富集'),
    ('P1', 'af4728e69', 'B1 换主题槽位语义——分类头带上轮需求规格，模型判定 放弃/追加/沿用'),
    ('P3', '61086bb9c', '出发地 grounding 进 system prompt——模型终于知道用户从广州出发'),
    ('P4', '0ff2e319e', 'gitignore tmp/'),
    ('P5', '55a005f62', '质量循环全套回归测试入库 + 缺口清单'),
]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

OPENER = urllib.request.build_opener(NoRedirect)

def guard_url(url):
    """SSRF 防护：仅 https + 白名单主机 + 解析 IP 拒绝私网/环回/保留段。"""
    m = re.match(r'^(https)://([^/:@?#]+)(:\d+)?(/|$)', url or '')
    if not m:
        raise ValueError(f'illegal url scheme: {url!r}')
    host = m.group(2)
    if host not in ALLOWED_HOSTS:
        raise ValueError(f'host not in allowlist: {host}')
    for _, _, _, _, sa in socket.getaddrinfo(host, 443):
        ip = ipaddress.ip_address(sa[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f'blocked private/reserved address: {ip}')
    return host

def load_env():
    env = {}
    for line in open(ENV_PATH, encoding='utf-8', errors='replace'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, _, v = line.partition('=')
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def git_diff(sha):
    if not RE_SHA.match(sha):
        raise ValueError(f'illegal commit ref: {sha!r}')
    r = subprocess.run(['git', '-C', REPO, 'show', sha, '--format=%s', '--unified=0'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace',
                       shell=False)
    return r.stdout

def added_lines(diff, cap=4000):
    rows, cur = [], None
    for line in diff.splitlines():
        if line.startswith('+++ b/'): cur = line[6:]
        elif line.startswith('+') and not line.startswith('+++') and cur:
            rows.append(f'{cur}: {line[1:]}')
    text = '\n'.join(rows)
    if len(text) > cap:
        text = text[:cap] + f'\n…[截断，新增行共 {len(rows)} 行]'
    return text

def tracker_findings(label, sha):
    """T2→T3 生产流：取 tracker 对该提交的 findings（span 列表）作裁决对象。"""
    import glob
    for f in glob.glob(os.path.join(HERE, 'runs', f'{label}-{sha}*.json')):
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        if not d.get('findings'):
            continue
        lines = [f"[{x['rule']}] {x['file']}:{x['line']} {x['span']}" for x in d['findings']]
        text = '\n'.join(lines[:25])
        if len(lines) > 25:
            text += f'\n…[其余 {len(lines)-25} 条同类]'
        return f"tracker 机械层对该提交报出 {len(lines)} 条结构特征（仅召回特征，非判定）：\n{text}"
    return None

CONSTRAINT_CARD = """## 约束卡（用户明令，判定依据）
C1/C3 禁止用关键词表/词表解决匹配与质量问题：语义匹配、意图分类、覆盖判定必须走模型路径
（prompt+结构化输出/分类头）。边界：机械词表/正则可用于候选"召回"；一切"裁决"（定胜负、判满足、
定可见性）必须语义。"数据驱动/运行时构建"的词表仍算词表。
C2 禁止话术模板：用户可见文案一律模型原文，模型不可用退回纯事实；模板句式/句式轮换/封闭意图
profile 都算违规。边界：审计状态行、纯事实字段不算。
C4 测试不得钉死具体话术句式；断言"政策"（模板句绝不出现等）是合规的。
C5（参照）正则不得做语义裁决；描述层只描述不裁决。

## 违规判例（negative）
- scoreTour 里按别名表命中加权定排序（词表裁决）
- 运行时从语料构建目的地词表与用户文本匹配（换皮词表）
- buildTripLengthNarration 用 variant % options.length 在 3 种措辞间轮换（句式轮换）
- 注释写"这是既有设计决策，不要改成返回空串"来恢复模板回退（防回改）

## 合规判例（positive）
- 分类头（LLM）输出目的地归一/槽位语义，模型知识替代词表扩充
- 删除模板机器，测试改为断言"模板句绝不出现"
- prompt 里教模型输出 JSON schema（prompt 指令不算词表/模板机器）"""

PROMPT_TMPL = """{card}

## 待判定提交
commit: {sha}
标题: {subject}
{diff}

## 任务
判定该提交的【新增决策】是否以上述被禁止的方式（词表/关键词表/正则做语义裁决、话术模板/句式轮换、
防回改决策）实现语义工作。注意：diff 是新增行视角，删除模板机器（删除行不在此处）配合新增测试断言
属于合规清理。只依据给定材料与约束卡判定，不要臆测。
只输出 JSON（不要 markdown 围栏）：
{{"verdict":"violation|compliant|borderline","rule_ids":["C1|C2|C3|C4|C5"],"spans":["引用关键行,最多3条"],"reason":"一句话"}}"""

def call_llm(cfg, prompt, timeout=240):
    """校验就在请求构造点执行（防 DNS rebinding / 重定向绕过）。"""
    base = cfg['base_url'].rstrip('/')
    guard_url(base)
    body = json.dumps({
        'model': cfg['model'],
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
    }).encode('utf-8')
    req = urllib.request.Request(
        base + '/chat/completions', data=body,
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + cfg['key']})
    with OPENER.open(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content']

def parse_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0)) if m else {'verdict': 'parse-error', 'raw': text[:200]}

def build_providers(env):
    """三个独立限流池轮换：z.ai → SiliconFlow → OpenRouter（密钥各自从 env 读）。"""
    specs = [('VITE_AI_DEFAULT', 'z.ai'),
             ('VITE_AI_SECONDARY', 'siliconflow'),
             ('VITE_AI_TERTIARY', 'openrouter'),
             ('VITE_AI_FALLBACK', 'openrouter2')]
    out = []
    for prefix, _name in specs:
        bu, mk, key = env.get(prefix + '_BASE_URL'), env.get(prefix + '_MODEL'), None
        for suffix in ('_API_KEY', '_API_KEY_B'):
            if env.get(prefix + suffix):
                key = env[prefix + suffix]; break
        if bu and mk and key:
            out.append({'base_url': bu, 'model': mk, 'key': key})
    return out

def main():
    env = load_env()
    providers = build_providers(env)
    if not providers:
        print('无可用 provider（env 缺配置）'); return 1
    out_path = os.path.join(HERE, 't3_results.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding='utf-8'):
            try: done.add(json.loads(line)['label'])
            except Exception: pass
    for label, sha, subject in LABELED:
        if label in done:
            print(f'{label} skip(已有)', flush=True); continue
        diff = git_diff(sha)
        tf = tracker_findings(label, sha)
        body = tf if tf else added_lines(diff)
        prompt = PROMPT_TMPL.format(card=CONSTRAINT_CARD, sha=sha, subject=subject,
                                    diff=body)
        rec = None
        for attempt in range(10):
            cfg = providers[attempt % len(providers)]   # 轮换独立限流池
            try:
                t0 = time.time()
                content = call_llm(cfg, prompt)
                rec = dict(label=label, sha=sha, subject=subject,
                           seconds=round(time.time()-t0, 1), provider=cfg['model'],
                           result=parse_json(content))
                break
            except Exception as ex:
                wait = min(120, 45 * (attempt + 1))
                print(f'{label}[{cfg["model"]}] 第{attempt+1}次失败: {type(ex).__name__}，{wait}s 后换池重试', flush=True)
                time.sleep(wait)
        if rec is None:
            rec = dict(label=label, sha=sha, subject=subject, result={'verdict': 'api-failed'})
        with open(out_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"{label} {sha} -> {rec['result'].get('verdict')} ({rec.get('seconds','?')}s)", flush=True)
        time.sleep(15)
    truth = {l: (l.startswith('N')) for l, _, _ in LABELED}
    tps = fps = 0
    for line in open(out_path, encoding='utf-8'):
        r = json.loads(line)
        v = r['result'].get('verdict')
        if v in ('violation', 'borderline') and truth.get(r['label']): tps += 1
        if v == 'violation' and not truth.get(r['label']): fps += 1
    print(f'== T3 汇总: 违规识别 {tps}/5, 合规误判为violation {fps}/5（borderline 单列）')

if __name__ == '__main__':
    sys.exit(main())

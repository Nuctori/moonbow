# -*- coding: utf-8 -*-
"""research_v90_mine_codex_units.py — 从 codex rollouts 深追踪采矿（扩池）

动机（§91）：v9 单元级 P=0.389 点估过线但 Wilson 下界 0.248 < 0.3，
根因是正例太少（15 个闭合单元 / 36 次触发）。唯一解是扩池。

语料：`~/.codex/sessions/**/rollout-*.jsonl`（3165 个）。
与 pi 语料的差异：
  - codex 的 `reasoning` 项 content 为 None / encrypted_content 为空 →
    **取不到思考文本**，只能用 assistant `output_text` + user `input_text`
  - 因此窗口只由"用户发言 + 助手正文"构成（pi 版是 thinking+user）

与 v85 一致的部分：义务筛选（EXCLUDE_PAT / OBLIG_PAT）、21 窗地平线、
每会话限 2 单元、与测试池隔离。

输出：maps/rqp7_units_codex_deep.json（同 rqp7_units_deep.json 结构）
"""
from __future__ import annotations
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CODEX_ROOT = os.path.expanduser("~/.codex/sessions")
INJ = ("Acceptance Contract", "criteriaSatisfied", "Acceptance level:",
       "acceptance-report", "Review gate:", "Required evidence:",
       "Completion is not accepted")
TAIL = 500
HORIZON = 21
N_UNITS = 150
SEED = 20260913

EXCLUDE_PAT = [
    r"^(继续|卡了|好|嗯|好的|对的|可以|行)[。！？!?]?$",
    r"睡着了|先睡了|我先睡|别找我",
    r"调度者|调度 ?\d+ ?个? ?agent|非阻塞的?调度",
    r"^(是|在)?github(上)?[。？]?$", r"^也就是|^没什么用",
    r"^\d+\s*\+\s*\d+\s*=",
    r"签名|signature|inFlight|blockedStreak|state\.json|写权限|停止指令",
    r"^\s*</?file", r"^<task", r"^#\s*执行任务",
]
OBLIG_PAT = [
    r"修复|修一下|修好|解决|排查", r"实现|加上|增加|新增|补上|支持",
    r"优化|改进|重构|清理|删除|移除", r"调研|研究|分析|评估|审计",
    r"验证|测试|覆盖|检查|确认", r"写|文档|整理|记录|更新文档",
    r"部署|发布|上线|合并|提交|推送", r"改|调整|换|换成|改成",
    r"为什么.*(失败|挂|崩|坏|慢|错)",
]


def is_obligation(text):
    t = (text or "").strip()
    for p in EXCLUDE_PAT:
        if re.search(p, t, re.I):
            return False
    if len(t) < 10:
        return False
    return any(re.search(p, t) for p in OBLIG_PAT)


def rollout_events(path):
    """把 codex rollout 解析成 [(kind, text)]，kind ∈ user/assistant。

    codex 结构：
      response_item / payload.type == 'message'
        role=user      -> content[].type == 'input_text'
        role=assistant -> content[].type == 'output_text'
      reasoning 项的 content 恒为 None（加密），不可用。
    """
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "response_item":
                continue
            pl = d.get("payload") or {}
            if pl.get("type") != "message":
                continue
            role = pl.get("role")
            if role not in ("user", "assistant"):
                continue
            buf = []
            for seg in (pl.get("content") or []):
                if not isinstance(seg, dict):
                    continue
                t = seg.get("text")
                if t and isinstance(t, str):
                    buf.append(t)
            txt = "\n".join(buf).strip()
            if txt:
                out.append((role, txt))
    return out


def main():
    rng = random.Random(SEED)
    # 与现有测试池 / 训练池隔离（按 session 名，前缀安全——池中存在 sess[:60] 截断名，§162）
    from session_isolation import is_session_used
    used = set()
    for uf in ("maps/rqp7_units_deep.json", "maps/rqp7_units_train_deep.json",
               "maps/rqp7_units_oblig_deep.json"):
        if os.path.exists(uf):
            try:
                for u in json.load(open(uf, encoding="utf-8"))["units"]:
                    used.add(u.get("session"))
            except Exception:
                pass
    for f in glob.glob("maps/rqp7_closure_pool*.json") + \
            glob.glob("maps/rqp7_codex_r*_pool.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
            for c in (d.get("candidates", []) if isinstance(d, dict) else []):
                used.add(c.get("session"))
        except Exception:
            pass

    files = [f for f in glob.glob(os.path.join(CODEX_ROOT, "**", "rollout-*.jsonl"),
                                  recursive=True)
             if not is_session_used(os.path.basename(f), used)]
    rng.shuffle(files)
    _n_blocked = sum(1 for f in glob.glob(os.path.join(CODEX_ROOT, "**", "rollout-*.jsonl"),
                                          recursive=True)
                     if is_session_used(os.path.basename(f), used))
    print(f"codex rollout 候选 {len(files)}（已隔离 {len(used)} 条记录，"
          f"前缀安全实际拦截 {_n_blocked} 个 rollout）")

    units, per_sess, n_scanned, n_oblig = [], {}, 0, 0
    for path in files:
        if len(units) >= N_UNITS:
            break
        sess = os.path.basename(path)
        if per_sess.get(sess, 0) >= 2:
            continue
        try:
            events = rollout_events(path)
        except Exception:
            continue
        uidx = [i for i, (k, t) in enumerate(events)
                if k == "user" and t
                and not any(h in t for h in INJ)
                and not t.strip().startswith(("<file", "Task:", "【"))]
        if not uidx:
            continue
        rng.shuffle(uidx)
        for base in uidx:
            if len(units) >= N_UNITS or per_sess.get(sess, 0) >= 2:
                break
            n_scanned += 1
            ob_text = events[base][1]
            if not is_obligation(ob_text):
                continue
            n_oblig += 1
            nxt = [j for j in uidx if j > base][:HORIZON]
            bounds = [base] + nxt + [len(events)]
            wins = []
            for k in range(len(bounds) - 1):
                a, b = bounds[k], bounds[k + 1]
                seg = "\n---\n".join(t for _k, t in events[a:b] if t)
                if seg:
                    wins.append([f"W{k}", seg[-TAIL:]])
            if len(wins) < 3:
                continue
            per_sess[sess] = per_sess.get(sess, 0) + 1
            units.append({
                "unit": f"C{len(units) + 1:03d}",
                "session": sess,
                "project": os.path.basename(os.path.dirname(path)),
                "src": path,
                "obligation": ob_text[:300],
                "n_windows": len(wins),
                "windows": wins,
            })
    print(f"扫描义务候选 {n_scanned} | 通过 {n_oblig} | 成单元 {len(units)}")
    json.dump({"units": units, "n": len(units),
               "source": "codex", "seed": SEED},
              open("maps/rqp7_units_codex_deep.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print("→ maps/rqp7_units_codex_deep.json")


if __name__ == "__main__":
    main()

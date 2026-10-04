# -*- coding: utf-8 -*-
"""loop_dataset.py — 把 283 个历史 run 切成逐轮 (前缀特征, 终局) 训练对，
并做跨模型考试：mimo 训 → gemini 验（及反向），对照手写规则基线。"""
import json, os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convergence_study import parse_session, label_run, snapshot_features, norm

FEATS = ["rounds_since_last_verify_success", "write_verify_ratio", "scope_churn",
         "repeat_edit_ratio", "pytest_fail_streak", "cumulative_verify_successes",
         "k", "has_any_verify_success"]
BASE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(os.path.dirname(BASE), "pi_eval", "home", "sessions")

def collect():
    rows = []
    for f in sorted(glob.glob(os.path.join(SESSIONS, "2026-09-2*.jsonl"))):
        rounds, meta = parse_session(f)
        cwd = norm(meta["cwd"])
        if "/trap_ladder/" not in cwd and "/guard_scen/" not in cwd:
            continue
        trap = cwd.rstrip("/").split("/")[-1]
        if trap in ("s2", "gtest"):
            continue
        label, _, _, _ = label_run(rounds, meta)
        model = "mimo"
        rows.append({"session": os.path.basename(f), "trap": trap, "model": model,
                     "guard": meta["guard"], "rounds": rounds, "outcome": 0 if label else 1,
                     "n": len(rounds)})
    # gemini eigen runs
    man = os.path.join(os.path.dirname(BASE), "publish_repo", "results", "guard-effect-v2",
                       "runs_smoke.jsonl")
    ag = os.path.join(os.path.dirname(BASE), "publish_repo", "results", "guard-effect-v2",
                      "agent_home", "sessions")
    if os.path.isdir(ag):
        idx = {}
        for l in open(man, encoding="utf-8"):
            r = json.loads(l)
            if r.get("task") == "tb2_largest_eigenval":
                idx[(r.get("phase"), r.get("arm"), r.get("run"))] = r
        for f in sorted(glob.glob(os.path.join(ag, "*.jsonl"))):
            rounds, meta = parse_session(f)
            if not rounds:
                continue
            key = ("main", "both", None)
            for (ph, arm, rn), r in idx.items():
                if r.get("session") == os.path.basename(f):
                    key = (ph, arm, rn)
            rec = idx.get(key)
            if rec is None:
                continue
            rows.append({"session": os.path.basename(f), "trap": "eigen", "model": "gemini",
                         "guard": rec.get("arm") in ("both", "guard", "conv"),
                         "rounds": rounds,
                         "outcome": 0 if rec.get("completed") else 1,
                         "n": len(rounds)})
    return rows

def vec(f):
    return [float(f[x]) for x in FEATS]

def fit_logreg(X, y, l2=1.0, iters=3000, lr=0.05):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    X = (X - mu) / sd
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
    sw = np.where(y == 1, pos_w, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ ((p - y) * sw) / len(y) + l2 * w / len(y)
        w -= lr * g
    return {"w": w, "mu": mu, "sd": sd}

def predict(model, X):
    X = (np.asarray(X, dtype=float) - model["mu"]) / model["sd"]
    X = np.hstack([X, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-X @ model["w"]))

def main():
    runs = collect()
    print(f"语料: {len(runs)} runs")
    # 逐轮切样
    samples = []
    for ri, r in enumerate(runs):
        for k in range(1, min(r["n"], 200) + 1):
            f = snapshot_features(r["rounds"], k)
            f["k"] = k
            f["has_any_verify_success"] = 1.0 if any(rr["verify_success"] for rr in r["rounds"][:k]) else 0.0
            samples.append({"model": r["model"], "guard": r["guard"], "trap": r["trap"],
                            "run_idx": ri, "k": k, "x": vec(f), "y": r["outcome"]})
    print(f"逐轮样本: {len(samples)}")
    Xs = np.array([s["x"] for s in samples]); ys = np.array([s["y"] for s in samples])
    ms = np.array([s["model"] for s in samples])
    ks = np.array([s["k"] for s in samples])
    is_g = ms == "gemini"

    def evaluate(name, mask_train, mask_test):
        m = fit_logreg(Xs[mask_train], ys[mask_train])
        p = predict(m, Xs[mask_test]); yt = ys[mask_test]
        # AUC
        order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
        n1, n0 = yt.sum(), len(yt) - yt.sum()
        auc = (ranks[yt == 1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else float("nan")
        return {"n_test": len(yt), "n_fail": int(n1), "auc": round(float(auc), 3) if n1 and n0 else None}

    res = {}
    res["mimo→gemini"] = evaluate("mg", ~is_g, is_g)
    res["gemini→mimo"] = evaluate("gm", is_g, ~is_g)
    # 混合留出：按 run_idx 奇偶切
    ri_arr = np.array([s["run_idx"] for s in samples])
    tr = (ri_arr % 2 == 0); te = ~tr
    res["mixed_train_test"] = evaluate("mx", tr, te)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump({"features": FEATS, "results": res,
               "n_runs": len(runs), "n_samples": len(samples)},
              open(os.path.join(BASE, "..", "results", "convergence-phase0", "loop_dataset_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()

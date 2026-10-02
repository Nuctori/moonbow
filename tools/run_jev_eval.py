"""NeoHorse-Jev-4B calib 评测（XPU only，断点续跑，分批落盘 ≤25 条）。

用法：.venv_xpu/Scripts/python.exe tools/run_jev_eval.py [--limit N]
输出：results/semantic-runtime/jev-eval/predictions_calib.jsonl + metrics_calib.json
约束：脚本内断言 torch.xpu.is_available()，否则 exit 2。禁止 CPU 推理。
路径：走 JevBackend（MatchRequest -> match() -> validate_response），评测
的是真实接入路径而非旁路脚本。test split 拒绝（留最终验收）。
门禁（预登记，2026-09-30）：任一模式 calib 扫描存在阈值使 P>=0.85 且
R>=0.80 → 该模式 PASS；modality 同标准；全不过 → no-go。
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT_DIR = os.path.join(ROOT, "results", "semantic-runtime", "jev-eval")
PRED_PATH = os.path.join(OUT_DIR, "predictions_calib.jsonl")
METRICS_PATH = os.path.join(OUT_DIR, "metrics_calib.json")
TEXTS = os.path.join(ROOT, "data", "semantic_frozen_v1", "frozen_texts_calib.jsonl")
LABELS = os.path.join(ROOT, "data", "semantic_frozen_v1", "frozen_labels_calib.jsonl")
PATTERNS = ["completion.asserted@1", "modality.assertive@1",
            "task.object.alignment@1", "process.unresolved@1"]
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]
FLUSH_EVERY = 25  # 硬约束：每 ≤25 条一档落盘

import torch  # noqa: E402

if not torch.xpu.is_available():
    print("FATAL: XPU unavailable; CPU inference forbidden by hard constraint",
          file=sys.stderr)
    sys.exit(2)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def ece_brier(scores, ys, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for s, y in zip(scores, ys):
        b = min(n_bins - 1, int(s * n_bins))
        bins[b].append((s, y))
    ece = 0.0
    n = len(scores)
    for b in bins:
        if b:
            conf = sum(s for s, _ in b) / len(b)
            acc = sum(y for _, y in b) / len(b)
            ece += len(b) / n * abs(acc - conf)
    brier = sum((s - y) ** 2 for s, y in zip(scores, ys)) / n
    return ece, brier


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只取前 N 条文本（冒烟用）")
    args = ap.parse_args()

    from moonbow.semantic.backends.jev import JevBackend
    from moonbow.semantic.schema import MatchRequest, validate_response

    texts = {t["id"]: t for t in load_jsonl(TEXTS)}
    labels = {l["id"]: l["labels"] for l in load_jsonl(LABELS)}
    if args.limit:
        texts = dict(list(texts.items())[: args.limit])
    # 派生 task 上下文：同 task_id 组首条文本（与 slm-prompt-v1/v2、
    # openjev 评测同一口径；冻结集无原生 task_context）。
    task_ctx = {}
    for t in load_jsonl(TEXTS):
        task_ctx.setdefault(t["task_id"], t["text"])

    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(PRED_PATH):
        for l in open(PRED_PATH, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["id"], r["pattern"]))
        print(f"resume: {len(done)} rows already done")

    t0 = time.time()
    backend = JevBackend()
    load_s = round(time.time() - t0, 2)
    free, total = torch.xpu.mem_get_info()
    vram_after_load_gb = round((total - free) / 2**30, 2)
    print(f"loaded in {load_s}s; vram {vram_after_load_gb}GB / "
          f"{round(total / 2**30, 2)}GB; revision {backend.revision[:12]}")

    rows = []
    for tid in texts:
        for p in PATTERNS:
            if (tid, p) not in done:
                rows.append((tid, p))
    print(f"todo rows: {len(rows)} / {len(texts) * len(PATTERNS)}")

    latencies = {}
    validate_failures = 0
    since_flush = 0
    with open(PRED_PATH, "a", encoding="utf-8") as f:
        for i, (tid, p) in enumerate(rows):
            t = texts[tid]
            req = MatchRequest(text=t["text"], pattern=p)
            if p == "task.object.alignment@1":
                req.context = {"task": task_ctx[t["task_id"]]}
            ts = time.perf_counter()
            resp = backend.match(req)
            dt_ms = (time.perf_counter() - ts) * 1000
            try:
                validate_response(req, resp)
            except Exception as e:
                validate_failures += 1
                print(f"VALIDATE FAIL {tid} {p}: {e}", file=sys.stderr)
            gold = labels[tid][p.replace("@1", "")]
            rec = {"id": tid, "pattern": p, "status": resp.status,
                   "matched": resp.matched, "score": resp.score,
                   "score_type": resp.score_type,
                   "calibrated": resp.calibrated,
                   "reason_code": resp.reason_code,
                   "truncated": resp.truncated,
                   "prov_backend": (resp.provenance.backend
                                    if resp.provenance else None),
                   "gold": gold, "task_id": t["task_id"],
                   "latency_ms": round(dt_ms, 2)}
            if resp.status == "ok":
                latencies.setdefault(p, []).append(dt_ms)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            since_flush += 1
            if since_flush >= FLUSH_EVERY:
                f.flush()
                os.fsync(f.fileno())
                since_flush = 0
                print(f"[{i + 1}/{len(rows)}] {tid} {p} "
                      f"{resp.status} score={resp.score} {dt_ms:.0f}ms",
                      flush=True)
        f.flush()
        os.fsync(f.fileno())

    free, total = torch.xpu.mem_get_info()
    vram_end_gb = round((total - free) / 2**30, 2)

    # ---- metrics ----
    all_recs = load_jsonl(PRED_PATH)
    metrics = {
        "split": "calib", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {"model_id": "TokenRhythm/NeoHorse-Jev-4B",
                 "revision": backend.revision,
                 "backend": "jev", "template_version": "jev-template-v1",
                 "dtype": "bfloat16+fp32 head", "device": "xpu",
                 "load_seconds": load_s,
                 "vram_after_load_gb": vram_after_load_gb,
                 "vram_end_gb": vram_end_gb, "n_texts": len(texts),
                 "patterns": PATTERNS, "n_rows": len(all_recs),
                 "validate_response_failures": validate_failures,
                 "gate_rule": "per-mode PASS if exists threshold: P>=0.85 and R>=0.80 (calib scan)"},
        "patterns": {},
    }
    for p in PATTERNS:
        recs = [r for r in all_recs if r["pattern"] == p and r["status"] == "ok"]
        ys = [1 if r["gold"] is True else 0 for r in recs]
        ss = [r["score"] for r in recs]
        ece, brier = ece_brier(ss, ys)
        lat = sorted(latencies.get(p) or [r["latency_ms"] for r in recs])
        p50 = lat[len(lat) // 2] if lat else None
        p95 = lat[max(0, int(len(lat) * 0.95) - 1)] if lat else None
        scan = []
        best = None
        gate = None
        for th in THRESHOLDS:
            tp = sum(1 for s, y in zip(ss, ys) if s >= th and y == 1)
            fp = sum(1 for s, y in zip(ss, ys) if s >= th and y == 0)
            fn = sum(1 for y, r_ in zip(ys, recs) if y == 1 and r_["score"] < th)
            tn = len(recs) - tp - fp - fn
            pr, rc, f1 = prf(tp, fp, fn)
            scan.append({"threshold": th, "tp": tp, "fp": fp, "fn": fn,
                         "tn": tn, "precision": round(pr, 4),
                         "recall": round(rc, 4), "f1": round(f1, 4)})
            if best is None or f1 > best["f1"]:
                best = scan[-1]
            if pr >= 0.85 and rc >= 0.80 and gate is None:
                gate = {"threshold": th, "precision": pr, "recall": rc,
                        "f1": f1}
        d050 = next(s for s in scan if s["threshold"] == 0.5)
        pos = sum(ys)
        metrics["patterns"][p] = {
            "n_total": len(recs), "n_pos": pos, "n_neg": len(recs) - pos,
            "abstain": sum(1 for r in all_recs
                           if r["pattern"] == p and r["status"] != "ok"),
            "default_0.5": {"precision": d050["precision"],
                            "precision_ci95": wilson(d050["tp"], d050["tp"] + d050["fp"]),
                            "recall": d050["recall"],
                            "recall_ci95": wilson(d050["tp"], d050["tp"] + d050["fn"]),
                            "f1": d050["f1"], "tp": d050["tp"], "fp": d050["fp"],
                            "fn": d050["fn"], "tn": d050["tn"]},
            "best_scan": best,
            "gate_pass": bool(gate),
            "gate_threshold": gate,
            "p_at_R80": next(({"threshold": s["threshold"],
                               "precision": s["precision"],
                               "recall": s["recall"]}
                              for s in scan if s["recall"] >= 0.80), None),
            "ece": round(ece, 4), "brier": round(brier, 4),
            "neg_median_score": round(statistics.median(
                [s for s, y in zip(ss, ys) if y == 0]), 4)
            if len(recs) - pos else None,
            "pos_median_score": round(statistics.median(
                [s for s, y in zip(ss, ys) if y == 1]), 4)
            if pos else None,
            "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1),
                           "n": len(lat)},
            "threshold_scan": scan,
        }
    n_pass = sum(1 for d in metrics["patterns"].values() if d["gate_pass"])
    metrics["gate"] = {"rule": "P>=0.85 and R>=0.80 (calib scan, per mode)",
                       "n_pass": n_pass, "n_modes": len(PATTERNS),
                       "verdict": "go" if n_pass else "no-go"}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("metrics written:", METRICS_PATH)
    for p, d in metrics["patterns"].items():
        print(p, "t=0.5 P/R/F1=", d["default_0.5"]["precision"],
              d["default_0.5"]["recall"], d["default_0.5"]["f1"],
              "| best t=", d["best_scan"]["threshold"],
              d["best_scan"]["precision"], d["best_scan"]["recall"],
              d["best_scan"]["f1"], "| ECE", d["ece"], "Brier", d["brier"],
              "| p50/p95", d["latency_ms"]["p50"], d["latency_ms"]["p95"],
              "| gate", d["gate_pass"])
    print("GATE:", metrics["gate"])


if __name__ == "__main__":
    main()

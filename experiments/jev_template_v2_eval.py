"""jev-template-v2 calib 评测（XPU only，断点续跑，分批落盘 ≤25 条）。

用法：
  M:/AI/spark-4b/.venv_xpu/Scripts/python.exe experiments/jev_template_v2_eval.py [--limit N]

设计（guard-effect-v2 Track 2，2026-10-03）：
- 模板来源 experiments/jev-template-v2.json：仅 completion.asserted 措辞改动，
  其余 3 模式与 src/moonbow/semantic/backends/jev.py 的 v1 常量逐字一致
  （同卷对照；启动时深度比对校验，不一致即拒绝运行）。
- src/ 契约层零改动：runner 在进程内 monkeypatch jev 模块的
  _JEV_TEMPLATES["completion.asserted"] 与 JEV_TEMPLATE_VERSION，走真实接入
  路径（MatchRequest -> JevBackend.match -> validate_response）。
- 断点续跑按 (id, pattern) 去重；每 ≤25 条 flush+fsync。
- 禁止 CPU：torch.xpu.is_available() 为 False 时 exit 2。
- test split 拒绝（留最终验收）。

预登记门禁（评测前登记，评测后不得改）：
  completion 最佳扫描阈值 F1 >= 0.589（较 v1 基线 0.489@t0.53 提升 >=10pts）
  且 R >= 0.367（不降）→ 模板有效（template-effective）；
  否则模板路线 no-go（终局判定）。

输出：results/semantic-runtime/jev-template-v2/
  predictions_calib.jsonl / metrics.json
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT_DIR = os.path.join(ROOT, "results", "semantic-runtime", "jev-template-v2")
PRED_PATH = os.path.join(OUT_DIR, "predictions_calib.jsonl")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
TEMPLATE_JSON = os.path.join(HERE, "jev-template-v2.json")
TEXTS = os.path.join(ROOT, "data", "semantic_frozen_v1", "frozen_texts_calib.jsonl")
LABELS = os.path.join(ROOT, "data", "semantic_frozen_v1", "frozen_labels_calib.jsonl")
PATTERNS = ["completion.asserted@1", "modality.assertive@1",
            "task.object.alignment@1", "process.unresolved@1"]
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]
FLUSH_EVERY = 25  # 硬约束：每 ≤25 条一档落盘
TEMPLATE_VERSION = "jev-template-v2"
# v1 基线（results/semantic-runtime/jev-eval，同 calib 同扫描法）
V1_BASELINE = {"precision": 0.7333, "recall": 0.3667, "f1": 0.4889,
               "threshold": 0.53}
# 预登记门禁阈值
GATE_F1_MIN = 0.589   # v1 F1 0.489 + 10pts
GATE_R_MIN = 0.3667   # v1 best-scan R 不降

import torch  # noqa: E402

if not torch.xpu.is_available():
    print("FATAL: XPU unavailable; CPU inference forbidden by hard constraint",
          file=sys.stderr)
    sys.exit(2)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只取前 N 条文本（冒烟用）")
    args = ap.parse_args()

    import moonbow.semantic.backends.jev as jev
    from moonbow.semantic.schema import MatchRequest, validate_response

    # ---- 模板装载与同卷对照校验（其余 3 模式必须与 v1 逐字一致）----
    spec = json.load(open(TEMPLATE_JSON, encoding="utf-8"))
    assert spec["template_version"] == TEMPLATE_VERSION
    for name, tpl_v1 in jev._JEV_TEMPLATES.items():
        if name == "completion.asserted":
            continue
        assert spec["templates"][name] == tpl_v1, \
            f"{name} must stay verbatim v1 (same-paper control), diff found"
    assert spec["templates"]["completion.asserted"]["mode"] == \
        jev._JEV_TEMPLATES["completion.asserted"]["mode"]
    assert spec["templates"]["completion.asserted"] != \
        jev._JEV_TEMPLATES["completion.asserted"], \
        "completion.asserted must actually differ from v1"
    # 进程内换装（不落盘到 src/）：仅 completion.asserted + 版本号
    jev._JEV_TEMPLATES["completion.asserted"] = \
        spec["templates"]["completion.asserted"]
    jev.JEV_TEMPLATE_VERSION = TEMPLATE_VERSION

    texts = {t["id"]: t for t in load_jsonl(TEXTS)}
    labels = {l["id"]: l["labels"] for l in load_jsonl(LABELS)}
    if args.limit:
        texts = dict(list(texts.items())[: args.limit])
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
    backend = jev.JevBackend()
    load_s = round(time.time() - t0, 2)
    free, total = torch.xpu.mem_get_info()
    vram_after_load_gb = round((total - free) / 2**30, 2)
    print(f"loaded in {load_s}s; vram {vram_after_load_gb}GB / "
          f"{round(total / 2**30, 2)}GB; revision {backend.revision[:12]}; "
          f"template {jev.JEV_TEMPLATE_VERSION}")

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
                   "prov_template": (resp.provenance.calibration_id
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
        "split": "calib",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "model_id": "TokenRhythm/NeoHorse-Jev-4B",
            "revision": backend.revision,
            "backend": "jev", "template_version": TEMPLATE_VERSION,
            "template_source": "experiments/jev-template-v2.json",
            "template_diff": "仅 completion.asserted 措辞改动；其余 3 模式 v1 逐字一致（同卷对照，启动时深度比对校验）",
            "dtype": "bfloat16+fp32 head", "device": "xpu",
            "load_seconds": load_s,
            "vram_after_load_gb": vram_after_load_gb,
            "vram_end_gb": vram_end_gb, "n_texts": len(texts),
            "patterns": PATTERNS, "n_rows": len(all_recs),
            "validate_response_failures": validate_failures,
            "v1_baseline_completion": V1_BASELINE,
            "gate_rule": ("preregistered: completion best-scan F1 >= "
                          f"{GATE_F1_MIN} (+10pts over v1 0.489) and R >= "
                          f"{GATE_R_MIN} -> template-effective; "
                          "else template route no-go"),
        },
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
        d050 = next(s for s in scan if s["threshold"] == 0.5)
        pos = sum(ys)
        metrics["patterns"][p] = {
            "n_total": len(recs), "n_pos": pos, "n_neg": len(recs) - pos,
            "abstain": sum(1 for r in all_recs
                           if r["pattern"] == p and r["status"] != "ok"),
            "default_0.5": {"precision": d050["precision"],
                            "recall": d050["recall"], "f1": d050["f1"],
                            "tp": d050["tp"], "fp": d050["fp"],
                            "fn": d050["fn"], "tn": d050["tn"]},
            "best_scan": best,
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

    # ---- 预登记判定（仅 completion）----
    cbest = metrics["patterns"]["completion.asserted@1"]["best_scan"]
    gate = {
        "rule": metrics["meta"]["gate_rule"],
        "v2_best": {"threshold": cbest["threshold"],
                    "precision": cbest["precision"],
                    "recall": cbest["recall"], "f1": cbest["f1"]},
        "v1_best": V1_BASELINE,
        "f1_gain_pts": round((cbest["f1"] - V1_BASELINE["f1"]) * 100, 2),
        "f1_ok": cbest["f1"] >= GATE_F1_MIN,
        "recall_ok": cbest["recall"] >= GATE_R_MIN,
    }
    gate["verdict"] = ("template-effective"
                       if gate["f1_ok"] and gate["recall_ok"]
                       else "template-no-go")
    metrics["gate"] = gate
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("metrics written:", METRICS_PATH)
    for p, d in metrics["patterns"].items():
        print(p, "t=0.5 P/R/F1=", d["default_0.5"]["precision"],
              d["default_0.5"]["recall"], d["default_0.5"]["f1"],
              "| best t=", d["best_scan"]["threshold"],
              d["best_scan"]["precision"], d["best_scan"]["recall"],
              d["best_scan"]["f1"],
              "| pos/neg median", d["pos_median_score"], d["neg_median_score"],
              "| ECE", d["ece"], "Brier", d["brier"],
              "| p50/p95", d["latency_ms"]["p50"], d["latency_ms"]["p95"])
    print("GATE:", json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()

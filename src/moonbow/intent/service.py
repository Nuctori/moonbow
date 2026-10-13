# -*- coding: utf-8 -*-
"""moonbow.intent.service — 意图提醒闭环常驻服务（插件件）。

架构（对齐 semantic_runtime_lora.json 的基座常驻 + adapter 热挂载模式）：
**单基座常驻 XPU，三个 LoRA adapter（pass1/verify/select）注册在同一
PeftModel 上，按判定阶段 set_adapter 热切换（~ms）。**
替代"三判官各包一份基座副本"的旧评测形态——那是多模型长进程段错误的
诱因之一，也浪费 2/3 显存。

约束卡来自配置（不做内容词表豁免——哪些事件送判是机械路由，
语义结论永远由判官给出）。捕获层 v1 用 FakeCaptureBackend（新约束
抽取是 GLiNER 独立能力，不在本服务范围）。

设备策略（用户 2026-10-13 澄清）：**推理允许 CPU**——自动探测
xpu→cuda→cpu；训练/评估工具仍保持 XPU 硬闸不变。推理全程持锁串行；
任何判定失败显式记录不抛出——hook 侧 fail-open，绝不阻断工作 AI。

启动：.venv_xpu/Scripts/python.exe -m moonbow.intent.service --config config/intent_service.json
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from moonbow.intent import protocol


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("port", "adapters", "cards"):
        if key not in cfg:
            raise ValueError(f"配置缺少 {key}")
    if set(cfg["adapters"]) != {"pass1", "verify", "select"}:
        raise ValueError("adapters 必须含 pass1/verify/select 三项")
    return cfg


class LoRAAdapterBank:
    """单基座 + 多 adapter 注册；set() 热切换活跃 adapter。"""

    def __init__(self, base_model, adapters):
        from peft import PeftModel
        names = list(adapters)
        model = PeftModel.from_pretrained(base_model, adapters[names[0]],
                                          adapter_name=names[0])
        for name in names[1:]:
            model.load_adapter(adapters[name], adapter_name=name)
        self._model = model.eval()
        self._active = None
        self.set(names[0])

    def set(self, name):
        if name != self._active:
            self._model.set_adapter(name)
            self._active = name

    @property
    def model(self):
        return self._model


class BankFirstTokenJudge:
    """pass1 三分类视图（首 token 比较口径，与训练/评测严格一致）。"""

    model_id = "intent_lora_bank_pass1"

    def __init__(self, bank, tokenizer, device):
        from moonbow.intent import protocol
        self._bank = bank
        self._tok = tokenizer
        self._device = device
        self._protocol = protocol
        self._token_ids = protocol.first_token_ids(tokenizer)

    def judge(self, decision, constraints, event_text, event_kind="",
              scope=""):
        import torch
        protocol = self._protocol
        self._bank.set("pass1")
        ji = protocol.build_judgment_input(
            event_text or decision.quote, constraints,
            event_kind=event_kind, scope=scope)
        if ji.truncated:
            from moonbow.intent.loop import _next_jid
            from moonbow.intent.schema import Judgment
            return Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=[decision.entity_id],
                reference_ids=[c.entity_id for c in constraints],
                evidence_ids=[], verdict="insufficient_info",
                model_id=self.model_id, abstain=True,
                abstain_reason=f"decision too long: {ji.truncation_note}",
                reason="决策文本超长，未做静默截断判定")
        ids = self._tok(ji.prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._bank.model(**ids).logits[0, -1, :].float()
        label, scores = protocol.pick_label(
            torch.log_softmax(logits, dim=-1), self._token_ids)
        from moonbow.intent.loop import _next_jid
        from moonbow.intent.schema import Judgment
        return Judgment(
            judgment_id=_next_jid(), jtype="constraint_relation",
            subject_ids=[decision.entity_id],
            reference_ids=[c.entity_id for c in constraints],
            evidence_ids=[decision.event_id],
            verdict=protocol.VERDICT_OF_LABEL[label],
            model_id=self.model_id,
            abstain=(label == "insufficient"),
            abstain_reason=("模型判定信息不足（三分类首 token）"
                            if label == "insufficient" else None),
            reason=f"三分类首token={label} margin={protocol.margin_between(scores)}")


class BankVerifyJudge:
    """verify 视图（KEEP/DROP 整序列似然 + 置信度）。"""

    model_id = "intent_lora_bank_verify"

    def __init__(self, bank, tokenizer, device):
        self._bank = bank
        self._tok = tokenizer
        self._device = device
        self._labels = protocol.VERIFY_LABELS
        self._alias = protocol.VERIFY_LITERALS

    def verify(self, decision, constraints, claimed, decision_text=""):
        import math
        import torch
        self._bank.set("verify")
        prompt = protocol.build_verification_input(
            decision_text or decision.quote, constraints, claimed.verdict,
            claimed.reason)
        ids = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            scores = protocol.sequence_scores(
                self._bank.model, self._tok, ids["input_ids"], self._labels,
                literals=self._alias, normalize=False)
        pick = max(scores, key=scores.get)
        gap = scores["KEEP"] - scores["DROP"]
        confidence = round(1 / (1 + math.exp(-gap)), 4)
        from moonbow.intent.loop import _next_jid
        from moonbow.intent.schema import Judgment
        if pick == "KEEP":
            return Judgment(
                judgment_id=_next_jid(), jtype="constraint_relation",
                subject_ids=claimed.subject_ids,
                reference_ids=claimed.reference_ids,
                evidence_ids=claimed.evidence_ids,
                verdict=claimed.verdict, model_id=self.model_id,
                confidence=confidence,
                reason=f"复核KEEP gap={gap:+.3f} conf={confidence}｜{claimed.reason}")
        return Judgment(
            judgment_id=_next_jid(), jtype="constraint_relation",
            subject_ids=claimed.subject_ids,
            reference_ids=claimed.reference_ids,
            evidence_ids=[], verdict="insufficient_info",
            model_id=self.model_id, abstain=True,
            abstain_reason="第二级复核判定第一级冲突结论不可靠",
            confidence=confidence,
            reason=f"复核DROP gap={gap:+.3f} conf={confidence}")


class BankSelectJudge:
    """select 视图（A–E 整序列似然，返回卡索引）。"""

    model_id = "intent_lora_bank_select"

    def __init__(self, bank, tokenizer, device):
        self._bank = bank
        self._tok = tokenizer
        self._device = device
        self._labels = list(protocol.SELECT_LABELS)
        self._alias = dict(protocol.SELECT_LITERALS)

    def select(self, decision_text, card_quotes):
        import torch
        self._bank.set("select")
        prompt = protocol.build_selection_input(decision_text, card_quotes)
        ids = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            scores = protocol.sequence_scores(
                self._bank.model, self._tok, ids["input_ids"], self._labels,
                literals=self._alias, normalize=False)
        pick = max(scores, key=scores.get)
        return self._labels.index(pick)


class IntentService:
    """会话化 IntentLoop 工厂 + 串行推理锁。"""

    @staticmethod
    def pick_device():
        """推理设备自动探测：xpu→cuda→cpu（CPU 推理合法）。"""
        import torch
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def __init__(self, cfg, ledger_root):
        import torch
        self._cfg = cfg
        self._torch = torch
        self._lock = threading.Lock()
        self._sessions = {}
        self.ledger_root = Path(ledger_root)
        self.device = self.pick_device()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_dir = cfg.get("base_model") or "Qwen/Qwen3.5-0.8B"
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base_dir,
                                                 trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=dtype, trust_remote_code=True).to(self.device)
        root = Path(__file__).resolve().parents[2]
        adapters = {name: (str(root / path) if not Path(path).is_absolute()
                           else path)
                    for name, path in cfg["adapters"].items()}
        missing = [p for p in adapters.values() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"adapter 权重缺失: {missing}")
        self.bank = LoRAAdapterBank(base, adapters)
        self.pass1 = BankFirstTokenJudge(self.bank, self.tok, self.device)
        self.verify = BankVerifyJudge(self.bank, self.tok, self.device)
        self.select = BankSelectJudge(self.bank, self.tok, self.device)

    def session_loop(self, session):
        if session in self._sessions:
            return self._sessions[session]
        from moonbow.intent.capture import CaptureStage, FakeCaptureBackend
        from moonbow.intent.ledger import Ledger
        from moonbow.intent.loop import IntentLoop
        from moonbow.intent.schema import Entity
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                       for ch in session)[:80] or "default"
        ledger = Ledger(self.ledger_root / f"svc_{safe}.jsonl")
        for i, card in enumerate(self._cfg["cards"]):
            ledger.add_entity(Entity(
                entity_id=f"card_{i}", kind="constraint", quote=card,
                span=None, event_id="cards", source_role="user_requirement",
                scope=self._cfg.get("scope", "live")))
        loop = IntentLoop(ledger, CaptureStage(FakeCaptureBackend([])),
                          self.pass1, self._cfg.get("scope", "live"),
                          verify_judge=self.verify,
                          select_judge=self.select)
        self._sessions[session] = loop
        return loop

    def judge(self, payload):
        session = str(payload.get("session") or "default")[:120]
        text = str(payload.get("text") or "")
        if not text.strip():
            return {"error": "text 不能为空", "reminders": []}
        event_id = str(payload.get("event_id") or f"evt-{threading.get_ident()}")
        from moonbow.intent.schema import Event
        ev = Event(event_id=event_id,
                   kind=payload.get("kind") or "tool_call",
                   source_role=payload.get("source_role") or "assistant_claim",
                   ts=payload.get("ts") or "", text=text)
        with self._lock:                                   # XPU 串行
            loop = self.session_loop(session)
            rep = loop.process_event(ev)
            ledger = loop.ledger
        return {
            "session": session, "event_id": event_id,
            "reminders": [r.to_dict() for r in rep.reminders_emitted],
            "reminders_deduped": rep.reminders_deduped,
            "judgments": rep.judgments,
            "abstains": sum(1 for j in ledger.all_judgments() if j.abstain),
            "failures": rep.failures,
        }


def build_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                     # noqa: N802
            pass

        def _json(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                              # noqa: N802
            if self.path == "/health":
                self._json(200, {"ok": True,
                                 "adapters": sorted(service._cfg["adapters"]),
                                 "cards": len(service._cfg["cards"]),
                                 "sessions": len(service._sessions)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):                             # noqa: N802
            if self.path != "/judge":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(
                    self.rfile.read(length).decode("utf-8") or "{}")
            except Exception as e:                     # noqa: BLE001
                self._json(400, {"error": f"bad request: {e}",
                                 "reminders": []})
                return
            try:
                self._json(200, service.judge(payload))
            except Exception as e:                     # noqa: BLE001 — fail-open
                self._json(200, {"error": f"{type(e).__name__}: {e}",
                                 "reminders": [], "failures": True})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/intent_service.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg.get("ledger_root") or
                ".mimosa/intent-loop/service_ledgers")
    service = IntentService(cfg, root)
    server = ThreadingHTTPServer(("127.0.0.1", cfg["port"]),
                                 build_handler(service))
    print(f"intent service on 127.0.0.1:{cfg['port']} "
          f"device={service.device} adapters={sorted(cfg['adapters'])} "
          f"cards={len(cfg['cards'])}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

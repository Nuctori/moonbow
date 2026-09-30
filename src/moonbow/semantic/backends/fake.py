# -*- coding: utf-8 -*-
"""moonbow.semantic.backends.fake

FakeBackend：可编程的确定性 backend，仅用于契约/故障/集成测试。
不计入真实模型验收（计划 §2.2）。

可注入项：
- rules：pattern ref -> FakeRule（固定 matched/score/evidence/relation），
  或 callable(request) -> MatchResponse 自定义函数。无规则时确定性返回
  matched=False 的 ok 负例（不从文本编造命中）。
- fault：None | "timeout" | "invalid_output" | "unavailable" | "overloaded"。
- delay：每次 match 的固定延迟秒数（测 deadline / health / 取消 / 排空）。
"""
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from moonbow.semantic.backends.base import (
    Backend,
    BackendInfo,
    BackendInvalidOutput,
    BackendOverloaded,
    BackendTimeout,
    BackendUnavailable,
)
from moonbow.semantic.schema import (
    Evidence,
    MatchRequest,
    MatchResponse,
    Provenance,
    validate_response,
)

FAULT_MODES = ("timeout", "invalid_output", "unavailable", "overloaded")

# 哨兵：请求 adapter 未注册（match() 据此返回契约内 unsupported error）。
_ADAPTER_UNSUPPORTED = object()


@dataclass
class FakeRule:
    """固定输出规则：确定性、与请求文本无关（除证据定位）。"""
    matched: bool
    score: Optional[float] = None
    quotes: Sequence[str] = ()          # 逐字引文；偏移由原文定位
    relation: str = "supports"
    truncated: bool = False


FakeRuleFn = Callable[[MatchRequest], MatchResponse]


class FakeBackend(Backend):
    def __init__(
        self,
        rules: Optional[Dict[str, Union[FakeRule, FakeRuleFn]]] = None,
        fault: Optional[str] = None,
        delay: float = 0.0,
        name: str = "fake",
        model_revision: str = "fake-rev-1",
        gate: Optional[threading.Event] = None,
    ):
        if fault is not None and fault not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {fault!r}; choose from {FAULT_MODES}")
        super().__init__(BackendInfo(
            name=name,
            model_revision=model_revision,
            fake=True,
            extra={"injectable_faults": list(FAULT_MODES)},
        ))
        self._rules: Dict[str, Union[FakeRule, FakeRuleFn]] = dict(rules or {})
        # adapter 注册表（R1）：逻辑名 -> 该适配器下的规则表。
        # 请求带未注册 adapter 时返回 unsupported error，绝不静默回退 base。
        self._adapter_rules: Dict[str, Dict[str, Union[FakeRule, FakeRuleFn]]] = {}
        self._fault = fault
        self._delay = float(delay)
        # gate：确定性地阻塞推理（set 之前 match 不返回），用于排队/关闭测试。
        self.gate = gate

    # -- 编程接口（测试用） ---------------------------------------------------

    def set_rule(self, ref: str, rule: Union[FakeRule, FakeRuleFn]) -> None:
        self._rules[ref] = rule

    # -- adapter 注册表（R1） -------------------------------------------------

    def register_adapter(self, name: str,
                         rules: Optional[Dict[str, Union[FakeRule, FakeRuleFn]]] = None
                         ) -> None:
        """注册 adapter 逻辑名（名字规则与契约一致）；rules 为该适配器下的
        规则覆盖表（缺省 ref 回落到 base 规则，仅供测试联调）。"""
        from moonbow.semantic.schema import _check_adapter
        _check_adapter(name, "adapter name")
        self._adapter_rules[name] = dict(rules or {})

    @property
    def registered_adapters(self) -> Tuple[str, ...]:
        return tuple(sorted(self._adapter_rules))

    def _effective_rule(self, request: MatchRequest
                        ) -> Union[FakeRule, FakeRuleFn, None]:
        """按请求 adapter 取规则；未注册的 adapter 返回哨兵 Unsupported。"""
        if request.adapter is not None:
            table = self._adapter_rules.get(request.adapter)
            if table is None:
                return _ADAPTER_UNSUPPORTED
            rule = table.get(request.matcher_ref)
            if rule is None and request.pattern is None:
                rule = table.get(f"requirement:{request.requirement}")
            if rule is not None:
                return rule
        ref = request.matcher_ref
        rule = self._rules.get(ref)
        if rule is None and request.pattern is None:
            rule = self._rules.get(f"requirement:{request.requirement}")
        return rule

    def set_fault(self, fault: Optional[str]) -> None:
        if fault is not None and fault not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {fault!r}")
        self._fault = fault

    # -- Backend 协议 ---------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps["adapters"] = list(self.registered_adapters)
        return caps

    def match(self, request: MatchRequest) -> MatchResponse:
        self.call_count += 1
        if self.gate is not None:
            self.gate.wait()
        if self._delay > 0:
            time.sleep(self._delay)
        if self._fault == "timeout":
            raise BackendTimeout("fake injected timeout")
        if self._fault == "invalid_output":
            raise BackendInvalidOutput("fake injected invalid output")
        if self._fault == "unavailable":
            raise BackendUnavailable("fake injected unavailable")
        if self._fault == "overloaded":
            raise BackendOverloaded("fake injected overloaded")
        return self._respond(request)

    # -- 确定性响应构造 -------------------------------------------------------

    def _respond(self, request: MatchRequest) -> MatchResponse:
        rule = self._effective_rule(request)
        if rule is _ADAPTER_UNSUPPORTED:
            # 硬要求：未注册 adapter 绝不静默回退 base，显式报 unsupported。
            return MatchResponse(
                status="error", reason_code="unsupported",
                request_id=request.request_id)
        if callable(rule):
            resp = rule(request)
            if not isinstance(resp, MatchResponse):
                raise BackendInvalidOutput("fake rule did not return MatchResponse")
            return resp

        return self._build_ok(request, rule)

    def _build_ok(self, request: MatchRequest,
                  rule: Optional[FakeRule]) -> MatchResponse:
        if rule is None:
            rule = FakeRule(matched=False)
        evidence: List[Evidence] = []
        for quote in rule.quotes:
            found = _locate(request.text, quote)
            if found is None:
                # 引文不是原文子串：fake 不修饰成可信命中，按无效输出处理。
                raise BackendInvalidOutput(
                    f"fake rule quote not verbatim in text: {quote!r}")
            start, end = found
            evidence.append(Evidence(quote=quote, start=start, end=end,
                                     relation=rule.relation))
        # match 是单一判定：最多一条证据（与契约一致，多引文只用于 find_all）。
        if request.operation == "match" and len(evidence) > 1:
            evidence = evidence[:1]
        resp = MatchResponse(
            status="ok",
            matched=bool(rule.matched),
            score=rule.score,
            evidence=evidence,
            truncated=bool(rule.truncated),
            provenance=Provenance(
                backend=self.name,
                model_revision=self.model_revision,
                pattern_version=request.pattern if request.pattern is not None
                else "requirement@0",
                calibrated=False,
                adapter=request.adapter,
            ),
            request_id=request.request_id,
        )
        validate_response(request, resp)
        return resp


def _locate(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """code point 偏移的首个 occurrence；找不到返回 None。"""
    if not quote:
        return None
    start = text.find(quote)
    if start < 0:
        return None
    return start, start + len(quote)


def make_fake_runtime(backend: Optional[FakeBackend] = None, **config):
    """便捷工厂：FakeBackend + 共享 runtime（测试常用）。"""
    from moonbow.semantic.runtime import SemanticRuntime
    return SemanticRuntime(backend if backend is not None else FakeBackend(),
                           **config)

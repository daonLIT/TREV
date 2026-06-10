"""A1: base 에이전트 루프 + 단일 검증 에이전트(tracer bullet).

native tool-calling 루프: (system 정책 + 도구셋)으로 LLM 호출 → tool_calls면 실행·관찰을
messages에 추가하고 반복 → `submit_verdict`로 종료(step 상한). 병렬 tool_calls를 리스트로
실행하고 각 결과를 매칭 tool_call_id로 회신한다.

단일 검증 에이전트는 `search_evidence`로 근거를 모은 뒤 스스로 `submit_verdict`(라벨·confidence·
justification·cited)를 호출 → 결정론과 동일 형태의 Verdict(5→4 매핑). CONFLICT는 내지 않는다
(멀티에이전트 R5는 A3). 멀티에이전트·tier/verify 도구는 후속 슬라이스에서 이 루프를 재사용한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from trev.llm import AssistantTurn
from trev.schemas import AgentStep, AgentTrace, Claim, Label5, ToolCall, Verdict
from trev.tools import (
    AgentContext,
    Tool,
    Toolbox,
    make_assess_source_tier_tool,
    make_rank_by_tier_tool,
    make_search_evidence_tool,
    make_verify_claim_tool,
)
from trev.verifier import VerifierLabel, _TO_LABEL5, to_averitec_label

DEFAULT_MAX_STEPS = 8


class _SubmitVerdict(BaseModel):
    label: VerifierLabel
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    cited: list[str] = Field(min_length=1)  # 무인용 판정 금지


SUBMIT_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit the final verdict once you have gathered evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["SUPPORT", "REFUTE", "PARTIAL", "NEI"]},
                "confidence": {"type": "number"},
                "justification": {"type": "string"},
                "cited": {"type": "array", "items": {"type": "string"},
                          "description": "doc_ids of evidence you relied on (non-empty)"},
            },
            "required": ["label", "confidence", "justification", "cited"],
        },
    },
}

VERIFIER_SYSTEM = """You are a fact-checking agent. Verify the CLAIM using ONLY evidence \
from the knowledge store tools. Do not use outside knowledge.

Tools:
- search_evidence: gather evidence (returns citable doc_ids).
- rank_by_tier: rank gathered evidence by source trust tier (T1 best); prefer high-tier sources.
- assess_source_tier: look up a domain's tier.
- verify_claim: delegate a labeled assessment with per-evidence stances.

Process:
1. search_evidence with focused queries (you may issue several).
2. Optionally rank_by_tier / assess_source_tier to weigh source trust, and verify_claim to \
get an assessment.
3. Judge: SUPPORT / REFUTE (core is false) / PARTIAL (exaggerated/cherry-picked) / NEI \
(no trustworthy evidence). Prefer trustworthy (T1-T3) evidence.
4. Call submit_verdict with label, confidence (0-1), justification, and non-empty cited \
(doc_ids). Do NOT use a CONFLICT label.
Ground every judgment in the retrieved evidence."""


def _assistant_msg(turn: AssistantTurn) -> dict:
    return {
        "role": "assistant",
        "content": turn.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in turn.tool_calls
        ],
    }


def _tool_msg(tool_call_id: str, result: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def _build_verdict(claim: Claim, args: dict) -> Verdict | None:
    """submit_verdict 인자를 검증해 Verdict로 만든다(검증 실패 시 None)."""
    try:
        out = _SubmitVerdict.model_validate(args)
    except ValidationError:
        return None
    label5 = _TO_LABEL5[out.label]
    return Verdict(
        claim_id=claim.claim_id, label5=label5,
        averitec_label=to_averitec_label(label5),
        confidence=out.confidence, justification=out.justification, cited=out.cited,
    )


def _fallback_nei(claim: Claim, why: str) -> Verdict:
    return Verdict(
        claim_id=claim.claim_id, label5=Label5.NEI,
        averitec_label=to_averitec_label(Label5.NEI),
        confidence=0.0, justification=why, cited=[],
    )


class _Continue:
    """종료 도구가 무효일 때 에이전트에 회신할 오류를 담아 루프를 계속하게 하는 신호."""

    def __init__(self, error: str):
        self.error = error


@dataclass
class Budget:
    """전 오케스트레이션 공유 예산(비용·시간 통제). tool-calling 1턴 ~40s(G-agent 실측)."""

    max_steps: int = 12
    max_tool_calls: int = 24
    steps: int = 0
    tool_calls: int = 0

    def step_exhausted(self) -> bool:
        return self.steps >= self.max_steps

    def tool_exhausted(self) -> bool:
        return self.tool_calls >= self.max_tool_calls


def run_tool_loop(
    ctx: AgentContext,
    llm,
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[Tool],
    terminal_specs: list[dict],
    on_terminal,
    nudge: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: "Budget | None" = None,
    trace: AgentTrace | None = None,
    agent: str = "agent",
):
    """일반 tool-calling 루프(단일 에이전트·searcher가 공유).

    tool_calls를 리스트로 실행하고, 종료 도구(`terminal_specs`)는 `on_terminal(name, args)`로
    처리한다. 반환값이 `_Continue`면 오류를 회신하고 계속, 아니면 그 값으로 종료.
    `budget`이 있으면 전역 step·tool 호출 상한을 적용한다. `trace`가 있으면 매 턴의 도구호출·
    관찰을 AgentStep으로 기록한다. 종료 도구 미호출 시 None.
    """
    toolbox = Toolbox(tools)
    specs = toolbox.specs() + terminal_specs
    terminal_names = {s["function"]["name"] for s in terminal_specs}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _record(calls, note):
        if trace is not None:
            trace.steps.append(AgentStep(agent=agent, tool_calls=calls, note=note))

    for _ in range(max_steps):
        if budget is not None and (budget.step_exhausted() or budget.tool_exhausted()):
            break
        if budget is not None:
            budget.steps += 1
        turn = llm.complete_with_tools(messages, specs)
        if not turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.content or ""})
            messages.append({"role": "user", "content": nudge})
            _record([], turn.content)
            continue
        messages.append(_assistant_msg(turn))
        step_calls: list[ToolCall] = []
        for tc in turn.tool_calls:
            if tc.name in terminal_names:
                out = on_terminal(tc.name, tc.arguments)
                if isinstance(out, _Continue):
                    messages.append(_tool_msg(tc.id, out.error))
                    step_calls.append(ToolCall(name=tc.name, args=tc.arguments, result=out.error))
                else:
                    step_calls.append(ToolCall(name=tc.name, args=tc.arguments, result="(terminal)"))
                    _record(step_calls, turn.content)
                    return out
            elif budget is not None and budget.tool_exhausted():
                msg = "error: tool budget exhausted — finish now"
                messages.append(_tool_msg(tc.id, msg))
                step_calls.append(ToolCall(name=tc.name, args=tc.arguments, result=msg))
            else:
                if budget is not None:
                    budget.tool_calls += 1
                result = toolbox.call(tc.name, tc.arguments)
                messages.append(_tool_msg(tc.id, result))
                step_calls.append(ToolCall(name=tc.name, args=tc.arguments, result=result))
        _record(step_calls, turn.content)
    return None


def run_agent(
    ctx: AgentContext,
    llm,
    *,
    system_prompt: str,
    tools: list[Tool],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Verdict:
    """단일 검증 에이전트 루프: submit_verdict로 종료(step 상한 시 NEI)."""

    def on_terminal(name, args):
        verdict = _build_verdict(ctx.claim, args)
        return verdict if verdict is not None else _Continue(
            "error: invalid verdict — need label, confidence 0-1, "
            "justification, and non-empty cited")

    result = run_tool_loop(
        ctx, llm, system_prompt=system_prompt, user_prompt=f"CLAIM: {ctx.claim.text}",
        tools=tools, terminal_specs=[SUBMIT_TOOL_SPEC], on_terminal=on_terminal,
        nudge="Call search_evidence to gather evidence, or submit_verdict to finish.",
        max_steps=max_steps,
    )
    return result if result is not None else _fallback_nei(
        ctx.claim, "step budget exhausted without a verdict")


def verify_with_agent(
    claim: Claim, index, embedder, llm, *,
    tier_config: dict | None = None, method: str = "dense",
    k: int = 10, candidate_n: int = 50, max_steps: int = DEFAULT_MAX_STEPS,
) -> Verdict:
    """단일 검증 에이전트로 claim을 관통시켜 Verdict를 만든다(검색·tier·verify 도구 사용)."""
    if tier_config is None:
        from trev.config import load_config
        tier_config = load_config().get("tier", {})
    ctx = AgentContext(claim=claim, index=index, embedder=embedder,
                       tier_config=tier_config, method=method, k=k, candidate_n=candidate_n)
    tools = [
        make_search_evidence_tool(ctx),
        make_rank_by_tier_tool(ctx),
        make_assess_source_tier_tool(ctx),
        make_verify_claim_tool(ctx, llm),
    ]
    return run_agent(ctx, llm, system_prompt=VERIFIER_SYSTEM, tools=tools, max_steps=max_steps)

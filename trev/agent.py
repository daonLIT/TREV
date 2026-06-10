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

from pydantic import BaseModel, Field, ValidationError

from trev.llm import AssistantTurn
from trev.schemas import Claim, Label5, Verdict
from trev.tools import AgentContext, Tool, Toolbox, make_search_evidence_tool
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
from the search_evidence tool (the knowledge store). Do not use outside knowledge.

Process:
1. Call search_evidence with focused queries to gather evidence (you may issue several).
2. Judge the claim: SUPPORT (evidence supports it), REFUTE (core is false), \
PARTIAL (exaggerated/cherry-picked/partly true), NEI (no trustworthy evidence).
3. Call submit_verdict with your label, confidence (0-1), justification, and cited \
(doc_ids you relied on — must be non-empty). Do NOT use a CONFLICT label.
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


def run_agent(
    ctx: AgentContext,
    llm,
    *,
    system_prompt: str,
    tools: list[Tool],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Verdict:
    """base 에이전트 루프: 도구 호출 ↔ 관찰 반복, submit_verdict로 종료(step 상한 시 NEI)."""
    toolbox = Toolbox(tools)
    specs = toolbox.specs() + [SUBMIT_TOOL_SPEC]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CLAIM: {ctx.claim.text}"},
    ]
    for _ in range(max_steps):
        turn = llm.complete_with_tools(messages, specs)
        if not turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.content or ""})
            messages.append({"role": "user",
                             "content": "Call search_evidence to gather evidence, "
                                        "or submit_verdict to finish."})
            continue
        messages.append(_assistant_msg(turn))
        for tc in turn.tool_calls:
            if tc.name == "submit_verdict":
                verdict = _build_verdict(ctx.claim, tc.arguments)
                if verdict is not None:
                    return verdict
                messages.append(_tool_msg(
                    tc.id, "error: invalid verdict — need label, confidence 0-1, "
                           "justification, and non-empty cited"))
            else:
                messages.append(_tool_msg(tc.id, toolbox.call(tc.name, tc.arguments)))
    return _fallback_nei(ctx.claim, "step budget exhausted without a verdict")


def verify_with_agent(
    claim: Claim, index, embedder, llm, *,
    tier_config: dict | None = None, method: str = "dense",
    k: int = 10, candidate_n: int = 50, max_steps: int = DEFAULT_MAX_STEPS,
) -> Verdict:
    """단일 검증 에이전트로 claim을 관통시켜 Verdict를 만든다(A1 진입점)."""
    ctx = AgentContext(claim=claim, index=index, embedder=embedder,
                       tier_config=tier_config, method=method, k=k, candidate_n=candidate_n)
    tools = [make_search_evidence_tool(ctx)]
    return run_agent(ctx, llm, system_prompt=VERIFIER_SYSTEM, tools=tools, max_steps=max_steps)

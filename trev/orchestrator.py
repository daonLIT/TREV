"""A3: 멀티에이전트 (planner/searcher/verifier) + orchestrator.

A1 base 루프(`run_tool_loop`)와 A2 도구를 재사용해 역할을 분담한다:
- planner: claim → 하위 질문(검색 전략). 단일 구조화 LLM 호출(분해).
- searcher: 하위 질문별로 search/rank/assess 도구로 근거를 모으고 finish_search로 종료(tool 루프).
- verifier: 수집 근거로 verify(run_verifier) → 라벨·stance·cited.
- orchestrator: planner→searcher→verifier 조율, **R5 CONFLICT**(T1~T2 stance 공존)와 cited 강제를
  적용해 단일 Verdict로 수렴. CONFLICT 규칙은 결정론 controller와 동일(`_high_tier_stance_conflict`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from trev.agent import Budget, run_tool_loop
from trev.controller import _high_tier_stance_conflict
from trev.schemas import AgentStep, AgentTrace, Claim, Label5, Verdict
from trev.tier import rank_evidence
from trev.tools import (
    AgentContext,
    make_assess_source_tier_tool,
    make_rank_by_tier_tool,
    make_search_evidence_tool,
)
from trev.verifier import VerifierOutput, _TO_LABEL5, run_verifier, to_averitec_label

DEFAULT_SEARCH_STEPS = 6


# --- planner ---------------------------------------------------------------

class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1)


_PLANNER_SYSTEM = (
    "You are a fact-checking planner. Decompose the CLAIM into 2-4 specific sub-questions "
    "whose answers would verify or refute it (AVeriTeC style). "
    'Return ONLY JSON: {"sub_questions": ["...", ...]}.'
)


def plan_claim(claim: Claim, llm) -> Plan:
    """claim을 하위 질문으로 분해한다(검색 전략)."""
    return llm.complete_json(
        [{"role": "system", "content": _PLANNER_SYSTEM},
         {"role": "user", "content": f"CLAIM: {claim.text}"}],
        schema=Plan,
    )


# --- searcher --------------------------------------------------------------

_SEARCHER_SYSTEM = (
    "You are a fact-checking searcher. Use search_evidence (issue several focused queries, "
    "including the sub-questions), then rank_by_tier to prefer trustworthy sources. "
    "Call finish_search when you have enough evidence."
)

_FINISH_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "finish_search",
        "description": "Call when enough evidence has been gathered.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def search_claim(
    ctx: AgentContext, llm, plan: Plan, *,
    max_steps: int = DEFAULT_SEARCH_STEPS, budget: Budget | None = None,
    feedback: str | None = None, trace: AgentTrace | None = None,
):
    """하위 질문별로 도구를 호출해 근거를 모은다(ctx.pool 채움). feedback은 재검색 힌트."""
    tools = [make_search_evidence_tool(ctx), make_rank_by_tier_tool(ctx),
             make_assess_source_tier_tool(ctx)]
    user = "Sub-questions to investigate:\n" + "\n".join(f"- {q}" for q in plan.sub_questions)
    if feedback:
        user += f"\n\n{feedback}"
    run_tool_loop(
        ctx, llm, system_prompt=_SEARCHER_SYSTEM, user_prompt=user, tools=tools,
        terminal_specs=[_FINISH_SEARCH_SPEC],
        on_terminal=lambda name, args: args or {},   # finish_search → 종료
        nudge="Call search_evidence for more, or finish_search to stop.",
        max_steps=max_steps, budget=budget, trace=trace, agent="searcher",
    )


# --- verifier --------------------------------------------------------------

def verify_pool(claim: Claim, evidence, llm) -> VerifierOutput:
    """수집 근거로 라벨·stance·cited를 산출한다(verifier 에이전트의 핵심 = run_verifier)."""
    return run_verifier(claim, evidence, llm)


# --- orchestrator ----------------------------------------------------------

def _nei(claim: Claim, why: str) -> Verdict:
    return Verdict(claim_id=claim.claim_id, label5=Label5.NEI,
                   averitec_label=to_averitec_label(Label5.NEI),
                   confidence=0.0, justification=why, cited=[])


_FEEDBACK = ("The previous evidence was insufficient (low confidence). "
             "Search with broader or alternative queries to find more trustworthy evidence.")


def _verdict_from(claim: Claim, out: VerifierOutput, evidence) -> Verdict:
    """verifier 출력 + R5 CONFLICT로 Verdict를 만든다."""
    label5 = _TO_LABEL5[out.label]
    if _high_tier_stance_conflict(evidence, out):   # R5: T1~T2 stance 공존 → CONFLICT
        label5 = Label5.CONFLICT
    return Verdict(
        claim_id=claim.claim_id, label5=label5,
        averitec_label=to_averitec_label(label5),
        confidence=out.confidence, justification=out.justification, cited=out.cited,
    )


def orchestrate(
    claim: Claim, index, embedder, llm, *,
    tier_config: dict | None = None, method: str = "dense",
    k: int = 10, candidate_n: int = 50, search_steps: int = DEFAULT_SEARCH_STEPS,
    low_confidence: float = 0.5, max_steps: int = 12, max_tool_calls: int = 24,
) -> tuple[Verdict, AgentTrace]:
    """planner→searcher→verifier 조율 + 저신뢰 재검색(1회) + R5 CONFLICT + cited 강제.

    반환: (Verdict, AgentTrace). trace는 에이전트별 도구호출 기록 + 비용(소비 step·tool 수).
    """
    if tier_config is None:
        from trev.config import load_config
        tier_config = load_config().get("tier", {})
    budget = Budget(max_steps=max_steps, max_tool_calls=max_tool_calls)
    trace = AgentTrace()
    ctx = AgentContext(claim=claim, index=index, embedder=embedder,
                       tier_config=tier_config, method=method, k=k, candidate_n=candidate_n)

    state: dict = {"evidence": []}

    def _finish(verdict: Verdict) -> tuple[Verdict, AgentTrace]:
        trace.tool_calls_used = budget.tool_calls
        trace.steps_used = budget.steps
        trace.retrieved_urls = [e.url for e in state["evidence"][:k] if e.url]
        return verdict, trace

    plan = plan_claim(claim, llm)
    trace.steps.append(AgentStep(agent="planner",
                                 note="sub_questions: " + "; ".join(plan.sub_questions)))
    search_claim(ctx, llm, plan, max_steps=search_steps, budget=budget, trace=trace)

    def _evaluate() -> VerifierOutput | None:
        if not ctx.pool:
            return None
        evidence = rank_evidence(claim, list(ctx.pool.values()), tier_config, weighted=True)
        state["evidence"] = evidence
        out = verify_pool(claim, evidence, llm)
        trace.steps.append(AgentStep(
            agent="verifier",
            note=f"label={out.label.value} confidence={out.confidence} cited={out.cited}"))
        return out

    out = _evaluate()
    if out is None:
        return _finish(_nei(claim, "no evidence gathered"))

    # R4 동등: 저신뢰 → searcher로 1회 피드백 재검색 후 재검증, 그래도 낮으면 NEI.
    if out.confidence < low_confidence:
        search_claim(ctx, llm, plan, max_steps=search_steps, budget=budget,
                     feedback=_FEEDBACK, trace=trace)
        out = _evaluate()
        if out is None or out.confidence < low_confidence:
            return _finish(_nei(claim, "low confidence after re-search"))

    if not out.cited:                       # cited 강제(무인용 금지)
        return _finish(_nei(claim, "verifier returned no citation"))

    return _finish(_verdict_from(claim, out, state["evidence"]))

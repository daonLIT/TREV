"""A3 멀티에이전트 orchestrator 단위 테스트(가짜 LLM: planner/searcher/verifier 역할)."""

from __future__ import annotations

from trev.data.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.agent.orchestrator import Plan, orchestrate, plan_claim, search_claim
from trev.schemas import AveritecLabel, Claim, ClaimType, Label5, Passage
from trev.agent.tools import AgentContext
from tests.test_indexing import FakeEmbedder

TIER_CFG = {
    "weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1},
    "overrides": {},
    "heuristics": {1: [".gov"], 2: ["factcheck"]},
}


def _claim():
    return Claim(claim_id=0, text="GDP grew strongly last year", type=ClaimType.NUMERICAL)


def _index(passages):
    return ClaimIndex.build(passages, FakeEmbedder())


def _search_then_finish(query="GDP grew"):
    return [
        AssistantTurn(tool_calls=[ToolCallRequest(id="s", name="search_evidence",
                                                  arguments={"query": query})]),
        AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})]),
    ]


class FakeOrchestratorLLM:
    """planner/verifier=complete_json(schema 분기), searcher=complete_with_tools."""

    def __init__(self, *, plan, search_turns, verifier):
        self._plan = plan
        self._search = list(search_turns)
        self._verifier = verifier

    def complete_json(self, messages, schema=None):
        data = self._plan if schema.__name__ == "Plan" else self._verifier
        return schema.model_validate(data)

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        return self._search.pop(0)


# --- planner ---------------------------------------------------------------

def test_plan_claim_decomposes():
    class _LLM:
        def complete_json(self, messages, schema=None):
            return schema.model_validate({"sub_questions": ["q1", "q2"]})
    plan = plan_claim(_claim(), _LLM())
    assert isinstance(plan, Plan) and plan.sub_questions == ["q1", "q2"]


# --- searcher --------------------------------------------------------------

def test_search_claim_populates_pool_with_tier_meta():
    ctx = AgentContext(claim=_claim(),
                       index=_index([Passage(claim_id=0, url="https://cdc.gov/a",
                                             text="GDP grew strongly per CDC", source_domain="cdc.gov")]),
                       embedder=FakeEmbedder(), tier_config=TIER_CFG)

    class _LLM:
        def __init__(self): self._t = _search_then_finish()
        def complete_with_tools(self, m, t, *, tool_choice="auto"): return self._t.pop(0)

    search_claim(ctx, _LLM(), Plan(sub_questions=["GDP growth?"]))
    assert ctx.pool                       # 근거 수집됨
    # rank 도구를 안 불러도 orchestrator가 확정하지만, 풀엔 evidence가 들어옴.


# --- orchestrate: 선형 조율 → Verdict --------------------------------------

def test_orchestrate_planner_searcher_verifier_to_verdict():
    index = _index([Passage(claim_id=0, url="https://cdc.gov/a",
                            text="GDP grew strongly last year per CDC", source_domain="cdc.gov")])
    llm = FakeOrchestratorLLM(
        plan={"sub_questions": ["GDP growth?"]},
        search_turns=_search_then_finish(),
        verifier={"label": "REFUTE", "confidence": 0.8, "justification": "j",
                  "cited": ["e0"], "stances": [{"doc_id": "e0", "stance": "REFUTE"}]},
    )
    v, trace = orchestrate(_claim(), index, FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.REFUTE and v.averitec_label is AveritecLabel.REFUTED
    assert trace.tool_calls_used >= 1  # searcher가 도구를 사용(비용 기록)
    assert any(s.agent == "planner" for s in trace.steps)   # 역할별 trace 기록
    assert any(s.agent == "verifier" for s in trace.steps)
    assert v.cited == ["e0"]


# --- orchestrate: R5 CONFLICT (T1~T2 stance 공존) --------------------------

def test_orchestrate_conflict_on_high_tier_stance_disagreement():
    index = _index([
        Passage(claim_id=0, url="https://cdc.gov/a", text="GDP grew strongly", source_domain="cdc.gov"),
        Passage(claim_id=0, url="https://factcheck.org/b", text="GDP grew weakly", source_domain="factcheck.org"),
    ])
    llm = FakeOrchestratorLLM(
        plan={"sub_questions": ["GDP?"]},
        search_turns=_search_then_finish(query="GDP grew"),
        # e0/e1 모두 고tier(T1/T2), stance 공존 → CONFLICT.
        verifier={"label": "SUPPORT", "confidence": 0.7, "justification": "mixed",
                  "cited": ["e0", "e1"],
                  "stances": [{"doc_id": "e0", "stance": "SUPPORT"},
                              {"doc_id": "e1", "stance": "REFUTE"}]},
    )
    v, _ = orchestrate(_claim(), index, FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.CONFLICT
    assert v.averitec_label is AveritecLabel.CONFLICTING


# --- orchestrate: 근거 없음 → NEI ------------------------------------------

def test_orchestrate_nei_when_no_evidence():
    empty = _index([])
    llm = FakeOrchestratorLLM(
        plan={"sub_questions": ["q"]},
        search_turns=_search_then_finish(),
        verifier={"label": "SUPPORT", "confidence": 0.9, "justification": "j", "cited": ["e0"], "stances": []},
    )
    v, _ = orchestrate(_claim(), empty, FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.NEI

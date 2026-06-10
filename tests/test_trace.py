"""A5 에이전트 trace 기록 + 비용 지표 단위 테스트."""

from __future__ import annotations

from trev.data.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.agent.orchestrator import orchestrate
from trev.schemas import AgentTrace, Claim, ClaimType, Passage
from tests.test_indexing import FakeEmbedder

TIER_CFG = {"weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}, "overrides": {},
            "heuristics": {1: [".gov"]}}


def _claim():
    return Claim(claim_id=0, text="GDP grew strongly", type=ClaimType.NUMERICAL)


def _index():
    return ClaimIndex.build(
        [Passage(claim_id=0, url="https://cdc.gov/a", text="GDP grew strongly per CDC",
                 source_domain="cdc.gov")], FakeEmbedder())


class ScriptedLLM:
    def __init__(self):
        self._search = [
            AssistantTurn(tool_calls=[ToolCallRequest(id="s", name="search_evidence",
                                                      arguments={"query": "GDP growth"})]),
            AssistantTurn(tool_calls=[ToolCallRequest(id="r", name="rank_by_tier",
                                                      arguments={"weighted": True})]),
            AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})]),
        ]

    def complete_json(self, messages, schema=None):
        if schema.__name__ == "Plan":
            return schema.model_validate({"sub_questions": ["GDP growth?", "GDP figures?"]})
        return schema.model_validate({"label": "REFUTE", "confidence": 0.9, "justification": "j",
                                      "cited": ["e0"], "stances": [{"doc_id": "e0", "stance": "REFUTE"}]})

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        return self._search.pop(0)


def test_trace_records_agents_tools_and_costs():
    v, trace = orchestrate(_claim(), _index(), FakeEmbedder(), ScriptedLLM(), tier_config=TIER_CFG)
    assert isinstance(trace, AgentTrace)

    agents = [s.agent for s in trace.steps]
    assert agents[0] == "planner" and "searcher" in agents and agents[-1] == "verifier"

    # planner 산출 기록
    planner = next(s for s in trace.steps if s.agent == "planner")
    assert "GDP growth?" in planner.note

    # searcher가 search_evidence·rank_by_tier를 인자와 함께 기록
    tool_names = [tc.name for s in trace.steps if s.agent == "searcher" for tc in s.tool_calls]
    assert "search_evidence" in tool_names and "rank_by_tier" in tool_names
    search_call = next(tc for s in trace.steps for tc in s.tool_calls if tc.name == "search_evidence")
    assert search_call.args == {"query": "GDP growth"} and search_call.result

    # 비용 집계
    assert trace.n_tool_calls() >= 2 and trace.n_steps() >= 3
    assert trace.tool_calls_used >= 2   # 실제 도구 실행 수(예산)


def test_trace_serializable():
    _, trace = orchestrate(_claim(), _index(), FakeEmbedder(), ScriptedLLM(), tier_config=TIER_CFG)
    blob = trace.model_dump()   # 직렬화 가능(분석 저장)
    assert "steps" in blob and blob["tool_calls_used"] >= 2

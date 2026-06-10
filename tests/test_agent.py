"""A1 도구 + 에이전트 루프 단위 테스트(가짜 LLM이 tool_calls를 스크립트, 실 임베더 없음)."""

from __future__ import annotations

from trev.agent.agent import run_agent, verify_with_agent, VERIFIER_SYSTEM
from trev.data.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.schemas import AveritecLabel, Claim, ClaimType, Label5, Passage
from trev.agent.tools import AgentContext, Toolbox, make_search_evidence_tool
from tests.test_indexing import FakeEmbedder


def _claim():
    return Claim(claim_id=0, text="GDP grew 2 percent last year", type=ClaimType.NUMERICAL)


def _index():
    passages = [
        Passage(claim_id=0, url="https://bls.gov/a", text="GDP grew 2 percent last year per BLS",
                source_domain="bls.gov"),
        Passage(claim_id=0, url="https://blog.x/b", text="unrelated cooking recipe",
                source_domain="blog.x"),
        Passage(claim_id=0, url="https://news.y/c", text="economic growth was modest",
                source_domain="news.y"),
    ]
    return ClaimIndex.build(passages, FakeEmbedder())


def _ctx():
    return AgentContext(claim=_claim(), index=_index(), embedder=FakeEmbedder())


def _search(**args):
    return ToolCallRequest(id="s1", name="search_evidence", arguments=args)


def _submit(**args):
    return ToolCallRequest(id="v1", name="submit_verdict", arguments=args)


class FakeToolLLM:
    """complete_with_tools가 스크립트된 AssistantTurn을 순서대로 반환."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        self.calls.append(messages)
        return self.turns.pop(0)


def _run(ctx, turns, max_steps=8):
    tools = [make_search_evidence_tool(ctx)]
    return run_agent(ctx, FakeToolLLM(turns), system_prompt=VERIFIER_SYSTEM,
                     tools=tools, max_steps=max_steps)


# --- 도구: search_evidence -------------------------------------------------

def test_search_tool_populates_pool_and_returns_doc_ids():
    ctx = _ctx()
    tool = make_search_evidence_tool(ctx)
    out = tool.handler({"query": "GDP growth"})
    assert "e0" in out and ctx.pool  # doc_id 부여 + 풀 적재


def test_search_tool_bad_args():
    ctx = _ctx()
    tool = make_search_evidence_tool(ctx)
    assert tool.handler({}).startswith("error:")              # query 누락
    assert tool.handler({"query": "x", "method": "xx"}).startswith("error:")


def test_toolbox_unknown_tool():
    assert Toolbox([]).call("nope", {}).startswith("error: unknown tool")


# --- 에이전트 루프 ---------------------------------------------------------

def test_agent_searches_then_submits_verdict():
    ctx = _ctx()
    turns = [
        AssistantTurn(tool_calls=[_search(query="GDP grew 2 percent")]),
        AssistantTurn(tool_calls=[_submit(label="REFUTE", confidence=0.8,
                                          justification="evidence shows modest growth",
                                          cited=["e0"])]),
    ]
    v = _run(ctx, turns)
    assert v.label5 is Label5.REFUTE and v.averitec_label is AveritecLabel.REFUTED
    assert v.cited == ["e0"]
    assert ctx.pool  # 검색이 실제로 실행됨


def test_partial_maps_to_conflicting():
    ctx = _ctx()
    turns = [
        AssistantTurn(tool_calls=[_search(query="GDP")]),
        AssistantTurn(tool_calls=[_submit(label="PARTIAL", confidence=0.5,
                                          justification="exaggerated", cited=["e0"])]),
    ]
    assert _run(ctx, turns).averitec_label is AveritecLabel.CONFLICTING


def test_parallel_searches_executed():
    ctx = _ctx()
    turns = [
        AssistantTurn(tool_calls=[_search(query="GDP growth"),
                                  ToolCallRequest(id="s2", name="search_evidence",
                                                  arguments={"query": "economic growth"})]),
        AssistantTurn(tool_calls=[_submit(label="REFUTE", confidence=0.7,
                                          justification="j", cited=["e0"])]),
    ]
    v = _run(ctx, turns)
    assert v.label5 is Label5.REFUTE
    assert len(ctx.pool) >= 1  # 두 병렬 검색 모두 실행


def test_invalid_submit_retries_then_succeeds():
    ctx = _ctx()
    turns = [
        AssistantTurn(tool_calls=[_search(query="GDP")]),
        AssistantTurn(tool_calls=[_submit(label="NEI", confidence=0.3,
                                          justification="none", cited=[])]),   # 빈 cited
        AssistantTurn(tool_calls=[_submit(label="NEI", confidence=0.3,
                                          justification="none", cited=["e0"])]),
    ]
    v = _run(ctx, turns)
    assert v.label5 is Label5.NEI and v.cited == ["e0"]


def test_step_cap_falls_back_to_nei():
    ctx = _ctx()
    # 계속 검색만 하고 submit 안 함 → 상한에서 NEI.
    turns = [AssistantTurn(tool_calls=[_search(query="GDP")]) for _ in range(5)]
    v = run_agent(ctx, FakeToolLLM(turns), system_prompt=VERIFIER_SYSTEM,
                  tools=[make_search_evidence_tool(ctx)], max_steps=3)
    assert v.label5 is Label5.NEI and "budget" in v.justification


def test_no_tool_calls_nudges_then_submits():
    ctx = _ctx()
    turns = [
        AssistantTurn(content="thinking..."),   # 도구 호출 없음 → 넛지
        AssistantTurn(tool_calls=[_submit(label="SUPPORT", confidence=0.9,
                                          justification="j", cited=["x"])]),
    ]
    v = _run(ctx, turns)
    assert v.label5 is Label5.SUPPORT

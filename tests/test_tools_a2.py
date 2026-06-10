"""A2 도구(verify_claim·rank_by_tier·assess_source_tier) 단위 테스트."""

from __future__ import annotations

import json

from trev.agent.agent import VERIFIER_SYSTEM, run_agent
from trev.data.indexing import ClaimIndex
from trev.llm import LLM, AssistantTurn, ToolCallRequest
from trev.schemas import AveritecLabel, Claim, ClaimType, Label5, Passage, Role
from trev.agent.tools import (
    AgentContext,
    make_assess_source_tier_tool,
    make_rank_by_tier_tool,
    make_search_evidence_tool,
    make_verify_claim_tool,
)
from tests.test_indexing import FakeEmbedder
from tests.test_llm import FakeClient

TIER_CFG = {
    "weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1},
    "overrides": {},
    "heuristics": {1: [".gov"], 3: ["reuters"]},
}


def _claim(source_domains=()):
    return Claim(claim_id=0, text="GDP grew 2 percent", type=ClaimType.NUMERICAL,
                 source_domains=list(source_domains))


def _ctx(claim=None):
    passages = [Passage(claim_id=0, url="https://cdc.gov/a", text="x", source_domain="cdc.gov")]
    idx = ClaimIndex.build(passages, FakeEmbedder())
    return AgentContext(claim=claim or _claim(), index=idx, embedder=FakeEmbedder(),
                        tier_config=TIER_CFG)


def _seed_pool(ctx, items):
    from trev.schemas import Evidence
    ctx.add_evidence([Evidence(doc_id="tmp", snippet=s, source_domain=d, url=u, sim=sim)
                      for s, d, u, sim in items])


# --- assess_source_tier ----------------------------------------------------

def test_assess_source_tier_lookup():
    tool = make_assess_source_tier_tool(_ctx())
    assert "T1" in tool.handler({"domain": "cdc.gov"})
    assert "T4" in tool.handler({"domain": "randomblog.com"})
    assert tool.handler({}).startswith("error:")


# --- rank_by_tier ----------------------------------------------------------

def test_rank_by_tier_annotates_pool_and_orders():
    ctx = _ctx()
    _seed_pool(ctx, [("s1", "randomblog.com", "u1", 0.9), ("s2", "cdc.gov", "u2", 0.5)])
    out = make_rank_by_tier_tool(ctx).handler({"weighted": True})
    # 가중 시 cdc.gov(T1)가 먼저, 풀에 tier 주석.
    assert out.index("cdc.gov") < out.index("randomblog.com")
    assert any(e.tier == 1 for e in ctx.pool.values())


def test_rank_by_tier_self_source_demotion():
    ctx = _ctx(_claim(source_domains=["cdc.gov"]))   # 자기출처 = cdc.gov
    _seed_pool(ctx, [("s", "cdc.gov", "u", 0.8)])
    make_rank_by_tier_tool(ctx).handler({"weighted": True})
    e = list(ctx.pool.values())[0]
    assert e.role is Role.TARGET and e.tier == 4   # 자기출처 → T4 강등


def test_rank_by_tier_empty_pool_errors():
    assert make_rank_by_tier_tool(_ctx()).handler({"weighted": True}).startswith("error:")


# --- verify_claim ----------------------------------------------------------

def _verify_llm(label="REFUTE"):
    return LLM(client=FakeClient([json.dumps({
        "label": label, "confidence": 0.8, "justification": "j",
        "cited": ["e0"], "stances": [{"doc_id": "e0", "stance": "REFUTE"}],
    })]))


def test_verify_claim_delegates():
    ctx = _ctx()
    _seed_pool(ctx, [("evidence text", "cdc.gov", "u", 0.5)])
    out = make_verify_claim_tool(ctx, _verify_llm()).handler({})
    assert "label=REFUTE" in out and "stances=[e0:REFUTE]" in out


def test_verify_claim_requires_evidence():
    out = make_verify_claim_tool(_ctx(), _verify_llm()).handler({})
    assert out.startswith("error:")


# --- 에이전트 연쇄: search → rank → verify → submit ------------------------

class FakeToolLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        return self.turns.pop(0)


def test_agent_chains_search_rank_verify_then_submits():
    ctx = _ctx()
    tools = [make_search_evidence_tool(ctx), make_rank_by_tier_tool(ctx),
             make_assess_source_tier_tool(ctx), make_verify_claim_tool(ctx, _verify_llm())]
    turns = [
        AssistantTurn(tool_calls=[ToolCallRequest(id="a", name="search_evidence",
                                                  arguments={"query": "GDP"})]),
        AssistantTurn(tool_calls=[ToolCallRequest(id="b", name="rank_by_tier",
                                                  arguments={"weighted": True})]),
        AssistantTurn(tool_calls=[ToolCallRequest(id="c", name="verify_claim", arguments={})]),
        AssistantTurn(tool_calls=[ToolCallRequest(id="d", name="submit_verdict",
                                                  arguments={"label": "REFUTE", "confidence": 0.85,
                                                             "justification": "j", "cited": ["e0"]})]),
    ]
    v = run_agent(ctx, FakeToolLLM(turns), system_prompt=VERIFIER_SYSTEM,
                  tools=tools, max_steps=8)
    assert v.label5 is Label5.REFUTE and v.averitec_label is AveritecLabel.REFUTED
    assert any(e.tier is not None for e in ctx.pool.values())   # rank 도구가 tier 주석

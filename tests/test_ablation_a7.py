"""A7 agentic ablation 단위 테스트(tier 도구 on/off, 비교, N=3 일치율 배선)."""

from __future__ import annotations

import pytest

from trev.ablation import agreement_rate, compare_methods
from trev.experiment import labels_by_claim, predictions_to_records, run_experiments
from trev.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.metrics import evaluate
from trev.orchestrator import search_claim, Plan
from trev.schemas import AveritecLabel, Claim, ClaimType, Passage
from trev.tools import AgentContext
from tests.test_indexing import FakeEmbedder

TIER_CFG = {"weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}, "overrides": {},
            "heuristics": {1: [".gov"]}}


def _claim():
    return Claim(claim_id=0, text="GDP grew strongly", type=ClaimType.NUMERICAL,
                 label=AveritecLabel.REFUTED, topic="economy")


def _index():
    return ClaimIndex.build(
        [Passage(claim_id=0, url="https://cdc.gov/a", text="GDP grew strongly per CDC",
                 source_domain="cdc.gov")], FakeEmbedder())


# --- tier 도구 on/off: searcher 도구셋이 달라짐 ----------------------------

class _Capture:
    """search_claim이 어떤 도구셋을 run_tool_loop에 넘기는지 가로채는 가짜 LLM."""

    def __init__(self):
        self.tool_names = None

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        self.tool_names = [t["function"]["name"] for t in tools]
        # 즉시 finish_search로 종료.
        return AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})])


def test_tier_tool_present_when_on_absent_when_off():
    ctx = AgentContext(claim=_claim(), index=_index(), embedder=FakeEmbedder(), tier_config=TIER_CFG)
    on = _Capture(); search_claim(ctx, on, Plan(sub_questions=["q"]), use_tier=True)
    off = _Capture(); search_claim(ctx, off, Plan(sub_questions=["q"]), use_tier=False)
    assert "rank_by_tier" in on.tool_names
    assert "rank_by_tier" not in off.tool_names
    assert "search_evidence" in off.tool_names    # 검색은 양쪽 다 존재


# --- 실행기: agentic + agentic_no_tier 두 조건 -----------------------------

class AgenticLLM:
    """searcher: search→finish를 교대로 무한 생성(여러 조건 실행 지원)."""

    def __init__(self):
        self._i = 0

    def complete_json(self, messages, schema=None):
        if schema.__name__ == "Plan":
            return schema.model_validate({"sub_questions": ["q"]})
        return schema.model_validate({"label": "REFUTE", "confidence": 0.9, "justification": "j",
                                      "cited": ["e0"], "stances": [{"doc_id": "e0", "stance": "REFUTE"}]})

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        self._i += 1
        if self._i % 2 == 1:
            return AssistantTurn(tool_calls=[ToolCallRequest(id="s", name="search_evidence",
                                                             arguments={"query": "GDP"})])
        return AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})])


def _records(modes):
    results = run_experiments([_claim()], FakeEmbedder(), AgenticLLM(),
                              index_provider=lambda c: _index(), tier_config=TIER_CFG, modes=modes)
    return predictions_to_records([_claim()], results,
                                  gold_urls_fn=lambda cid: ["https://cdc.gov/a"],
                                  ks_urls_fn=lambda cid: ["https://cdc.gov/a"])


def test_runner_supports_both_agentic_conditions():
    records = _records(("agentic", "agentic_no_tier"))
    modes = {r["mode"] for r in records}
    assert modes == {"agentic", "agentic_no_tier"}
    table = evaluate(records, k=10)
    assert "agentic" in table and "agentic_no_tier" in table


# --- 비교(재사용) + N=3 일치율 ---------------------------------------------

def test_compare_methods_on_agentic_vs_deterministic():
    a = {"agentic": {"macro_f1": 0.8}}
    d = {"agentic": {"macro_f1": 0.6}}
    assert compare_methods(a, d)["agentic"]["delta"] == pytest.approx(0.2)  # 향상폭


def test_n3_agreement_from_records():
    # 동일 라벨 3회 → 일치율 1.0.
    runs = [labels_by_claim(_records(("agentic",)), "agentic") for _ in range(3)]
    out = agreement_rate(runs)
    assert out["agreement"] == 1.0 and out["n_runs"] == 3

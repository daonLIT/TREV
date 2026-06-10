"""A6 agentic 조건 실험·평가 통합 테스트(가짜 LLM·임베더, end-to-end)."""

from __future__ import annotations

from trev.experiment import predictions_to_records, run_experiments
from trev.data.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.eval.metrics import evaluate
from trev.schemas import AveritecLabel, Claim, ClaimType, Passage
from tests.test_indexing import FakeEmbedder

TIER_CFG = {"weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}, "overrides": {},
            "heuristics": {1: [".gov"]}}


def _claim():
    return Claim(claim_id=0, text="GDP grew strongly", type=ClaimType.NUMERICAL,
                 label=AveritecLabel.REFUTED)


def _index():
    return ClaimIndex.build(
        [Passage(claim_id=0, url="https://cdc.gov/a", text="GDP grew strongly per CDC",
                 source_domain="cdc.gov")], FakeEmbedder())


class AgenticLLM:
    """planner/verifier=complete_json, searcher=complete_with_tools."""

    def __init__(self):
        self._search = [
            AssistantTurn(tool_calls=[ToolCallRequest(id="s", name="search_evidence",
                                                      arguments={"query": "GDP"})]),
            AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})]),
        ]

    def complete_json(self, messages, schema=None):
        if schema.__name__ == "Plan":
            return schema.model_validate({"sub_questions": ["GDP?"]})
        return schema.model_validate({"label": "REFUTE", "confidence": 0.9, "justification": "j",
                                      "cited": ["e0"], "stances": [{"doc_id": "e0", "stance": "REFUTE"}]})

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        return self._search.pop(0)


def test_agentic_condition_runs_and_records_same_format():
    results = run_experiments(
        [_claim()], FakeEmbedder(), AgenticLLM(), index_provider=lambda c: _index(),
        tier_config=TIER_CFG, modes=("agentic",))
    assert set(results) == {"agentic"}
    pred = results["agentic"][0]
    assert pred["retrieved_urls"]                 # searcher 회수 url
    assert pred["cost"]["tool_calls"] >= 1        # 비용 기록

    records = predictions_to_records(
        [_claim()], results, gold_urls_fn=lambda cid: ["https://cdc.gov/a"],
        ks_urls_fn=lambda cid: ["https://cdc.gov/a"])
    rec = records[0]
    # 결정론과 동일 포맷(회수 url·gold url·분류·비용 포함).
    assert {"pred_label", "gold_label", "retrieved_urls", "gold_urls",
            "retrieval_category", "cost"} <= set(rec)
    assert rec["pred_label"] == "Refuted" and rec["retrieval_category"] == "retrieved"


def test_metrics_scores_agentic_with_cost():
    results = run_experiments(
        [_claim()], FakeEmbedder(), AgenticLLM(), index_provider=lambda c: _index(),
        tier_config=TIER_CFG, modes=("agentic",))
    records = predictions_to_records(
        [_claim()], results, gold_urls_fn=lambda cid: ["https://cdc.gov/a"],
        ks_urls_fn=lambda cid: ["https://cdc.gov/a"])
    table = evaluate(records, k=10)
    assert "agentic" in table
    assert table["agentic"]["accuracy"] == 1.0           # Refuted == gold
    assert table["agentic"]["avg_tool_calls"] is not None  # 비용 지표 집계

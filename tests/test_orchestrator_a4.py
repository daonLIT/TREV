"""A4 피드백 루프(저신뢰 재검색) + 예산 상한 단위 테스트."""

from __future__ import annotations

from trev.indexing import ClaimIndex
from trev.llm import AssistantTurn, ToolCallRequest
from trev.orchestrator import orchestrate
from trev.schemas import Claim, ClaimType, Label5, Passage
from tests.test_indexing import FakeEmbedder

TIER_CFG = {"weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}, "overrides": {},
            "heuristics": {1: [".gov"]}}


def _claim():
    return Claim(claim_id=0, text="GDP grew strongly", type=ClaimType.NUMERICAL)


def _index():
    return ClaimIndex.build(
        [Passage(claim_id=0, url="https://cdc.gov/a", text="GDP grew strongly per CDC",
                 source_domain="cdc.gov")], FakeEmbedder())


def _search_finish():
    return [
        AssistantTurn(tool_calls=[ToolCallRequest(id="s", name="search_evidence",
                                                  arguments={"query": "GDP"})]),
        AssistantTurn(tool_calls=[ToolCallRequest(id="f", name="finish_search", arguments={})]),
    ]


class ScriptedLLM:
    """planner/verifier=complete_json(순서), searcher=complete_with_tools(턴 큐)."""

    def __init__(self, *, plan, search_rounds, verifier_outs):
        self._plan = plan
        self._search = [t for r in search_rounds for t in r]
        self._verifier = list(verifier_outs)

    def complete_json(self, messages, schema=None):
        if schema.__name__ == "Plan":
            return schema.model_validate(self._plan)
        return schema.model_validate(self._verifier.pop(0))   # 호출마다 다음 verifier 출력

    def complete_with_tools(self, messages, tools, *, tool_choice="auto"):
        return self._search.pop(0)


_GOOD = {"label": "REFUTE", "confidence": 0.9, "justification": "j", "cited": ["e0"],
         "stances": [{"doc_id": "e0", "stance": "REFUTE"}]}
_LOW = {"label": "NEI", "confidence": 0.2, "justification": "weak", "cited": ["e0"],
        "stances": []}


def test_low_confidence_triggers_one_research_then_recovers():
    # 1차 저신뢰 → 재검색 → 2차 고신뢰 회복.
    llm = ScriptedLLM(plan={"sub_questions": ["q"]},
                      search_rounds=[_search_finish(), _search_finish()],
                      verifier_outs=[_LOW, _GOOD])
    v, trace = orchestrate(_claim(), _index(), FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.REFUTE          # 재검색 후 회복
    assert trace.tool_calls_used >= 2         # 두 라운드 검색


def test_low_confidence_still_low_after_research_is_nei():
    llm = ScriptedLLM(plan={"sub_questions": ["q"]},
                      search_rounds=[_search_finish(), _search_finish()],
                      verifier_outs=[_LOW, _LOW])
    v, _ = orchestrate(_claim(), _index(), FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.NEI and "low confidence" in v.justification


def test_no_research_when_confident():
    # 1차 고신뢰 → 재검색 없음(verifier 1회만 소비).
    llm = ScriptedLLM(plan={"sub_questions": ["q"]},
                      search_rounds=[_search_finish()],
                      verifier_outs=[_GOOD])
    v, _ = orchestrate(_claim(), _index(), FakeEmbedder(), llm, tier_config=TIER_CFG)
    assert v.label5 is Label5.REFUTE


def test_tool_budget_caps_search():
    # 검색만 무한 반복(finish 안 함) → tool 예산에서 안전 종료, Verdict 반환(예외 없음).
    many_searches = [AssistantTurn(tool_calls=[ToolCallRequest(
        id=f"s{i}", name="search_evidence", arguments={"query": "GDP"})]) for i in range(20)]

    class CappedLLM:
        def __init__(self): self._v = [_GOOD]
        def complete_json(self, messages, schema=None):
            from trev.orchestrator import Plan
            if schema is Plan:
                return schema.model_validate({"sub_questions": ["q"]})
            return schema.model_validate(self._v.pop(0))
        def complete_with_tools(self, m, t, *, tool_choice="auto"):
            return many_searches.pop(0)

    v, trace = orchestrate(_claim(), _index(), FakeEmbedder(), CappedLLM(),
                           tier_config=TIER_CFG, max_tool_calls=3, search_steps=50)
    assert trace.tool_calls_used <= 3        # 상한 준수
    assert v.label5 in set(Label5)           # 예외 없이 Verdict

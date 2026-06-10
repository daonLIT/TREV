"""S11 보조지표 단위 테스트 — LLM-judge는 스텁, 근사 score는 순수."""

from __future__ import annotations

import json

from trev.eval.auxmetrics import (
    approx_averitec_score,
    generate_cited_qa,
    geval_faithfulness,
    ragas_faithfulness,
)
from trev.llm import LLM
from trev.schemas import Claim, ClaimType
from tests.test_llm import FakeClient


def _claim():
    return Claim(claim_id=0, text="GDP grew 2% last year", type=ClaimType.NUMERICAL)


# --- RAGAS faithfulness (스텁) ---------------------------------------------

def test_ragas_faithfulness_parses_score():
    llm = LLM(client=FakeClient([json.dumps({"score": 0.9, "reasoning": "grounded"})]))
    out = ragas_faithfulness(_claim(), "GDP rose 2%", ["GDP increased by 2 percent"], llm)
    assert out["score"] == 0.9 and out["reasoning"] == "grounded"


# --- G-Eval (1~5 → 0~1) ----------------------------------------------------

def test_geval_normalizes_1_to_5():
    llm = LLM(client=FakeClient([json.dumps({"score": 5, "reasoning": "ok"})]))
    out = geval_faithfulness(_claim(), "j", ["e"], llm)
    assert out["raw"] == 5 and out["score"] == 1.0
    llm2 = LLM(client=FakeClient([json.dumps({"score": 1, "reasoning": "hallucinated"})]))
    assert geval_faithfulness(_claim(), "j", ["e"], llm2)["score"] == 0.0


# --- QA 생성 어댑터 (스텁) -------------------------------------------------

def test_generate_cited_qa():
    payload = {"qa": [{"question": "How much did GDP grow?", "answer": "2 percent"}]}
    llm = LLM(client=FakeClient([json.dumps(payload)]))
    qa = generate_cited_qa(_claim(), ["GDP grew 2%"], llm)
    assert qa == [{"question": "How much did GDP grow?", "answer": "2 percent"}]


# --- 공식 score 근사 (순수) ------------------------------------------------

def test_approx_score_covers_matching_answers():
    gold = [{"question": "q1", "answer": "the economy grew two percent"}]
    gen = [{"question": "g1", "answer": "economy grew two percent last year"}]
    assert approx_averitec_score(gold_qa=gold, generated_qa=gen) == 1.0


def test_approx_score_zero_when_no_overlap():
    gold = [{"question": "q1", "answer": "completely different topic here"}]
    gen = [{"question": "g1", "answer": "unrelated cooking recipe"}]
    assert approx_averitec_score(gold_qa=gold, generated_qa=gen) == 0.0


def test_approx_score_empty_gold():
    assert approx_averitec_score(generated_qa=[{"question": "x", "answer": "y"}], gold_qa=[]) == 0.0

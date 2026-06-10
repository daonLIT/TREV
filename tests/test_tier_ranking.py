"""S4 동적 tier + 가중 랭킹 단위 테스트(핵심 기여 — 가장 촘촘히)."""

from __future__ import annotations

import json

import pytest

from trev.llm import LLM
from trev.schemas import Claim, ClaimType, Evidence, Role
from trev.tier import (
    DEFAULT_WEIGHTS,
    LLMDomainClassifier,
    assign_tier,
    rank_evidence,
)
from tests.test_llm import FakeClient

TIER_CFG = {
    "weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1},
    "overrides": {"politifact.com": 2},
    "heuristics": {1: [".gov"], 2: ["factcheck"], 3: ["reuters", "bbc."]},
}


def _claim(source_domains=()):
    return Claim(claim_id=0, text="t", type=ClaimType.NUMERICAL,
                 source_domains=list(source_domains))


# --- assign_tier (type × role × domain 케이스 테이블) ----------------------

@pytest.mark.parametrize(
    "domain,expected_tier,expected_weight",
    [
        ("cdc.gov", 1, 1.0),
        ("politifact.com", 2, 0.7),     # override
        ("reuters.com", 3, 0.4),
        ("randomblog.com", 4, 0.1),     # 미분류 기본 T4
    ],
)
def test_assign_tier_by_domain(domain, expected_tier, expected_weight):
    tier, weight = assign_tier(ClaimType.NUMERICAL, Role.GENERAL, domain, TIER_CFG)
    assert (tier, weight) == (expected_tier, expected_weight)


def test_self_source_demoted_to_t4_regardless_of_domain():
    # 자기출처(role=target)면 .gov여도 T4 강등 — 동적 tier의 실제 발화.
    tier, weight = assign_tier(ClaimType.NUMERICAL, Role.TARGET, "cdc.gov", TIER_CFG)
    assert tier == 4 and weight == 0.1


def test_none_domain_defaults_t4():
    assert assign_tier(ClaimType.NUMERICAL, Role.GENERAL, None, TIER_CFG) == (4, 0.1)


# --- 잔여 도메인 LLM 분류 + 캐시 ------------------------------------------

def test_ambiguous_domain_llm_classified_once_and_cached():
    client = FakeClient([json.dumps({"tier": 3})])  # 호출 1회분만 제공
    clf = LLMDomainClassifier(LLM(client=client))
    # 첫 호출: heuristic 미적중(T4) → LLM이 T3으로 분류.
    t1, w1 = assign_tier(ClaimType.NUMERICAL, Role.GENERAL, "obscurenews.com", TIER_CFG,
                         classifier=clf)
    assert (t1, w1) == (3, 0.4)
    # 둘째 호출: 캐시 적중(추가 LLM 호출 없음 — 응답 1개뿐이라 호출 시 IndexError 났을 것).
    t2, _ = assign_tier(ClaimType.NUMERICAL, Role.GENERAL, "obscurenews.com", TIER_CFG,
                        classifier=clf)
    assert t2 == 3 and len(client.calls) == 1


def test_classifier_not_called_for_known_domain():
    client = FakeClient([])  # 호출되면 IndexError
    clf = LLMDomainClassifier(LLM(client=client))
    tier, _ = assign_tier(ClaimType.NUMERICAL, Role.GENERAL, "cdc.gov", TIER_CFG, classifier=clf)
    assert tier == 1 and client.calls == []


# --- rank_evidence (가중 vs unweighted) ------------------------------------

def _ev(doc_id, domain, sim):
    return Evidence(doc_id=doc_id, snippet="s", source_domain=domain, sim=sim)


def test_weighted_ranking_promotes_high_tier():
    # T4 문서가 sim은 더 높지만, 가중(score=sim*weight) 시 T1이 앞서야 한다.
    evidence = [
        _ev("low", "randomblog.com", 0.9),  # T4: 0.9*0.1 = 0.09
        _ev("high", "cdc.gov", 0.5),         # T1: 0.5*1.0 = 0.50
    ]
    ranked = rank_evidence(_claim(), evidence, TIER_CFG, weighted=True)
    assert ranked[0].doc_id == "high"
    assert ranked[0].tier == 1 and ranked[0].weight == 1.0
    assert ranked[0].score == pytest.approx(0.5)


def test_unweighted_keeps_sim_order():
    evidence = [_ev("low", "randomblog.com", 0.9), _ev("high", "cdc.gov", 0.5)]
    ranked = rank_evidence(_claim(), evidence, TIER_CFG, weighted=False)
    assert ranked[0].doc_id == "low"            # sim 순서 유지
    assert ranked[0].score == pytest.approx(0.9)


def test_self_source_demotion_in_ranking():
    claim = _claim(source_domains=["cdc.gov"])    # claim 자기출처 = cdc.gov
    ranked = rank_evidence(claim, [_ev("self", "cdc.gov", 0.8)], TIER_CFG, weighted=True)
    assert ranked[0].role is Role.TARGET
    assert ranked[0].tier == 4 and ranked[0].score == pytest.approx(0.08)  # 0.8*0.1


def test_input_evidence_not_mutated():
    e = _ev("x", "cdc.gov", 0.5)
    rank_evidence(_claim(), [e], TIER_CFG, weighted=True)
    assert e.tier == 4 and e.weight == 0.1  # 원본 기본값 유지(복사본만 변경)

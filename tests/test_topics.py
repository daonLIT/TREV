"""S8 토픽 태깅 + 분해 단위 테스트."""

from __future__ import annotations

import json

from trev.llm import LLM
from trev.schemas import Claim, ClaimType
from trev.topics import (
    LLMTopicClassifier,
    Topic,
    tag_claims,
    topic_breakdown,
    topic_tagger,
)
from tests.test_llm import FakeClient


def _claim(text, speaker=None, publisher=None, claim_id=0):
    return Claim(claim_id=claim_id, text=text, type=ClaimType.NUMERICAL,
                 speaker=speaker, publisher=publisher)


# --- topic_tagger (키워드) -------------------------------------------------

def test_keyword_topic_matches():
    assert topic_tagger(_claim("The president won the election")) is Topic.POLITICS
    assert topic_tagger(_claim("COVID vaccine reduces hospital cases")) is Topic.HEALTH
    assert topic_tagger(_claim("Unemployment and inflation rose")) is Topic.ECONOMY


def test_speaker_publisher_feed_tagging():
    # 텍스트엔 키워드 없지만 speaker로 정치 태깅.
    assert topic_tagger(_claim("He said it would happen", speaker="Joe Biden")) is Topic.POLITICS


def test_ambiguous_without_llm_is_other():
    assert topic_tagger(_claim("The cat sat on the mat")) is Topic.OTHER


def test_ambiguous_uses_llm_and_caches():
    client = FakeClient([json.dumps({"topic": "science_environment"})])
    clf = LLMTopicClassifier(LLM(client=client))
    c = _claim("an obscure statement with no keywords")
    assert topic_tagger(c, clf) is Topic.SCIENCE_ENVIRONMENT
    topic_tagger(c, clf)                       # 캐시 적중(추가 호출 없음)
    assert len(client.calls) == 1


def test_tag_claims_fills_topic():
    tagged = tag_claims([_claim("election results", claim_id=3)])
    assert tagged[0].topic == "politics"


# --- topic_breakdown -------------------------------------------------------

def test_topic_breakdown_per_topic_and_delta():
    records = [
        # politics: proposed 맞고 naive 틀림 → proposed F1 우위.
        {"claim_id": 0, "mode": "proposed", "pred_label": "Refuted", "gold_label": "Refuted"},
        {"claim_id": 1, "mode": "proposed", "pred_label": "Supported", "gold_label": "Supported"},
        {"claim_id": 0, "mode": "naive_rag", "pred_label": "Supported", "gold_label": "Refuted"},
        {"claim_id": 1, "mode": "naive_rag", "pred_label": "Supported", "gold_label": "Supported"},
    ]
    topic_by_claim = {0: "politics", 1: "politics"}
    table = topic_breakdown(records, topic_by_claim, baseline="naive_rag", target="proposed")
    assert "politics" in table
    assert table["politics"]["macro_f1"]["proposed"] > table["politics"]["macro_f1"]["naive_rag"]
    assert table["politics"]["delta"] > 0

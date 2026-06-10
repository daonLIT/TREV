"""S2 verifier 단위 테스트 — LLM seam을 canned JSON으로 스텁(실 GPT-5 불필요)."""

from __future__ import annotations

import json

import pytest

from trev.llm import LLM
from trev.schemas import AveritecLabel, Claim, ClaimType, Evidence, Label5
from trev.verifier import to_averitec_label, verify
from tests.test_llm import FakeClient


def _claim():
    return Claim(claim_id=7, text="The economy grew 10% last year.", type=ClaimType.NUMERICAL)


def _evidence():
    return [
        Evidence(doc_id="g0", snippet="GDP grew 2% last year.", source_domain="bls.gov"),
        Evidence(doc_id="g1", snippet="Growth was modest.", source_domain="reuters.com"),
    ]


def _llm(payload: dict):
    return LLM(client=FakeClient([json.dumps(payload)]), model="gpt-5")


# --- 5→4 매핑 (순수) -------------------------------------------------------

def test_label_mapping_5_to_4():
    assert to_averitec_label(Label5.SUPPORT) is AveritecLabel.SUPPORTED
    assert to_averitec_label(Label5.REFUTE) is AveritecLabel.REFUTED
    assert to_averitec_label(Label5.NEI) is AveritecLabel.NOT_ENOUGH_EVIDENCE
    assert to_averitec_label(Label5.PARTIAL) is AveritecLabel.CONFLICTING
    assert to_averitec_label(Label5.CONFLICT) is AveritecLabel.CONFLICTING  # R5 라벨도 매핑


# --- verify (스텁 LLM) -----------------------------------------------------

def test_verify_produces_verdict_with_mapping():
    payload = {
        "label": "REFUTE", "confidence": 0.9, "justification": "Only 2% growth.",
        "cited": ["g0"], "stances": [{"doc_id": "g0", "stance": "REFUTE"}],
    }
    v = verify(_claim(), _evidence(), _llm(payload))
    assert v.claim_id == 7
    assert v.label5 is Label5.REFUTE
    assert v.averitec_label is AveritecLabel.REFUTED
    assert v.cited == ["g0"]
    assert v.confidence == 0.9


def test_partial_maps_to_conflicting():
    payload = {
        "label": "PARTIAL", "confidence": 0.5, "justification": "exaggerated",
        "cited": ["g1"], "stances": [],
    }
    v = verify(_claim(), _evidence(), _llm(payload))
    assert v.label5 is Label5.PARTIAL
    assert v.averitec_label is AveritecLabel.CONFLICTING


def test_verifier_cannot_emit_conflict_retries_then_fails():
    # CONFLICT는 VerifierLabel에 없음 → 스키마 검증 실패 → 재시도 소진 후 LLMError.
    from trev.llm import LLMError

    bad = json.dumps({"label": "CONFLICT", "confidence": 0.5, "justification": "x",
                      "cited": ["g0"], "stances": []})
    client = FakeClient([bad, bad, bad])
    with pytest.raises(LLMError):
        verify(_claim(), _evidence(), LLM(client=client, max_retries=3))


def test_empty_cited_rejected_then_retry_succeeds():
    bad = json.dumps({"label": "NEI", "confidence": 0.3, "justification": "none",
                      "cited": [], "stances": []})
    good = json.dumps({"label": "NEI", "confidence": 0.3, "justification": "none",
                       "cited": ["g0"], "stances": []})
    client = FakeClient([bad, good])
    v = verify(_claim(), _evidence(), LLM(client=client, max_retries=3))
    assert v.cited == ["g0"]  # 무인용 거부 후 재시도로 회복
    assert len(client.calls) == 2


def test_json_parse_failure_retries():
    client = FakeClient([
        "not json at all",
        json.dumps({"label": "SUPPORT", "confidence": 0.8, "justification": "ok",
                    "cited": ["g0"], "stances": []}),
    ])
    v = verify(_claim(), _evidence(), LLM(client=client, max_retries=3))
    assert v.label5 is Label5.SUPPORT
    assert len(client.calls) == 2

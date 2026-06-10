"""스키마 형태·enum 값 검증(실데이터 표기 고정)."""

import pytest
from pydantic import ValidationError

from trev.schemas import (
    AveritecLabel,
    Claim,
    ClaimType,
    Evidence,
    Label5,
    Role,
    Verdict,
)


def test_gold_label_strings_match_data():
    # 채점이 어긋나지 않도록 데이터의 정확 표기 고정.
    assert AveritecLabel.CONFLICTING.value == "Conflicting Evidence/Cherrypicking"
    assert AveritecLabel.NOT_ENOUGH_EVIDENCE.value == "Not Enough Evidence"


def test_claim_type_values_cover_dev():
    assert ClaimType("Numerical Claim") is ClaimType.NUMERICAL
    assert ClaimType("Event/Property Claim") is ClaimType.EVENT_PROPERTY


def test_claim_minimal_and_optional_date():
    c = Claim(claim_id=0, text="x", type=ClaimType.NUMERICAL)
    assert c.claim_date is None and c.label is None


def test_evidence_defaults_to_t4_general():
    e = Evidence(doc_id="d", snippet="s")
    assert e.tier == 4 and e.role is Role.GENERAL and e.weight == 0.1


def test_verdict_confidence_bounds():
    Verdict(
        claim_id=0,
        label5=Label5.SUPPORT,
        averitec_label=AveritecLabel.SUPPORTED,
        confidence=1.0,
        justification="j",
        cited=["d1"],
    )
    with pytest.raises(ValidationError):
        Verdict(
            claim_id=0,
            label5=Label5.NEI,
            averitec_label=AveritecLabel.NOT_ENOUGH_EVIDENCE,
            confidence=1.5,  # 범위 밖
            justification="j",
        )

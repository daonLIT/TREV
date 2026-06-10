"""D1 적재·필터·날짜·카운트 단위 테스트(픽스처 기반, 외부 행동 검증)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trev.data.dataset import (
    conflicting_count,
    load_averitec,
    normalize_claim_type,
    parse_claim_date,
    subset_gate,
)
from trev.guards import DataHygieneError
from trev.schemas import AveritecLabel, ClaimType

FIXTURE = Path(__file__).parent / "fixtures" / "dev.json"


# --- parse_claim_date (순수 케이스 테이블) ---------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("31-10-2020", "2020-10-31"),   # DD-MM-YYYY (실데이터 포맷)
        ("2019-05-01", "2019-05-01"),   # YYYY-MM-DD
        ("", None),                      # 빈 문자열
        (None, None),                    # 결측
        ("not-a-date", None),            # 파싱 실패 → None(시점필터 통과)
        ("  30-10-2020  ", "2020-10-30"), # 공백 트림
    ],
)
def test_parse_claim_date(raw, expected):
    assert parse_claim_date(raw) == expected


# --- normalize_claim_type (멀티타입 우선순위) ------------------------------

def test_single_subset_type():
    assert normalize_claim_type(["Numerical Claim"]) is ClaimType.NUMERICAL


def test_multi_type_priority_numerical_wins():
    assert (
        normalize_claim_type(["Event/Property Claim", "Numerical Claim"])
        is ClaimType.NUMERICAL
    )


def test_non_subset_type_excluded():
    assert normalize_claim_type(["Position Statement"]) is None


def test_quote_toggle():
    assert normalize_claim_type(["Quote Verification"]) is None
    assert (
        normalize_claim_type(["Quote Verification"], include_quote=True)
        is ClaimType.QUOTE
    )


def test_multi_type_keeps_subset_member_only():
    # Causal은 대상 아님, Event/Property로 통과.
    assert (
        normalize_claim_type(["Causal Claim", "Event/Property Claim"])
        is ClaimType.EVENT_PROPERTY
    )


# --- load_averitec (픽스처) -----------------------------------------------

def test_filter_excludes_position_and_quote_by_default():
    claims = load_averitec(FIXTURE)
    # A, C, E 통과 (B=position, D=quote 제외)
    assert len(claims) == 3
    assert {c.type for c in claims} == {ClaimType.NUMERICAL, ClaimType.EVENT_PROPERTY}


def test_quote_toggle_includes_quote_claim():
    claims = load_averitec(FIXTURE, include_quote=True)
    assert len(claims) == 4
    assert any(c.type is ClaimType.QUOTE for c in claims)


def test_claim_id_is_original_index_not_renumbered():
    claims = load_averitec(FIXTURE)
    # A(0), C(2), E(4) — B(1)/D(3)는 제외되어 인덱스가 비연속.
    assert [c.claim_id for c in claims] == [0, 2, 4]


def test_claim_date_normalized():
    claims = {c.claim_id: c for c in load_averitec(FIXTURE)}
    assert claims[0].claim_date == "2020-10-31"  # A: DD-MM-YYYY
    assert claims[2].claim_date == "2019-05-01"  # C: YYYY-MM-DD
    assert claims[4].claim_date is None          # E: 결측 필드


def test_gold_label_preserved():
    claims = {c.claim_id: c for c in load_averitec(FIXTURE)}
    assert claims[0].label is AveritecLabel.REFUTED
    assert claims[2].label is AveritecLabel.CONFLICTING


# --- Conflicting 카운트 + 게이트 -------------------------------------------

def test_conflicting_count_and_qualitative_branch():
    claims = load_averitec(FIXTURE)
    assert conflicting_count(claims) == 1
    gate = subset_gate(claims)
    assert gate == {"subset_size": 3, "conflicting": 1, "eval_mode": "qualitative"}


# --- 위생 가드 연동 --------------------------------------------------------

def test_loader_rejects_train_split():
    with pytest.raises(DataHygieneError):
        load_averitec("data_store/averitec/train.json")

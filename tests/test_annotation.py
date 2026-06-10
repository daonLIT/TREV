"""S9 사람 주석 하네스 단위 테스트."""

from __future__ import annotations

import pytest

from trev.eval.annotation import (
    cohens_kappa,
    export_conflicting_for_refinement,
    export_topic_sample,
    ingest_5label,
    read_csv,
    refinement_branch,
    topic_agreement,
    write_csv,
)
from trev.schemas import AveritecLabel, Claim, ClaimType


def _claim(cid, text="t", topic=None, label=None):
    return Claim(claim_id=cid, text=text, type=ClaimType.NUMERICAL, topic=topic, label=label)


# --- Cohen's κ -------------------------------------------------------------

def test_kappa_perfect_agreement():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_chance_level_near_zero():
    # 한쪽이 전부 같은 값이면 pe 큼 → κ 낮음.
    k = cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
    assert -0.1 <= k <= 0.1


def test_kappa_empty_none():
    assert cohens_kappa([], []) is None


# --- 토픽 표본 export + 취합 ----------------------------------------------

def test_export_topic_sample_deterministic_and_capped():
    claims = [_claim(i, topic="politics") for i in range(10)]
    s1 = export_topic_sample(claims, n=5, seed=1)
    s2 = export_topic_sample(claims, n=5, seed=1)
    assert [r["claim_id"] for r in s1] == [r["claim_id"] for r in s2]  # 시드 고정
    assert len(s1) == 5
    assert s1[0]["human_topic"] == "" and s1[0]["llm_topic"] == "politics"


def test_topic_agreement_skips_unfilled():
    records = [
        {"llm_topic": "politics", "human_topic": "politics"},
        {"llm_topic": "health", "human_topic": "economy"},
        {"llm_topic": "economy", "human_topic": ""},   # 미검수 → 제외
    ]
    out = topic_agreement(records)
    assert out["n"] == 2 and out["agreement"] == 0.5
    assert out["cohens_kappa"] is not None


# --- Conflicting 5라벨 세분 ------------------------------------------------

def test_export_conflicting_only():
    claims = [
        _claim(0, label=AveritecLabel.CONFLICTING),
        _claim(1, label=AveritecLabel.REFUTED),
        _claim(2, label=AveritecLabel.CONFLICTING),
    ]
    rows = export_conflicting_for_refinement(claims)
    assert [r["claim_id"] for r in rows] == [0, 2]
    assert all(r["human_label5"] == "" for r in rows)


def test_refinement_branch_quantitative_when_ge_threshold():
    many = [_claim(i, label=AveritecLabel.CONFLICTING) for i in range(20)]
    few = [_claim(i, label=AveritecLabel.CONFLICTING) for i in range(5)]
    assert refinement_branch(many) == "quantitative"
    assert refinement_branch(few) == "qualitative"


def test_ingest_5label_validates():
    records = [
        {"claim_id": "0", "human_label5": "partial"},   # 대소문자 무관
        {"claim_id": "2", "human_label5": "CONFLICT"},
        {"claim_id": "3", "human_label5": ""},           # 미기입 제외
        {"claim_id": "4", "human_label5": "BOGUS"},       # 무효 제외
    ]
    assert ingest_5label(records) == {0: "PARTIAL", 2: "CONFLICT"}


# --- CSV round-trip --------------------------------------------------------

def test_csv_roundtrip(tmp_path):
    records = export_conflicting_for_refinement([_claim(0, label=AveritecLabel.CONFLICTING)])
    path = tmp_path / "ann" / "conflicting.csv"
    write_csv(path, records)
    loaded = read_csv(path)
    assert loaded[0]["claim_id"] == "0" and "human_label5" in loaded[0]

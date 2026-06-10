"""D5 Recall 매칭 단위 테스트 — URL 정규화·Recall/Precision·원인 분류."""

from __future__ import annotations

import pytest

from trev.recall import (
    classify_retrieval,
    normalize_url,
    precision_at_k,
    recall_at_k,
)
from trev.schemas import AveritecLabel

ARCHIVE = "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed/"
ORIGINAL = "http://www.scoopertino.com/exposed?utm=1#frag"


# --- normalize_url ---------------------------------------------------------

def test_archive_and_original_normalize_equal():
    # 아카이브 스냅샷과 원본(스킴/www/쿼리/fragment 차이)이 같은 키로 정규화.
    assert normalize_url(ARCHIVE) == normalize_url(ORIGINAL) == "scoopertino.com/exposed"


def test_normalize_trailing_slash_and_empty():
    assert normalize_url("https://x.com/a/") == "x.com/a"
    assert normalize_url(None) == "" and normalize_url("") == ""


# --- recall@k / precision@k ------------------------------------------------

def test_recall_hits_after_normalization():
    gold = [ARCHIVE]                       # 아카이브 gold
    retrieved = ["https://other.com/x", ORIGINAL]  # 원본으로 회수
    assert recall_at_k(retrieved, gold, k=5) == 1.0
    assert recall_at_k(retrieved, gold, k=1) == 0.0   # top-1엔 other.com만


def test_recall_none_when_no_gold():
    assert recall_at_k(["https://x.com/a"], [], k=5) is None


def test_precision_at_k():
    gold = ["https://g.com/a"]
    retrieved = ["https://g.com/a", "https://b.com/c"]
    assert precision_at_k(retrieved, gold, k=2) == 0.5
    assert precision_at_k([], gold, k=2) is None


# --- 검색실패 vs 실제NEI 구분 ----------------------------------------------

def test_classify_retrieved():
    assert classify_retrieval([ARCHIVE], [ARCHIVE], [ORIGINAL],
                              gold_label=AveritecLabel.REFUTED, k=5) == "retrieved"


def test_classify_retrieval_failure():
    # gold가 KS엔 있지만 top-k엔 없음 → 검색 실패.
    out = classify_retrieval([ARCHIVE], [ARCHIVE], ["https://other.com/x"],
                             gold_label=AveritecLabel.REFUTED, k=5)
    assert out == "retrieval_failure"


def test_classify_gold_absent_from_ks():
    out = classify_retrieval([ARCHIVE], ["https://nothere.com/y"], ["https://other.com/x"],
                             gold_label=AveritecLabel.REFUTED, k=5)
    assert out == "gold_absent_from_ks"


def test_classify_true_nei():
    # gold 근거 URL 없음 + label=NEI → 실제 부재.
    out = classify_retrieval([], [], [], gold_label=AveritecLabel.NOT_ENOUGH_EVIDENCE, k=5)
    assert out == "true_nei"

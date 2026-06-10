"""D2 KS 로더 단위 테스트(픽스처 JSONL 주입)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trev.knowledge_store import (
    derive_published_at,
    extract_domain,
    load_claim_passages,
    recover_archive_url,
)

KS_DIR = Path(__file__).parent / "fixtures" / "knowledge_store" / "dev"

ARCHIVE_URL = (
    "https://web.archive.org/web/20201129141238/"
    "https://scoopertino.com/exposed-the-imac-disaster/"
)


# --- published_at 유도 (순수 케이스) --------------------------------------

def test_derive_published_at_from_archive():
    assert derive_published_at(ARCHIVE_URL) == "2020-11-29"


def test_derive_published_at_non_archive_is_none():
    assert derive_published_at("https://www.example.com/article") is None


def test_derive_published_at_bad_timestamp_is_none():
    bad = "https://web.archive.org/web/20209999999999/https://x.com/"
    assert derive_published_at(bad) is None


# --- 도메인 추출 (아카이브 원본 복원) -------------------------------------

def test_recover_archive_url():
    assert recover_archive_url(ARCHIVE_URL) == (
        "https://scoopertino.com/exposed-the-imac-disaster/"
    )


def test_extract_domain_recovers_original_and_strips_www():
    assert extract_domain(ARCHIVE_URL) == "scoopertino.com"
    assert extract_domain("https://www.example.com/article?utm=1") == "example.com"


# --- load_claim_passages (매핑·추출·메타·dedup·gold) ----------------------

def test_passages_extracted_with_meta():
    passages = load_claim_passages(0, KS_DIR)
    bbc = [p for p in passages if "bbc" in p.url]
    assert bbc and bbc[0].source_domain == "news.bbc.co.uk"  # netloc 그대로(축약 X)
    assert all(p.claim_id == 0 for p in passages)


def test_dedup_collapses_duplicate_url_text():
    passages = load_claim_passages(0, KS_DIR)
    # example.com 본문 2줄이 question + question_duplicate로 2회 등장 → dedup 후 2개만.
    example = [p for p in passages if "example.com" in p.url]
    assert len(example) == 2


def test_empty_and_blank_passages_skipped():
    passages = load_claim_passages(0, KS_DIR)
    # most_similar의 "  "/"" 라인 제외 → BBC는 1줄만.
    bbc = [p for p in passages if "bbc" in p.url]
    assert len(bbc) == 1
    # url이 빈 gpt_url_only 레코드는 통째로 제외.
    assert all(p.url for p in passages)


def test_gold_type_kept_as_normal_candidate():
    passages = load_claim_passages(0, KS_DIR)
    gold = [p for p in passages if p.ks_type == "gold"]
    # 격리·제거가 아니라 일반 후보로 유지(부스팅은 하지 않음).
    assert len(gold) == 2
    assert gold[0].published_at == "2020-11-29"  # gold는 아카이브 → 시점 유도됨


def test_claim_id_mismatch_raises():
    with pytest.raises(ValueError):
        load_claim_passages(9, KS_DIR)  # 9.json 레코드 claim_id="0"


def test_loader_rejects_non_dev_split(tmp_path):
    # knowledge_store/test/... 경로는 위생 가드가 차단.
    from trev.guards import DataHygieneError

    bad = tmp_path / "knowledge_store" / "test"
    bad.mkdir(parents=True)
    (bad / "0.json").write_text('{"claim_id":"0","url":"x","url2text":[]}', encoding="utf-8")
    with pytest.raises(DataHygieneError):
        load_claim_passages(0, bad)

"""D3 tier 입력 단위 테스트: 도메인 빈도·화이트리스트 분류·자기출처매칭."""

from __future__ import annotations

from pathlib import Path

from trev.knowledge_store import claim_source_domains
from trev.schemas import Claim, ClaimType
from trev.tier import classify_domain, is_self_source, subset_domain_frequency

KS_DIR = Path(__file__).parent / "fixtures" / "knowledge_store" / "dev"

TIER_CFG = {
    "overrides": {"example.com": 2},
    "heuristics": {
        1: [".gov", "court"],
        2: ["politifact", "factcheck"],
        3: ["bbc.", "reuters"],
    },
}


# --- classify_domain (overrides → heuristics → T4) -------------------------

def test_override_wins():
    assert classify_domain("example.com", TIER_CFG) == 2


def test_suffix_rule_gov():
    assert classify_domain("cdc.gov", TIER_CFG) == 1
    assert classify_domain("cancercontrol.cancer.gov", TIER_CFG) == 1


def test_substring_rule_factcheck():
    assert classify_domain("politifact.com", TIER_CFG) == 2


def test_major_media_t3():
    assert classify_domain("news.bbc.co.uk", TIER_CFG) == 3
    assert classify_domain("reuters.com", TIER_CFG) == 3


def test_unknown_defaults_t4():
    assert classify_domain("randomblog.wordpress.com", TIER_CFG) == 4
    assert classify_domain(None, TIER_CFG) == 4


# --- claim_source_domains (D1이 사용) -------------------------------------

def test_source_domains_from_archive_url_and_social_name():
    domains = claim_source_domains(
        "https://web.archive.org/web/20201113115127/https://twitter.com/x/status/1",
        "Facebook",
    )
    assert "twitter.com" in domains   # original_claim_url(아카이브 복원)
    assert "facebook.com" in domains  # reporting_source 이름→도메인


def test_source_domains_unknown_name_skipped():
    assert claim_source_domains(None, "The Federalist") == []  # 추측 안 함


# --- is_self_source --------------------------------------------------------

def _claim(source_domains):
    return Claim(claim_id=0, text="t", type=ClaimType.NUMERICAL, source_domains=source_domains)


def test_self_source_exact_and_subdomain():
    claim = _claim(["twitter.com"])
    assert is_self_source("twitter.com", claim) is True
    assert is_self_source("mobile.twitter.com", claim) is True   # 서브도메인
    assert is_self_source("nytimes.com", claim) is False


def test_self_source_empty_when_no_source():
    assert is_self_source("twitter.com", _claim([])) is False


# --- subset_domain_frequency (픽스처 KS) ----------------------------------

def test_domain_frequency_over_fixture():
    # 픽스처 claim0: example.com·scoopertino.com·news.bbc.co.uk (고유 URL 단위).
    claim = Claim(claim_id=0, text="t", type=ClaimType.NUMERICAL)
    freq = dict(subset_domain_frequency([claim], KS_DIR))
    assert freq.get("example.com") == 1   # question/question_duplicate 동일 URL → 1
    assert "scoopertino.com" in freq       # 아카이브 원본 복원
    assert "news.bbc.co.uk" in freq

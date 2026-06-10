"""S3 retriever 단위 테스트(가짜 임베더 — 실모델 없이 검색·시점필터·Evidence 검증)."""

from __future__ import annotations

from trev.data.indexing import ClaimIndex
from trev.pipeline.retriever import build_queries, passes_time_filter, retrieve
from trev.schemas import Claim, ClaimType, Passage
from tests.test_indexing import FakeEmbedder


def _claim(text="alpha numerical claim", date=None):
    return Claim(claim_id=0, text=text, type=ClaimType.NUMERICAL, claim_date=date)


def _index(passages):
    return ClaimIndex.build(passages, FakeEmbedder())


def _p(url, text, **kw):
    return Passage(claim_id=0, url=url, text=text, **kw)


# --- 질의 템플릿 -----------------------------------------------------------

def test_build_queries_returns_multiple_with_claim_text():
    qs = build_queries(_claim("the moon is made of cheese"))
    assert len(qs) >= 2
    assert all("the moon is made of cheese" in q for q in qs)


# --- 시점 필터 (순수 경계) -------------------------------------------------

def test_time_filter_boundaries():
    assert passes_time_filter(None, "2020-10-31") is True          # 유도 실패 → 통과
    assert passes_time_filter("2020-01-01", None) is True          # claim_date 결측 → 통과
    assert passes_time_filter("2020-10-30", "2020-10-31") is True  # 이전
    assert passes_time_filter("2020-10-31", "2020-10-31") is True  # 같은 날(<=)
    assert passes_time_filter("2020-11-01", "2020-10-31") is False # 이후 → 차단


# --- retrieve --------------------------------------------------------------

def test_gpt_only_skips_search():
    idx = _index([_p("u", "alpha")])
    assert retrieve(_claim(), idx, FakeEmbedder(), mode="gpt_only") == []


def test_returns_evidence_with_meta():
    idx = _index([_p("uA", "alpha numerical claim", source_domain="bbc.co.uk",
                     published_at="2019-01-01")])
    ev = retrieve(_claim(date="2020-01-01"), idx, FakeEmbedder(), k=5)
    assert ev and ev[0].url == "uA"
    assert ev[0].source_domain == "bbc.co.uk"
    assert ev[0].published_at == "2019-01-01"
    assert ev[0].sim is not None and ev[0].score == ev[0].sim
    assert ev[0].tier == 4 and ev[0].weight == 0.1  # 기본값(가중은 S4)


def test_time_filter_excludes_future_evidence():
    passages = [
        _p("past", "alpha numerical claim", published_at="2019-01-01"),
        _p("future", "alpha numerical claim", published_at="2099-01-01"),
        _p("undated", "alpha numerical claim"),  # published_at None → 통과
    ]
    ev = retrieve(_claim(date="2020-01-01"), _index(passages), FakeEmbedder(), k=10)
    urls = {e.url for e in ev}
    assert "past" in urls and "undated" in urls
    assert "future" not in urls  # 시점 누수 차단


def test_topk_respected():
    passages = [_p(f"u{i}", f"alpha token{i} claim") for i in range(8)]
    ev = retrieve(_claim(), _index(passages), FakeEmbedder(), k=3)
    assert len(ev) == 3

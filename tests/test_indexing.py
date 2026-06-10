"""D4 인덱싱 단위 테스트 — 가짜 임베더로 실모델 없이 FAISS/BM25 seam 검증."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from trev.indexing import ClaimIndex, build_claim_index, chunk_passages
from trev.schemas import Passage

KS_DIR = Path(__file__).parent / "fixtures" / "knowledge_store" / "dev"


class FakeEmbedder:
    """토큰 bag-of-words를 고정 차원에 해싱(결정론적). 토큰 공유 시 유사도 ↑."""

    def __init__(self, dim: int = 512):  # 충돌 회피용 충분한 차원
        self.dim = dim

    def encode(self, texts, *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                out[r, h] += 1.0
        return out


def _p(claim_id, url, text, **kw):
    return Passage(claim_id=claim_id, url=url, text=text, **kw)


# --- 재청킹 ----------------------------------------------------------------

def test_chunk_groups_by_url_and_merges_lines():
    passages = [
        _p(0, "u1", "alpha beta", source_domain="d1", published_at="2020-01-01"),
        _p(0, "u1", "gamma"),
        _p(0, "u2", "delta"),
    ]
    chunks = chunk_passages(passages, max_words=180)
    by_url = {c.url: c for c in chunks}
    assert by_url["u1"].text == "alpha beta gamma"  # 같은 url 라인 병합
    assert by_url["u1"].source_domain == "d1"        # 메타 상속
    assert by_url["u1"].published_at == "2020-01-01"
    assert set(by_url) == {"u1", "u2"}


def test_chunk_splits_by_max_words_and_caps_per_url():
    passages = [_p(0, "u1", " ".join(str(i) for i in range(10)))]
    assert len(chunk_passages(passages, max_words=4)) == 3            # 10/4 -> 3 청크
    assert len(chunk_passages(passages, max_words=4, max_chunks_per_url=2)) == 2  # 상한


# --- per-claim 인덱스 ------------------------------------------------------

def _index(passages):
    return ClaimIndex.build(passages, FakeEmbedder())


def test_dense_search_ranks_token_match_first_with_meta():
    passages = [
        _p(0, "uA", "bbc news report", source_domain="bbc.co.uk", published_at="2019-01-01"),
        _p(0, "uB", "weather forecast sunny"),
    ]
    hits = _index(passages).search_dense("bbc", FakeEmbedder(), k=2)
    assert hits[0].passage.url == "uA"                  # 토큰 매칭 1위
    assert hits[0].passage.source_domain == "bbc.co.uk" # 메타 동반
    assert hits[0].passage.published_at == "2019-01-01"


def test_bm25_search_ranks_token_match_first():
    # 3+ 문서라야 희소 term의 IDF가 0이 아님(소코퍼스 퇴화 회피).
    passages = [
        _p(0, "uA", "election results numerical figures"),
        _p(0, "uB", "cooking recipe food"),
        _p(0, "uC", "sports match score"),
    ]
    hits = _index(passages).search_bm25("numerical", k=3)
    assert hits[0].passage.url == "uA"


def test_topk_respects_k():
    passages = [_p(0, f"u{i}", f"text token{i}") for i in range(5)]
    assert len(_index(passages).search_dense("token1", FakeEmbedder(), k=3)) == 3


def test_per_claim_isolation():
    idx_a = _index([_p(0, "uA", "claim zero doc")])
    idx_b = _index([_p(1, "uB", "claim one doc")])
    assert {p.url for p in idx_a.passages} == {"uA"}
    assert {p.url for p in idx_b.passages} == {"uB"}
    assert all(p.claim_id == 0 for p in idx_a.passages)


def test_empty_index_returns_no_hits():
    idx = _index([])
    assert idx.search_dense("x", FakeEmbedder(), k=5) == []
    assert idx.search_bm25("x", k=5) == []


# --- D2 → 청킹 → 인덱스 결선 (픽스처 KS) ----------------------------------

def test_build_claim_index_end_to_end():
    idx = build_claim_index(0, FakeEmbedder(), ks_dir=KS_DIR)
    urls = {p.url for p in idx.passages}
    assert any("example.com" in u for u in urls)
    hits = idx.search_dense("BBC", FakeEmbedder(), k=3)
    assert hits and any("bbc" in h.passage.url for h in hits)

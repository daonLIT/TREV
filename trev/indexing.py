"""D4: per-claim 인덱싱 (e5 FAISS + BM25) + 메타 보존.

D2 passage(=페이지 라인)를 URL별로 병합·재청킹한 뒤, claim마다 그 claim의 청크만으로
dense(FAISS) + lexical(BM25) 인덱스를 만든다. 전역 단일 인덱스는 금지(시점 누수).
각 청크는 Passage 메타(url·source_domain·published_at)를 그대로 유지해 검색 결과가
tier·시점필터·Recall@k에 바로 쓰인다.

embedder는 주입 가능 인터페이스(`encode(texts, is_query=False) -> np.ndarray`)다.
테스트는 가짜 임베더로 모델 다운로드 없이 검증하고, 실험은 `E5Embedder`(lazy)를 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import json
from pathlib import Path

import numpy as np

from trev.knowledge_store import (
    DEFAULT_KS_DIR,
    derive_published_at,
    extract_domain,
    iter_claim_records,
)
from trev.schemas import Passage


# --- 재청킹 ----------------------------------------------------------------

def chunk_passages(
    passages: list[Passage],
    *,
    max_words: int = 180,
    max_chunks_per_url: int | None = None,
) -> list[Passage]:
    """passage(라인)를 URL별로 병합한 뒤 ~max_words 단어 청크로 재구성한다.

    메타(url·source_domain·published_at·ks_type)는 상속한다. `max_chunks_per_url`은
    페이지당 청크 수 상한(앞부분=lead 우선) — claim당 passage 폭증 방지.
    """
    grouped: dict[str, dict] = {}
    for p in passages:
        g = grouped.setdefault(
            p.url,
            {"words": [], "domain": p.source_domain, "published_at": p.published_at,
             "ks_type": p.ks_type, "claim_id": p.claim_id},
        )
        g["words"].extend(p.text.split())

    chunks: list[Passage] = []
    for url, g in grouped.items():
        words = g["words"]
        n_chunks = 0
        for i in range(0, len(words), max_words):
            if max_chunks_per_url is not None and n_chunks >= max_chunks_per_url:
                break
            chunks.append(
                Passage(
                    claim_id=g["claim_id"],
                    url=url,
                    text=" ".join(words[i:i + max_words]),
                    source_domain=g["domain"],
                    published_at=g["published_at"],
                    ks_type=g["ks_type"],
                )
            )
            n_chunks += 1
    return chunks


# --- embedder 인터페이스 ---------------------------------------------------

class Embedder(Protocol):
    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


class E5Embedder:
    """`intfloat/multilingual-e5-large` 임베더. sentence-transformers를 lazy 로드한다.

    e5 규약대로 passage/query에 접두사를 붙인다.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        vecs = self._ensure().encode([prefix + t for t in texts], convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def _normalize(mat: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화(코사인=내적). 0 벡터는 그대로 둔다."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --- per-claim 인덱스 ------------------------------------------------------

@dataclass
class SearchHit:
    passage: Passage
    score: float


class ClaimIndex:
    """단일 claim의 청크에 대한 dense(FAISS) + lexical(BM25) 인덱스."""

    def __init__(self, passages: list[Passage], faiss_index, bm25):
        self.passages = passages
        self._faiss = faiss_index
        self._bm25 = bm25

    @classmethod
    def build(cls, passages: list[Passage], embedder: Embedder) -> "ClaimIndex":
        import faiss
        from rank_bm25 import BM25Okapi

        dim = 1
        faiss_index = faiss.IndexFlatIP(dim)
        if passages:
            emb = _normalize(embedder.encode([p.text for p in passages], is_query=False))
            dim = emb.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(emb)
        bm25 = BM25Okapi([_tokenize(p.text) for p in passages] or [[""]])
        return cls(passages, faiss_index, bm25)

    def save(self, directory: str | Path) -> None:
        """FAISS 인덱스 + passage 메타를 디스크에 저장한다(BM25는 재구성). 빌드 1회용."""
        import faiss

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(d / "faiss.index"))
        (d / "passages.json").write_text(
            json.dumps([p.model_dump() for p in self.passages], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ClaimIndex":
        """save된 인덱스를 재로드한다(e5 재임베딩 없이 — 실험 재실행 가속)."""
        import faiss
        from rank_bm25 import BM25Okapi

        d = Path(directory)
        faiss_index = faiss.read_index(str(d / "faiss.index"))
        passages = [
            Passage(**o)
            for o in json.loads((d / "passages.json").read_text(encoding="utf-8"))
        ]
        bm25 = BM25Okapi([_tokenize(p.text) for p in passages] or [[""]])
        return cls(passages, faiss_index, bm25)

    def search_dense(self, query: str, embedder: Embedder, k: int = 5) -> list[SearchHit]:
        if not self.passages:
            return []
        q = _normalize(embedder.encode([query], is_query=True))
        scores, idx = self._faiss.search(q, min(k, len(self.passages)))
        return [SearchHit(self.passages[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def search_bm25(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self.passages:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [SearchHit(self.passages[i], float(scores[i])) for i in order]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_claim_chunks(
    claim_id: int,
    *,
    ks_dir=DEFAULT_KS_DIR,
    max_words: int = 180,
    max_chunks_per_url: int | None = None,
    dedup: bool = True,
) -> list[Passage]:
    """KS 레코드를 스트리밍하며 URL별 단어를 모아 청크를 만든다(메모리 효율).

    `chunk_passages(load_claim_passages(...))`와 동일 결과지만, url2text를 line별
    Passage로 펼치지 않아 대용량 claim(라인 수십만)에서 객체 생성 비용을 피한다.
    `max_chunks_per_url`이 있으면 URL당 단어 예산(=cap×max_words)에서 조기 종료한다.
    """
    budget = max_words * max_chunks_per_url if max_chunks_per_url else None
    groups: dict[str, dict] = {}  # url -> {words, domain, published_at, ks_type, seen}
    for rec in iter_claim_records(claim_id, ks_dir):
        url = rec.get("url") or ""
        if not url:
            continue
        g = groups.get(url)
        if g is None:
            g = {
                "words": [], "domain": extract_domain(url),
                "published_at": derive_published_at(url),
                "ks_type": rec.get("type"), "seen": set(),
            }
            groups[url] = g
        if budget is not None and len(g["words"]) >= budget:
            continue  # 이 URL은 이미 충분(조기 종료)
        for text in rec.get("url2text") or []:
            if not isinstance(text, str) or not text.strip():
                continue
            if dedup:
                if text in g["seen"]:
                    continue
                g["seen"].add(text)
            g["words"].extend(text.split())
            if budget is not None and len(g["words"]) >= budget:
                break

    chunks: list[Passage] = []
    for url, g in groups.items():
        words = g["words"]
        n = 0
        for i in range(0, len(words), max_words):
            if max_chunks_per_url is not None and n >= max_chunks_per_url:
                break
            chunks.append(
                Passage(
                    claim_id=claim_id, url=url, text=" ".join(words[i:i + max_words]),
                    source_domain=g["domain"], published_at=g["published_at"],
                    ks_type=g["ks_type"],
                )
            )
            n += 1
    return chunks


def build_claim_index(
    claim_id: int,
    embedder: Embedder,
    *,
    ks_dir=DEFAULT_KS_DIR,
    max_words: int = 180,
    max_chunks_per_url: int | None = None,
) -> ClaimIndex:
    """KS 스트리밍 청킹 → per-claim 인덱스 빌드(전 과정 결선)."""
    chunks = build_claim_chunks(
        claim_id, ks_dir=ks_dir, max_words=max_words,
        max_chunks_per_url=max_chunks_per_url,
    )
    return ClaimIndex.build(chunks, embedder)

"""D4 실 e5 인덱스 빌드 데모: 단일 claim의 per-claim FAISS+BM25 인덱스 + 질의.

실 e5(`sentence-transformers`, torch) lazy 로드 — 무거우므로 상한(config) 권장.
실행: python -m scripts.build_index --claim 0 --query "Sean Connery letter" --cap 30
"""

from __future__ import annotations

import argparse

from trev.config import load_config
from trev.indexing import E5Embedder, build_claim_index


def main() -> None:
    cfg = load_config().get("index", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", type=int, default=0)
    ap.add_argument("--query", default="claim evidence")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--cap", type=int, default=cfg.get("max_chunks_per_url"),
        help="URL당 청크 상한(미지정 시 config)",
    )
    args = ap.parse_args()

    embedder = E5Embedder(cfg.get("e5_model", "intfloat/multilingual-e5-large"))
    index = build_claim_index(
        args.claim, embedder,
        max_words=cfg.get("chunk_max_words", 180),
        max_chunks_per_url=args.cap,
    )
    print(f"claim {args.claim}: 청크 {len(index.passages)}개")
    print(f"\n[dense] query={args.query!r}")
    for h in index.search_dense(args.query, embedder, k=args.k):
        print(f"  {h.score:.3f}  {h.passage.source_domain}  {h.passage.text[:80]}")
    print(f"\n[bm25] query={args.query!r}")
    for h in index.search_bm25(args.query, k=args.k):
        print(f"  {h.score:.3f}  {h.passage.source_domain}  {h.passage.text[:80]}")


if __name__ == "__main__":
    main()

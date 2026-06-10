"""S6: 4조건 실험 실행기 (03_run_experiments).

서브셋 claim마다 per-claim 인덱스를 빌드(또는 캐시 로드)하고 4조건을 실행해
조건별 예측을 outputs/predictions_{method}.json에 저장한다(재현·재채점).

실행: python -m scripts.run_experiments --method dense --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from trev.config import load_config
from trev.data.dataset import load_averitec
from trev.experiment import DEFAULT_MODES, predictions_to_records, run_experiments
from trev.data.indexing import E5Embedder, ClaimIndex, build_claim_index
from trev.llm import LLM

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "index"
OUT_DIR = REPO / "outputs"


def _index_provider(embedder, idx_cfg, *, cache: bool):
    """claim → ClaimIndex. 캐시되어 있으면 로드, 아니면 빌드 후 저장(빌드 1회)."""
    def provide(claim):
        cdir = INDEX_DIR / str(claim.claim_id)
        if cache and (cdir / "faiss.index").exists():
            return ClaimIndex.load(cdir)
        index = build_claim_index(
            claim.claim_id, embedder,
            max_words=idx_cfg.get("chunk_max_words", 180),
            max_chunks_per_url=idx_cfg.get("max_chunks_per_url"),
        )
        if cache:
            index.save(cdir)
        return index
    return provide


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dense", "bm25"], default="dense")
    ap.add_argument("--limit", type=int, default=None, help="claim 수 제한(스모크용)")
    ap.add_argument("--no-cache", action="store_true", help="인덱스 디스크 캐시 비활성")
    ap.add_argument("--agentic", action="store_true", help="멀티에이전트 조건만 실행")
    args = ap.parse_args()

    cfg = load_config()
    claims = load_averitec(
        include_quote=cfg.get("data", {}).get("include_quote_verification", False)
    )
    if args.limit:
        claims = claims[: args.limit]

    embedder = E5Embedder(cfg.get("index", {}).get("e5_model", "intfloat/multilingual-e5-large"))
    provider = _index_provider(embedder, cfg.get("index", {}), cache=not args.no_cache)
    llm = LLM.from_config(cfg)

    modes = ("agentic",) if args.agentic else DEFAULT_MODES
    results = run_experiments(
        claims, embedder, llm, index_provider=provider,
        tier_config=cfg.get("tier", {}), method=args.method, modes=modes,
    )
    records = predictions_to_records(claims, results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "agentic" if args.agentic else args.method
    out = OUT_DIR / f"predictions_{suffix}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out}  ({len(claims)} claims × {len(results)} 조건 = {len(records)} 예측)")
    for mode in results:
        from collections import Counter
        dist = Counter(p["verdict"].averitec_label.value for p in results[mode])
        print(f"  {mode:16s} {dict(dist)}")


if __name__ == "__main__":
    main()

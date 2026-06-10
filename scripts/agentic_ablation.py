"""A7: agentic ablation + agentic vs deterministic + N=3 일치율.

- tier 도구 on/off: agentic vs agentic_no_tier 비교.
- agentic vs deterministic proposed 비교 + 토픽별 분해.
- agentic N=3 반복 라벨 일치율('결정론' 아님).

실행: python -m scripts.agentic_ablation --limit 5 --n 3
(주의: agentic은 claim당 도구호출이 많아 비용·시간이 큼)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from trev.eval.ablation import agreement_rate, compare_methods
from trev.config import load_config
from trev.data.dataset import load_averitec
from trev.experiment import (
    labels_by_claim,
    predictions_to_records,
    run_experiments,
)
from trev.data.indexing import ClaimIndex, E5Embedder, build_claim_index
from trev.llm import LLM
from trev.eval.metrics import evaluate
from trev.eval.topics import tag_claims, topic_breakdown

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "index"
OUT_DIR = REPO / "outputs"


def _provider(embedder, idx_cfg):
    def provide(claim):
        cdir = INDEX_DIR / str(claim.claim_id)
        if (cdir / "faiss.index").exists():
            return ClaimIndex.load(cdir)
        index = build_claim_index(claim.claim_id, embedder,
                                  max_words=idx_cfg.get("chunk_max_words", 180),
                                  max_chunks_per_url=idx_cfg.get("max_chunks_per_url"))
        index.save(cdir)
        return index
    return provide


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--n", type=int, default=3, help="agentic N회 반복(일치율)")
    args = ap.parse_args()

    cfg = load_config()
    claims = load_averitec(
        include_quote=cfg.get("data", {}).get("include_quote_verification", False))[: args.limit]
    embedder = E5Embedder(cfg.get("index", {}).get("e5_model", "intfloat/multilingual-e5-large"))
    provider = _provider(embedder, cfg.get("index", {}))
    llm = LLM.from_config(cfg)
    tier_cfg = cfg.get("tier", {})

    def _run(modes):
        results = run_experiments(claims, embedder, llm, index_provider=provider,
                                  tier_config=tier_cfg, modes=modes)
        return predictions_to_records(claims, results)

    # 1) tier 도구 on/off + agentic vs deterministic proposed.
    records = _run(("agentic", "agentic_no_tier", "proposed"))
    table = evaluate(records, k=10)
    topic_by_claim = {c.claim_id: c.topic for c in tag_claims(claims)}

    report = {
        "metrics": table,
        "tier_tool_ablation": compare_methods(
            {"x": table.get("agentic", {})}, {"x": table.get("agentic_no_tier", {})}),
        "agentic_vs_deterministic": compare_methods(
            {"x": table.get("agentic", {})}, {"x": table.get("proposed", {})}),
        "topic_breakdown": topic_breakdown(records, topic_by_claim,
                                           baseline="proposed", target="agentic"),
    }

    # 2) agentic N회 반복 일치율.
    runs = [labels_by_claim(_run(("agentic",)), "agentic") for _ in range(args.n)]
    report["agentic_agreement"] = agreement_rate(runs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "agentic_ablation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out}")
    for mode, m in table.items():
        print(f"  {mode:16s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"avg_tool_calls={m.get('avg_tool_calls')}")
    print(f"  agentic N={args.n} agreement={report['agentic_agreement']['agreement']}")


if __name__ == "__main__":
    main()

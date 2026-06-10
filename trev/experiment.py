"""S6: 4개 baseline 조건 실행기 + 예측 직렬화.

controller(S5)를 실 retriever(S3)·verifier(S2)·tier(S4)와 묶어, 4개 조건
(gpt_only / naive_rag / unweighted_rag / proposed)을 **동일 GPT-5·동일 KS·동일 프롬프트·
동일 검색 method**로 실행한다(검색·가중 외 변인 통제). dense vs bm25는 method로 ablation.

per-claim 인덱스는 claim당 1회 빌드해 4조건이 공유한다(불필요한 재임베딩 방지).
"""

from __future__ import annotations

from trev import controller
from trev.controller import ControllerConfig
from trev.indexing import ClaimIndex, Embedder
from trev.retriever import retrieve
from trev.schemas import Claim, Verdict
from trev.verifier import gpt_only_verdict, run_verifier

DEFAULT_MODES = ("gpt_only", "naive_rag", "unweighted_rag", "proposed")


def predict_claim(
    claim: Claim,
    index: ClaimIndex,
    embedder: Embedder,
    llm,
    *,
    mode: str,
    method: str,
    tier_config: dict,
    k: int = 10,
    candidate_n: int = 50,
    config: ControllerConfig = ControllerConfig(),
) -> Verdict:
    """단일 claim·단일 조건의 예측 Verdict를 만든다(seam을 controller에 결선)."""

    def retrieve_fn(c, expand=False):
        return retrieve(c, index, embedder, k=k, candidate_n=candidate_n,
                        method=method, expand=expand)

    return controller.run(
        claim, mode=mode,
        retrieve_fn=retrieve_fn,
        verify_fn=lambda c, ev: run_verifier(c, ev, llm),
        gpt_only_fn=lambda c: gpt_only_verdict(c, llm),
        tier_config=tier_config, config=config,
    )


def run_experiments(
    claims: list[Claim],
    embedder: Embedder,
    llm,
    *,
    index_provider,
    tier_config: dict,
    modes=DEFAULT_MODES,
    method: str = "dense",
    k: int = 10,
    candidate_n: int = 50,
    config: ControllerConfig = ControllerConfig(),
) -> dict[str, list[Verdict]]:
    """claim들 × 조건의 예측을 만든다. `index_provider(claim)`가 인덱스를 빌드/로드한다."""
    results: dict[str, list[Verdict]] = {m: [] for m in modes}
    for claim in claims:
        index = index_provider(claim)
        for mode in modes:
            results[mode].append(
                predict_claim(claim, index, embedder, llm, mode=mode, method=method,
                              tier_config=tier_config, k=k, candidate_n=candidate_n,
                              config=config)
            )
    return results


def predictions_to_records(
    claims: list[Claim], results: dict[str, list[Verdict]]
) -> list[dict]:
    """예측을 재현·재채점용 레코드로 직렬화한다(조건별 예측 + gold label)."""
    by_id = {c.claim_id: c for c in claims}
    records = []
    for mode, verdicts in results.items():
        for v in verdicts:
            gold = by_id.get(v.claim_id)
            records.append({
                "claim_id": v.claim_id,
                "mode": mode,
                "label5": v.label5.value,
                "pred_label": v.averitec_label.value,
                "gold_label": gold.label.value if gold and gold.label else None,
                "confidence": v.confidence,
                "cited": v.cited,
                "justification": v.justification,
            })
    return records

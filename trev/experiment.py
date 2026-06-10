"""S6: 4개 baseline 조건 실행기 + 예측 직렬화.

controller(S5)를 실 retriever(S3)·verifier(S2)·tier(S4)와 묶어, 4개 조건
(gpt_only / naive_rag / unweighted_rag / proposed)을 **동일 GPT-5·동일 KS·동일 프롬프트·
동일 검색 method**로 실행한다(검색·가중 외 변인 통제). dense vs bm25는 method로 ablation.

per-claim 인덱스는 claim당 1회 빌드해 4조건이 공유한다(불필요한 재임베딩 방지).
"""

from __future__ import annotations

from trev import controller
from trev.controller import ControllerConfig
from trev.dataset import gold_source_urls
from trev.indexing import ClaimIndex, Embedder
from trev.knowledge_store import load_claim_urls
from trev.orchestrator import orchestrate
from trev.recall import classify_retrieval
from trev.retriever import retrieve
from trev.schemas import Claim, Verdict
from trev.tier import rank_evidence
from trev.verifier import gpt_only_verdict, run_verifier

DEFAULT_MODES = ("gpt_only", "naive_rag", "unweighted_rag", "proposed")


def ranked_topk_urls(
    claim: Claim,
    index: ClaimIndex,
    embedder: Embedder,
    *,
    mode: str,
    method: str,
    tier_config: dict,
    k: int = 10,
    candidate_n: int = 50,
) -> list[str]:
    """모드별 ranking을 적용한 top-k 회수 URL(Recall@k 채점 입력). gpt_only는 빈 리스트."""
    if mode == "gpt_only":
        return []
    rp = {"weighted": mode == "proposed", "dynamic_role": mode != "naive_rag"}
    evidence = retrieve(claim, index, embedder, k=candidate_n, candidate_n=candidate_n,
                        method=method)
    ranked = rank_evidence(claim, evidence, tier_config, **rp)
    return [e.url for e in ranked[:k] if e.url]


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
) -> dict[str, list[dict]]:
    """claim들 × 조건의 예측 + 회수 URL + 비용을 만든다.

    `agentic` 조건은 멀티에이전트 orchestrator(동일 KS·인덱스·GPT-5)로, 나머지 4조건은
    결정론 controller로 실행한다 — 동일 레코드 포맷으로 head-to-head 채점한다.
    """
    results: dict[str, list[dict]] = {m: [] for m in modes}
    for claim in claims:
        index = index_provider(claim)
        for mode in modes:
            if mode in ("agentic", "agentic_no_tier"):
                verdict, trace = orchestrate(
                    claim, index, embedder, llm, tier_config=tier_config,
                    method=method, k=k, candidate_n=candidate_n,
                    use_tier=(mode == "agentic"))   # ablation: tier 도구·가중 on/off
                results[mode].append({
                    "verdict": verdict, "retrieved_urls": trace.retrieved_urls,
                    "cost": {"steps": trace.steps_used, "tool_calls": trace.tool_calls_used},
                })
            else:
                verdict = predict_claim(claim, index, embedder, llm, mode=mode, method=method,
                                        tier_config=tier_config, k=k, candidate_n=candidate_n,
                                        config=config)
                urls = ranked_topk_urls(claim, index, embedder, mode=mode, method=method,
                                        tier_config=tier_config, k=k, candidate_n=candidate_n)
                results[mode].append({"verdict": verdict, "retrieved_urls": urls, "cost": {}})
    return results


def labels_by_claim(records: list[dict], mode: str | None = None) -> dict[int, str]:
    """레코드를 {claim_id: pred_label}로 변환한다(N=3 일치율 입력)."""
    return {r["claim_id"]: r["pred_label"]
            for r in records if mode is None or r["mode"] == mode}


def predictions_to_records(
    claims: list[Claim],
    results: dict[str, list[dict]],
    *,
    k: int = 10,
    gold_urls_fn=gold_source_urls,
    ks_urls_fn=load_claim_urls,
) -> list[dict]:
    """예측을 재현·재채점용 레코드로 직렬화한다(예측 + 회수url + gold url + 검색분류)."""
    by_id = {c.claim_id: c for c in claims}
    # claim별 gold url·KS url은 mode마다 동일 → 1회만 조회해 캐시.
    gold_cache: dict[int, list[str]] = {}
    ks_cache: dict[int, list[str]] = {}

    records = []
    for mode, preds in results.items():
        for pred in preds:
            v: Verdict = pred["verdict"]
            cid = v.claim_id
            gold = by_id.get(cid)
            gold_urls = gold_cache.setdefault(cid, gold_urls_fn(cid))
            ks_urls = ks_cache.setdefault(cid, ks_urls_fn(cid))
            retrieved = pred["retrieved_urls"]
            records.append({
                "claim_id": cid,
                "mode": mode,
                "label5": v.label5.value,
                "pred_label": v.averitec_label.value,
                "gold_label": gold.label.value if gold and gold.label else None,
                "confidence": v.confidence,
                "cited": v.cited,
                "retrieved_urls": retrieved,
                "gold_urls": gold_urls,
                "retrieval_category": classify_retrieval(
                    gold_urls, ks_urls, retrieved,
                    gold_label=gold.label if gold else None, k=k,
                ),
                "cost": pred.get("cost", {}),
            })
    return records

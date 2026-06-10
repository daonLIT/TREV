"""S3: per-claim dense 검색 + 시점 필터 → Evidence.

D4(`ClaimIndex`)가 만든 per-claim 인덱스에 claim type별 질의 템플릿(2~3개)으로 dense
검색을 돌려 top-N 후보를 받고, 시점 필터(`published_at <= T_evidence(=claim_date)`)로
누수를 차단한 뒤 메타(url·source_domain·published_at) 보존된 `Evidence`를 반환한다.
tier/role/weight는 기본값(가중·역할은 S4/#6 소관).

분기: `gpt_only`는 검색을 생략(claim만으로 판정 — controller/#7), `naive`/`proposed`는
동일하게 dense top-k를 반환(가중 재정렬은 S4). BM25 ablation은 S6/#8.

G0 주의: 검색 코퍼스의 published_at 커버리지는 사실상 0%(아카이브는 gold에 집중)라
시점 필터는 대부분 통과(no-op)로 동작한다 — 누수 안전판으로 유지한다.
"""

from __future__ import annotations

from trev.indexing import ClaimIndex, Embedder
from trev.schemas import Claim, ClaimType, Evidence

# claim type별 질의 템플릿(2~3개). `{c}` = claim 텍스트.
QUERY_TEMPLATES: dict[ClaimType, list[str]] = {
    ClaimType.NUMERICAL: ["{c}", "statistics and figures about: {c}", "numerical evidence for: {c}"],
    ClaimType.EVENT_PROPERTY: ["{c}", "did this happen: {c}", "facts and details about: {c}"],
    ClaimType.QUOTE: ["{c}", "who said this: {c}", "exact statement: {c}"],
    ClaimType.POSITION: ["{c}", "stance and position on: {c}"],
    ClaimType.CAUSAL: ["{c}", "cause and effect of: {c}"],
}


def build_queries(claim: Claim) -> list[str]:
    """claim type에 맞는 질의 문자열들을 만든다."""
    templates = QUERY_TEMPLATES.get(claim.type, ["{c}"])
    return [t.format(c=claim.text) for t in templates]


def passes_time_filter(published_at: str | None, t_evidence: str | None) -> bool:
    """시점 누수 차단: published_at(ISO date)이 T_evidence 이하인지. None은 통과(US11)."""
    if published_at is None or t_evidence is None:
        return True
    return published_at <= t_evidence


def _to_evidence(claim_id: int, i: int, passage, score: float) -> Evidence:
    return Evidence(
        doc_id=f"{claim_id}-{i}",
        snippet=passage.text,
        source=passage.source_domain,
        url=passage.url,
        published_at=passage.published_at,
        source_domain=passage.source_domain,
        sim=score,
        score=score,  # S3: 가중 없음(score=sim). 가중 재정렬은 S4/#6.
    )


def retrieve(
    claim: Claim,
    index: ClaimIndex,
    embedder: Embedder,
    *,
    k: int = 10,
    candidate_n: int = 50,
    mode: str = "proposed",
) -> list[Evidence]:
    """claim에 대해 dense 검색 → 시점 필터 → top-k Evidence를 반환한다.

    `mode="gpt_only"`면 검색을 생략하고 빈 리스트를 반환한다.
    """
    if mode == "gpt_only":
        return []

    # 여러 질의 템플릿 결과를 (url, text) 기준으로 병합(최고 점수 유지).
    best: dict[tuple[str, str], tuple] = {}
    for query in build_queries(claim):
        for hit in index.search_dense(query, embedder, k=candidate_n):
            key = (hit.passage.url, hit.passage.text)
            if key not in best or hit.score > best[key][1]:
                best[key] = (hit.passage, hit.score)

    survivors = [
        (passage, score)
        for passage, score in best.values()
        if passes_time_filter(passage.published_at, claim.claim_date)
    ]
    survivors.sort(key=lambda ps: ps[1], reverse=True)
    return [
        _to_evidence(claim.claim_id, i, passage, score)
        for i, (passage, score) in enumerate(survivors[:k])
    ]

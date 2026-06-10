"""D3: tier 입력 데이터 — 도메인 빈도목록 + 화이트리스트 분류 + 자기출처매칭.

동적 tier(S4/#6)가 소비할 결정론적 입력을 준비한다:
- `subset_domain_frequency`: 서브셋 KS 고유 도메인 빈도순 목록(화이트리스트 작성 입력).
- `classify_domain`: config 화이트리스트(overrides → heuristics → T4)로 base tier 부여.
- `is_self_source`: 문서 도메인이 claim 자기출처(original_claim_url/reporting_source)와
  일치하면 True → S4에서 role=target, T4 강등.

애매 도메인의 LLM 1회 분류·캐시와 type×role 결합(assign_tier)은 S4/#6 소관이다.
"""

from __future__ import annotations

from collections import Counter

from trev.knowledge_store import DEFAULT_KS_DIR, extract_domain, load_claim_urls
from trev.schemas import Claim

DEFAULT_TIER = 4


def subset_domain_frequency(
    claims: list[Claim], ks_dir=DEFAULT_KS_DIR
) -> list[tuple[str, int]]:
    """서브셋 KS 고유 도메인을 빈도순(내림차순)으로 반환한다(화이트리스트 작성 입력)."""
    counter: Counter[str] = Counter()
    for claim in claims:
        for url in load_claim_urls(claim.claim_id, ks_dir):
            domain = extract_domain(url)
            if domain:
                counter[domain] += 1
    return counter.most_common()


def classify_domain(domain: str | None, tier_config: dict) -> int:
    """도메인의 base tier(1~4)를 결정한다: overrides → heuristics → T4 기본.

    heuristics 규칙: `.`로 시작하면 suffix 매칭(`.gov`), 아니면 substring 매칭(`politifact`).
    """
    if not domain:
        return DEFAULT_TIER
    overrides = tier_config.get("overrides") or {}
    if domain in overrides:
        return int(overrides[domain])
    heuristics = tier_config.get("heuristics") or {}
    for tier in (1, 2, 3):
        for rule in heuristics.get(tier, []):
            if rule.startswith("."):
                if domain.endswith(rule):
                    return tier
            elif rule in domain:
                return tier
    return DEFAULT_TIER


def is_self_source(doc_domain: str | None, claim: Claim) -> bool:
    """문서 도메인이 claim 자기출처 도메인과 일치하는지(서브도메인 허용)."""
    if not doc_domain:
        return False
    for s in claim.source_domains:
        if doc_domain == s or doc_domain.endswith("." + s) or s.endswith("." + doc_domain):
            return True
    return False

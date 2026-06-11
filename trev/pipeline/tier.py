"""tier 입력(D3/#19) + 동적 tier·가중 랭킹(S4/#6).

D3 — 결정론적 입력:
- `subset_domain_frequency`: 서브셋 KS 고유 도메인 빈도순 목록(화이트리스트 작성 입력).
- `classify_domain`: config 화이트리스트(overrides → heuristics → T4)로 base tier 부여.
- `is_self_source`: 문서 도메인이 claim 자기출처와 일치하면 True → role=target 판정.

S4 — 핵심 기여:
- `assign_tier(type×role×domain)`: role=target→T4 강등, 그 외 화이트리스트(+잔여 LLM 1회).
- `LLMDomainClassifier`: 잔여 애매 도메인 LLM 1회 분류·캐시(선택적 주입).
- `rank_evidence`: `score = sim*weight`(proposed) / `sim`(unweighted)로 재정렬.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from trev.data.knowledge_store import DEFAULT_KS_DIR, extract_domain, load_claim_urls
from trev.schemas import Claim, ClaimType, Evidence, Role

DEFAULT_TIER = 4

# tier 가중치(연구 기본값). config `tier.weights`가 있으면 그쪽을 우선한다.
DEFAULT_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}


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


# ===========================================================================
# S4/#6: 동적 tier + 가중 랭킹 (핵심 기여)
# ===========================================================================


class _DomainTier(BaseModel):
    tier: int = Field(ge=1, le=4)


class LLMDomainClassifier:
    """heuristic/override로 못 잡은 잔여 도메인을 LLM 1회로 분류하고 캐시한다.

    선택적 컴포넌트 — assign_tier에 주입될 때만 동작한다(기본 경로는 결정론).
    """

    _SYSTEM = (
        "Classify a news/source domain into an evidence tier for fact-checking:\n"
        "1 = government/official statistics/courts; 2 = dedicated fact-checkers;\n"
        "3 = major established news outlets; 4 = social media, blogs, or unknown.\n"
        'Return ONLY JSON: {"tier": 1-4}.'
    )

    def __init__(self, llm, cache: dict[str, int] | None = None):
        self.llm = llm
        self.cache = cache if cache is not None else {}

    def tier(self, domain: str) -> int:
        if domain in self.cache:
            return self.cache[domain]
        out = self.llm.complete_json(
            [{"role": "system", "content": self._SYSTEM},
             {"role": "user", "content": domain}],
            schema=_DomainTier,
        )
        self.cache[domain] = out.tier
        return out.tier


def assign_tier(
    claim_type: ClaimType,
    role: Role,
    source_domain: str | None,
    tier_config: dict,
    *,
    classifier: LLMDomainClassifier | None = None,
) -> tuple[int, float]:
    """(type × role × domain)으로 (tier, weight)를 부여한다 — 동적 tier의 핵심.

    - role=target(자기출처매칭): T4 강등(동적 tier의 실제 발화).
    - 그 외: classify_domain(overrides→heuristics). 잔여 T4이고 classifier가 있으면 LLM 1회.
    - weight: config `tier.weights` 우선, 없으면 DEFAULT_WEIGHTS.

    `claim_type`은 시그니처·케이스테이블 차원이며, 현재 결정론 규칙은 role+domain으로
    tier를 정한다(타입별 정책 예: quote 자기출처 예외는 미적용 — 결정 사항으로 보류).
    """
    if role is Role.TARGET:
        tier = 4
    else:
        tier = classify_domain(source_domain, tier_config)
        if tier == DEFAULT_TIER and classifier is not None and source_domain:
            tier = classifier.tier(source_domain)
    weights = tier_config.get("weights") or DEFAULT_WEIGHTS
    return tier, float(weights.get(tier, DEFAULT_WEIGHTS[4]))


def rank_evidence(
    claim: Claim,
    evidence: list[Evidence],
    tier_config: dict,
    *,
    weighted: bool = True,
    dynamic_role: bool = True,
    classifier: LLMDomainClassifier | None = None,
    top_k: int | None = None,
) -> list[Evidence]:
    """근거에 tier/role/weight를 부여하고 점수로 재정렬한다.

    - `weighted`: proposed면 `score = sim*weight`, 아니면 `score = sim`(unweighted/naive).
    - `dynamic_role`: True면 자기출처→target(T4 강등). False(naive)면 역할 미적용(도메인 tier만).
    - `top_k`: 지정 시 가중 정렬 후 상위 k개로 절단(가중을 절단보다 먼저 적용 — 고tier 승격 보존).
    입력 Evidence는 보존하고 복사본을 반환한다.
    """
    ranked: list[Evidence] = []
    for e in evidence:
        role = (
            Role.TARGET
            if dynamic_role and is_self_source(e.source_domain, claim)
            else Role.GENERAL
        )
        tier, weight = assign_tier(
            claim.type, role, e.source_domain, tier_config, classifier=classifier
        )
        sim = e.sim if e.sim is not None else 0.0
        score = sim * weight if weighted else sim
        ranked.append(
            e.model_copy(update={"tier": tier, "role": role, "weight": weight, "score": score})
        )
    ranked.sort(key=lambda ev: ev.score, reverse=True)
    return ranked[:top_k] if top_k is not None else ranked

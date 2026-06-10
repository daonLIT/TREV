"""S8: 토픽 태깅 + 토픽별 Macro-F1 분해.

`topic_tagger`로 claim 텍스트·speaker·publisher를 입력해 토픽을 태깅한다(키워드 규칙 1차
→ 애매하면 LLM 분류·캐시). `topic_breakdown`으로 토픽별 Macro-F1을 분해해 baseline 대비
proposed의 향상폭을 토픽별로 보고한다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from trev.eval.metrics import macro_f1
from trev.schemas import Claim


class Topic(str, Enum):
    POLITICS = "politics"
    HEALTH = "health"
    ECONOMY = "economy"
    JUSTICE_CRIME = "justice_crime"
    SCIENCE_ENVIRONMENT = "science_environment"
    OTHER = "other"


# 토픽별 키워드(1차 규칙). 소문자 부분일치.
TOPIC_KEYWORDS: dict[Topic, list[str]] = {
    Topic.POLITICS: ["election", "president", "government", "senate", "congress", "vote",
                     "parliament", "minister", "democrat", "republican", "campaign",
                     "biden", "trump", "policy", "governor", "candidate"],
    Topic.HEALTH: ["covid", "vaccine", "health", "hospital", "disease", "virus",
                   "medical", "drug", "cdc", "patient", "cancer", "mask", "pandemic",
                   "infection", "mortality"],
    Topic.ECONOMY: ["economy", "gdp", "unemployment", "inflation", "tax", "trade",
                    "stock", "jobs", "wage", "budget", "deficit", "dollar", "economic",
                    "income", "poverty"],
    Topic.JUSTICE_CRIME: ["court", "police", "crime", "arrest", "murder", "prison",
                          "judge", "lawsuit", "fraud", "shooting", "violence",
                          "convicted", "illegal", "killed"],
    Topic.SCIENCE_ENVIRONMENT: ["climate", "environment", "carbon", "emission",
                                "temperature", "species", "energy", "pollution",
                                "nasa", "research", "scientist", "wildfire", "renewable"],
}


def _keyword_topic(text: str) -> Topic | None:
    """키워드 점수 최댓값 토픽을 반환한다. 매칭 0 또는 동점이면 None(애매)."""
    low = text.lower()
    scores = {t: sum(1 for kw in kws if kw in low) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return None
    winners = [t for t, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else None


class _TopicOut(BaseModel):
    topic: Topic


class LLMTopicClassifier:
    """애매 claim을 LLM 1회로 분류하고 캐시한다(선택적 주입)."""

    _SYSTEM = (
        "Classify the claim into one topic: politics, health, economy, "
        "justice_crime, science_environment, or other.\n"
        'Return ONLY JSON: {"topic": "..."}.'
    )

    def __init__(self, llm, cache: dict[str, Topic] | None = None):
        self.llm = llm
        self.cache = cache if cache is not None else {}

    def classify(self, text: str) -> Topic:
        if text in self.cache:
            return self.cache[text]
        out = self.llm.complete_json(
            [{"role": "system", "content": self._SYSTEM},
             {"role": "user", "content": text}],
            schema=_TopicOut,
        )
        self.cache[text] = out.topic
        return out.topic


def topic_tagger(claim: Claim, classifier: LLMTopicClassifier | None = None) -> Topic:
    """claim 텍스트+speaker+publisher로 토픽을 태깅한다(키워드 → 애매하면 LLM → other)."""
    text = " ".join(filter(None, [claim.text, claim.speaker, claim.publisher]))
    topic = _keyword_topic(text)
    if topic is not None:
        return topic
    if classifier is not None:
        return classifier.classify(claim.text)
    return Topic.OTHER


def tag_claims(
    claims: list[Claim], classifier: LLMTopicClassifier | None = None
) -> list[Claim]:
    """각 claim의 `topic`을 채운 복사본 리스트를 반환한다."""
    return [c.model_copy(update={"topic": topic_tagger(c, classifier).value}) for c in claims]


def topic_breakdown(
    records: list[dict],
    topic_by_claim: dict[int, str],
    *,
    baseline: str = "naive_rag",
    target: str = "proposed",
) -> dict[str, dict]:
    """토픽 × 조건별 Macro-F1과 target−baseline 향상폭을 산출한다.

    어떤 토픽에서 tier 가중(proposed)이 특히 효과적인지 본다.
    """
    # (topic, mode) -> (pred, gold) 쌍.
    groups: dict[tuple[str, str], list[tuple]] = {}
    for r in records:
        topic = topic_by_claim.get(r["claim_id"], Topic.OTHER.value)
        groups.setdefault((topic, r["mode"]), []).append((r["pred_label"], r["gold_label"]))

    topics = sorted({t for t, _ in groups})
    table: dict[str, dict] = {}
    for topic in topics:
        per_mode = {
            mode: macro_f1([p for p, _ in pairs], [g for _, g in pairs])
            for (t, mode), pairs in groups.items() if t == topic
        }
        n = sum(len(pairs) for (t, _), pairs in groups.items() if t == topic) // max(
            1, len({m for (t, m) in groups if t == topic})
        )
        entry = {"n": n, "macro_f1": per_mode}
        if target in per_mode and baseline in per_mode:
            entry["delta"] = per_mode[target] - per_mode[baseline]
        table[topic] = entry
    return table

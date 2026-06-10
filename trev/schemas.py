"""TREV 핵심 pydantic 스키마: Claim / Evidence / Verdict.

enum 값은 AVeriTeC dev 실데이터(G0 점검)에 맞춘다. claim_types·gold label 문자열은
데이터에서 확인한 정확한 표기를 사용한다(철자 고정 — 채점 어긋남 방지).
정규화·매핑 로직은 각 슬라이스(D1/#17, S2/#4)가 담당하고, 여기서는 형태만 정의한다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """AVeriTeC dev `claim_types` 값(5종)."""

    NUMERICAL = "Numerical Claim"
    EVENT_PROPERTY = "Event/Property Claim"
    QUOTE = "Quote Verification"
    POSITION = "Position Statement"
    CAUSAL = "Causal Claim"


class AveritecLabel(str, Enum):
    """AVeriTeC gold `label`(4라벨, 채점 기준). 문자열은 데이터에서 확인한 정확 표기."""

    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
    CONFLICTING = "Conflicting Evidence/Cherrypicking"


class Label5(str, Enum):
    """내부 5라벨. verifier는 SUPPORT/REFUTE/PARTIAL/NEI를, controller(R5)가 CONFLICT를 낸다."""

    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    NEI = "NEI"


class Stance(str, Enum):
    """근거 1건이 claim에 대해 갖는 입장(R5 CONFLICT 판정 입력)."""

    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    NEUTRAL = "NEUTRAL"


class Role(str, Enum):
    """근거의 역할. target=자기출처(claim 출처 도메인과 일치) → tier T4 강등."""

    GENERAL = "general"
    TARGET = "target"


class Claim(BaseModel):
    """검증 대상 주장. `claim_date`는 정규화된 T_claim(결측 시 None=시점필터 통과)."""

    claim_id: int
    text: str
    type: ClaimType
    claim_date: str | None = None
    topic: str | None = None
    checkworthiness: float | None = None
    label: AveritecLabel | None = None  # gold(평가 기준). 예측 경로에서는 사용 금지.


class Passage(BaseModel):
    """KS에서 추출한 검색 단위(url2text 1개). D2가 생산, D4 인덱싱·D5 Recall이 소비.

    `ks_type`은 검색전략 provenance(14종)로 분석용 보존일 뿐 부스팅에 쓰지 않는다.
    `published_at`은 아카이브 스냅샷에서 유도(없으면 None=시점필터 통과).
    """

    claim_id: int
    url: str
    text: str
    source_domain: str | None = None
    published_at: str | None = None
    ks_type: str | None = None


class Evidence(BaseModel):
    """검색된 근거 1건. published_at은 아카이브 스냅샷에서 유도(없으면 None)."""

    doc_id: str
    snippet: str
    source: str | None = None
    url: str | None = None
    published_at: str | None = None
    source_domain: str | None = None
    tier: int = 4
    role: Role = Role.GENERAL
    weight: float = 0.1
    stance: Stance | None = None
    sim: float | None = None
    score: float | None = None


class Verdict(BaseModel):
    """claim에 대한 최종 판정. averitec_label은 label5→4 매핑 결과(채점용)."""

    claim_id: int
    label5: Label5
    averitec_label: AveritecLabel
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    cited: list[str] = Field(default_factory=list)

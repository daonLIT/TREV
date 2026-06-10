"""D1: AVeriTeC dev 적재 + claim type 필터 + claim_date 정규화 + Conflicting 카운트.

데이터 계약 진입점. `dev.json`(500 claim)을 `list[Claim]`로 적재하고,
Numerical + Event/Property 서브셋(+Quote 토글)만 통과시킨다. `claim_id`는 원본 위치
인덱스로 보존(= knowledge_store/dev/{index}.json 매핑 키). 적재 직후 서브셋 내
Conflicting gold 개수로 5라벨 평가 분기(≥20 정량 / 미만 정성)를 결정한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trev.guards import assert_data_file_allowed
from trev.data.knowledge_store import (
    claim_source_domains,
    derive_published_at,
    extract_domain,
)
from trev.schemas import AveritecLabel, Claim, ClaimType, Evidence

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEV_PATH = _REPO_ROOT / "data_store" / "averitec" / "dev.json"

# 서브셋 핵심 타입(항상 포함) + Quote(토글). 우선순위 = 단일 Claim.type 선택 순서.
_BASE_TYPES = (ClaimType.NUMERICAL, ClaimType.EVENT_PROPERTY)
_PRIORITY = (ClaimType.NUMERICAL, ClaimType.EVENT_PROPERTY, ClaimType.QUOTE)

# 5라벨(정량 세분 F1) vs 정성(사례연구) 분기 임계값.
CONFLICTING_QUANT_THRESHOLD = 20

# claim_date 파서 시도 순서(PRD): dev는 DD-MM-YYYY이나 결측·타포맷에 견고하게.
_DATE_FORMATS = ("%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y")


def parse_claim_date(raw: str | None) -> str | None:
    """claim_date를 ISO `YYYY-MM-DD`로 정규화한다(T_claim). 결측·파싱 실패 → None.

    None은 시점필터 통과 처리(US11)를 위한 값이다.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_claim_type(
    claim_types: list[str], *, include_quote: bool = False
) -> ClaimType | None:
    """claim_types(복수 가능)를 서브셋 단일 ClaimType으로 정규화한다.

    {Numerical, Event/Property}(+include_quote 시 Quote) 중 하나라도 있으면 통과시키고,
    우선순위(Numerical > Event/Property > Quote)로 1개를 고른다. 해당 없으면 None(서브셋 제외).
    """
    target = set(_BASE_TYPES)
    if include_quote:
        target.add(ClaimType.QUOTE)
    present = {ClaimType(t) for t in claim_types if t in ClaimType._value2member_map_}
    for ct in _PRIORITY:
        if ct in present and ct in target:
            return ct
    return None


def _normalize_label(raw: str | None) -> AveritecLabel | None:
    """gold label 문자열을 AveritecLabel로 보존(미지 값 → None)."""
    if raw in AveritecLabel._value2member_map_:
        return AveritecLabel(raw)
    return None


def load_averitec(
    path: str | Path = DEFAULT_DEV_PATH, *, include_quote: bool = False
) -> list[Claim]:
    """dev.json을 적재해 서브셋 `list[Claim]`을 반환한다.

    claim_id는 **원본 위치 인덱스**(KS 파일 매핑 키)이며, 필터로 제외된 claim의
    인덱스는 건너뛴다(번호 재배열하지 않음).
    """
    path = Path(path)
    assert_data_file_allowed(path.name)  # 데이터 위생 가드(#16): dev만 허용
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    claims: list[Claim] = []
    for idx, obj in enumerate(raw):
        ctype = normalize_claim_type(
            obj.get("claim_types") or [], include_quote=include_quote
        )
        if ctype is None:
            continue
        claims.append(
            Claim(
                claim_id=idx,
                text=obj["claim"],
                type=ctype,
                claim_date=parse_claim_date(obj.get("claim_date")),
                label=_normalize_label(obj.get("label")),
                source_domains=claim_source_domains(
                    obj.get("original_claim_url"), obj.get("reporting_source")
                ),
                speaker=obj.get("speaker") or None,
                publisher=obj.get("reporting_source") or None,
            )
        )
    return claims


def load_gold_evidence(
    claim_id: int, path: str | Path = DEFAULT_DEV_PATH
) -> list[Evidence]:
    """claim의 gold 근거(questions[].answers[])를 Evidence stub으로 반환한다(S2 워킹 스켈레톤).

    검색 미구현 단계에서 verifier 입력으로 주입한다(S3/S4에서 실검색으로 교체).
    gold는 평가/스켈레톤 전용 — 검색 코퍼스에 섞지 않는다.
    """
    path = Path(path)
    assert_data_file_allowed(path.name)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    obj = raw[claim_id]

    evidence: list[Evidence] = []
    for q in obj.get("questions") or []:
        question = q.get("question") or ""
        for a in q.get("answers") or []:
            answer = (a.get("answer") or "").strip()
            if not answer:
                continue
            url = a.get("source_url") or None
            evidence.append(
                Evidence(
                    doc_id=f"g{len(evidence)}",
                    snippet=f"{question} — {answer}" if question else answer,
                    url=url,
                    source_domain=extract_domain(url) if url else None,
                    source=extract_domain(url) if url else None,
                    published_at=derive_published_at(url) if url else None,
                )
            )
    return evidence


def gold_source_urls(claim_id: int, path: str | Path = DEFAULT_DEV_PATH) -> list[str]:
    """claim의 gold 근거 URL(questions[].answers[].source_url, 비어있지 않은 것)."""
    path = Path(path)
    assert_data_file_allowed(path.name)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)[claim_id]
    urls: list[str] = []
    for q in obj.get("questions") or []:
        for a in q.get("answers") or []:
            url = a.get("source_url")
            if url:
                urls.append(url)
    return urls


def conflicting_count(claims: list[Claim]) -> int:
    """서브셋 내 gold label == Conflicting Evidence/Cherrypicking 개수."""
    return sum(1 for c in claims if c.label == AveritecLabel.CONFLICTING)


def subset_gate(claims: list[Claim]) -> dict:
    """서브셋 크기·Conflicting 개수·5라벨 평가 분기를 산출한다."""
    n = len(claims)
    conf = conflicting_count(claims)
    mode = "quantitative" if conf >= CONFLICTING_QUANT_THRESHOLD else "qualitative"
    return {"subset_size": n, "conflicting": conf, "eval_mode": mode}

"""S2: verifier — 근거로 claim을 판정해 Verdict를 만든다(GPT-5 1회).

verifier는 제공된 근거에만 정초해 4라벨(SUPPORT/REFUTE/PARTIAL/NEI) + confidence +
justification + cited + per-evidence stance를 스키마 검증된 JSON으로 출력한다.
**CONFLICT는 여기서 내지 않는다**(상충 판정은 controller R5/#7). 이후 5→4 매핑으로
`Verdict.averitec_label`을 채운다. `cited`는 비어 있을 수 없다(무인용 판정 금지).

S2는 gold 근거 stub을 입력으로 받지만, verify 자체는 근거 출처(gold/실검색)에 무관하다
— S3/S4에서 실검색 Evidence로 그대로 교체된다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from trev.schemas import AveritecLabel, Claim, Evidence, Label5, Stance, Verdict


class VerifierLabel(str, Enum):
    """verifier가 낼 수 있는 라벨(CONFLICT 제외 — R5 소관)."""

    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    PARTIAL = "PARTIAL"
    NEI = "NEI"


class EvidenceStance(BaseModel):
    doc_id: str
    stance: Stance


class VerifierOutput(BaseModel):
    """verifier의 스키마 검증된 JSON 출력."""

    label: VerifierLabel
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    cited: list[str] = Field(min_length=1)  # 무인용 판정 금지
    stances: list[EvidenceStance] = Field(default_factory=list)


# verifier 라벨 → 내부 Label5(이름 1:1, CONFLICT 없음).
_TO_LABEL5 = {
    VerifierLabel.SUPPORT: Label5.SUPPORT,
    VerifierLabel.REFUTE: Label5.REFUTE,
    VerifierLabel.PARTIAL: Label5.PARTIAL,
    VerifierLabel.NEI: Label5.NEI,
}

# 5→4 매핑: PARTIAL·CONFLICT → Conflicting/Cherry-picking, 나머지 1:1.
_TO_AVERITEC = {
    Label5.SUPPORT: AveritecLabel.SUPPORTED,
    Label5.REFUTE: AveritecLabel.REFUTED,
    Label5.NEI: AveritecLabel.NOT_ENOUGH_EVIDENCE,
    Label5.PARTIAL: AveritecLabel.CONFLICTING,
    Label5.CONFLICT: AveritecLabel.CONFLICTING,
}


def to_averitec_label(label5: Label5) -> AveritecLabel:
    """내부 5라벨을 AVeriTeC 4라벨로 매핑한다(채점 기준)."""
    return _TO_AVERITEC[label5]


SYSTEM_PROMPT = """You are a careful fact-checking verifier. Judge the CLAIM using ONLY \
the provided EVIDENCE. Do not use outside knowledge.

Labels (choose exactly one):
- SUPPORT: the evidence supports the claim's core assertion.
- REFUTE: the evidence shows the claim's core assertion is false (the core is wrong).
- PARTIAL: the claim is exaggerated, cherry-picked, or only partly supported.
- NEI: there is no trustworthy evidence to decide.

Rules:
- Ground every judgment strictly in the given evidence.
- Do NOT output CONFLICT. The strongest "mixed" verdict you may give is PARTIAL.
- "cited" must contain at least one evidence doc_id you relied on.
- "stances" must give a stance (SUPPORT/REFUTE/NEUTRAL) for each evidence doc_id you considered.

Return ONLY a JSON object:
{"label": "...", "confidence": 0.0-1.0, "justification": "...", \
"cited": ["doc_id", ...], "stances": [{"doc_id": "...", "stance": "..."}, ...]}"""


def _format_evidence(claim: Claim, evidence: list[Evidence]) -> str:
    lines = [f"CLAIM: {claim.text}", "", "EVIDENCE:"]
    for e in evidence:
        domain = e.source_domain or "unknown"
        lines.append(f"[{e.doc_id}] ({domain}) {e.snippet}")
    return "\n".join(lines)


def run_verifier(claim: Claim, evidence: list[Evidence], llm) -> VerifierOutput:
    """GPT-5 1회 호출로 스키마 검증된 VerifierOutput을 받는다(stance 포함 — R5가 소비)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_evidence(claim, evidence)},
    ]
    return llm.complete_json(messages, schema=VerifierOutput)


def verify(claim: Claim, evidence: list[Evidence], llm) -> Verdict:
    """claim을 판정해 Verdict(5라벨 + 4라벨 매핑)를 만든다. CONFLICT는 내지 않는다."""
    out = run_verifier(claim, evidence, llm)
    label5 = _TO_LABEL5[out.label]
    return Verdict(
        claim_id=claim.claim_id,
        label5=label5,
        averitec_label=to_averitec_label(label5),
        confidence=out.confidence,
        justification=out.justification,
        cited=out.cited,
    )

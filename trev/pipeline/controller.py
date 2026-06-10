"""S5: controller R1~R5 라우팅 + stance기반 CONFLICT + baseline 분기.

retrieve(S3) → rank_evidence(S4) → verify(S2)를 묶어 라우팅한다:
- R1: checkworthiness < τ → 폐기(NEI).
- R2: evidence 0건 또는 max_tier=T4뿐 → 즉시 NEI.
- R3: T1~T3 근거 ≥1 → verify 호출.
- R4: confidence < 0.5 → 질의 확장 후 1회 재검색·재검증, 그래도 낮으면 NEI.
- R5: T1~T2 근거에 support·refute stance가 공존 → CONFLICT(verifier가 아닌 controller가 결정).
- cited 비어있지 않음 강제(무인용 → NEI).

baseline 분기(동일 full controller 통과, ranking만 다름):
- proposed: dynamic tier + weighted(score=sim*weight).
- unweighted_rag: dynamic tier + unweighted(score=sim).
- naive_rag: 도메인 tier만(자기출처 강등 없음) + unweighted.
- gpt_only: 검색 생략 + R1/R2/R5·cited 우회(claim만으로 라벨 직출) — NEI로 붕괴하지 않음.

seam: retrieve_fn / verify_fn / gpt_only_fn을 주입받아 LLM·데이터 없이 라우팅을 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from trev.schemas import Claim, Evidence, Label5, Stance, Verdict
from trev.pipeline.tier import rank_evidence
from trev.pipeline.verifier import VerifierOutput, _TO_LABEL5, to_averitec_label

RAG_MODES = ("naive_rag", "unweighted_rag", "proposed")


@dataclass
class ControllerConfig:
    checkworthiness_tau: float = 0.5
    low_confidence: float = 0.5


def _rank_params(mode: str) -> dict:
    """모드별 ranking 파라미터(weighted, dynamic_role)."""
    return {
        "weighted": mode == "proposed",
        "dynamic_role": mode != "naive_rag",
    }


def _nei(claim: Claim, why: str, cited: list[str] | None = None) -> Verdict:
    return Verdict(
        claim_id=claim.claim_id, label5=Label5.NEI,
        averitec_label=to_averitec_label(Label5.NEI),
        confidence=0.0, justification=why, cited=cited or [],
    )


def _high_tier_stance_conflict(evidence: list[Evidence], out: VerifierOutput) -> bool:
    """T1~T2 근거에 support·refute stance가 공존하는지(R5 CONFLICT 판정)."""
    tier_by_id = {e.doc_id: e.tier for e in evidence}
    stances = {s.stance for s in out.stances if tier_by_id.get(s.doc_id) in (1, 2)}
    return Stance.SUPPORT in stances and Stance.REFUTE in stances


def _verdict(claim: Claim, out: VerifierOutput, evidence: list[Evidence]) -> Verdict:
    label5 = _TO_LABEL5[out.label]
    if _high_tier_stance_conflict(evidence, out):  # R5: controller가 stance로 결정
        label5 = Label5.CONFLICT
    return Verdict(
        claim_id=claim.claim_id, label5=label5,
        averitec_label=to_averitec_label(label5),
        confidence=out.confidence, justification=out.justification, cited=out.cited,
    )


def run(
    claim: Claim,
    *,
    mode: str,
    retrieve_fn,
    verify_fn,
    gpt_only_fn,
    tier_config: dict,
    config: ControllerConfig = ControllerConfig(),
) -> Verdict:
    """단일 claim을 모드에 따라 라우팅해 Verdict를 만든다.

    retrieve_fn(claim, expand=False)->list[Evidence], verify_fn(claim, evidence)->VerifierOutput,
    gpt_only_fn(claim)->Verdict.
    """
    if mode == "gpt_only":  # R1/R2/R5·cited 우회
        return gpt_only_fn(claim)

    # R1: not check-worthy → 폐기
    if claim.checkworthiness is not None and claim.checkworthiness < config.checkworthiness_tau:
        return _nei(claim, "R1: checkworthiness below threshold")

    rp = _rank_params(mode)
    evidence = rank_evidence(claim, retrieve_fn(claim), tier_config, **rp)

    # R2: 근거 없음 또는 최상 tier가 T4(=T1~T3 없음) → 즉시 NEI
    if not evidence or all(e.tier == 4 for e in evidence):
        return _nei(claim, "R2: no trustworthy (T1-T3) evidence")

    # R3: T1~T3 근거 ≥1 → verify
    out = verify_fn(claim, evidence)

    # R4: 저신뢰 → 질의 확장 1회 재검색·재검증, 그래도 낮으면 NEI
    if out.confidence < config.low_confidence:
        evidence = rank_evidence(claim, retrieve_fn(claim, expand=True), tier_config, **rp)
        if not evidence or all(e.tier == 4 for e in evidence):
            return _nei(claim, "R4: no trustworthy evidence after expansion")
        out = verify_fn(claim, evidence)
        if out.confidence < config.low_confidence:
            return _nei(claim, "R4: low confidence after re-retrieval", cited=out.cited)

    # cited 강제(무인용 판정 금지)
    if not out.cited:
        return _nei(claim, "no citation")

    # R5: stance 공존 시 CONFLICT
    return _verdict(claim, out, evidence)

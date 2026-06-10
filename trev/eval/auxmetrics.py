"""S11 (stretch): 보조 지표 — RAGAS faithfulness · G-Eval + 공식 AVeriTeC score 근사.

메인 지표는 S7(metrics.py). 여기는 보조다:
- `ragas_faithfulness`: justification이 근거에 정초됐는지(환각 여부) LLM-judge 0~1.
- `geval_faithfulness`: G-Eval식(CoT → 1~5 → 0~1) 충실성 평가.
- `generate_cited_qa` + `approx_averitec_score`: cited 근거로 QA를 생성해 gold QA와 근사 매칭
  (공식 Ev2R 완전구현은 Out of Scope — 토큰 중첩 근사).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _Faithfulness(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class _GEval(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class _QAItem(BaseModel):
    question: str
    answer: str


class _QAList(BaseModel):
    qa: list[_QAItem]


def _evidence_block(evidence_texts: list[str]) -> str:
    return "\n".join(f"- {t}" for t in evidence_texts)


_RAGAS_PROMPT = (
    "Rate how faithful the JUSTIFICATION is to the EVIDENCE (is every statement "
    "supported by the evidence, with no hallucination?). 1.0 = fully grounded, "
    "0.0 = unsupported. Return ONLY JSON: {\"score\": 0.0-1.0, \"reasoning\": \"...\"}."
)


def ragas_faithfulness(claim, justification: str, evidence_texts: list[str], llm) -> dict:
    """RAGAS faithfulness(0~1): justification이 근거에 정초됐는지 LLM-judge."""
    msg = (f"CLAIM: {claim.text}\n\nJUSTIFICATION: {justification}\n\n"
           f"EVIDENCE:\n{_evidence_block(evidence_texts)}")
    out = llm.complete_json(
        [{"role": "system", "content": _RAGAS_PROMPT}, {"role": "user", "content": msg}],
        schema=_Faithfulness,
    )
    return {"score": out.score, "reasoning": out.reasoning}


_GEVAL_PROMPT = (
    "You evaluate FAITHFULNESS using G-Eval. Think step by step: (1) list the claims in "
    "the justification, (2) check each against the evidence, (3) note any unsupported "
    "statement. Then give an integer score 1-5 (5 = fully faithful, 1 = hallucinated). "
    "Return ONLY JSON: {\"score\": 1-5, \"reasoning\": \"...\"}."
)


def geval_faithfulness(claim, justification: str, evidence_texts: list[str], llm) -> dict:
    """G-Eval식 충실성(1~5 → 0~1 정규화) + CoT 근거."""
    msg = (f"CLAIM: {claim.text}\n\nJUSTIFICATION: {justification}\n\n"
           f"EVIDENCE:\n{_evidence_block(evidence_texts)}")
    out = llm.complete_json(
        [{"role": "system", "content": _GEVAL_PROMPT}, {"role": "user", "content": msg}],
        schema=_GEval,
    )
    return {"score": (out.score - 1) / 4, "raw": out.score, "reasoning": out.reasoning}


_QA_PROMPT = (
    "Given the CLAIM and cited EVIDENCE, generate the question-answer pairs that the "
    "evidence resolves about the claim (AVeriTeC style). Return ONLY JSON: "
    "{\"qa\": [{\"question\": \"...\", \"answer\": \"...\"}, ...]}."
)


def generate_cited_qa(claim, cited_texts: list[str], llm) -> list[dict]:
    """cited 근거로 QA를 생성한다(공식 score 근사용 어댑터)."""
    msg = f"CLAIM: {claim.text}\n\nEVIDENCE:\n{_evidence_block(cited_texts)}"
    out = llm.complete_json(
        [{"role": "system", "content": _QA_PROMPT}, {"role": "user", "content": msg}],
        schema=_QAList,
    )
    return [{"question": q.question, "answer": q.answer} for q in out.qa]


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def approx_averitec_score(
    generated_qa: list[dict], gold_qa: list[dict], *, threshold: float = 0.2
) -> float:
    """공식 AVeriTeC score 근사: gold 답변별 최적 생성답변 토큰 Jaccard로 커버리지.

    완전한 Ev2R/QA 매칭이 아님(Out of Scope) — 보조 근사치다.
    """
    if not gold_qa:
        return 0.0
    gen_tokens = [_tokens(g["answer"]) for g in generated_qa] or [set()]
    covered = 0
    for gold in gold_qa:
        gt = _tokens(gold["answer"])
        best = max((_jaccard(gt, gt2) for gt2 in gen_tokens), default=0.0)
        if best >= threshold:
            covered += 1
    return covered / len(gold_qa)

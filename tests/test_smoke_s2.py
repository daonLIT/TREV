"""S2 워킹 스켈레톤 실 GPT-5 스모크: gold 근거 → verify → Verdict 관통.

`.env`에 HAI_GPT_API_KEY가 있으면 단일 claim을 실제로 관통시켜 Verdict 형태를 확인한다.
키가 없으면 skip(CI 보호). 정답 일치까지는 단정하지 않는다(모델 비결정성).

실행: python -m pytest tests/test_smoke_s2.py -v -s
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from trev.config import load_config
from trev.dataset import load_averitec, load_gold_evidence
from trev.llm import LLM
from trev.schemas import AveritecLabel, Label5
from trev.verifier import verify

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("HAI_GPT_API_KEY"),
    reason="HAI_GPT_API_KEY 없음 — .env에 키를 채우면 S2 스모크가 실행됩니다.",
)


def test_gold_to_verdict_end_to_end():
    claim = load_averitec()[0]
    evidence = load_gold_evidence(claim.claim_id)
    assert evidence, "gold 근거 stub이 비어 있음"

    verdict = verify(claim, evidence, LLM.from_config(load_config()))
    assert isinstance(verdict.label5, Label5)
    assert isinstance(verdict.averitec_label, AveritecLabel)
    assert verdict.cited, "무인용 판정(비어있음) 금지"
    assert 0.0 <= verdict.confidence <= 1.0
    print(f"\n[S2] claim{claim.claim_id} pred={verdict.label5.value}"
          f"->{verdict.averitec_label.value} gold={claim.label and claim.label.value}")

"""G1 게이트: GPT-5 실호출 1건 스모크 테스트.

실제 HAI-GPT 게이트웨이로 1회 호출해 응답을 받는다. `.env`에 키가 없으면 skip한다
(CI·키 없는 환경 보호). 게이트 통과는 키를 채운 뒤 이 테스트가 PASS함으로 확인한다.

실행: python -m pytest tests/test_smoke_gpt5.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from trev.config import load_config
from trev.llm import LLM

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("HAI_GPT_API_KEY"),
    reason="HAI_GPT_API_KEY 없음 — .env에 키를 채우면 G1 스모크가 실행됩니다.",
)


def test_gpt5_ping():
    llm = LLM.from_config(load_config())
    reply = llm.complete(
        [{"role": "user", "content": "Reply with exactly: pong"}]
    )
    assert reply and reply.strip(), "빈 응답"
    print(f"\n[G1] model={llm.model} reply={reply!r}")

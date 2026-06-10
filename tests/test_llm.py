"""LLM 래퍼 단위 테스트 — 주입된 가짜 클라이언트로 실호출 없이 seam 검증.

검증 대상(외부 행동): temperature 전달/생략, JSON 파싱, 파싱 실패 시 재시도,
pydantic 스키마 검증, 검증 실패 max_retries 초과 시 예외.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from trev.llm import LLM, LLMError


class FakeClient:
    """`chat.completions.create`를 흉내내고, 스크립트된 응답을 순서대로 반환한다."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class Out(BaseModel):
    label: str
    confidence: float


def test_complete_returns_text_and_passes_temperature():
    fake = FakeClient(["hello"])
    llm = LLM(client=fake, model="gpt-5", temperature=0)
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert fake.calls[0]["temperature"] == 0
    assert fake.calls[0]["model"] == "gpt-5"


def test_temperature_none_is_omitted():
    fake = FakeClient(["ok"])
    LLM(client=fake, temperature=None).complete([{"role": "user", "content": "hi"}])
    assert "temperature" not in fake.calls[0]


def test_complete_json_parses_plain_and_code_fence():
    fake = FakeClient(['```json\n{"a": 1}\n```'])
    llm = LLM(client=fake)
    assert llm.complete_json([{"role": "user", "content": "x"}]) == {"a": 1}


def test_complete_json_retries_on_bad_json_then_succeeds():
    fake = FakeClient(["not json", '{"a": 2}'])
    llm = LLM(client=fake, max_retries=3)
    assert llm.complete_json([{"role": "user", "content": "x"}]) == {"a": 2}
    assert len(fake.calls) == 2  # 재시도 1회


def test_complete_json_validates_schema():
    fake = FakeClient(['{"label": "SUPPORT", "confidence": 0.9}'])
    llm = LLM(client=fake)
    out = llm.complete_json([{"role": "user", "content": "x"}], schema=Out)
    assert isinstance(out, Out) and out.label == "SUPPORT"


def test_complete_json_raises_after_max_retries():
    fake = FakeClient(["bad", "still bad", "nope"])
    llm = LLM(client=fake, max_retries=3)
    with pytest.raises(LLMError):
        llm.complete_json([{"role": "user", "content": "x"}])
    assert len(fake.calls) == 3

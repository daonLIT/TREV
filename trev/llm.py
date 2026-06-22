"""GPT-5 LLM 래퍼 (HAI-GPT API Gateway).

게이트웨이는 OpenAI 호환(Chat Completions)이라 OpenAI SDK를 그대로 쓰고 base_url만 바꾼다:
    base_url = https://factchat-cloud.mindlogic.ai/v1/gateway
    인증     = Authorization: Bearer <HAI_GPT_API_KEY>  (OpenAI SDK가 처리)

`LLM`은 프롬프트·tenacity 재시도·JSON 파싱·pydantic 스키마 검증을 캡슐화한
**단일 주입 가능 인터페이스**다. 테스트·baseline은 OpenAI 호환 `client`를 주입해
실호출 없이 교체할 수 있다(`client.chat.completions.create(...)`만 만족하면 됨).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from trev.config import load_config

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"

# tool-calling은 게이트웨이가 ~40s 걸릴 수 있어(G-agent 실측) 넉넉한 timeout을 둔다.
TOOL_CALL_TIMEOUT = 180


@dataclass
class ToolCallRequest:
    """LLM이 요청한 단일 도구 호출(정규화 — SDK 객체 누출 없음)."""

    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    """tool-calling 1턴의 어시스턴트 응답: 도구 호출들 또는 최종 텍스트."""

    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


class LLMError(Exception):
    """LLM 호출/파싱/검증 실패."""


def build_client(api_key: str | None = None, base_url: str | None = None):
    """HAI-GPT 게이트웨이를 가리키는 OpenAI 호환 클라이언트를 만든다.

    키·URL은 인자 → 환경변수(`HAI_GPT_API_KEY`/`HAI_GPT_BASE_URL`) 순으로 해석한다.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("HAI_GPT_API_KEY")
    if not key:
        raise LLMError("HAI_GPT_API_KEY가 없음(.env 또는 인자로 제공).")
    url = base_url or os.environ.get("HAI_GPT_BASE_URL") or DEFAULT_BASE_URL
    client = OpenAI(api_key=key, base_url=url)

    # LangSmith 추적(선택): LANGSMITH_TRACING=true면 클라이언트를 wrap해 모든 호출을 추적.
    # langsmith 미설치/미설정이면 그대로 통과(동작 영향 없음).
    if os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1"):
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except ImportError:
            pass
    return client


class LLM:
    """주입 가능한 GPT-5 인터페이스.

    Parameters
    ----------
    client : OpenAI 호환 클라이언트(`chat.completions.create`). 미지정 시 게이트웨이로 생성.
    model : 게이트웨이 모델 id(예: "gpt-5").
    temperature : 0 고정 권장. `None`이면 요청에서 생략(일부 모델은 비기본 temperature 거부).
    max_retries : 호출·JSON 파싱·검증 재시도 횟수.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = "gpt-5",
        temperature: float | None = 0,
        max_retries: int = 3,
    ) -> None:
        self.client = client if client is not None else build_client()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    @classmethod
    def from_config(cls, config: dict | None = None, *, client: Any = None) -> "LLM":
        """config.yaml의 `llm` 섹션으로 LLM을 구성한다."""
        cfg = (config or load_config()).get("llm", {})
        return cls(
            client=client,
            model=cfg.get("model", "gpt-5"),
            temperature=cfg.get("temperature", 0),
            max_retries=cfg.get("max_retries", 3),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        """messages를 보내고 응답 텍스트를 반환한다(네트워크 오류는 tenacity 재시도)."""
        return self._call(messages)

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T] | None = None,
    ) -> T | dict:
        """JSON 응답을 받아 파싱(+스키마 검증)한다.

        파싱·검증 실패 시 교정 지시를 덧붙여 최대 `max_retries`회 재호출한다.
        `schema`가 주어지면 검증된 모델 인스턴스를, 아니면 dict를 반환한다.
        """
        convo = list(messages)
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            raw = self._call(convo)
            try:
                data = json.loads(_strip_code_fence(raw))
            except json.JSONDecodeError as e:
                last_err = e
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "유효한 JSON만 출력하세요. 코드펜스·설명 금지."},
                ]
                continue
            if schema is None:
                return data
            try:
                return schema.model_validate(data)
            except ValidationError as e:
                last_err = e
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"스키마 검증 실패. 다음 오류를 고쳐 JSON만 출력:\n{e}"},
                ]
        raise LLMError(f"{self.max_retries}회 시도 후 JSON 파싱/검증 실패: {last_err}")

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, max=60), reraise=True)
    def _call(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "timeout": 120}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, max=60), reraise=True)
    def complete_with_tools(
        self, messages: list[dict], tools: list[dict], *, tool_choice: str = "auto"
    ) -> AssistantTurn:
        """tools(JSON-Schema)와 함께 호출하고, 정규화된 AssistantTurn을 반환한다.

        병렬 tool_calls(GPT-5 실측)를 리스트로 파싱한다. 도구 호출이 없으면 최종 텍스트.
        """
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "tools": tools, "tool_choice": tool_choice, "timeout": TOOL_CALL_TIMEOUT,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        msg = self.client.chat.completions.create(**kwargs).choices[0].message
        calls = [
            ToolCallRequest(
                id=tc.id, name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        return AssistantTurn(content=msg.content, tool_calls=calls)


def _strip_code_fence(text: str) -> str:
    """```json ... ``` 코드펜스를 제거해 순수 JSON 본문만 남긴다."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -len("```")]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[len("json"):]
    return s.strip()

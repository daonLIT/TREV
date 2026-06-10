"""G-agent 게이트: 게이트웨이 native tool-calling 지원 스모크.

OpenAI 호환 게이트웨이가 `tools`/`tool_calls`를 지원하는지 실호출 1건으로 확인한다.
키가 없으면 skip. TREV-Agent(#22)의 핵심 전제(native tool-calling)를 보장한다.

발견: GPT-5는 병렬 tool_calls(한 턴 다중 호출)를 낸다 → 에이전트 루프는 tool_calls를
리스트로 실행하고 각 결과를 매칭 tool_call_id로 회신해야 한다.

실행: python -m pytest tests/test_smoke_tool_calling.py -v -s
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from trev.llm import build_client

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("HAI_GPT_API_KEY"),
    reason="HAI_GPT_API_KEY 없음 — .env에 키를 채우면 G-agent 스모크가 실행됩니다.",
)

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": "Search the knowledge store for evidence about a claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "method": {"type": "string", "enum": ["dense", "bm25"]},
            },
            "required": ["query"],
        },
    },
}]


def test_gateway_supports_native_tool_calling():
    client = build_client()
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Use the search_evidence tool before answering."},
            {"role": "user", "content": "Verify: GDP grew 10 percent last year. Search first."},
        ],
        tools=_TOOLS,
        tool_choice="auto",
        timeout=180,  # GPT-5 추론+병렬 tool_calls는 느릴 수 있음
    )
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls, "게이트웨이가 tool_calls를 반환하지 않음(native tool-calling 미지원?)"
    first = tool_calls[0]
    assert first.function.name == "search_evidence"
    assert first.id  # 결과 회신에 필요한 tool_call_id
    print(f"\n[G-agent] {len(tool_calls)}개 tool_calls (병렬), 첫 호출 args={first.function.arguments}")

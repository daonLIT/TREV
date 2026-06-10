"""LLM tool-calling 파싱 단위 테스트 — 가짜 클라이언트로 tool_calls 응답을 정규화."""

from __future__ import annotations

from types import SimpleNamespace

from trev.llm import LLM


def _resp(content=None, tool_calls=None):
    tcs = [
        SimpleNamespace(id=tc["id"],
                        function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]))
        for tc in (tool_calls or [])
    ]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeToolClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_parses_parallel_tool_calls():
    client = FakeToolClient([_resp(tool_calls=[
        {"id": "c1", "name": "search_evidence", "arguments": '{"query": "gdp 2020"}'},
        {"id": "c2", "name": "search_evidence", "arguments": '{"query": "imf 2020", "method": "bm25"}'},
    ])])
    turn = LLM(client=client).complete_with_tools([{"role": "user", "content": "x"}], tools=[])
    assert turn.content is None
    assert [tc.id for tc in turn.tool_calls] == ["c1", "c2"]   # 병렬 보존
    assert turn.tool_calls[0].arguments == {"query": "gdp 2020"}
    assert turn.tool_calls[1].arguments["method"] == "bm25"
    # tools/tool_choice/timeout가 요청에 전달됨
    assert "tools" in client.calls[0] and client.calls[0]["tool_choice"] == "auto"


def test_final_text_when_no_tool_calls():
    client = FakeToolClient([_resp(content="done")])
    turn = LLM(client=client).complete_with_tools([{"role": "user", "content": "x"}], tools=[])
    assert turn.content == "done" and turn.tool_calls == []

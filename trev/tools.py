"""에이전트 도구 레지스트리 + 기존 능력 래핑.

도구 = {name, description, JSON-Schema, 핸들러}. 핸들러는 기존 함수로 위임하고 문자열을
반환한다(에이전트가 읽는 관찰). A1은 `search_evidence`만 — 검증 위임·tier 랭킹 도구는 A2.

`AgentContext`는 도구가 공유하는 실행 맥락(claim·index·embedder·근거 풀)이다. 도구는 dev
split·KS만 접근한다(retriever가 가드 통과). 라이브 웹 도구 없음.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from trev.indexing import ClaimIndex, Embedder
from trev.retriever import retrieve_for_queries
from trev.schemas import Claim, Evidence


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON-Schema
    handler: Callable[[dict], str]

    def spec(self) -> dict:
        """OpenAI tool-calling 형식."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Toolbox:
    """이름→도구 디스패치. 미지 도구·핸들러 예외를 안전한 오류 문자열로 회신한다."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            return tool.handler(args or {})
        except Exception as e:  # 에이전트가 복구하도록 오류를 회신
            return f"error: {type(e).__name__}: {e}"


@dataclass
class AgentContext:
    """에이전트 실행 맥락(도구가 공유). `pool`은 누적 근거(doc_id→Evidence)."""

    claim: Claim
    index: ClaimIndex
    embedder: Embedder
    tier_config: dict | None = None
    k: int = 10
    candidate_n: int = 50
    method: str = "dense"
    pool: dict[str, Evidence] = field(default_factory=dict)
    _seen: set = field(default_factory=set)

    def add_evidence(self, evidences: list[Evidence]) -> list[Evidence]:
        """근거를 풀에 추가(중복 제거)하고 안정적 doc_id를 부여한 것만 반환."""
        added: list[Evidence] = []
        for e in evidences:
            key = (e.url, e.snippet)
            if key in self._seen:
                continue
            self._seen.add(key)
            doc_id = f"e{len(self.pool)}"
            stored = e.model_copy(update={"doc_id": doc_id})
            self.pool[doc_id] = stored
            added.append(stored)
        return added


def make_search_evidence_tool(ctx: AgentContext) -> Tool:
    """`search_evidence(query, method)` — KS 검색(시점필터 내장)을 retriever에 위임."""

    def handler(args: dict) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: 'query' (non-empty string) is required"
        method = args.get("method", ctx.method)
        if method not in ("dense", "bm25"):
            return f"error: 'method' must be 'dense' or 'bm25', got {method!r}"
        evidences = retrieve_for_queries(
            ctx.claim, ctx.index, ctx.embedder, [query],
            method=method, k=ctx.k, candidate_n=ctx.candidate_n,
        )
        added = ctx.add_evidence(evidences)
        if not added:
            return "No new evidence found for that query."
        lines = [
            f"[{e.doc_id}] ({e.source_domain or 'unknown'}) {e.snippet[:200]}"
            for e in added
        ]
        return "Found evidence (cite these doc_ids):\n" + "\n".join(lines)

    return Tool(
        name="search_evidence",
        description="Search the per-claim knowledge store for evidence. "
                    "Returns doc_ids you can cite.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "method": {"type": "string", "enum": ["dense", "bm25"]},
            },
            "required": ["query"],
        },
        handler=handler,
    )

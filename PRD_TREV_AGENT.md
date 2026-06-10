# PRD — TREV-Agent: 다중 에이전트 기반 근거 검증 프레임워크 (native tool-calling)

> 짝 문서: `PRD_TREV.md`(#1, 결정론 controller 시스템), `PRD_TREV_DATA.md`(#15, 데이터 계층)
> 범위: 기존 TREV(데이터 계층 + 결정론적 controller R1~R5)를 **다중 에이전트 + 네이티브
> tool-calling** 구조로 재설계한다. 기존 능력(검색·tier 가중·verify)을 **도구**로 노출하고,
> planner/searcher/verifier 에이전트가 자율적으로 조율한다.
> 원칙: 결정론 controller는 **baseline으로 보존**(에이전트 vs 규칙 head-to-head), KS-only 검색·
> 시점 통제·dev-only·동일 평가 하니스 유지(재현성).

---

## Problem Statement

연구자는 결정론적 controller(R1~R5)로 근거 검증 파이프라인을 구현했지만, 이 구조는 검증 절차를
**하드코딩**한다: 라우팅 순서가 고정이라 claim마다 전략을 바꾸지 못하고, 복잡한 주장을
하위 질문으로 분해하지 못하며, "근거가 충분한지 / 더 검색할지"를 동적으로 판단하지 못한다.
또한 controller는 재사용 가능한 **프레임워크**가 아니라 단일 실험용 흐름에 가깝다.

연구자는 LLM 에이전트에게 검색·검증 과정의 **자율성**(계획 수립, 도구 사용, 역할 분담)을 주면
veracity 예측과 근거 품질이 향상되는지 정량 검증하고 싶다. 동시에 이 연구의 핵심 기여인
**동적 tier 가중**과 **KS-only 재현성**은 잃지 않아야 하며, 결정론 controller와 **동일 지표로
공정 비교**할 수 있어야 한다.

## Solution

planner·searcher·verifier 에이전트가 **네이티브 tool-calling**으로 협업하는 다중 에이전트
프레임워크를 만든다. 기존 모듈을 도구로 노출한다: per-claim 검색(dense/bm25), tier 가중 랭킹,
verify, 도메인 tier 평가. orchestrator가 planner→searcher→verifier를 조율하고, verifier
저신뢰 시 searcher로 피드백 루프를 돈다. 결정론 controller(R1~R5)는 baseline 조건으로 남겨,
**agentic vs deterministic**을 동일 KS·동일 GPT-5·동일 평가 지표(Acc·Macro-F1·Recall@k·
인용율·검색실패vsNEI)로 측정한다. 에이전트의 전체 추론·도구호출 trace를 보존해 분석한다.

핵심 가치(연구자 관점):
- 에이전트 자율성의 효과를 결정론 baseline 대비 **정량 입증**(향상폭·실패모드).
- 핵심 기여(tier 가중)를 **도구로 보존** → 에이전트가 tier 도구를 쓸 때/안 쓸 때 ablation.
- KS-only·시점 통제·dev-only 유지 → 결정론 결과와 **재현성·비교 가능성** 보장.
- 재사용 가능한 에이전트 프레임워크(도구 레지스트리·에이전트 루프·trace).

## User Stories

1. 연구자로서, 결정론 controller를 그대로 baseline 조건으로 유지하고 싶다. 그래야 에이전트와 규칙기반을 공정 비교한다.
2. 연구자로서, 기존 retriever(dense/bm25)를 `search_evidence` 도구로 노출하고 싶다. 그래야 에이전트가 검색을 호출한다.
3. 연구자로서, 기존 tier 가중 랭킹을 `rank_by_tier` 도구로 노출하고 싶다. 그래야 핵심 기여를 에이전트가 사용한다.
4. 연구자로서, 기존 verifier를 `verify_claim` 도구로 노출하고 싶다. 그래야 에이전트가 판정을 위임한다.
5. 연구자로서, 도메인 tier 평가를 `assess_source_tier` 도구로 노출하고 싶다. 그래야 에이전트가 출처 신뢰도를 질의한다.
6. 에이전트 개발자로서, OpenAI 호환 게이트웨이의 native tool-calling으로 LLM이 도구를 직접 호출하게 하고 싶다. 그래야 관용적 에이전트 루프를 만든다.
7. 에이전트 개발자로서, `llm.py`에 tool-calling 인터페이스(tools 전달 + tool_calls 수신·디스패치·결과 회신)를 추가하고 싶다. 그래야 단일 주입 가능 seam을 유지한다.
8. 에이전트 개발자로서, 단일 에이전트 루프(LLM↔도구 반복, 최종 응답으로 종료)를 base로 두고 싶다. 그래야 역할별 에이전트가 이를 상속한다.
9. 연구자로서, planner 에이전트가 claim을 하위 질문으로 분해하고 검색 전략을 세우게 하고 싶다. 그래야 복잡한 주장을 다룬다.
10. 연구자로서, searcher 에이전트가 하위 질문별로 검색·tier 랭킹·시점필터 도구를 호출해 근거를 모으게 하고 싶다. 그래야 근거를 수집한다.
11. 연구자로서, verifier 에이전트가 모은 근거로 라벨·stance·cited를 산출하게 하고 싶다. 그래야 판정을 낸다.
12. 연구자로서, orchestrator가 planner→searcher→verifier를 조율하게 하고 싶다. 그래야 다중 에이전트가 하나의 Verdict로 수렴한다.
13. 연구자로서, verifier가 저신뢰면 searcher로 피드백(질의 확장)해 1회 재검색하게 하고 싶다. 그래야 R4와 동등한 보강을 에이전트가 자율 수행한다.
14. 연구자로서, 에이전트의 step·도구호출·토큰 예산에 상한을 두고 싶다. 그래야 무한 루프·비용 폭주를 막는다.
15. 연구자로서, CONFLICT는 verifier 에이전트가 T1~T2 stance 공존을 근거로 판정하게 하고 싶다. 그래야 결정론 R5와 동일 의미를 유지한다.
16. 연구자로서, cited가 비어있지 않도록 강제하고 싶다. 그래야 무인용 판정을 막는다(인용율 지표).
17. 연구자로서, 에이전트가 KS 위에서만 검색하게 강제하고 싶다. 그래야 라이브 웹 누수 없이 재현성을 지킨다.
18. 연구자로서, 시점필터(`published_at <= T_claim`)를 검색 도구 내부에 유지하고 싶다. 그래야 에이전트도 시점 누수를 차단한다.
19. 연구자로서, 에이전트 전체 trace(에이전트·도구·인자·관찰·계획)를 구조화해 저장하고 싶다. 그래야 의사결정을 분석한다.
20. 연구자로서, 'agentic' 조건을 기존 4조건(gpt_only/naive/unweighted/proposed) 실행기에 추가하고 싶다. 그래야 동일 하니스로 비교한다.
21. 연구자로서, 에이전트 예측을 결정론과 동일한 예측 JSON 포맷으로 저장하고 싶다. 그래야 동일 채점(metrics.py)을 쓴다.
22. 연구자로서, agentic 조건도 동일 Acc·Macro-F1·Recall@k·Precision@k·인용율·검색실패vsNEI로 채점하고 싶다.
23. 연구자로서, tier 도구를 끈 agentic ablation을 돌리고 싶다. 그래야 동적 tier 기여를 에이전트 맥락에서 분리한다.
24. 연구자로서, 에이전트 vs 결정론 controller의 향상폭·토픽별 분해를 보고하고 싶다. 그래야 어디서 자율성이 효과적인지 안다.
25. 연구자로서, agentic 조건의 N=3 반복 라벨 일치율을 보고하고 싶다. 그래야 재현성을 '결정론'이 아닌 일치율로 서술한다.
26. 에이전트 개발자로서, 도구를 JSON-Schema로 선언하고 레지스트리로 디스패치하고 싶다. 그래야 도구 추가가 쉽다.
27. 에이전트 개발자로서, 도구 호출 실패·잘못된 인자에 안전한 오류 메시지를 에이전트에 회신하고 싶다. 그래야 에이전트가 복구한다.
28. 연구자로서, 에이전트가 사용한 평균 도구호출 수·step 수를 비용 지표로 보고하고 싶다. 그래야 자율성의 비용을 정량화한다.
29. 에이전트 개발자로서, 가짜 LLM(스크립트된 tool_calls)과 가짜 도구로 orchestrator를 테스트하고 싶다. 그래야 실 LLM·데이터 없이 조율·종료를 검증한다.
30. 연구자로서, native tool-calling 게이트웨이 지원을 스모크로 1건 확인하고 싶다(G-gate). 그래야 본 구현 전 가용성을 보장한다.
31. 연구자로서, 결정론 baseline과 agentic이 **동일 GPT-5·프롬프트 정책**을 공유하게 하고 싶다. 그래야 모델 변인을 통제한다.
32. 연구자로서, 에이전트 인덱스 빌드도 영속화(빌드 1회)를 재사용하고 싶다. 그래야 전체 실행이 가능하다.

## Implementation Decisions

### 재사용(불변) 모듈 — 도구·하부 능력
- 데이터 계층: `dataset`, `knowledge_store`, `indexing`(per-claim FAISS+BM25, 영속화), `guards`,
  `schemas`, `config`, `llm`(확장), `recall`, `tier`, `retriever`, `verifier`, `metrics`,
  `topics`, `ablation`, `annotation`, `auxmetrics` — 그대로 유지.
- 결정론 `controller`(R1~R5)는 **삭제하지 않고 baseline 조건으로 보존**.

### LLM tool-calling (seam 확장) — G-agent 실측 반영
- `llm`에 tool-calling 메서드를 추가: `tools`(JSON-Schema 선언)와 messages를 받아 OpenAI 호환
  `chat.completions.create(tools=..., tool_choice="auto")`를 호출하고, 응답의 `tool_calls`
  (없으면 최종 텍스트/JSON)를 반환한다. tenacity 재시도·검증을 기존과 동일하게 캡슐화한다.
- 기존 `complete`/`complete_json`은 유지. tool-calling은 **추가** 인터페이스.
- **병렬 tool_calls(실측)**: GPT-5는 한 턴에 여러 도구 호출을 동시에 낸다 → 루프는 `tool_calls`를
  **리스트로 실행**하고, 각 결과를 매칭 `tool_call_id`의 `role="tool"` 메시지로 회신해야 한다.
- **지연(실측 ~40s/턴)**: GPT-5 추론+병렬 호출로 tool-calling 1턴이 ~40초 → 요청 timeout을
  넉넉히(예: 180s) 두고, step 상한으로 턴 수를 제한한다.

### 도구 레지스트리 (기존 능력 래핑)
- 각 도구 = `{name, JSON-Schema, 핸들러}`. 핸들러는 기존 함수로 위임:
  - `search_evidence(query, method=dense|bm25)` → `retriever.retrieve`(시점필터 내장, KS-only).
  - `rank_by_tier(weighted: bool)` → `tier.rank_evidence`(동적 tier·자기출처 강등; 핵심 기여).
  - `verify_claim()` → `verifier.run_verifier`(라벨·stance·cited, 스키마 검증).
  - `assess_source_tier(domain)` → `tier.assign_tier`/`classify_domain`.
- 디스패처: 이름으로 핸들러 호출, 인자 검증, 실패 시 에이전트에 안전한 오류 문자열 회신.
- 위생: 도구는 dev split·KS만 접근(가드 통과). 라이브 웹 도구 없음.

### 에이전트
- **base 에이전트 루프**: (system 정책 + 도구셋)으로 LLM 호출 → tool_calls면 실행·관찰을 messages에
  추가하고 반복 → 최종 응답이면 종료. step·도구호출 상한.
- **planner**: claim → 하위 질문/검색 전략(JSON). (AVeriTeC questions와 동형의 분해.)
- **searcher**: 하위 질문별로 `search_evidence`+`rank_by_tier`(+`assess_source_tier`) 호출 →
  tier 메타 보존 Evidence 집합.
- **verifier(agent)**: claim + 수집 근거로 `verify_claim` 호출/추론 → 라벨·stance·cited.
  CONFLICT는 T1~T2 stance 공존 시(결정론 R5와 동일 규칙) verifier 에이전트가 결정.
- **orchestrator**: planner→searcher→verifier 조율. verifier confidence<τ면 searcher로 1회
  피드백(질의 확장) 후 재검증, 그래도 낮으면 NEI. cited 강제. 최종 `Verdict` 반환.
- 정책(프롬프트)은 R1~R5의 **의미를 가이드라인**으로 담되 순서는 에이전트가 자율 결정.

### 실험·평가 통합
- `experiment` 실행기에 **`agentic` 조건**을 추가(기존 gpt_only/naive/unweighted/proposed와 공존).
  동일 GPT-5·동일 KS·동일 인덱스 영속화·동일 예측 JSON 포맷 사용.
- agentic도 `metrics.evaluate`로 동일 채점. `ranked_topk_urls` 동등물(searcher가 모은 top-k url)을
  Recall@k에 공급.
- **ablation**: agentic(+tier 도구) vs agentic(−tier 도구) vs deterministic proposed.
- 비용 지표: 평균 step·도구호출 수를 레코드에 포함.

### 스키마/계약
- 신규: `ToolCall`(name, args, result), `AgentStep`(agent, thought?, tool_calls, observations),
  `AgentTrace`(steps[], final Verdict). 기존 `Claim`/`Evidence`/`Verdict`는 재사용.
- 출력 Verdict는 결정론과 동일 형태(label5·averitec_label·confidence·justification·cited).

### 게이트
- **G-agent: 통과(확인 완료)**. 게이트웨이 native tool-calling 실호출 성공 — `finish_reason=tool_calls`,
  병렬 tool_calls 반환. JSON-action 폴백 불요. (`tests/test_smoke_tool_calling.py`로 재현.)
- 비용(실측 반영): tool-calling 1턴 ~40s × 멀티에이전트 다(多)턴(planner+searcher 루프+verifier)
  × claim 수 → 결정론(1~2회)보다 훨씬 무겁다. **step·도구호출 상한 필수**, 인덱스 영속화 재사용,
  전체 실행은 GPU/예산·시간 예산을 별도 관리. 평균 step/호출 수를 비용 지표로 보고.

## Testing Decisions

좋은 테스트 = **외부 행동만 검증**(조율 결과·종료·도구 디스패치·Verdict 형태). 실 LLM·실 임베딩·
실 데이터 불필요. 기존 패턴 재사용: `controller` 테스트(가짜 retriever/verifier 주입)와
`llm` 테스트(`FakeClient`)의 철학을 그대로 잇는다.

- **orchestrator (최상위 seam)**: 가짜 LLM(스크립트된 tool_calls/최종응답) + 가짜 도구 주입 →
  planner→searcher→verifier 흐름, 저신뢰 피드백 1회, cited 강제, step 상한, 최종 Verdict 검증.
  CONFLICT가 T1~T2 stance 공존 시에만 나는지.
- **base 에이전트 루프**: 가짜 LLM이 tool_call→관찰→최종응답 시퀀스를 낼 때 도구 실행·종료·상한 동작.
- **planner/searcher/verifier (각 seam)**: 가짜 tool_call 스크립트로 분해·근거수집·라벨 산출 검증.
- **llm tool-calling**: 가짜 client가 `tool_calls` 응답을 줄 때 파싱·디스패치·결과 회신.
- **도구 어댑터 (순수/모킹)**: 각 도구가 기존 함수로 올바른 계약으로 위임하는지(인자 검증·오류 회신 포함).
- **trace**: 도구호출·step이 trace에 정확히 기록되는지.
- **평가 통합 (픽스처)**: agentic 레코드가 결정론과 동일 포맷으로 `metrics.evaluate`에 들어가는지.

Prior art: `tests/test_controller.py`(가짜 seam 주입), `tests/test_llm.py`(`FakeClient`),
`tests/test_experiment.py`(실행기). 신규 픽스처는 스크립트된 tool_calls 응답 + 가짜 도구.

## Out of Scope

- **외부 라이브 웹 검색** — KS-only 유지(재현성). 에이전트도 KS 도구만.
- **운영 데모 파이프라인(S12, #14)** — 별도·시연용.
- **공식 AVeriTeC Ev2R 완전구현** — 보조 근사만(#13).
- **에이전트 학습/파인튜닝** — 프롬프트·도구 설계만.
- **다중 claim·교차 claim 에이전트** — 단일 claim 검증에 한정.
- **JSON-action 폴백** — native tool-calling 우선. G-agent 미통과 시에만 재논의.
- **test/`test_2025`/train 사용** — blind·Out of Scope 유지.

## Further Notes

- **결정론 보존이 핵심**: controller R1~R5는 비교 baseline이자 정책 명세의 출처. 에이전트 정책
  프롬프트는 R1~R5의 의미(무근거·T4뿐→NEI / T1~3→verify / 저신뢰→확장재검색 / stance공존→CONFLICT)를
  가이드로 담는다.
- **핵심 기여 보존**: 동적 tier 가중은 `rank_by_tier` 도구로 살아 있으며, on/off ablation으로
  에이전트 맥락에서도 효과를 분리한다.
- **재현성**: GPT-5 temp=0 tool-calling도 완전 결정론이 아님 → N=3 일치율로 보고('결정론' 표현 금지).
- **데이터 계약 불변**: G0 사실(claim↔KS 위치인덱스, published_at 아카이브 유도, gold 격리,
  url 정규화 Recall)과 가드 규칙을 그대로 따른다.
- **라이선스**: AVeriTeC CC BY-NC 4.0 — 비영리·출처표시·데이터 재배포 금지.
- **인용**: Schlichtkrull, Guo, Vlachos. AVeriTeC. NeurIPS D&B, 2023.

# PRD — TREV: Tier-weighted Retrieval for EVidence-based verification

> 근거 문서: `TREV_시스템구성.md`, `TREV_연구_상세설계서_AVeriTeC_20215255_최다온.docx`
> 결정 출처: grilling 세션(설계 결정 기록 10건) 반영
> 상태: ready-for-agent (발행 대기 — gh 미설치)

---

## Problem Statement

연구자(최다온)는 AVeriTeC 벤치마크의 **수치·기록성 주장(Numerical · Event/Property Claim)**을 대상으로,
"근거 출처만으로 신뢰도를 고정하지 않고 `(주장 유형 × 근거 역할)`로 동적으로 가중하는 검색이
일반 RAG보다 라벨 정확도와 환각 억제에서 우수하다"는 가설을 **실험적으로 입증**해야 한다.

현재 레포는 설계 문서만 있는 그린필드 상태이며, 가설을 측정 가능한 시스템으로 구현한 코드가 없다.
또한 설계서에는 다음과 같은 미해소 위험이 있어 그대로 구현하면 실험이 무효화되거나 대량 재작업이 발생한다:

- AVeriTeC knowledge store(KS)를 "전역 단일 인덱스"로 가정 → 시점 누수·스케일 위험
- CONFLICT 판정 책임이 verifier와 controller에 이중으로 존재 → 모순
- "AVeriTeC 공식 score"를 메인 지표로 가정 → QA 생성 모듈이 빠져 실측 불가
- `gpt_only` baseline이 controller 하드 규칙(R2/cited)과 충돌 → baseline이 통째로 NEI로 붕괴
- 동적 tier의 "역할(role)" 판정 방식이 모호 → 핵심 기여가 단순 도메인 화이트리스트로 붕괴

## Solution

claim 하나를 입력받아 → KS 위에서 **per-claim 검색** → **동적 tier 가중** 랭킹 →
**5단계 라벨 + confidence + justification + 근거 인용**으로 판정하는 모듈형 RAG 에이전트를 구현한다.

핵심 가치(연구자 관점):

- 동일 LLM(GPT-5) · 동일 KS · 동일 프롬프트로 4개 조건(`gpt_only` / `naive_rag` / `unweighted_rag` / `proposed`)을
  비교해 **검색·가중 구조의 효과만 분리** 측정한다.
- 동적 tier가 "출처만으로 고정하지 않는다"를 **결정론적으로 실제 발화**시켜(자기출처 매칭 → `target` 강등),
  핵심 기여가 측정 가능하게 만든다.
- 메인 지표(Label Accuracy · Macro-F1 · Evidence Recall@k/Precision@k)만으로 가설을 완전히 입증하고,
  공식 AVeriTeC score는 보조로 둔다.

## User Stories

1. 연구자로서, AVeriTeC `dev.json`을 적재하고 싶다. 그래야 실험 대상 claim을 확보한다.
2. 연구자로서, claim의 `claim_types`로 Numerical·Event/Property 서브셋만 필터링하고 싶다. 그래야 핵심 검증 대상에 집중한다.
3. 연구자로서, 서브셋이 작으면 Quote Verification을 보조로 포함하는 옵션을 갖고 싶다. 그래야 표본 부족에 대응한다.
4. 연구자로서, 각 claim의 `claim_date`를 `T_claim`으로 보존하고 싶다. 그래야 시점 통제의 기준을 갖는다.
5. 연구자로서, AVeriTeC 4라벨 gold를 보존하고 싶다. 그래야 채점 기준으로 쓴다.
6. 연구자로서, claim 텍스트·speaker·publisher 메타로 토픽(정치/보건/경제/사법·범죄/과학·환경)을 태깅하고 싶다. 그래야 토픽별 분석을 한다.
7. 연구자로서, 토픽 태깅을 키워드 규칙 1차 → 애매하면 LLM으로 분류하고 싶다. 그래야 비용을 줄이면서 정확도를 확보한다.
8. 연구자로서, 표본 200건을 사람이 검수해 LLM 태깅과의 Cohen's κ를 보고하고 싶다. 그래야 태깅 신뢰도를 입증한다.
9. 연구자로서, 적재 직후 서브셋 내 Conflicting/Cherry-picking gold 개수를 카운트하고 싶다. 그래야 5라벨 평가 방식을 분기한다.
10. 연구자로서, 각 claim의 KS 후보 문서 풀만 임베딩해 per-claim FAISS 인덱스를 만들고 싶다. 그래야 시점 누수 없이 검색한다.
11. 연구자로서, 문서를 청크로 분할해 e5 임베딩하고 싶다. 그래야 긴 문서를 검색 가능 단위로 만든다.
12. 연구자로서, BM25 인덱스도 만들고 싶다. 그래야 dense vs lexical ablation을 한다.
13. 연구자로서, 각 문서의 `published_at`·`source`(도메인)·url을 보존하고 싶다. 그래야 시점 필터·tier 부여에 쓴다.
14. 연구자로서, claim type별 질의 템플릿으로 2~3개 질의를 생성하고 싶다. 그래야 검색 적중률을 높인다.
15. 연구자로서, 질의 임베딩으로 per-claim 인덱스에서 top-N 후보를 받고 싶다. 그래야 재랭킹 대상을 좁힌다.
16. 연구자로서, `published_at <= T_evidence(=claim_date)` 문서만 통과시키고 싶다. 그래야 시점 누수를 차단한다.
17. 연구자로서, 후보 문서의 도메인을 휴리스틱 규칙으로 1차 tier(T1~T4)에 매핑하고 싶다. 그래야 신뢰도 계층을 부여한다.
18. 연구자로서, 서브셋에 실제 등장하는 고유 도메인을 수동 감수해 config로 오버라이드하고 싶다. 그래야 long-tail 오분류를 막는다.
19. 연구자로서, 문서 도메인이 claim의 `original_claim_url`/`reporting_source`와 일치하면 `role=target`으로 T4 강등하고 싶다. 그래야 동적 tier가 실제로 발화한다.
20. 연구자로서, 잔여 애매 도메인만 LLM으로 1회 분류·캐시하고 싶다. 그래야 비용·비결정성을 최소화한다.
21. 연구자로서, `score = sim * weight`(또는 `sim + λ·log weight`)로 재랭킹해 top-k를 받고 싶다. 그래야 신뢰 근거를 우선한다.
22. 연구자로서, baseline 분기에서 `unweighted`는 weight를 무시(`score=sim`)하고 싶다. 그래야 가중 효과를 분리한다.
23. 연구자로서, `naive_rag`는 top-k 그대로, `gpt_only`는 검색을 건너뛰고 싶다. 그래야 조건을 통제한다.
24. 연구자로서, verifier 한 번의 LLM 호출에서 5라벨·confidence·justification·cited와 함께 **per-evidence stance**까지 받고 싶다. 그래야 별도 NLI 없이 CONFLICT를 판정한다.
25. 연구자로서, verifier가 `{SUPPORT, REFUTE, PARTIAL, NEI}`만 판정하게 하고 싶다. 그래야 CONFLICT 책임을 controller로 일원화한다.
26. 연구자로서, verifier가 검색된 근거에만 정초해 판정하게 하고 싶다. 그래야 환각을 차단한다.
27. 연구자로서, verifier 출력의 cited가 비어있지 않도록 강제하고 싶다(gpt_only 제외). 그래야 무인용 판정을 막는다.
28. 연구자로서, 경계 규칙(핵심 틀림→REFUTE, 과장→PARTIAL, 신뢰근거 없음→NEI)을 프롬프트에 명시하고 싶다. 그래야 라벨 일관성을 높인다.
29. 연구자로서, 5라벨을 4라벨로 매핑(PARTIAL·CONFLICT→Conflicting/Cherry-picking, 나머지 1:1)하고 싶다. 그래야 공식 채점에 정렬한다.
30. 연구자로서, controller R1로 `checkworthiness < τ`인 claim을 폐기하고 싶다. 그래야 검증 가치 없는 claim을 거른다.
31. 연구자로서, controller R2로 evidence 0건 또는 최대 tier가 T4뿐이면 즉시 NEI 처리하고 싶다. 그래야 근거 없는 판정을 막는다.
32. 연구자로서, controller R3로 T1~T3 근거가 1건 이상일 때만 verify를 호출하고 싶다.
33. 연구자로서, controller R4로 confidence<0.5이면 질의 확장 후 1회 재검색·재검증하고, 그래도 낮으면 NEI로 두고 싶다. 그래야 저신뢰 판정을 보정한다.
34. 연구자로서, controller R5로 T1~T2 근거에 support·refute stance가 공존하면 CONFLICT로 결정하고 싶다.
35. 연구자로서, `gpt_only` 조건만 R1/R2/R5·cited 강제를 우회해 claim만으로 라벨을 직접 받게 하고 싶다. 그래야 baseline이 NEI로 붕괴하지 않는다.
36. 연구자로서, `naive_rag`/`unweighted_rag`/`proposed`는 동일 full controller를 통과시키고 싶다. 그래야 검색·가중 외 변인을 통제한다.
37. 연구자로서, 4개 조건을 동일 GPT-5·동일 KS·동일 프롬프트로 실행하고 싶다. 그래야 모델 성능과 구조 효과를 분리한다.
38. 연구자로서, 조건별 예측을 JSON으로 저장하고 싶다. 그래야 재현·재채점이 가능하다.
39. 연구자로서, Label Accuracy와 Macro-F1(5/4라벨)을 계산하고 싶다. 그래야 메인 정확도를 본다.
40. 연구자로서, Evidence Recall@k/Precision@k를 계산하고 싶다(검색된 top-k 문서 URL이 gold QA 근거 URL과 일치하는 비율). 그래야 검색 품질을 본다.
41. 연구자로서, 검색 실패(gold 근거 미회수)와 실제 NEI(gold가 NEI)를 구분하고 싶다. 그래야 오류 원인을 분리한다.
42. 연구자로서, 유효 인용 비율(cited)을 측정하고 무인용 판정 0을 확인하고 싶다. 그래야 근거 추적성을 입증한다.
43. 연구자로서, 토픽별로 baseline 대비 proposed의 향상폭을 분해 보고하고 싶다. 그래야 tier 가중이 효과적인 토픽을 보인다.
44. 연구자로서, Conflicting 표본이 ≥20이면 사람 세분 라벨로 PARTIAL/CONFLICT F1을, 부족하면 정성 사례연구로 보고하고 싶다.
45. 연구자로서, 3라벨 vs 5라벨, tier 가중 민감도(±0.1), BM25 vs dense ablation을 돌리고 싶다.
46. 연구자로서, 보조로 RAGAS faithfulness·G-Eval을 측정하고 싶다(LLM-as-judge 보조 지표).
47. 연구자로서, 공식 AVeriTeC score는 cited 근거에 한해 QA를 생성해 근사하는 stretch 옵션으로 두고 싶다.
48. 연구자로서, N=3 반복 실행의 라벨 일치율(agreement rate)을 재현성 지표로 보고하고 싶다. 그래야 "결정론" 과장 없이 안정성을 입증한다.
49. 연구자로서, 1주차에 GPT-5(HAI-GPT) ping이 성공하는지 먼저 확인하고 싶다. 그래야 LLM 의존 단계 착수 전 게이트를 통과한다.
50. 연구자로서, LLM 호출의 JSON 파싱 실패에 재시도·스키마 검증으로 대응하고 싶다. 그래야 파이프라인이 견고하다.
51. 연구자로서, 고정 시드·프롬프트·KS로 실험을 재현하고 싶다.
52. (운영 데모, 선택) 사용자로서, 뉴스 수집→클러스터링→요약→주장추출→검증의 일일 다이제스트를 보고 싶다. 그래야 실사용 흐름을 시연한다.

## Implementation Decisions

### 데이터 & 시점
- **검색 대상**: AVeriTeC **knowledge store 문서 위 실검색**(gold QA 직접 투입 아님). tier·Recall@k·baseline 비교가 의미를 갖기 위한 전제.
- **인덱싱**: **per-claim 인덱싱**. claim마다 그 claim의 KS 후보 문서 풀만 임베딩해 FAISS 인덱스 구성 → top-k 검색. 전역 단일 인덱스 금지(시점 누수·스케일 회피).
- **시점 통제**: 세 시점 분리(`T_claim`=claim_date, `T_evidence`=T_claim 기본, `T_label`=AVeriTeC verdict 기준). 검색은 항상 `published_at <= T_evidence`.
- **claim type 필터**: Numerical + Event/Property. 표본 부족 시 Quote Verification 보조 포함(config 토글).
- **토픽 태깅**: 키워드 규칙 1차 → 애매건 LLM. 200건 사람 검수로 κ 보고.
- **게이트(적재 직후)**: 서브셋 내 Conflicting/Cherry-picking gold 개수 카운트 → 5라벨 평가 분기 결정.

### 동적 tier (핵심 기여)
- `assign_tier(claim_type, evidence_role, source_domain) -> (tier, weight)`, 가중치 `{T1:1.0, T2:0.7, T3:0.4, T4:0.1}`.
- **role 판정은 결정론적**: 도메인 화이트리스트로 base tier → 문서 도메인이 claim의 `original_claim_url`/`reporting_source`와 일치하면 `role=target`으로 **T4 강등**. LLM 미사용.
- **화이트리스트 구축(하이브리드)**: 휴리스틱 규칙(`.gov`/통계기관/`court`→T1, 팩트체크→T2, 주요 언론→T3, 소셜·블로그→T4) base + 서브셋 실등장 고유 도메인 수동 감수·오버라이드 → `config.yaml`에 고정. 잔여 애매 도메인만 LLM 1회 분류+캐시.

### 검색
- 질의 생성: claim type별 템플릿(2~3개 질의).
- 랭킹: `score = sim * weight`(기본) 또는 `sim + λ·log(weight)`. top-N(=50) 후보 → 시점필터·tier부여 → top-k(=5).
- baseline 분기: `proposed`=동적 tier 가중, `unweighted`=weight 무시(`score=sim`), `naive`=top-k 그대로, `gpt_only`=검색 생략.
- BM25 인덱스 병행(dense vs lexical ablation).

### 검증 (책임 분리 — 설계서 모순 해소)
- **verifier**: LLM 1회 호출로 `{label∈{SUPPORT,REFUTE,PARTIAL,NEI}, confidence, justification, cited[], per-evidence stance[]}`를 JSON으로 출력. 검색 근거에만 정초.
- **CONFLICT는 verifier가 내지 않고 controller R5가 stance로 결정**. (별도 NLI 모델 없음.)
- 5→4 매핑: `PARTIAL, CONFLICT → "Conflicting/Cherry-picking"`, 나머지 1:1.

### Controller 라우팅 (R1~R5) + baseline 예외
- R1 `checkworthiness<τ`→폐기 / R2 evidence 0 또는 max_tier=T4→NEI / R3 T1~T3≥1→verify / R4 confidence<0.5→질의확장 1회 재검색·재검증, 그래도 낮으면 NEI / R5 T1~T2에 support·refute 공존→CONFLICT.
- `cited` 비어있지 않음 강제(무인용 판정 금지).
- **`gpt_only`만 R1/R2/R5·cited 강제를 우회**(claim만→라벨 직출). `naive_rag`/`unweighted_rag`/`proposed`는 동일 full controller 통과.

### LLM 래퍼 (신규 seam)
- `llm.py`는 프롬프트·재시도(tenacity)·JSON 파싱·스키마 검증(pydantic)을 캡슐화한 **단일 주입 가능 인터페이스**로 만든다. 테스트·baseline에서 스텁/교체 가능해야 한다.
- 모든 조건 동일 GPT-5(temperature=0, max_retries=3) 고정.

### 스키마 (pydantic, `schemas.py`)
- `Claim`(claim_id, text, type, topic?, claim_date, checkworthiness), `Evidence`(doc_id, snippet, source, url?, published_at?, tier, role, weight, stance, sim, score), `Verdict`(claim_id, label5, averitec_label, confidence, justification, cited[]).

### 평가
- **메인**: Label Accuracy, Macro-F1(5/4), Evidence Recall@k/Precision@k, 검색실패 vs 실제NEI 구분, 인용율.
- **보조**: RAGAS faithfulness, G-Eval.
- **stretch**: 공식 AVeriTeC score(`evaluate_veracity.py`) — cited 근거에 한해 QA 생성 어댑터로 근사.
- **재현성**: N=3 반복 라벨 일치율(agreement rate). "결정론" 표현 사용 금지.
- **분해/ablation**: 토픽별 Macro-F1, 3 vs 5 라벨, tier 가중 민감도(±0.1), BM25 vs dense.

## Testing Decisions

좋은 테스트 = **외부 행동만 검증**(구현 세부 아님). 입력 → 관찰 가능한 출력(라벨·tier·랭킹 순서·라우팅 결과)으로 단정한다.

- **`assign_tier` (순수·결정론, 최우선)**: `(claim_type × role × domain)` 케이스 테이블 테스트. 특히 자기출처매칭→T4 강등, 도메인 화이트리스트/오버라이드 적중. 핵심 기여라 가장 촘촘히.
- **시점 필터 (순수)**: `published_at <= claim_date` 경계 케이스(같은 날/이후/누락) 통과·차단 검증.
- **가중 랭킹 (순수)**: 동일 sim·다른 weight 입력 시 순서가 tier 우선으로 재정렬되는지. `unweighted` 분기는 sim 순서 유지.
- **`verifier.verify` (LLM seam 모킹)**: `llm.py`를 canned JSON으로 스텁 → 라벨 파싱, stance 파싱, cited 비어있지 않음, 5→4 매핑, JSON 파싱 실패 시 재시도/스키마 검증.
- **`controller.run` (최상위 seam, 가짜 retriever/verifier 주입)**: R1~R5 각 분기와 `gpt_only` 예외(검색 생략·cited 면제·R2 미발화)를 LLM·데이터 없이 검증. CONFLICT가 stance 공존 시에만 결정되는지.
- **`metrics` (순수)**: 소규모 예측/gold 픽스처로 Accuracy·Macro-F1·Recall@k 값 검증, 검색실패 vs 실제NEI 분류.
- **`retriever.search` (픽스처 KS 주입)**: 작은 in-memory 인덱스로 top-N→시점필터→tier부여→top-k end-to-end.

Prior art: 현재 그린필드라 기존 테스트 없음 → pytest 기반으로 위 seam별 단위 테스트를 신규 수립. 픽스처(소형 claim/evidence/KS 샘플)를 `tests/fixtures`에 둔다.

## Out of Scope

- 운영(데모) 1~4단계(뉴스 수집·전처리·클러스터링·요약)의 정량 평가 — 시연용일 뿐 평가 대상 아님(선택 구현).
- AVeriTeC **test 분할** 평가 — 라벨 blind. 평가는 dev로만, train은 few-shot 예시에만.
- 외부 라이브 웹 검색 — KS 위에서만 검색(재현성).
- 공식 AVeriTeC score의 완전 구현(QA 생성 + Ev2R 매칭) — stretch.
- 한국어/다국어 운영 평가 — 임베딩은 다국어 호환이나 평가는 영어.
- 전체 KS(train/test) 다운로드·전역 인덱싱.

## Further Notes

- **대기 게이트**: (G0) KS 실제 구조 확인 — 데이터 다운로드는 연구자가 직접 진행 중; (G1) 1주차 GPT-5 ping 성공.
- **설계서 대비 변경(반영 필요)**: ① 전역 인덱싱→per-claim, ② CONFLICT 이중책임→verifier(stance)/controller(R5) 분리, ③ 공식 score 메인→보조, ④ gpt_only controller 예외 명시, ⑤ 재현성=N=3 일치율.
- **라이선스**: AVeriTeC CC BY-NC 4.0 — 비영리·출처표시, 데이터 재배포 주의.
- **발행**: `gh` 미설치로 자동 이슈 발행 보류. 설치·인증 후 `ready-for-agent` 라벨로 GitHub Issues(daonLIT/TREV) 발행 예정.

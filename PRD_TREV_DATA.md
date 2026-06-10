# PRD — TREV 데이터 계층 (AVeriTeC dev 소비 계약)

> 짝 문서: `PRD_TREV.md`(시스템 PRD, 이슈 #1), `TREV_시스템구성.md`, `TREV_데이터사용가이드.md`
> 범위: TREV 각 모듈이 **AVeriTeC dev 데이터를 정확히 어떻게 소비하는지**의 계약을 규정한다.
> 원칙: per-claim 검색 · 시점 통제 · dev-only · knowledge store 위 실검색(gold QA 직접 투입 금지).
> 상태: 실데이터 G0 점검 결과 반영(아래 사실 확정).

---

## Problem Statement

연구자는 `data_store/`와 `knowledge_store/`에 AVeriTeC dev 데이터를 적재해 두었지만,
각 모듈(적재·필터·인덱싱·tier·검증·평가)이 이 데이터의 **실제 필드와 포맷**을 어떻게 읽고 보존해야 하는지에 대한
검증된 계약이 없다. 데이터 사용 가이드는 KS에 `published_at`이 있을 수 있다고 가정했으나
실데이터에는 **그 필드가 없다.** 또한 gold 근거(`questions`)를 검색에 섞으면 실험이 무효가 되고,
KS에 섞인 `type="gold"` 문서를 부스팅해도 무효가 된다. 잘못된 가정 위에 인덱싱·시점필터·Recall@k를 구현하면
실험 결과 자체가 신뢰를 잃는다.

## Solution

dev 데이터를 TREV 스키마로 적재하는 **검증된 데이터 계약 계층**을 만든다.
실데이터 G0 점검으로 확정한 포맷에 맞춰: claim ↔ KS 파일 위치-인덱스 매핑, KS JSONL(`url`, `url2text` passages) 소비,
web.archive.org 스냅샷 타임스탬프에서 `published_at` 유도(시점필터 복구), gold 근거의 검색 격리,
gold URL ↔ 검색 결과 URL 정규화 매칭(Recall@k)을 규정한다.
LLM이 필요 없는 단계(적재·필터·도메인 추출·인덱싱)를 먼저 끝내 GPT-5 게이트(G1) 이전에 진척을 낸다.

핵심 가치(연구자 관점):
- 모든 모듈이 동일한 검증된 데이터 계약을 공유 → 실험 재현성·무결성 보장.
- 시점 통제 기여를 데이터로 실제 뒷받침(아카이브 타임스탬프 유도).
- gold 격리·`type` 위생 규칙으로 누수 차단.

## 확정된 데이터 사실 (G0 점검 결과)

- `data_store/averitec/dev.json`: **500 claim**. claim 객체 키:
  `claim, required_reannotation, label, justification, claim_date, speaker, original_claim_url,
  fact_checking_article, reporting_source, location_ISO_code, claim_types, fact_checking_strategies,
  questions, cached_original_claim_url`.
- **gold 근거 URL은 `questions[].answers[].source_url`** (claim 최상위 아님).
- `knowledge_store/dev/`: **500개 파일**, 이름 `{index}.json`(예 `0.json`). 레코드 `claim_id` 필드 = 그 인덱스
  → **claim ↔ KS 파일은 위치 인덱스로 매핑**.
- KS 포맷: **JSONL**, 레코드 키 `{claim_id, type, query, url, url2text}`. `url2text`는 **passage 리스트**(예 23개).
- **KS에 `published_at` 없음.** 단 `url`이 web.archive.org 스냅샷(`/web/YYYYMMDDhhmmss/...`)
  → 스냅샷 타임스탬프를 `published_at`으로 유도 가능.
- KS 레코드에 **`type` 필드 존재(예 `"gold"`)** — gold 문서가 코퍼스에 섞여 있음.
- gold URL 포함률: claim0 기준 gold 2개가 KS 825개 중 **2개 다 포함(overlap 2/2)**
  → exact/정규화 URL Recall@k 실현 가능.

## User Stories

1. 데이터 엔지니어로서, `dev.json` 500 claim을 `list[Claim]`로 적재하고 싶다. 그래야 실험 대상을 확보한다.
2. 데이터 엔지니어로서, `claim_types`로 Numerical + Event/Property 서브셋을 필터링하고 싶다. 그래야 핵심 검증 대상에 집중한다.
3. 데이터 엔지니어로서, 서브셋이 작으면 Quote Verification을 보조 포함하는 config 토글을 갖고 싶다. 그래야 표본 부족에 대응한다.
4. 데이터 엔지니어로서, `claim_date`(d-m-yyyy·결측 혼재)를 정규화해 `T_claim`으로 보존하고 싶다. 그래야 시점필터 기준을 갖는다.
5. 데이터 엔지니어로서, gold `label`을 4라벨 채점 기준으로 보존하고 싶다. 그래야 평가 기준을 확보한다.
6. 데이터 엔지니어로서, 적재 직후 서브셋 내 `Conflicting Evidence/Cherrypicking` gold 개수를 카운트하고 싶다. 그래야 5라벨 평가 방식을 분기한다.
7. 데이터 엔지니어로서, claim ↔ KS 파일을 위치 인덱스로 매핑하고 `claim_id`로 교차검증하고 싶다. 그래야 올바른 후보 풀을 로드한다.
8. 데이터 엔지니어로서, KS JSONL 레코드에서 `url`과 `url2text` passage 리스트를 읽고 싶다. 그래야 검색 코퍼스를 만든다.
9. 데이터 엔지니어로서, `url`에서 도메인을 추출(`urllib.parse`)해 보존하고 싶다. 그래야 tier 부여에 쓴다.
10. 데이터 엔지니어로서, web.archive.org 스냅샷 타임스탬프를 `published_at`으로 유도하고 싶다. 그래야 시점필터를 살린다.
11. 데이터 엔지니어로서, `published_at` 유도 실패 시 그 문서를 시점필터 통과로 처리하고 싶다. 그래야 날짜 결측에 안전하다.
12. 데이터 엔지니어로서, claim별로 그 claim의 KS 문서만 e5 임베딩해 per-claim FAISS 인덱스를 만들고 싶다. 그래야 시점 누수 없이 검색한다.
13. 데이터 엔지니어로서, 동일 코퍼스로 BM25 인덱스도 만들고 싶다. 그래야 dense vs lexical ablation을 한다.
14. 데이터 엔지니어로서, 인덱싱 시 `url`·도메인·`published_at`(유도분) 메타를 passage에 보존하고 싶다. 그래야 tier·시점필터·Recall@k가 동작한다.
15. 데이터 엔지니어로서, 서브셋에 실제 등장하는 고유 도메인 목록을 빈도순으로 추출하고 싶다. 그래야 tier 화이트리스트를 작성한다.
16. 데이터 엔지니어로서, `original_claim_url`/`reporting_source`의 도메인을 추출해 보존하고 싶다. 그래야 `role=target` 자기출처매칭에 쓴다.
17. 데이터 엔지니어로서, `speaker`·`reporting_source`·`location_ISO_code`를 토픽 태깅 보조 입력으로 보존하고 싶다.
18. 데이터 엔지니어로서, gold `questions`를 검색 코퍼스에서 격리하고 싶다. 그래야 gold 누수로 실험이 무효화되지 않는다.
19. 데이터 엔지니어로서, KS의 `type="gold"` 문서를 검색·랭킹에서 부스팅하지 않고 일반 후보로만 다루고 싶다. 그래야 누수를 막는다.
20. 데이터 엔지니어로서, 검색 top-k 문서 `url`을 gold `source_url`과 정규화(아카이브 prefix·스킴·www·쿼리 제거) 후 매칭하고 싶다. 그래야 Recall@k/Precision@k를 측정한다.
21. 데이터 엔지니어로서, gold 근거가 KS에 있는데 미회수면 검색실패, gold label이 NEI면 실제부재로 구분하고 싶다. 그래야 오류 원인을 분리한다.
22. 데이터 엔지니어로서, `train.json`은 few-shot 예시로만 접근하고 평가에 쓰지 않게 강제하고 싶다.
23. 데이터 엔지니어로서, test/`test_2025`/train KS를 사용하지 않게 가드하고 싶다. 그래야 blind·Out of Scope를 지킨다.
24. 데이터 엔지니어로서, `data_store/`·`knowledge_store/`·`index/`·`outputs/`·`.env`가 `.gitignore`에 있도록 보장하고 싶다. 그래야 데이터를 재배포하지 않는다(CC BY-NC 4.0).
25. 데이터 엔지니어로서, 5→4 매핑의 gold 문자열(`"Conflicting Evidence/Cherrypicking"`)을 데이터에서 정확히 확인하고 싶다. 그래야 채점이 어긋나지 않는다.

## Implementation Decisions

### 적재 & 정규화
- `dev.json`(500) → `list[Claim]`. `claim_types`를 `ClaimType`으로 정규화, Numerical + Event/Property 필터(config로 Quote 보조 토글).
- `claim_date` 파서: `%d-%m-%Y`/`%Y-%m-%d`/`%m-%d-%Y` 순 시도, 실패·결측 → `None`(시점필터 통과). 성공 시 `T_claim = T_evidence`.
- 적재 직후 게이트: 서브셋 내 `label == "Conflicting Evidence/Cherrypicking"` 개수 카운트 → ≥20 정량(세분 F1) / 미만 정성(사례연구).
- gold `label` 보존, gold 문자열은 데이터에서 정확히 확인(철자 고정).

### knowledge store 소비 (per-claim)
- **매핑**: claim 위치 인덱스 → `knowledge_store/dev/{index}.json`. 레코드 `claim_id`로 교차검증(불일치 시 오류).
- **포맷**: JSONL. 각 레코드 `{claim_id, type, query, url, url2text}`. `url2text`(passage 리스트)를 검색 단위로 사용(추가 청킹은 길이 초과 passage에만 적용).
- **시점 메타 유도**: `url`이 `web.archive.org/web/(\d{14})/<original>` 패턴이면 14자리를 `published_at`(스냅샷 시각)으로 파싱. 비-아카이브/파싱 실패 → `published_at=None`(시점필터 통과).
- **보존 메타(필수)**: `url`, `source_domain`(url netloc), `published_at`(유도분). tier·시점필터·Recall@k가 전적으로 의존.
- **인덱싱**: claim마다 그 claim의 passage만 e5 임베딩 → per-claim FAISS + BM25. 전역 단일 인덱스 금지(시점 누수).
- **데이터 위생**: `questions`(gold)는 검색 코퍼스 제외. KS `type` 필드는 분석용으로만 보존하고 **검색·랭킹 부스팅 금지**.

### tier 입력 데이터 (결정론)
- `source_domain` = KS `url`의 netloc(아카이브 URL이면 원본 URL을 복원 후 netloc).
- base tier = 도메인 화이트리스트(`config.yaml`): `.gov`/통계/`court`→T1, 팩트체크→T2, 주요언론→T3, 소셜·블로그→T4.
- `role=target` 강등: KS 문서 도메인이 claim의 `original_claim_url`/`reporting_source` 도메인과 일치하면 T4.
- 화이트리스트 작성 입력: 서브셋 실등장 고유 도메인 빈도순 목록 → 사람 감수·config 오버라이드, 잔여만 LLM 1회+캐시.

### 평가 데이터
- Label Acc·Macro-F1: 예측 `label5`→4 매핑 vs gold `label`.
- Recall@k/Precision@k: 검색 top-k `url` vs gold `questions[].answers[].source_url`, **URL 정규화 후 매칭**(아카이브 prefix 제거→원본 URL, 스킴/`www`/쿼리 정규화).
- 검색실패 vs 실제 NEI: gold URL이 KS에 존재하나 미회수=검색실패 / gold label=NEI=실제부재.
- 5→4 매핑: `PARTIAL, CONFLICT → "Conflicting Evidence/Cherrypicking"`, 나머지 1:1.

## Testing Decisions

좋은 테스트 = 외부 행동만 검증(입력 데이터 → 관찰 가능한 출력: 파싱 결과·보존 메타·매칭 집합). 실 LLM·실 임베딩 불필요.

- **`load_averitec` (픽스처 dev.json 일부)**: 필터 결과 수, ClaimType 정규화, `claim_date`→`T_claim`, gold label 보존, Conflicting 카운트.
- **`parse_date` / `archive_ts` (순수 케이스 테이블)**: d-m-yyyy·y-m-d·결측·비아카이브 URL·`/web/14digit/` 패턴 경계.
- **`load_ks` (픽스처 JSONL 주입)**: `{index}.json` ↔ `claim_id` 교차검증, `url2text` passage 추출, 보존 메타(url·domain·published_at) 채움, gold `questions` 미포함, `type` 부스팅 없음.
- **도메인 추출 (순수)**: 아카이브 URL→원본 netloc 복원, 자기출처매칭(`original_claim_url`/`reporting_source`)→target.
- **Recall 매칭 (순수)**: 정규화 후 gold `source_url` ↔ top-k `url` 교집합 계산, 아카이브/원본 혼재·쿼리·www 차이 케이스.
- **하네스 가드 (순수)**: train/test KS 접근 시 예외, `.gitignore` 항목 존재 확인.

Prior art: 그린필드 → pytest 신규. 소형 픽스처(claim 2~3개 + 미니 KS JSONL)를 `tests/fixtures`에 둔다. 시스템 PRD(#1)의 seam 규약과 동일 철학.

## Out of Scope

- AVeriTeC **test/`test_2025`·train KS** 사용 — blind·Out of Scope.
- `train.json` — few-shot 예시 외 사용 금지(평가 비대상).
- gold QA·외부 라이브 웹 검색 — KS 위에서만(재현성).
- 시점필터의 완전성 보장 — 아카이브 타임스탬프가 없는 문서는 통과 처리(시점 통제 기여는 "AVeriTeC 구축단계 누수 통제 + 아카이브 타임스탬프 보강"으로 서술).
- 임베딩/검색/검증 로직 자체 — 시스템 PRD(#1) 슬라이스 S3~S7 소관. 본 PRD는 그 모듈들이 소비하는 **데이터 계약**만 규정.

## Further Notes

- **가이드 대비 정정**: ① KS에 `published_at` 없음 → 아카이브 타임스탬프 유도로 대체. ② gold URL은 `questions[].answers[].source_url`. ③ claim↔KS는 위치-인덱스 매핑(`claim_id` 교차검증). ④ KS `type="gold"` 위생 규칙 추가.
- **LLM 불요 선행 작업**: US1~16(적재·필터·날짜·도메인·인덱싱·화이트리스트 추출)은 GPT-5 게이트(G1) 이전에 완료 가능.
- **시스템 PRD 슬라이스 연계**: 본 계약은 #2(S0 스키마/구조점검)·#3(S1 적재·필터·카운트)·#5(S3 per-claim 인덱싱)·#6(S4 tier 입력)·#9(S7 Recall@k)에서 소비된다.
- **라이선스**: AVeriTeC CC BY-NC 4.0 — 비영리·출처표시, 데이터 재배포 금지. 인용: `Schlichtkrull, Guo, Vlachos. AVeriTeC. NeurIPS D&B, 2023.`

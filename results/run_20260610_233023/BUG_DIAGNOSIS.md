# 버그 진단 — 26% 강제 NEI의 근본 원인

작성: 2026-06-11. 대상: `run_20260610_233023`. 방법: predictions 포렌식 + 코드 검증(실행 불요).

## 결론 (한 줄)

**tier 가중(논문 핵심 기여)이 "후보를 top-10으로 잘라낸 뒤"에 적용돼, 후보 풀 깊숙한 T1~T3 신뢰 소스를 검증 대상으로 끌어올리지 못한다. 그 결과 raw-유사도 top-10이 전부 T4가 되는 claim에서 controller R2가 발동해 강제 NEI가 된다.** 코드만으로 확정되는 구조적 버그이며, 재실행 없이 검증됨.

## 증상 (관측)

- dense run: 모든 RAG mode의 ~26%(101~103/394)가 `confidence=0`. **예외/크래시는 0개** → controller `_nei()` 라우팅.
- 그 NEI의 ~70%(68/97)는 회수 URL에 **T1~T3 신뢰 소스가 실제로 존재**하는데도 NEI. (예: claim 16은 T1 소스 8개 회수하고도 NEI)
- `proposed ≈ unweighted ≈ naive` 정확도 거의 동일(dense 0.444/0.452/0.449).
- 그런데 R@10은 갈림: unweighted 0.105 vs proposed 0.057.

## 근본 원인 (코드 검증)

두 retrieve 경로의 파라미터가 **불일치**:

| 경로 | 호출 | 동작 |
|---|---|---|
| **검증**(`experiment.predict_claim` L64) | `retrieve(k=10, candidate_n=50)` | raw 유사도 **top-10**만 받아 → 그 10개에 `rank_evidence` |
| **Recall 채점**(`experiment.ranked_topk_urls` L42) | `retrieve(k=50)` | **50개** 받아 → `rank_evidence` → top-10 |

- `retrieve`는 `survivors[:k]` 반환(retriever.py L113). 검증 경로는 k=10이므로 **가중 이전에 이미 top-10으로 절단**.
- `rank_evidence`(tier 가중, tier.py L138)는 받은 것만 재정렬 → **절단된 10개를 reorder만 할 뿐, 풀 깊숙한 고tier 소스를 승격 불가**.
- dense 유사도는 블로그·소셜이 상위에 잘 잡혀 raw top-10이 전부 T4가 되기 쉬움 → controller R2(`all(e.tier==4)`, controller.py L97) 발동 → `_nei()` → conf=0 강제 NEI.

## 이 하나로 3개 증상이 전부 설명됨

1. **26% 강제 NEI**: raw-sim top-10이 전부 T4 → R2. T1~T3이 풀에 있어도 못 봄.
2. **proposed≈unweighted≈naive (정확도)**: 검증 경로에서 가중은 같은 10개를 재정렬만 → verdict가 mode-무관 → **tier 기여가 무력화(neutered)**.
3. **proposed R@10 < unweighted**: Recall 경로(가중을 50개에 적용)에서는 가중이 작동하지만, gold URL을 강등 → R@10 하락. (검증 경로와 Recall 경로가 서로 다르게 동작하는 불일치)

## 영향 추정

- conf=0 폴백 NEI 97개 중 **90개가 gold≠NEI**(Refuted 60·Supported 27·Conflicting 3) → 손해. 약 70%(T1~T3 존재)는 버그 수정 시 **회복 가능** → acc 최대 +20%p대 잠재(상한, 재실행 필요).
- 29/68 분류는 근사치(Recall 경로 URL 기반). 정확한 R2 비율은 수정 후 재실행으로 확정.

## 수정 방향 (소규모)

검증 경로도 **후보 50개를 받아 → 가중 랭킹 → top-k 절단** 순서로 바꾼다(가중을 절단 전에 적용). 예: `predict_claim`의 `retrieve_fn`이 `k=candidate_n`로 받고, `controller.run`(또는 `rank_evidence)`이 가중 후 top-k(10)로 잘라 R2/verify에 전달.
- 효과: tier 기여가 비로소 작동(T1~T3 승격) → R2 오발동 급감, proposed≠unweighted 분리, RAG가 gpt_only와 공정 비교됨.
- 부수: Recall 경로와 검증 경로의 회수 정의를 일치시킬지 결정 필요.

## 재실행 판단

- **현재 RAG 숫자(naive/unweighted/proposed)는 무효** — 버그 증상이라 논문에 findings로 못 씀.
- 수정은 코드 ~10줄·원인 명확. **수정 → GPU 재실행(3~4h 비동기) → 그동안 숫자 무관 파트 집필 → 야간에 깨끗한 숫자 삽입**이 최적.
- gpt_only·dense vs bm25(검색기 자체)·agentic은 이 버그 영향 적음(단 agentic은 별도 재현성 5% 문제).

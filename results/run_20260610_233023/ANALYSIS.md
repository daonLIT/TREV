# 실험 결과 분석 — run_20260610_233023

작성: 2026-06-11. 대상: `results/run_20260610_233023/`
모델: `gpt-5.4-mini`, N=3 반복, AVeriTeC dev 서브셋 **394 claim**(agentic만 20).
원자료: `summary.txt`, `metrics_{dense,bm25,agentic}.json`, `agentic_ablation.json`, `predictions_*.json`.

---

## 0. 한눈에 보는 결과표

| method  | mode            | acc   | macroF1 | R@10  | P@10  | NEI예측 | avg_tools |
|---------|-----------------|-------|---------|-------|-------|--------|-----------|
| dense   | gpt_only        | 0.571 | 0.432   | 0.000 | —     | 43     | —         |
| dense   | naive_rag       | 0.449 | 0.368   | 0.105 | 0.026 | —      | —         |
| dense   | unweighted_rag  | 0.452 | 0.369   | 0.105 | 0.026 | —      | —         |
| dense   | proposed        | 0.444 | 0.367   | 0.057 | 0.014 | 123    | —         |
| bm25    | gpt_only        | 0.513 | 0.409   | 0.000 | —     | —      | —         |
| bm25    | naive_rag       | 0.429 | 0.354   | 0.067 | 0.017 | —      | —         |
| bm25    | unweighted_rag  | 0.431 | 0.355   | 0.070 | 0.018 | —      | —         |
| bm25    | proposed        | 0.406 | 0.334   | 0.041 | 0.010 | —      | —         |
| agentic | agentic         | 0.450 | 0.265   | 0.125 | 0.020 | 1      | 11.55     |

- gold 라벨 분포(394): Refuted **226(57.4%)** / Supported 112 / NEI 29 / Conflicting 27.
- **다수 클래스(전부 Refuted) 베이스라인 = 0.574**.

---

## 1. 결과를 보기 전 세운 가설

인수인계 문서(`TREV_handoff.md` §3·§4)에서 분석할 질문으로 명시했던 것을 가설 형태로 고정한다.

- **H1 (tier 가중 효과)**: 동적 tier 가중을 쓴 `proposed`가 `naive_rag`/`unweighted_rag`/`gpt_only`보다 acc·macroF1이 높을 것이다. (이 프로젝트의 핵심 기여)
- **H2 (dense vs bm25)**: dense 검색이 bm25보다 검색 품질(R@10)과 정확도가 높을 것이다.
- **H3 (RAG > LLM 단독)**: 근거를 붙인 RAG가 `gpt_only`보다 정확할 것이다. 단, handoff §4 경고대로 gpt_only가 사전지식으로 높게 나올 수 있어 효과는 시간민감·생소 claim에서만 드러날 수 있다.
- **H4 (agentic vs deterministic)**: 멀티에이전트(`agentic`)가 결정론 controller(`proposed`)보다 우수할 것이다.
- **H5 (agentic ± tier 도구)**: `rank_by_tier` 도구를 쓴 `agentic`이 `agentic_no_tier`보다 우수할 것이다.
- **H6 (재현성)**: N=3 반복 간 verdict 일치율이 높아 결과가 안정적일 것이다.

---

## 2. 봐야 하는 지표 (왜 보는가)

- **accuracy / macroF1** — 4라벨 분류 성능. 클래스 불균형(Refuted 57%)이 크므로 **macroF1이 주지표**, accuracy는 다수 클래스 베이스라인(0.574)과 비교해야 의미.
- **Recall@10 / Precision@10** — 검색 품질. **gold-URL 기준**이라 다른 유효 근거를 써도 실패로 잡힘(handoff §4 주의). 검색 모드 비교(H1·H2)에만 사용.
- **NEI 예측 수** — 과도 기권(abstention) 진단. 실패 claim NEI 폴백·검색 노이즈로 인한 기권을 잡아냄.
- **retrieval_breakdown(retrieved vs retrieval_failure)** — 394 중 gold-URL을 실제로 회수한 claim 수.
- **avg_tool_calls** — agentic 효율(예산) 지표.
- **agentic_agreement** — N=3 verdict 일치율, H6(재현성)의 직접 지표.
- **노이즈 하한**: `gpt_only`는 검색을 안 쓰므로 dense/bm25 런에서 **동일해야 하지만 0.571 vs 0.513**로 갈렸다. 이 **5.8pp**가 LLM 샘플링 노이즈의 실측 하한 → **약 6pp 미만 차이는 노이즈로 간주**.

---

## 3. 결과 해석 (가설별)

### H1 — tier 가중 효과: **기각 (오히려 악화)**
`proposed`(가중)가 모든 비교군보다 **낮다**.
- dense: proposed 0.444 / 0.367 vs unweighted 0.452 / 0.369 vs naive 0.449 / 0.368 → 거의 동률이나 proposed가 미세 열위.
- 특히 **R@10이 절반으로 떨어짐**: unweighted 0.105 → proposed 0.057 (dense), 0.070 → 0.041 (bm25). retrieval_failure도 322 → 353으로 증가.
- 즉 동적 tier 가중이 **gold-URL 회수를 깎아먹어** 검색·정확도 모두 손해. naive ≈ unweighted (Δ<0.01, 노이즈 내)라 **동적 tier 자체도 무효과**.

### H2 — dense vs bm25: **채택**
모든 동일 mode에서 dense ≥ bm25.
- gpt_only 0.571 vs 0.513, naive 0.449 vs 0.429, R@10도 dense가 일관되게 높음(0.105 vs 0.067).
- 검색기로는 **dense가 명확히 우세**. 다만 gpt_only 차이는 H2 근거가 아님(검색 무관 → 노이즈, §2 참조).

### H3 — RAG > LLM 단독: **기각**
모든 RAG mode가 `gpt_only`보다 **낮다**.
- dense: gpt_only 0.571 vs 최고 RAG(unweighted) 0.452 → **약 12pp 하락**(노이즈 하한 6pp의 2배, 유의미).
- gpt_only(dense) 0.571 ≈ 다수 클래스 베이스라인 0.574. 모델이 사실상 사전지식+다수 라벨로 베이스라인을 찍고 있고, **근거를 붙이면 오히려 떨어진다**.
- 원인 진단: `proposed`의 **NEI 예측이 43 → 123**으로 폭증(gold NEI는 29뿐). 검색 노이즈/폴백이 모델을 **과도 기권**으로 몰아 Refuted 정답을 NEI로 놓침. → handoff §4 경고("gpt_only가 높게, RAG 효과는 잘 안 드러남")가 그대로 재현.

### H4 — agentic vs deterministic: **부분 채택 (소표본)**
- agentic(20) acc 0.450, macroF1 0.265. 동일 20-subset ablation에서 **agentic macroF1 0.306 vs deterministic proposed 0.153, Δ+0.153** → 어려운 subset에서 agentic이 결정론을 크게 앞섬.
- 단 **n=20**이라 신뢰구간 매우 넓음. 전체 394 proposed(0.367)와 직접 비교는 subset이 달라 불가.
- agentic은 R@10 0.125로 검색 mode 중 최고(다만 P@10 0.02, **tool 11.55회/claim**로 고비용).

### H5 — agentic ± tier 도구: **기각 (무효과)**
ablation(20): agentic macroF1 0.306 vs agentic_no_tier 0.310, **Δ-0.0048**.
- tier 도구를 빼도 성능 동일(미세 우위). avg_tools는 오히려 no_tier가 더 많음(12.9 vs 11.0).
- **H1과 같은 결론**: tier 가중/도구는 이 데이터셋·모델에서 verdict 정확도에 기여하지 않음.

### H6 — 재현성: **기각 (심각한 불안정)**
- `agentic_agreement = 0.05` (N=3, n=20). **20개 중 19개 claim이 3회 실행에서 verdict 불일치**.
- agentic 멀티에이전트 경로는 **재현성이 거의 없음**. 단일 실행 결과를 결론으로 쓰면 위험.

---

## 4. 인사이트 / 시사점

1. **현재 세팅에서 RAG가 LLM 단독을 못 이긴다 (핵심 음성 결과).** gpt-5.4-mini의 사전지식만으로 다수 클래스 베이스라인(0.574)에 도달하고, 검색 근거를 더하면 과도 기권(NEI 43→123)으로 오히려 12pp 하락. → "근거를 붙이는 행위"가 아니라 "**근거 품질**"이 문제. R@10 최고가 0.125(gold-URL 기준)인 점이 이를 뒷받침.

2. **프로젝트의 핵심 기여인 동적 tier 가중이 효과 없음 — 오히려 검색을 망친다.** proposed가 R@10을 절반으로 깎고(0.105→0.057) retrieval_failure를 늘림. naive≈unweighted까지 고려하면 tier 관련 기여 전체가 무효~음성. **가중 로직/캡(`max_chunks_per_url`) 재점검**이 시급(handoff §5 cap 튜닝과 연결).

3. **gold-URL 기준 Recall이 성능을 가린다.** handoff §4대로 모델이 다른 유효 근거를 써도 retrieval_failure로 집계됨(394 중 322~368이 실패). 진짜 검색 효과를 보려면 **근거 정합성을 사람/LLM-judge로 재채점**하거나 NEI 폴백을 분리 집계해야 한다.

4. **dense는 채택, bm25는 폐기 후보.** 검색기 선택은 명확히 dense.

5. **agentic은 가능성은 보이나 (결정론 대비 +0.15 macroF1) 두 가지 큰 결함**: (a) **재현성 5%** — 평가 신뢰 불가, (b) **claim당 tool 11.5회** 고비용. 다음 단계는 전체 394 확장 전에 **재현성 안정화**(temperature·종료조건·tool 예산 통제)가 선결.

6. **노이즈 하한 ~6pp를 분석 기준선으로.** gpt_only가 런 간 5.8pp 흔들렸으므로, 이보다 작은 mode 간 차이(naive vs unweighted 등)는 **노이즈로 보고 결론 내지 말 것**. N을 키우거나 동일 시드 고정 필요.

---

## 5. 후속 작업 제안 (우선순위)

1. **NEI 폴백 vs 실제 기권 분리 집계** — proposed의 NEI 123개 중 폴백("error:")이 몇 개인지 확인해 H3 하락이 검색 품질 탓인지 폴백 탓인지 규명.
2. **tier 가중 로직 디버그** — proposed가 R@10을 깎는 원인(가중이 gold-tier를 눌렀는지) 추적. cap 튜닝 실험(handoff §5)과 병행.
3. **agentic 재현성 고정** 후 소표본(n≥50) 재실행 → 그 다음 전체 394 확장.
4. **근거 정합성 재채점**(LLM-judge/RAGAS, handoff §5) — gold-URL Recall의 과소평가 보정.

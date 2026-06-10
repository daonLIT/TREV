# TREV — Tier-weighted Retrieval for EVidence-based verification

AVeriTeC(dev) 위에서 **출처 신뢰도(tier)를 가중한 근거 검색**으로 주장의 사실성을 판정하는
연구 프레임워크. 두 가지 검증 시스템을 **동일한 데이터·검색·평가 하니스**로 비교한다:

- **결정론 파이프라인** — controller가 규칙(R1~R5)으로 검색·검증을 라우팅 (baseline).
- **TREV-Agent (멀티에이전트)** — planner/searcher/verifier가 native tool-calling으로 자율 조율.
  기존 능력(검색·tier 가중·verify)을 **도구**로 노출하고, 결정론과 head-to-head로 측정한다.

> 핵심 기여: **동적 tier 가중**(claim_type × evidence_role × source_domain → tier·weight).
> 자기출처(claim 자체 출처) 매칭 시 T4로 강등 — "동적 tier의 실제 발화".

---

## 패키지 구조

```
trev/
  schemas.py  config.py  llm.py  guards.py     # foundation (공유 계약·게이트웨이·위생 가드)
  experiment.py                                 # 실험 실행기 (pipeline+agent → eval 결선)
  data/        dataset · knowledge_store · indexing       # AVeriTeC 데이터 계층 (적재·KS·인덱싱)
  pipeline/    retriever · tier · verifier · controller   # 결정론 검증 (baseline)
  agent/       tools · agent · orchestrator               # 멀티에이전트 (TREV-Agent)
  eval/        metrics · recall · topics · ablation · annotation · auxmetrics
scripts/       run_experiments · run_eval · agentic_ablation · build_index · …
tests/         29개 파일, 201 테스트 (외부 행동만 검증, 실 LLM/임베딩 불요)
```

| 계층 | 역할 |
|---|---|
| `data/` | `dev.json`(500 claim) 적재·필터, KS(per-claim JSONL) 로더, per-claim FAISS+BM25 인덱싱(영속화) |
| `pipeline/` | 검색(시점필터)·동적 tier 가중 랭킹·verifier·controller R1~R5 |
| `agent/` | 도구 레지스트리(기존 능력 래핑) + base 루프 + planner/searcher/verifier orchestrator |
| `eval/` | Acc·Macro-F1·Recall@k·검색실패vsNEI·인용율·토픽분해·ablation·사람주석·보조지표 |

---

## 빠른 시작

```bash
python -m venv .venv && .venv/Scripts/activate      # (Windows)
pip install -r requirements.txt                      # 검색은 faiss-cpu·rank-bm25, 실 e5는 sentence-transformers
cp .env.example .env                                 # HAI_GPT_API_KEY 채우기
```

`.env` (OpenAI 호환 HAI-GPT 게이트웨이, 모델 `gpt-5`):
```
HAI_GPT_API_KEY=...
HAI_GPT_BASE_URL=https://factchat-cloud.mindlogic.ai/v1/gateway
```

데이터(`data_store/averitec/`, `knowledge_store/dev/`)는 AVeriTeC에서 직접 적재 — **재배포 금지**라
`.gitignore`로 제외(코드 `trev.guards`가 강제). test/`test_2025`/train은 로드 차단(blind).

---

## 검증 조건 (동일 GPT-5·KS·프롬프트로 통제)

| 조건 | 검색 | 동적 tier | 가중 랭킹 | 라우팅 |
|---|---|---|---|---|
| `gpt_only` | 없음 | - | - | claim만으로 라벨 직출 |
| `naive_rag` | dense/bm25 | ✗(도메인 tier만) | ✗ | controller R1~R5 |
| `unweighted_rag` | dense/bm25 | ✓ | ✗ | controller R1~R5 |
| `proposed` | dense/bm25 | ✓ | ✓ (sim×weight) | controller R1~R5 |
| `agentic` | 도구 | ✓ | ✓ | 멀티에이전트 orchestrator |
| `agentic_no_tier` | 도구 | ✗ | ✗ | 멀티에이전트 (tier 도구 off) |

ablation 축: **dense vs bm25**(검색), **proposed vs unweighted vs naive**(tier 효과),
**agentic vs deterministic**(자율성), **agentic ± rank_by_tier 도구**(에이전트 맥락 tier 효과).

---

## 실행 (scripts)

```bash
python -m scripts.subset_gate            # 서브셋 크기·Conflicting 카운트·정량/정성 분기
python -m scripts.inspect_ks             # KS 구조 점검(G0) → docs/ks_structure.md
python -m scripts.domain_freq            # 도메인 빈도 → docs/domain_frequency.md (tier 화이트리스트 입력)
python -m scripts.run_experiments --method dense          # 4 baseline 조건 → outputs/predictions_dense.json
python -m scripts.run_experiments --agentic               # agentic 조건 → outputs/predictions_agentic.json
python -m scripts.run_eval --method dense                 # 채점표 → outputs/metrics_dense.json
python -m scripts.agentic_ablation --limit 5 --n 3        # tier on/off · agentic vs det · N=3 일치율
python -m scripts.export_annotations     # 토픽 κ(200) + Conflicting 5라벨 CSV (사람 주석)
```

per-claim 인덱스는 **빌드 1회→디스크 영속화**(`index/`), 재실행 시 로드만(재임베딩 0).

---

## 확정된 데이터 사실 (G0 실데이터 점검)

- `dev.json` **500 claim**, 서브셋(Numerical+Event/Property) **394**, Conflicting **27**(≥20 → 정량 모드).
- claim ↔ KS 파일 = **위치 인덱스**(`{index}.json`), `claim_id`는 **문자열**.
- KS `type` **14종**(전부 url+url2text 웹문서); `url2text`는 **페이지 라인**(claim당 ~수십만) → URL별
  **재청킹 + 캡** 필요(config `index.max_chunks_per_url`).
- `published_at` 없음 → web.archive.org 스냅샷에서 유도. 단 아카이브는 **gold에 집중** →
  검색 코퍼스 커버리지 ~0%, **시점필터는 대부분 통과**(누수 안전판으로 유지).
- gold 근거 URL = `questions[].answers[].source_url`; Recall@k는 **URL 정규화**(아카이브 prefix·스킴·
  www·쿼리 제거) 후 매칭. 검색실패(gold가 KS에 있는데 미회수) vs 실제 NEI 구분.

---

## 테스트

```bash
python -m pytest                                   # 전체 (실 API 스모크는 키 없으면 skip)
python -m pytest --ignore=tests/test_smoke_gpt5.py --ignore=tests/test_smoke_s2.py --ignore=tests/test_smoke_tool_calling.py
```

좋은 테스트 = **외부 행동만 검증**. 가짜 LLM(스크립트된 응답·tool_calls)·가짜 임베더 주입으로 실
GPT-5·실 임베딩 없이 라우팅·도구·채점을 검증한다. 게이트: G0(KS 구조)·G1(GPT-5 ping)·
G-agent(native tool-calling) 스모크.

---

## 비용·운영 메모

- tool-calling 1턴 ~40s(GPT-5 실측). **agentic은 claim당 도구호출 多**(예: ~14회) → 결정론(1~2회)보다
  훨씬 무거움. `agent`는 step·도구호출 **예산 상한**으로 통제, trace에 비용 기록.
- CPU e5-large는 180단어 청크당 ~0.5~1s → **전체 394 인덱싱은 CPU로 ~수시간~1일**(영속화로 1회).
  대규모 실험은 **GPU 권장**.

## 라이선스 / 인용

데이터: **AVeriTeC, CC BY-NC 4.0** (비영리·출처표시, 재배포 금지).
Schlichtkrull, Guo, Vlachos. *AVeriTeC*. NeurIPS D&B, 2023.

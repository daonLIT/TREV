# TREV — 데이터 사용 가이드 (AVeriTeC)

> 짝 문서: `PRD_TREV.md`, `TREV_시스템구성.md`
> 목적: PRD에 맞춰 **AVeriTeC dev 데이터를 TREV 각 모듈이 정확히 어떻게 소비하는지**를 규정한다.
> 원칙: **per-claim 검색 · 시점 통제 · dev-only · KS 위 실검색**(gold QA 직접 투입 금지).

---

## 0. 데이터 인벤토리 (받은 상태)

| 경로 | 내용 | TREV 용도 |
|---|---|---|
| `data_store/averitec/dev.json` | dev 500개 주장 + gold(label·근거 QA·justification) + 메타 | 평가 기준·① 필터·토픽·시점 기준 |
| `data_store/averitec/train.json` | train 주장 | few-shot 프롬프트 예시(선택) |
| `knowledge_store/dev/` | claim별 후보 문서 풀 (500 files, 1 claim ↔ 1 file) | **검색 코퍼스 (per-claim 인덱싱 대상)** |

> `test_2025`/train KS는 **사용 안 함**(Out of Scope). git에는 `data_store/`·`knowledge_store/` 모두 **.gitignore**.

---

## 1. ⚠️ 먼저 통과할 검증 게이트 (G0 — 구현 착수 전 1회)

실데이터로 아래 2개를 **반드시 먼저 확인**한다. 결과에 따라 시점필터·Recall@k 구현이 갈린다.

```python
import json, glob, os

dev = json.load(open("data_store/averitec/dev.json", encoding="utf-8"))
print("claims:", len(dev), "| keys:", list(dev[0].keys()))

# (G0-a) KS 파일 1개 구조 — JSONL? 필드명? published_at 있나?
f = sorted(glob.glob("knowledge_store/dev/*"))[0]
print("KS file:", f)
line = open(f, encoding="utf-8").readline()
rec = json.loads(line)
print("KS record keys:", list(rec.keys()))   # 보통 url, url2text ... 'published'/'date' 있는지 확인

# (G0-b) gold QA source_url 이 그 claim의 KS 안에 들어있나? (Recall@k 가능성)
def ks_urls(path):
    s=set()
    for ln in open(path, encoding="utf-8"):
        try: s.add(json.loads(ln).get("url"))
        except: pass
    return s
c0 = dev[0]
gold = {a["source_url"] for q in c0.get("questions",[]) for a in q.get("answers",[]) if a.get("source_url")}
ks = ks_urls(f)
print("gold URLs:", len(gold), "| KS URLs:", len(ks), "| 교집합:", len(gold & ks))
```

판단:
- **G0-a `published_at`가 없으면** → 시점필터는 "날짜 있는 문서만 적용, 없으면 통과"로 구현하고, 시점 통제 기여는 "AVeriTeC가 구축단계서 누수 통제 + 가능한 범위 보강"으로 약하게 서술.
- **G0-b 교집합이 거의 0이면** → exact URL Recall@k는 무의미. **URL 정규화**(스킴·www·쿼리 제거) 후 매칭, 그래도 낮으면 **텍스트 근사 매칭** 또는 Ev2R로 전환.

---

## 2. `dev.json` 필드 → TREV 사용 매핑

각 claim 객체에서 TREV가 쓰는 필드:

| 필드 | 타입 | TREV 사용처 | 관련 PRD |
|---|---|---|---|
| `claim` | str | 검증 대상 텍스트, 질의 생성·verifier 입력 | US1 |
| `claim_types` | list[str] | **① 서브셋 필터** (Numerical·Event/Property) | US2,3 |
| `claim_date` | str (d-m-yyyy) | **`T_claim`= `T_evidence`** (시점 필터 기준) | US4, 결정98 |
| `label` | str | **gold 4라벨** (채점 기준) | US5 |
| `questions[].answers[].source_url` | str | **gold 근거 URL** (Recall@k 기준) | US40 |
| `questions[].answers[].answer` | str | gold 근거 텍스트(근사 매칭·QA 어댑터 stretch) | US47 |
| `speaker`, `reporting_source` | str | **토픽 태깅 보조 입력** + tier 자기출처매칭 | US6, 결정105 |
| `original_claim_url` | str/null | **tier `role=target` 강등 매칭** | US19, 결정105 |
| `location_ISO_code` | str/null | 토픽 태깅 보조 | US6 |
| `justification` | str | (보조) gold 설명 — 직접 학습엔 미사용 | — |

> **주의(데이터 위생)**: `questions`(gold QA·근거)는 **검색 대상이 아니라 채점용 정답**이다. 검색은 반드시 `knowledge_store/dev/`에서 한다(US 96/156). gold를 검색에 넣으면 실험이 무효.

### 2.1 claim_date 정규화
AVeriTeC 날짜는 `"25-8-2020"`(d-m-yyyy)·결측 등 들쭉날쭉.
```python
from datetime import datetime
def parse_date(s):
    if not s: return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try: return datetime.strptime(s.strip(), fmt).date()
        except: pass
    return None     # 결측 → 시점필터 통과 처리
```

### 2.2 ① claim type 필터 (적재 직후)
```python
KEEP = {"Numerical Claim", "Event/Property Claim"}        # 부족 시 "Quote Verification" 추가
sub = [c for c in dev if set(c.get("claim_types",[])) & KEEP]
print("서브셋:", len(sub), "/", len(dev))
# 게이트: Conflicting/Cherry-picking gold 개수 → 5라벨 평가 분기 (US9,44)
n_conf = sum(c["label"]=="Conflicting Evidence/Cherrypicking" for c in sub)
print("Conflicting gold:", n_conf, "→", "세분라벨 F1" if n_conf>=20 else "정성 사례연구")
```

---

## 3. `knowledge_store/dev/` 사용 (per-claim, 핵심)

### 3.1 포맷 (G0에서 확정)
- 파일 1개 = claim 1개의 **후보 문서 풀**. 보통 **JSONL**, 각 줄 ≈ `{"url": ..., "url2text": [text, ...]}` (필드명은 G0로 확정).
- claim ↔ 파일 매칭 키 확인: 파일명이 dev 순서 인덱스인지, claim_id인지 G0에서 확인해 매핑 규칙 고정.

### 3.2 per-claim 인덱싱 (전역 인덱스 금지)
PRD 결정 97: **claim마다 그 claim의 KS 문서만** 인덱싱한다.
```
for claim in subset:
    docs = load_ks(claim)                       # 그 claim의 후보 문서들
    chunks = [chunk(d.text) for d in docs]       # 긴 문서 → passage 분할
    meta   = {url, source_domain, published_at}  # tier·시점필터·Recall@k용 보존
    faiss_index = embed_e5(chunks)               # per-claim FAISS
    bm25_index  = build_bm25(chunks)             # dense vs lexical ablation
```
- **보존 메타(필수)**: `url`, `source`(도메인), `published_at`(있으면). → tier 부여·시점필터·Recall@k가 전부 이 메타에 의존.
- 전역 단일 인덱스를 만들면 다른 claim·다른 시점 문서가 섞여 **시점 누수**가 난다 → 금지.

### 3.3 검색 시 데이터 흐름
```
질의(2~3개, claim type별) → per-claim 인덱스 top-N(=50)
   → 시점필터(published_at <= T_evidence; 결측은 통과)
   → assign_tier(claim_type, role, source_domain)  # role=target 강등 포함
   → score = sim*weight (proposed) / sim (unweighted) / top-k 그대로 (naive)
   → top-k(=5)
```

---

## 4. 동적 tier가 쓰는 데이터 (결정론)

`assign_tier(claim_type, evidence_role, source_domain)`의 입력 출처:

| 입력 | 데이터 출처 |
|---|---|
| `source_domain` | KS 문서 `url`의 도메인 추출 (`urllib.parse`) |
| base tier | **도메인 화이트리스트**(`.gov`/통계/`court`→T1, 팩트체크→T2, 주요언론→T3, 소셜·블로그→T4) — `config.yaml`에 고정 |
| `role=target` 강등 | KS 문서 도메인이 claim의 `original_claim_url`/`reporting_source` 도메인과 **일치하면 T4** |
| long-tail 보정 | 서브셋 실등장 고유 도메인 수동 감수 → config 오버라이드, 잔여만 LLM 1회 분류+캐시 |

> 화이트리스트를 만들려면 **서브셋에 실제 등장하는 고유 도메인 목록**을 먼저 뽑아야 한다:
> ```python
> from urllib.parse import urlparse
> doms = {}
> for c in subset:
>     for ln in open(ks_path(c), encoding="utf-8"):
>         u = json.loads(ln).get("url","")
>         d = urlparse(u).netloc
>         doms[d] = doms.get(d,0)+1
> # 빈도순 상위 도메인을 사람이 보고 T1~T4 매핑 → config.yaml
> ```

---

## 5. 평가가 쓰는 데이터

| 지표 | 사용 데이터 |
|---|---|
| Label Accuracy · Macro-F1(5/4) | 예측 `label5`→4 매핑 vs `dev.json` gold `label` |
| Evidence Recall@k / Precision@k | 검색 top-k 문서 `url` vs gold `questions[].answers[].source_url` (정규화 후) |
| 검색실패 vs 실제 NEI | gold 근거 URL이 KS에 있는데 미회수=검색실패 / gold label이 NEI=실제부재 |
| 인용율 | 예측 `cited` 비어있지 않은 비율 |
| 토픽별 분해 | 토픽 태그 × 위 지표 |
| (stretch) AVeriTeC score | cited 근거로 QA 생성 어댑터 → `evaluate_veracity.py` 근사 |

5→4 매핑: `PARTIAL, CONFLICT → "Conflicting Evidence/Cherrypicking"`, 나머지 1:1.
> gold label 문자열은 데이터에서 정확히 확인할 것(예: `"Conflicting Evidence/Cherrypicking"` 철자).

---

## 6. 데이터 위생·라이선스

- **dev만 평가**, train은 few-shot 예시에만, **test 미사용**(blind).
- 검색은 **KS 위에서만** — gold QA·외부 라이브 웹 금지(재현성).
- `.gitignore`: `data_store/`, `knowledge_store/`, `index/`, `outputs/`, `.venv/`, `.env`.
- **AVeriTeC = CC BY-NC 4.0** — 비영리·출처표시, **데이터 재배포 금지**. 논문에 인용:
  `Schlichtkrull, Guo, Vlachos. AVeriTeC. NeurIPS D&B, 2023.`

---

## 7. 요약 — "데이터로 뭘 먼저 하나" 순서

1. **G0 확인**(§1): KS 포맷·`published_at` 유무·gold URL 포함률 → 시점필터/Recall@k 방식 확정
2. **적재+① 필터+claim_date 정규화**(§2) → 서브셋 + Conflicting gold 카운트 게이트
3. **서브셋 도메인 목록 추출**(§4) → 화이트리스트 `config.yaml` 작성
4. **토픽 태깅**(키워드→LLM, 200건 κ 검수)
5. **per-claim 인덱싱**(§3): 청크→e5 임베딩→FAISS + BM25, 메타(url·도메인·날짜) 보존
6. 이후 검색→tier→검증→평가는 위 데이터·메타를 그대로 사용

> 1·2·3번은 LLM 없이 가능 → **GPT-5 ping 게이트(G1) 전에 먼저 끝낼 수 있는 작업**.

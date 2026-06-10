# HANDOFF — GPU PC에서 전체 실험 실행

대상: SSH로 접속하는 **GPU PC**. **개발 없이 실험만** 실행한다(394 claim × baseline 4 + agentic).
이 문서 하나로 셋업 → 데이터 전송 → 실행 → 채점까지 끝낸다.

> 짝 문서(레포에 포함): `README.md`(시스템 전반), `PRD_TREV.md`/`PRD_TREV_DATA.md`/`PRD_TREV_AGENT.md`,
> GitHub 이슈 `daonLIT/TREV` #1~#29. 코드 구조·조건·지표는 `README.md` 참조 — 여기서 중복하지 않는다.

---

## 0. 한눈에

| 무엇 | 어떻게 |
|---|---|
| **코드** | `git clone https://github.com/daonLIT/TREV.git` (전부 커밋됨) |
| **데이터·키** | git에 **없음** → 아래 표대로 **수동 전송**(가장 큰 건 knowledge_store 35GB) |
| **환경** | venv + `requirements.txt` + **CUDA torch**(GPU 임베딩) |
| **실행** | `scripts.run_experiments` → `scripts.run_eval` (인덱스 1회 빌드·영속화) |

---

## 1. git에 없어서 수동 전송해야 하는 파일 (`.gitignore` 기준)

`.gitignore`가 데이터·비밀키·산출물을 제외한다(AVeriTeC 재배포 금지 + 보안). 실험에 **필수**인 것:

| 경로 | 크기 | 필수? | 비고 |
|---|---|---|---|
| `knowledge_store/dev/` (500개 `{i}.json`) | **~35 GB** | **필수** | 검색 코퍼스. 전송 핵심 — rsync 권장(이어받기) |
| `data_store/averitec/dev.json` | 1.7 MB | **필수** | 500 claim + gold |
| `data_store/averitec/train.json` | 10 MB | 불필요 | few-shot 전용 — 실험 코드가 로드 안 함 |
| `.env` | <1 KB | **필수** | `HAI_GPT_API_KEY` — **이 문서/깃에 키 넣지 말 것**. 직접 생성 또는 안전 전송 |
| `index/` | 8.9 MB | 불필요 | per-claim 인덱스 캐시 — **GPU에서 재생성**(전송 X) |
| `outputs/` | - | 불필요 | 실행 시 생성됨 |

> **실험은 35GB 전부를 쓰지 않는다**: ① claim은 서브셋 **394/500**만 사용(나머지 106은 KS 파일도
> 로드 안 함) ② 각 URL은 페이지 lead만 인덱싱(`max_chunks_per_url`, 5절). 단 사용하는 파일은
> **전체가 있어야 함**(로더가 URL 목록을 파일 전체에서 추출). 전송을 줄이려면 서브셋 394 파일만 보내도
> 되지만(106개·~7GB 절약), 간단히는 500개 전부 보내는 게 안전(Quote 토글 시 455개 필요).

전송 안 해도 되는 것(개발 전용·재생성): `.venv/`, `.claude/`, `.agents/`, `CLAUDE.md`,
`TREV_시스템구성.md`, `*.docx`, `hai-gpt 사용문서.md` 등.

### 전송 예시 (로컬 → GPU PC)
```bash
# 35GB는 rsync(중단 시 이어받기). 압축 전송이 더 빠를 수 있음.
rsync -avP knowledge_store/  user@gpu-pc:/path/TREV/knowledge_store/
rsync -avP data_store/       user@gpu-pc:/path/TREV/data_store/
# .env는 키 포함 → 안전 채널로 전송하거나 GPU PC에서 직접 작성(아래 3단계)
```

---

## 2. 코드 가져오기 (GPU PC)

```bash
git clone https://github.com/daonLIT/TREV.git && cd TREV
# (이미 클론돼 있으면) git pull
```
전송한 `knowledge_store/`·`data_store/`를 레포 루트에 그대로 둔다(`.gitignore`라 git이 무시).

---

## 3. 환경 셋업 (GPU)

```bash
python -m venv .venv && source .venv/bin/activate     # Linux는 bin/activate
# 1) CUDA용 torch 먼저 (CUDA 버전에 맞는 인덱스로 — 예: cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) 나머지 의존성 (faiss-cpu·rank-bm25·sentence-transformers·openai 등)
pip install -r requirements.txt
```

`.env` 작성(키는 직접):
```bash
cp .env.example .env
# .env 편집: HAI_GPT_API_KEY=<본인 키>, HAI_GPT_BASE_URL은 그대로
```

GPU 인식 확인:
```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"   # True 여야 함
```
e5(`sentence-transformers`)는 CUDA가 보이면 **자동으로 GPU 사용**(코드 변경 불요).

### 셋업 검증 (실 데이터/키 없이도 통과)
```bash
python -m pytest --ignore=tests/test_smoke_gpt5.py --ignore=tests/test_smoke_s2.py --ignore=tests/test_smoke_tool_calling.py   # 201 통과
python -m scripts.inspect_ks        # KS 전송 정상 확인(파일 500, 구조 점검)
python -m scripts.subset_gate       # 서브셋 394 / Conflicting 27 출력되면 데이터 OK
# 키 채운 뒤 게이트:
python -m pytest tests/test_smoke_gpt5.py tests/test_smoke_tool_calling.py -v   # G1·G-agent
```

---

## 4. 실험 실행

인덱스는 **빌드 1회 → `index/`에 영속화**, 이후 조건·재실행은 로드만(재임베딩 0).

```bash
# 1) 결정론 4조건 (dense). 첫 실행이 394 claim 인덱스를 빌드·캐시.
python -m scripts.run_experiments --method dense      # → outputs/predictions_dense.json
# 2) BM25 ablation (캐시 인덱스 재사용)
python -m scripts.run_experiments --method bm25       # → outputs/predictions_bm25.json
# 3) 채점
python -m scripts.run_eval --method dense
python -m scripts.run_eval --method bm25
# 4) agentic 조건 (아래 비용 주의)
python -m scripts.run_experiments --agentic           # → outputs/predictions_agentic.json
python -m scripts.run_eval --method agentic
# 5) 종합 ablation: tier on/off · agentic vs deterministic · N=3 일치율
python -m scripts.agentic_ablation --limit 50 --n 3   # → outputs/agentic_ablation.json
```

스모크: 먼저 `--limit 3`으로 파이프라인 확인 후 전체 실행 권장.
청크 캡은 `config.yaml`의 `index.max_chunks_per_url`(현재 3, 페이지 lead). GPU라 임베딩이
빠르니 **캡을 올려 Recall 변화를 측정**해볼 수 있음(#21 Recall@k로 검증).

---

## 5. (선택·권장) 청크 cap 튜닝 실험 — "본문을 더 쓸지" 데이터로 결정

현재 `config.yaml`의 `index.max_chunks_per_url: 3`은 각 URL의 **페이지 lead(~540단어)만** 인덱싱한다
(CPU 한계 + boilerplate 노이즈 회피). **GPU면 더 많은 본문 임베딩이 시간상 가능**하므로, cap을 올려
Recall@k·accuracy가 **실제로 좋아지는지** 측정해 정한다(추측 금지 — #21/#9가 답함).

GPU 임베딩 비용(대략):

| `max_chunks_per_url` | claim당 청크 | 394 전체 | GPU 임베딩(인덱스 빌드) | 인덱스 디스크 |
|---|---|---|---|---|
| 3 (현재) | ~1,900 | ~75만 | 수분 | ~9 GB |
| 20 | ~12,000 | ~470만 | ~1시간 | ~수십 GB |
| `null`(무제한) | ~40,000 | ~1,600만 | ~수시간 | ~60 GB+ |

절차:
```bash
# cap 바꾸면 캐시 인덱스 무효 → 재빌드 필요(기존 cap으로 빌드됨).
# config.yaml에서 max_chunks_per_url 수정 후:
rm -rf index/                                   # 또는 run_experiments에 --no-cache
python -m scripts.run_experiments --method dense
python -m scripts.run_eval --method dense       # Recall@k / accuracy 비교
```
판단: Recall@k가 **오르면** 더 큰 cap 채택, **노이즈만 늘면**(precision↓, Recall 정체) cap 유지.

주의: cap은 **인덱싱(임베딩)** 비용만 바꾼다. **GPT-5 호출 수·verify 시간은 그대로**라 실험 wall-time의
주 병목(아래 6절)은 cap과 무관하다. 즉 "GPU로 다 넣자"가 아니라 "Recall이 오르는 만큼만 넣자".

---

## 6. ⚠️ 비용·시간 (정직하게)

**GPU가 빠르게 하는 건 e5 인덱스 빌드뿐.** 실험 wall-time은 **GPT-5 API 지연**이 지배한다(GPU 무관):

- **결정론 4조건**: claim당 ~1 verify 호출 × 394 × 조건 ≈ **수천 호출(~7s/호출) → ~수시간**. 감당 가능.
- **agentic**: claim당 tool-calling 多턴(예 ~14 호출), **1턴 ~40s** → 394 전체는 **수십 시간~며칠**.
  → 권장: agentic은 **표본(`--limit`)**으로 돌리거나, 백그라운드 장시간 실행.
- 인덱스 빌드(394): GPU에서 수십 분 내(영속화로 1회만).

산출물은 `outputs/predictions_*.json`·`metrics_*.json`·`agentic_ablation.json`. 재현·재채점 가능.

---

## 7. 잘 빠지는 함정

- **경로**: 코드는 `pathlib`라 OS 무관. venv 활성화만 Linux는 `source .venv/bin/activate`.
- **`.env` 미작성** → `LLMError: HAI_GPT_API_KEY가 없음`. 키 채우면 해결.
- **knowledge_store 누락/부분 전송** → `subset_gate`/`inspect_ks`가 실패하거나 파일 수<500. rsync 재실행.
- **train/test 로드 차단**: `trev.guards`가 dev만 허용(의도된 동작). test/test_2025/train 접근 시 예외.
- **agentic 비용 폭주 방지**: `orchestrate`의 `max_tool_calls`·`max_steps` 예산 상한이 통제(기본 24/12).

---

## 8. Suggested skills (다음 에이전트용)

이 PC는 **실험 실행 전용**이라 개발 스킬은 불필요. 도움이 될 수 있는 것:

- `/verify` — 변경 없이 실행 결과(예측·채점)가 기대대로인지 확인할 때.
- (개발이 필요해지면) `/code-review`, `/simplify` — 단, 이 PC는 실험만 하기로 했으므로 보통 불필요.

가장 정확한 다음 행동: **3단계 셋업 검증 → 4단계 실행**. 막히면 `README.md`·이슈(#1~#29) 참조.

---

## 참조

- 시스템 전반: `README.md`
- PRD: `PRD_TREV.md`(결정론) · `PRD_TREV_DATA.md`(데이터 계약) · `PRD_TREV_AGENT.md`(멀티에이전트)
- 이슈: https://github.com/daonLIT/TREV/issues (#1~#29, 전부 구현·테스트 완료)
- G0 데이터 점검: `docs/ks_structure.md` · `docs/domain_frequency.md`

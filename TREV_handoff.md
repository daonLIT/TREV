# 인수인계 — TREV-Agent (다음 세션용)

작성: 2026-06-10. 다음 세션 초점: **GPU PC에서 도는 전체 394 실험 결과 분석** 또는 작업 이어받기.

> 중복 금지: 시스템 설명은 `README.md`, 설계는 `PRD_TREV.md`/`PRD_TREV_DATA.md`/`PRD_TREV_AGENT.md`,
> 작업 내역은 GitHub 이슈 `daonLIT/TREV` #1~#29, GPU 실행 절차는 레포의 `HANDOFF_GPU.md` 참조.
> 이 문서는 **현재 진행 상황 + 다음 할 일**만 담는다.

---

## 1. 지금 무슨 상태인가

- **TREV-Agent 멀티에이전트 프레임워크 완성** (이슈 #22 PRD, 슬라이스 A1~A7 = #23~#29 전부 구현·테스트).
- 코드/문서 전부 커밋·푸시됨. GPU PC도 `git pull` 완료.
- **전체 394 실험이 GPU PC에서 실행 중** (이 대화 끝 시점에 사용자가 켜놓고 감):
  - 머신: 사용자 GPU PC(NVIDIA **GB10**, Linux, 경로 `~/TREV` 추정 `/home/litailab01/daon/TREV`), SSH 접속.
  - 실행: `tmux` 세션 **`trev`** 안에서 `bash run_all.sh`.
  - 명령: `LANGSMITH_TRACING=false AGENTIC_LIMIT=394 ABLATION_LIMIT=50 N=3 WORKERS=8 bash run_all.sh`
  - 모델: **`gpt-5.4-mini`** (config.yaml). LangSmith 추적 **off**(실험 중엔 끔).
  - 예상: ~3-4시간, ~11,700 크레딧 (예산 ~16,500 남았었음).
  - 결과 위치: `results/run_<타임스탬프>/` (predictions_*.json, metrics_*.json, agentic_ablation.json, summary.txt, run.log).

### 결과 확인 방법 (다음 세션 첫 행동)
```bash
# GPU PC에서
tmux attach -t trev                  # 도는 중이면 진행률, 끝났으면 "===== 완료 ====="
cat  results/run_*/summary.txt       # 조건별 acc·macroF1·R@10·avg_tools
tail -30 results/run_*/run.log
```
결과 JSON을 이 PC로 가져오려면 `scp -r litailab01@GPU_IP:~/TREV/results ./results` 또는 git.

---

## 2. 프로젝트 궤적 (왜 지금 이 모습인가)

1. 원래 **TREV(결정론)**: AVeriTeC dev 위 tier 가중 검색 + controller R1~R5. 데이터계층·검증 파이프라인·평가·ablation까지 구현(이슈 #1~#21, 슬라이스 S0~S12 / D1~D6).
2. **에이전트 기반으로 피벗**: 사용자가 "에이전트 프레임워크"로 목표 변경 → PRD #22 작성 → **멀티에이전트(planner/searcher/verifier + orchestrator, native tool-calling)** 를 A1~A7로 구현. 결정론 controller는 **baseline으로 보존**, 핵심 기여(동적 tier 가중)는 `rank_by_tier` **도구로 보존**.
3. **구조 정리**: `trev/`를 `data/ pipeline/ agent/ eval/` 서브패키지로 재편(foundation: schemas/config/llm/guards + experiment는 top-level).
4. **실험 인프라**: per-claim 인덱스 영속화, GPT-5 호출 **병렬화(ThreadPool, WORKERS)** + `LockedEmbedder`, claim 실패 **NEI 폴백** + 120s 타임아웃(무인 실행 안정화), 진행률 로그.
5. **비용 최적화**: gpt-5(63 cr/claim baseline) → **gpt-5.4-mini(4.4)** 로 전환(품질 검증됨: claim0 정답·근거·tool-calling OK). 이걸로 394 전체가 예산 안에 들어옴.

---

## 3. 실험 조건 (결과 분석 시 필요)

| 조건 | 검색 | 동적 tier | 가중 | 라우팅 |
|---|---|---|---|---|
| gpt_only | 없음 | - | - | claim만 |
| naive_rag | dense/bm25 | ✗(도메인 tier만) | ✗ | controller R1~R5 |
| unweighted_rag | dense/bm25 | ✓ | ✗ | controller R1~R5 |
| proposed | dense/bm25 | ✓ | ✓ | controller R1~R5 |
| agentic | 도구 | ✓ | ✓ | 멀티에이전트 |
| agentic_no_tier | 도구 | ✗ | ✗ | 멀티에이전트 |

분석할 질문: ① proposed vs naive/unweighted/gpt_only(tier 가중 효과) ② dense vs bm25 ③ agentic vs deterministic
④ agentic ± tier 도구 ⑤ N=3 일치율(재현성). run_all.sh가 1~4단계로 다 산출.

---

## 4. 결과 해석 시 알아둘 것 (스모크에서 관찰)

- **숫자는 표본 크기 보고 판단** — 스모크(5/3 claim)는 노이즈였음. 394는 의미 있음.
- **gpt_only가 높게 나올 수 있음** — GPT-5/mini의 사전지식. RAG 효과가 시간민감·생소 claim에서 드러남.
- **agentic이 gold URL 대신 다른 유효 근거(팩트체커) 사용** → verdict 맞아도 `retrieval_category=retrieval_failure`로 잡힘(Recall은 gold-URL 기준). 연구적으로 의미있는 관찰.
- **실패한 claim은 결과에 NEI로** 표시됨(폴백). summary/predictions에서 NEI 비율·justification("error:")로 확인.
- 데이터 사실(서브셋 394, Conflicting 27, KS published_at~0%, cap 등)은 `docs/ks_structure.md`·`docs/domain_frequency.md`·PRD 참조.

---

## 5. 미해결/후속 가능 작업

- **cap 튜닝 실험**(`config.yaml index.max_chunks_per_url`, 현재 3): GPU에서 캡 올려 Recall@k 변화 측정(HANDOFF_GPU 5절). 인덱스 재빌드 필요.
- **agentic LangSmith 분석**: 프레임워크 사용/디버깅 시 `LANGSMITH_TRACING=true`(.env) + 소표본 `--agentic --limit 10` → 대시보드에서 멀티턴 추론 관찰.
- **5라벨 / 토픽 κ 사람주석**: `scripts/export_annotations.py`로 CSV export됨(`outputs/annotations/`), 사람이 채워야 κ·5라벨 gold 나옴(이슈 #11).
- 보조지표(RAGAS·G-Eval, `trev/eval/auxmetrics.py`)·공식 score 어댑터는 stretch.

---

## 6. 환경/주의

- 로컬(이 PC): Windows, `.venv`(Python 3.12), `python -m pytest`로 204 통과(실 API 스모크 3개는 키 있으면 실행).
- GPU PC: Linux, `python3 -m venv`, CUDA torch(cu128 류), gpt-5.4-mini.
- 데이터(`data_store/`,`knowledge_store/`)·`.env`·`index/`·`outputs/`·`results/`는 **gitignore**(루트 앵커). 코드만 git, 데이터는 수동 전송됨.
- `.env`에 `HAI_GPT_API_KEY`(키), `HAI_GPT_BASE_URL`, (선택)`LANGSMITH_*`. **키는 절대 커밋/문서화 금지.**
- 테스트 픽스처는 `tests/fixtures/`(KS 픽스처는 루트 앵커 수정으로 커밋됨 — 한 번 누락됐다 고침).

---

## 7. Suggested skills (다음 세션)

- **`/verify`** — 실험 결과(summary.txt·metrics)가 기대대로인지 확인할 때.
- **`/code-review`** 또는 **`/simplify`** — agentic 코드(trev/agent/) 추가 개발·정리 시.
- (분석이 주 목적이면 스킬 없이) `results/run_*/`의 metrics·predictions·agentic_ablation.json을 읽어 표로 정리 → proposed vs baseline, agentic vs deterministic 비교 보고.

가장 정확한 다음 행동: **GPU PC `tmux attach -t trev` → `summary.txt` 확인 → 조건별 비교 분석.**

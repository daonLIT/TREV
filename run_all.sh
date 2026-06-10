#!/usr/bin/env bash
# run_all.sh — TREV 전체 실험을 한 번에 실행하고 산출물을 정리한다 (GPU PC, Linux).
#
# 사용법:
#   bash run_all.sh                         # 기본(WORKERS=8 병렬)으로 전체 실행
#   WORKERS=10 bash run_all.sh              # 호출 10개 동시 (rate limit 걸리면 줄이기)
#   # 추천(mini 등 저비용 모델): baseline·agentic은 394 전체, ablation/N=3만 50 표본
#   AGENTIC_LIMIT=394 ABLATION_LIMIT=50 N=3 WORKERS=10 bash run_all.sh
#   LIMIT=20 AGENTIC_LIMIT=10 ABLATION_LIMIT=5 bash run_all.sh   # 빠른 스모크
#
# 변수: LIMIT=baseline claim수(빈값=394) · AGENTIC_LIMIT=step3 agentic 표본 ·
#       ABLATION_LIMIT=step4 ablation/N회 표본(비싸니 작게) · N=반복수 · WORKERS=동시호출수
#
# WORKERS = GPT-5 호출 동시 처리 수(claim 병렬). 병목이 API 지연이라 이게 가장 큰 가속.
# 기본 8 → 전체 실행이 ~하루에서 ~2-3시간으로. 로그에 429/재시도 많으면 줄이기.
#
# 비용(반드시 읽기):
#   - 결정론 baseline(dense+bm25): claim당 GPT-5 ~1회 → 전체 394는 ~수시간(감당 가능).
#   - agentic: claim당 tool-calling 多턴(1턴 ~40s) → 전체 394는 수십 시간~며칠.
#     그래서 AGENTIC_LIMIT 기본은 표본 20. 전체로 돌리려면 AGENTIC_LIMIT=394.
#   - GPU는 e5 인덱스 빌드만 가속. GPT-5 API 지연은 GPU와 무관.

set -uo pipefail

# ── 설정 (환경변수로 덮어쓰기 가능) ─────────────────────────────
LIMIT="${LIMIT:-}"                    # baseline claim 수 (빈값 = 전체 394)
AGENTIC_LIMIT="${AGENTIC_LIMIT:-20}"  # step3 agentic 조건 표본
ABLATION_LIMIT="${ABLATION_LIMIT:-$AGENTIC_LIMIT}"  # step4 ablation(tier on/off·N회) 표본 — 작게 두면 비용↓
N="${N:-3}"                           # N회 반복 일치율(재현성)
PY="${PY:-python}"
# ────────────────────────────────────────────────────────────

cd "$(dirname "$0")"                   # 레포 루트로 이동
[ -f .venv/bin/activate ] && source .venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
RUN="results/run_$TS"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1   # 화면 + 로그 동시 기록

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }
limit_arg() { [ -n "$1" ] && printf -- "--limit %s" "$1"; }

echo "TREV 전체 실험 시작 — $TS"
echo "  baseline: ${LIMIT:-전체(394)} | agentic: $AGENTIC_LIMIT | ablation: $ABLATION_LIMIT | N=$N | workers=${WORKERS:-8}"
echo "  결과 폴더: $RUN"

step "0) 환경·데이터 점검"
$PY -m scripts.subset_gate || { echo "데이터 점검 실패 — data_store/knowledge_store 배치 확인"; exit 1; }
$PY -m scripts.inspect_ks  || true

step "1) 결정론 baseline (dense)"
$PY -m scripts.run_experiments --method dense $(limit_arg "$LIMIT")
$PY -m scripts.run_eval --method dense

step "2) BM25 ablation (캐시 인덱스 재사용)"
$PY -m scripts.run_experiments --method bm25 $(limit_arg "$LIMIT")
$PY -m scripts.run_eval --method bm25

step "3) agentic 조건 (표본 $AGENTIC_LIMIT)"
$PY -m scripts.run_experiments --agentic $(limit_arg "$AGENTIC_LIMIT")
$PY -m scripts.run_eval --method agentic

step "4) 종합 ablation (tier on/off · agentic vs deterministic · N=$N 일치율, 표본 $ABLATION_LIMIT)"
$PY -m scripts.agentic_ablation --limit "$ABLATION_LIMIT" --n "$N"

step "5) 산출물 정리 → $RUN"
cp -f outputs/predictions_*.json outputs/metrics_*.json outputs/agentic_ablation.json "$RUN/" 2>/dev/null || true
cp -f docs/ks_structure.md docs/domain_frequency.md "$RUN/" 2>/dev/null || true

$PY - "$RUN" <<'PYEOF'
# 조건별 핵심 지표 요약표 → results/run_*/summary.txt
import glob, json, os, sys
run = sys.argv[1]
lines = [f"{'method':8} {'mode':16} {'acc':>6} {'macroF1':>8} {'R@10':>6} {'P@10':>6} {'avg_tools':>9}"]
lines.append("-" * len(lines[0]))
def f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
for path in sorted(glob.glob("outputs/metrics_*.json")):
    method = os.path.basename(path)[len("metrics_"):-len(".json")]
    table = json.load(open(path, encoding="utf-8"))
    for mode, m in table.items():
        lines.append(f"{method:8} {mode:16} {f(m.get('accuracy')):>6} "
                     f"{f(m.get('macro_f1')):>8} {f(m.get('recall@10')):>6} "
                     f"{f(m.get('precision@10')):>6} {str(m.get('avg_tool_calls')):>9}")
summary = "\n".join(lines)
print(summary)
open(os.path.join(run, "summary.txt"), "w", encoding="utf-8").write(summary + "\n")
PYEOF

echo
echo "===== 완료 ====="
echo "결과: $RUN/  (예측·지표 JSON, summary.txt, run.log)"
ls -la "$RUN"

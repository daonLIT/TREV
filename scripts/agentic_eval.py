"""A8: 에이전트 본평가(RQ5) — 체크포인트·재개 지원.

agentic / agentic_unweighted 조건을 claim×run 단위로 실행하되, 완료된 레코드를
즉시 JSONL 체크포인트에 append한다. 중단(크레딧 소진·Ctrl-C·프로세스 종료) 후
다시 실행하면 이미 끝난 (claim, run, mode)는 건너뛰고 그다음부터 이어서 돈다.
종료 시(정상/중단) 체크포인트로부터 run별 predictions_agentic_run{id}.json을 만든다
→ `python -m scripts.aggregate_n3 --method agentic --runs 1 2 3 [--exclude-timeouts]`로 집계.

실행 예: python -m scripts.agentic_eval --limit 50 --runs 1 2 3 --workers 4
(주의: agentic은 claim당 도구호출이 많아 비용·시간이 큼. 결정론 proposed/gpt_only는
 이미 predictions_dense_run{1,2,3}.json에 있으니 여기선 에이전트 조건만 돌린다.)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from trev.agent.orchestrator import orchestrate
from trev.config import load_config
from trev.data.dataset import load_averitec
from trev.data.indexing import ClaimIndex, E5Embedder, LockedEmbedder, build_claim_index
from trev.experiment import predictions_to_records
from trev.llm import LLM

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "index"
OUT_DIR = REPO / "outputs"
CKPT = OUT_DIR / "agentic_eval_ckpt.jsonl"

# 일시적 레이트리밋이 이 횟수만큼 연속되면 게이트웨이 장애로 보고 중단(부분결과 보존).
MAX_CONSEC_TRANSIENT = 20
# 일시적 실패(레이트리밋) 발생 시 다음 시도 전 짧은 백오프.
TRANSIENT_BACKOFF_S = 10


def _is_fatal(e: Exception) -> bool:
    """크레딧/쿼터 소진·인증 실패처럼 '재시도해도 소용없는' 오류인가?
    레이트리밋(429 rate_limit)은 일시적이라 fatal 아님 — 백오프 후 재실행으로 흡수."""
    try:
        from openai import AuthenticationError, PermissionDeniedError
        if isinstance(e, (AuthenticationError, PermissionDeniedError)):
            return True
    except Exception:
        pass
    msg = str(e).lower()
    return any(k in msg for k in ("insufficient_quota", "quota", "billing",
                                  "exceeded your current quota", "credit"))


def _provider(embedder, idx_cfg):
    def provide(claim):
        cdir = INDEX_DIR / str(claim.claim_id)
        if (cdir / "faiss.index").exists():
            return ClaimIndex.load(cdir)
        index = build_claim_index(claim.claim_id, embedder,
                                  max_words=idx_cfg.get("chunk_max_words", 180),
                                  max_chunks_per_url=idx_cfg.get("max_chunks_per_url"))
        index.save(cdir)
        return index
    return provide


def _load_done() -> set[tuple[int, str, str]]:
    """체크포인트에서 완료된 (claim_id, run, mode) 키 집합을 읽는다."""
    done: set[tuple[int, str, str]] = set()
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["claim_id"], r["run"], r["mode"]))
    return done


def _emit_predictions() -> None:
    """체크포인트를 run별 predictions_agentic_run{id}.json으로 변환(부분 결과도 즉시 사용 가능)."""
    if not CKPT.exists():
        return
    by_run: dict[str, list[dict]] = defaultdict(list)
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        run = rec.pop("run")
        by_run[run].append(rec)
    for run, recs in by_run.items():
        out = OUT_DIR / f"predictions_agentic_run{run}.json"
        out.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[written] {out}  ({len(recs)} 레코드)")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="claim 수(앞에서부터)")
    ap.add_argument("--runs", nargs="+", default=["1", "2", "3"], help="반복 run-id 목록")
    ap.add_argument("--modes", nargs="+", default=["agentic", "agentic_unweighted"],
                    help="실행할 에이전트 조건")
    ap.add_argument("--method", choices=["dense", "bm25"], default="dense")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "2")),
                    help="동시 claim 수. 레이트리밋(429) 잦으면 1~2로 낮출 것")
    args = ap.parse_args()

    cfg = load_config()
    claims = load_averitec(
        include_quote=cfg.get("data", {}).get("include_quote_verification", False))[: args.limit]
    embedder = E5Embedder(cfg.get("index", {}).get("e5_model", "intfloat/multilingual-e5-large"))
    if args.workers > 1:
        embedder = LockedEmbedder(embedder)
    provider = _provider(embedder, cfg.get("index", {}))
    llm = LLM.from_config(cfg)
    tier_cfg = cfg.get("tier", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done()
    print(f"[resume] 체크포인트 완료 {len(done)}건 — 이미 끝난 (claim,run,mode)는 건너뜀", flush=True)

    write_lock = threading.Lock()
    fail_lock = threading.Lock()
    state = {"consec_transient": 0, "stop": False, "written": 0}

    def _append(rec: dict, run: str) -> None:
        with write_lock:
            with CKPT.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "run": run}, ensure_ascii=False) + "\n")
                f.flush()
            state["written"] += 1

    def _on_failure(claim, run: str, e: Exception) -> None:
        """실패 분류: 치명적(쿼터/인증)이면 즉시 중단, 일시적(레이트리밋)이면 백오프 후 계속."""
        if _is_fatal(e):
            state["stop"] = True
            print(f"\n[중단] claim {claim.claim_id} run{run} 치명적 오류 "
                  f"{type(e).__name__}: {e} — 크레딧/쿼터 소진으로 판단. "
                  f"여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
            return
        with fail_lock:
            state["consec_transient"] += 1
            n = state["consec_transient"]
        print(f"  [일시오류] claim {claim.claim_id} run{run}: {type(e).__name__} "
              f"(연속 {n}) — {TRANSIENT_BACKOFF_S}s 백오프 후 계속(이 claim은 재실행 시 재시도)",
              flush=True)
        if n >= MAX_CONSEC_TRANSIENT:
            state["stop"] = True
            print(f"\n[중단] 일시오류 {n}회 연속 — 게이트웨이 장애로 판단. "
                  f"여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
            return
        time.sleep(TRANSIENT_BACKOFF_S)

    def process_claim(claim, run: str) -> None:
        """한 claim의 미완료 mode들을 실행하고 즉시 체크포인트에 기록.
        LLM 오류는 위로 전파(_on_failure가 치명/일시 분류)."""
        todo = [m for m in args.modes if (claim.claim_id, run, m) not in done]
        if not todo:
            return
        index = provider(claim)
        for mode in todo:
            if state["stop"]:
                return
            verdict, trace = orchestrate(
                claim, index, embedder, llm, tier_config=tier_cfg,
                method=args.method, k=10, candidate_n=50,
                use_tier=(mode == "agentic"))
            out = {"verdict": verdict, "retrieved_urls": trace.retrieved_urls,
                   "cost": {"steps": trace.steps_used, "tool_calls": trace.tool_calls_used}}
            rec = predictions_to_records([claim], {mode: [out]})[0]
            _append(rec, run)
            with fail_lock:
                state["consec_transient"] = 0  # 성공 → 연속 일시오류 리셋
            print(f"  run{run} claim {claim.claim_id} {mode:18} → "
                  f"{verdict.averitec_label.value} (tools {trace.tool_calls_used})", flush=True)

    try:
        for run in args.runs:
            if state["stop"]:
                break
            pending = [c for c in claims
                       if any((c.claim_id, run, m) not in done for m in args.modes)]
            print(f"\n[run {run}] 처리 대상 {len(pending)} claim "
                  f"(modes={args.modes}, workers={args.workers})", flush=True)
            if args.workers > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(process_claim, c, run): c for c in pending}
                    for fut in as_completed(futs):
                        try:
                            fut.result()
                        except Exception as e:
                            _on_failure(futs[fut], run, e)
                            if state["stop"]:
                                break
            else:
                for c in pending:
                    if state["stop"]:
                        break
                    try:
                        process_claim(c, run)
                    except Exception as e:
                        _on_failure(c, run, e)
    except KeyboardInterrupt:
        print("\n[중단] KeyboardInterrupt — 여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
    finally:
        print(f"\n[이번 실행 신규 기록 {state['written']}건] 체크포인트: {CKPT}", flush=True)
        _emit_predictions()
        print("\n집계: python -m scripts.aggregate_n3 --method agentic "
              f"--runs {' '.join(args.runs)} --exclude-timeouts", flush=True)


if __name__ == "__main__":
    main()

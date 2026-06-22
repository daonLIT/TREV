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

# 연속 실패가 이만큼 쌓이면 크레딧 소진/장애로 보고 깔끔히 중단(부분 결과는 보존).
MAX_CONSEC_FAILS = 5


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
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
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
    state = {"consec_fail": 0, "stop": False, "written": 0}

    def _append(rec: dict, run: str) -> None:
        with write_lock:
            with CKPT.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "run": run}, ensure_ascii=False) + "\n")
                f.flush()
            state["written"] += 1

    def process_claim(claim, run: str) -> None:
        """한 claim의 미완료 mode들을 실행하고 즉시 체크포인트에 기록.
        LLM/크레딧 오류는 위로 전파(연속 실패 카운트로 중단 판단)."""
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
                state["consec_fail"] = 0  # 성공 → 연속 실패 리셋
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
                        except Exception as e:  # 크레딧 소진/장애 → 연속 실패 누적
                            c = futs[fut]
                            with fail_lock:
                                state["consec_fail"] += 1
                                n = state["consec_fail"]
                            print(f"  [실패] claim {c.claim_id} run{run}: "
                                  f"{type(e).__name__}: {e} (연속 {n})", flush=True)
                            if n >= MAX_CONSEC_FAILS:
                                state["stop"] = True
                                print(f"\n[중단] 연속 실패 {n}회 — 크레딧 소진/장애로 판단. "
                                      f"여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
                                break
            else:
                for c in pending:
                    if state["stop"]:
                        break
                    try:
                        process_claim(c, run)
                        state["consec_fail"] = 0
                    except Exception as e:
                        state["consec_fail"] += 1
                        print(f"  [실패] claim {c.claim_id} run{run}: "
                              f"{type(e).__name__}: {e} (연속 {state['consec_fail']})", flush=True)
                        if state["consec_fail"] >= MAX_CONSEC_FAILS:
                            state["stop"] = True
                            print(f"\n[중단] 연속 실패 {state['consec_fail']}회 — 크레딧 소진/장애로 "
                                  f"판단. 여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
    except KeyboardInterrupt:
        print("\n[중단] KeyboardInterrupt — 여기까지 저장하고 종료(재실행하면 이어서 진행).", flush=True)
    finally:
        print(f"\n[이번 실행 신규 기록 {state['written']}건] 체크포인트: {CKPT}", flush=True)
        _emit_predictions()
        print("\n집계: python -m scripts.aggregate_n3 --method agentic "
              f"--runs {' '.join(args.runs)} --exclude-timeouts", flush=True)


if __name__ == "__main__":
    main()

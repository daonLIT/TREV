"""S11: N=3 반복 집계 + 유의성 검정 (RQ1 유의성 확정).

run_experiments.py --run-id 1/2/3 로 생성한
outputs/predictions_{method}_run{1,2,3}.json 를 읽어:
  - 조건별 accuracy·macro_f1 평균±표준편차 (run 간)
  - 실행 간 라벨 일치율 (agreement_rate)
  - 다수결(majority-vote) 라벨 기준 조건 쌍 McNemar 검정 (proposed vs unweighted/naive)
  - accuracy 차이의 bootstrap 95% CI
를 산출해 results/n3_{method}.json + 콘솔 표로 출력한다.

McNemar: temperature=0이어도 비결정이므로 run별이 아닌 '다수결 라벨'로 paired 검정.
판정 정답 여부(pred==gold)를 두 조건에서 비교, 불일치쌍(b,c)에 이항검정(scipy binomtest).

실행: python -m scripts.aggregate_n3 --method dense --runs 1 2 3
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from trev.eval.metrics import evaluate

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"


def _load_run(in_dir: Path, method: str, run_id: str) -> list[dict]:
    path = in_dir / f"predictions_{method}_run{run_id}.json"
    if not path.exists():
        raise SystemExit(f"[누락] {path} — run-id={run_id} 예측이 없음. 먼저 run_experiments 실행.")
    return json.loads(path.read_text(encoding="utf-8"))


def _timeout_excluded_ids(runs: list[list[dict]]) -> set[int]:
    """타임아웃/에러 시그니처(confidence=0 AND 검색결과 없음)로 실패한 claim 집합.
    한 조건·한 run에서라도 실패하면 공정 비교 위해 모든 조건에서 제외(RESULTS clean 방식).
    RAG의 정상 NEI(conf=0이나 검색결과 존재)는 제외 대상 아님."""
    bad: set[int] = set()
    for recs in runs:
        for r in recs:
            if (r.get("confidence") or 0) == 0 and not r.get("retrieved_urls"):
                bad.add(r["claim_id"])
    return bad


def _filter(runs: list[list[dict]], drop_ids: set[int]) -> list[list[dict]]:
    return [[r for r in recs if r["claim_id"] not in drop_ids] for recs in runs]


def _majority_label(labels: list[str]) -> str:
    """run 간 다수결 라벨. 동률이면 첫 run 우선(Counter는 입력 순서 보존)."""
    return Counter(labels).most_common(1)[0][0]


def _majority_by_mode(runs: list[list[dict]]) -> dict[str, dict[int, dict]]:
    """{mode: {claim_id: {pred(다수결), gold}}} — run들을 합쳐 claim별 다수결."""
    acc: dict[str, dict[int, dict]] = {}
    for records in runs:
        for r in records:
            slot = acc.setdefault(r["mode"], {}).setdefault(
                r["claim_id"], {"preds": [], "gold": r["gold_label"]}
            )
            slot["preds"].append(r["pred_label"])
    out: dict[str, dict[int, dict]] = {}
    for mode, claims in acc.items():
        out[mode] = {
            cid: {"pred": _majority_label(s["preds"]), "gold": s["gold"]}
            for cid, s in claims.items()
        }
    return out


def _agreement(runs: list[list[dict]], mode: str) -> dict:
    """해당 mode에서 run 간 라벨 일치율(모든 run이 같은 라벨인 claim 비율)."""
    per_run = [
        {r["claim_id"]: r["pred_label"] for r in rec if r["mode"] == mode}
        for rec in runs
    ]
    common = set(per_run[0])
    for pr in per_run[1:]:
        common &= set(pr)
    disagree = [c for c in common if len({pr[c] for pr in per_run}) > 1]
    n = len(common)
    return {"agreement": (n - len(disagree)) / n if n else None, "n": n}


def _mcnemar(maj_a: dict[int, dict], maj_b: dict[int, dict]) -> dict:
    """두 조건의 정답여부 paired 비교. b=A만 정답, c=B만 정답, 이항검정 p값."""
    shared = set(maj_a) & set(maj_b)
    b = c = 0  # b: A correct & B wrong, c: A wrong & B correct
    for cid in shared:
        ga = maj_a[cid]["gold"]
        a_ok = maj_a[cid]["pred"] == ga
        b_ok = maj_b[cid]["pred"] == maj_b[cid]["gold"]
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    n_disc = b + c
    # 양측 이항검정(p=0.5): 불일치쌍이 한쪽으로 치우치는지.
    p = binomtest(b, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    return {"b_only_A_correct": b, "c_only_B_correct": c, "n_discordant": n_disc, "p_value": p}


def _bootstrap_acc_delta(
    maj_a: dict[int, dict], maj_b: dict[int, dict], *, n_boot: int = 10000, seed: int = 0
) -> dict:
    """acc(A)-acc(B)의 bootstrap 95% CI (claim 단위 재표집)."""
    shared = sorted(set(maj_a) & set(maj_b))
    a_ok = np.array([maj_a[c]["pred"] == maj_a[c]["gold"] for c in shared], dtype=float)
    b_ok = np.array([maj_b[c]["pred"] == maj_b[c]["gold"] for c in shared], dtype=float)
    diff = a_ok - b_ok
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "delta_acc": float(diff.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dense", "bm25"], default="dense")
    ap.add_argument("--runs", nargs="+", default=["1", "2", "3"], help="run-id 목록")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--in-dir", type=str, default="outputs",
                    help="예측 json 위치(기본 outputs). results 폴더에 두면 --in-dir results")
    ap.add_argument("--exclude-timeouts", action="store_true",
                    help="타임아웃(conf=0 & 검색실패) claim을 모든 조건에서 공통 제외(clean 비교)")
    ap.add_argument("--tag", type=str, default=None, help="출력 파일명 접미사(예: clean, gem)")
    args = ap.parse_args()

    in_dir = REPO / args.in_dir
    runs = [_load_run(in_dir, args.method, rid) for rid in args.runs]
    n_before = len({r["claim_id"] for r in runs[0]})
    n_excluded = 0
    if args.exclude_timeouts:
        drop = _timeout_excluded_ids(runs)
        runs = _filter(runs, drop)
        n_excluded = len(drop)
    tables = [evaluate(rec, k=args.k) for rec in runs]
    modes = sorted(tables[0].keys())

    # 1) 조건별 평균±표준편차 + 일치율
    summary = {}
    for mode in modes:
        accs = [t[mode]["accuracy"] for t in tables]
        f1s = [t[mode]["macro_f1"] for t in tables]
        summary[mode] = {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "macro_f1_mean": statistics.mean(f1s),
            "macro_f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "per_run_acc": accs,
            "agreement": _agreement(runs, mode),
        }

    # 2) 다수결 라벨로 McNemar + bootstrap (proposed 기준 비교)
    maj = _majority_by_mode(runs)
    comparisons = {}
    if "proposed" in maj:
        for other in ("unweighted_rag", "naive_rag"):
            if other in maj:
                comparisons[f"proposed_vs_{other}"] = {
                    "mcnemar": _mcnemar(maj["proposed"], maj[other]),
                    "bootstrap": _bootstrap_acc_delta(maj["proposed"], maj[other]),
                }

    n_eval = len({r["claim_id"] for r in runs[0]})
    result = {"method": args.method, "n_runs": len(runs),
              "n_total": n_before, "n_excluded_timeout": n_excluded, "n_eval": n_eval,
              "exclude_timeouts": args.exclude_timeouts,
              "summary": summary, "comparisons": comparisons}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"n3_{args.method}{tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 출력
    excl = f", 타임아웃 제외 {n_excluded} → N_eval={n_eval}" if args.exclude_timeouts else f", N_eval={n_eval}"
    print(f"\n[N={len(runs)} {args.method}]  (per-run acc id: {args.runs}{excl})\n")
    hdr = f"{'mode':16s} {'acc(mean±std)':>16s} {'macroF1(mean±std)':>18s} {'agree':>7s}"
    print(hdr); print("-" * len(hdr))
    for mode in modes:
        s = summary[mode]
        ag = s["agreement"]["agreement"]
        print(f"{mode:16s} {s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}   "
              f"  {s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}    "
              f"{(ag if ag is not None else float('nan')):.3f}")
    print()
    for name, comp in comparisons.items():
        m, bs = comp["mcnemar"], comp["bootstrap"]
        print(f"{name}: Δacc={bs['delta_acc']:+.3f} "
              f"CI95[{bs['ci95'][0]:+.3f},{bs['ci95'][1]:+.3f}]  "
              f"McNemar b={m['b_only_A_correct']} c={m['c_only_B_correct']} "
              f"p={m['p_value']:.4f}")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()

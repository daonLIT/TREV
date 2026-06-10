"""S7: 예측 채점 — 조건별 지표 표 생성(run_eval).

outputs/predictions_{method}.json(enriched: 회수url·gold url·검색분류 포함)을 읽어
조건별 Accuracy·Macro-F1·Recall@k·Precision@k·검색실패vsNEI·인용율을 산출하고
outputs/metrics_{method}.json에 저장한다.

실행: python -m scripts.run_eval --method dense --k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trev.eval.metrics import evaluate

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dense", "bm25", "agentic"], default="dense")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    pred_path = OUT_DIR / f"predictions_{args.method}.json"
    records = json.loads(pred_path.read_text(encoding="utf-8"))
    table = evaluate(records, k=args.k)

    out = OUT_DIR / f"metrics_{args.method}.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.method}] {pred_path.name} → {out.name}\n")
    header = f"{'mode':16s} {'acc':>6s} {'macroF1':>8s} {'R@k':>6s} {'P@k':>6s} {'uncited':>8s}"
    print(header)
    print("-" * len(header))
    for mode, m in table.items():
        r = m.get(f"recall@{args.k}")
        p = m.get(f"precision@{args.k}")
        print(f"{mode:16s} {m['accuracy']:6.3f} {m['macro_f1']:8.3f} "
              f"{(r if r is not None else float('nan')):6.3f} "
              f"{(p if p is not None else float('nan')):6.3f} "
              f"{m['citation']['uncited']:8d}")
        if m["retrieval_breakdown"]:
            print(f"{'':16s} breakdown: {m['retrieval_breakdown']}")


if __name__ == "__main__":
    main()

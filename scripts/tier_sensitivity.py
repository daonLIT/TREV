"""1-5: tier 가중치 민감도 (dense, proposed 1회씩).

기본 가중(1.0/0.7/0.4/0.1) 대비 약/강/이진 가중과 self-source 강등 단독 ablation을
비교해 "효과가 특정 가중값에 민감한가 / self-source 규칙만으로도 효과가 있나"를 본다.

변형(dense, proposed 모드만):
  weak             weights {1:1.0,2:0.85,3:0.70,4:0.55}  (균등에 가깝게)
  strong           weights {1:1.0,2:0.50,3:0.20,4:0.05}  (가파르게)
  binary           weights {1:1.0,2:1.0,3:1.0,4:0.0}      (신뢰 tier=1 / T4=0)
  self_source_only weights 전부 1.0 + self_source_weight 0.1 (tier 무가중, 자기출처만 강등)

변형 단위 체크포인트: outputs/predictions_dense_tier_{name}.json 있으면 건너뜀(재실행 이어가기).
기존 baseline(proposed, 기본 가중)은 predictions_dense_run*.json / n3_dense_clean.json 참고.

실행: python -m scripts.tier_sensitivity --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from trev.config import load_config
from trev.data.dataset import load_averitec
from trev.data.indexing import ClaimIndex, E5Embedder, LockedEmbedder, build_claim_index
from trev.eval.metrics import evaluate
from trev.experiment import predictions_to_records, run_experiments
from trev.llm import LLM

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "index"
OUT_DIR = REPO / "outputs"
RESULTS_DIR = REPO / "results"

VARIANTS = {
    "weak": {"weights": {1: 1.0, 2: 0.85, 3: 0.70, 4: 0.55}},
    "strong": {"weights": {1: 1.0, 2: 0.50, 3: 0.20, 4: 0.05}},
    "binary": {"weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}},
    "self_source_only": {"weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
                         "self_source_weight": 0.1},
}


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


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="claim 수 제한(스모크용)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    help="실행할 변형(기본 전체)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    args = ap.parse_args()

    cfg = load_config()
    claims = load_averitec(
        include_quote=cfg.get("data", {}).get("include_quote_verification", False))
    if args.limit:
        claims = claims[: args.limit]
    embedder = E5Embedder(cfg.get("index", {}).get("e5_model", "intfloat/multilingual-e5-large"))
    if args.workers > 1:
        embedder = LockedEmbedder(embedder)
    provider = _provider(embedder, cfg.get("index", {}))
    llm = LLM.from_config(cfg)
    base_tier = cfg.get("tier", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in args.variants:
        if name not in VARIANTS:
            print(f"[건너뜀] 알 수 없는 변형: {name}", flush=True)
            continue
        out_path = OUT_DIR / f"predictions_dense_tier_{name}.json"
        if out_path.exists():  # 변형 단위 재개
            print(f"[skip] {name} — 이미 완료({out_path.name})", flush=True)
            records = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            tier_cfg = copy.deepcopy(base_tier)
            tier_cfg.update(VARIANTS[name])  # weights / self_source_weight 덮어쓰기
            print(f"\n[실행] variant={name} weights={VARIANTS[name]} "
                  f"claims={len(claims)} workers={args.workers}", flush=True)
            results = run_experiments(
                claims, embedder, llm, index_provider=provider,
                tier_config=tier_cfg, method="dense", modes=("proposed",),
                max_workers=args.workers)
            records = predictions_to_records(claims, results)
            out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"[written] {out_path}", flush=True)
        m = evaluate(records, k=10)["proposed"]
        summary[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "citation_uncited": m["citation"]["uncited"], "n": m["n"],
                         "weights": VARIANTS[name]}

    out = RESULTS_DIR / "tier_sensitivity.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[tier 민감도 — proposed, dense]")
    print(f"  (baseline 1.0/0.7/0.4/0.1: n3_dense_clean.json 참고, acc≈0.493)")
    print(f"{'variant':18s} {'acc':>6s} {'macroF1':>8s}")
    print("-" * 36)
    for name, s in summary.items():
        print(f"{name:18s} {s['accuracy']:6.3f} {s['macro_f1']:8.3f}")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()

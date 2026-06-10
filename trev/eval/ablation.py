"""S10: Ablation + 재현성(agreement rate).

순수 분석 함수:
- `granularity_macro_f1`: 3라벨 vs 4라벨 Macro-F1(과장·상충 포착력). 5라벨은 사람주석(#11) 필요.
- `weight_topk_stability`: tier 가중 ±0.1 섭동 시 top-k 랭킹 강건성(Jaccard).
- `compare_methods`: BM25 vs dense 지표 나란히 비교.
- `agreement_rate`: 고정 시드·프롬프트·KS에서 N회 반복 라벨 일치율.

주의: GPT-5는 temperature=0이어도 완전 재현이 아니므로 '결정론'이 아닌 '일치율'로 보고한다.
"""

from __future__ import annotations

from trev.eval.metrics import macro_f1

# 4라벨 → 3라벨 축약(상충/과장 구분을 버림): Conflicting → NEI.
_TO_3LABEL = {
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Not Enough Evidence": "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking": "Not Enough Evidence",
}


def relabel(label: str | None, granularity: int) -> str | None:
    """granularity=3이면 Conflicting을 NEI로 축약, 4면 원본 유지."""
    if label is None:
        return None
    return _TO_3LABEL.get(label, label) if granularity == 3 else label


def granularity_macro_f1(records: list[dict], granularity: int = 4) -> float:
    """주어진 라벨 입도에서 Macro-F1(pred vs gold). 3 vs 4 비교로 상충 포착력을 본다."""
    pred = [relabel(r["pred_label"], granularity) for r in records]
    gold = [relabel(r["gold_label"], granularity) for r in records]
    return macro_f1(pred, gold)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _weighted_topk(evidence, weights: dict[int, float], k: int) -> set[str]:
    """(url, sim, tier) 리스트를 score=sim*weight로 정렬해 top-k url 집합."""
    ranked = sorted(evidence, key=lambda e: e[1] * weights.get(e[2], 0.0), reverse=True)
    return {url for url, _, _ in ranked[:k]}


def weight_topk_stability(
    per_claim_evidence: list[list[tuple]],
    base_weights: dict[int, float],
    *,
    delta: float = 0.1,
    k: int = 10,
) -> dict:
    """가중치를 ±delta 섭동했을 때 top-k url 집합의 평균 Jaccard(강건성)."""
    variants = []
    for sign in (+1, -1):
        variants.append({t: min(1.0, max(0.0, w + sign * delta))
                         for t, w in base_weights.items()})
    jaccards = []
    for ev in per_claim_evidence:
        base_top = _weighted_topk(ev, base_weights, k)
        for w in variants:
            jaccards.append(_jaccard(base_top, _weighted_topk(ev, w, k)))
    mean = sum(jaccards) / len(jaccards) if jaccards else 1.0
    return {"delta": delta, "mean_jaccard": mean, "n_comparisons": len(jaccards)}


def compare_methods(table_a: dict, table_b: dict, *, metric: str = "macro_f1") -> dict:
    """두 evaluate() 표(예: dense vs bm25)를 조건별로 나란히 비교한다."""
    out = {}
    for mode in sorted(set(table_a) | set(table_b)):
        a = table_a.get(mode, {}).get(metric)
        b = table_b.get(mode, {}).get(metric)
        out[mode] = {"a": a, "b": b,
                     "delta": (a - b) if (a is not None and b is not None) else None}
    return out


def agreement_rate(runs: list[dict[int, str]]) -> dict:
    """N회 반복 라벨 일치율: 모든 run이 같은 라벨을 낸 claim 비율('결정론' 아님)."""
    if not runs:
        return {"agreement": None, "n": 0, "n_runs": 0}
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    disagree = [cid for cid in common if len({r[cid] for r in runs}) > 1]
    n = len(common)
    return {
        "agreement": (n - len(disagree)) / n if n else None,
        "n": n,
        "n_runs": len(runs),
        "disagree_claims": sorted(disagree),
    }

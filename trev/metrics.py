"""S7: 평가 지표 — Label Accuracy · Macro-F1 · Recall@k/Precision@k · 검색실패vsNEI · 인용율.

순수 함수(외부 의존 없음). 예측 레코드(조건별 pred/gold + 회수 url + gold url + 검색 분류)를
받아 조건별 표를 만든다. 검색 url ↔ gold url 매칭은 D5(recall.py)를 재사용한다.
"""

from __future__ import annotations

from collections import Counter

from trev.recall import precision_at_k, recall_at_k


def accuracy(pred_labels, gold_labels) -> float:
    """Label Accuracy(pred == gold). gold 결측 쌍은 제외."""
    pairs = [(p, g) for p, g in zip(pred_labels, gold_labels) if g is not None]
    if not pairs:
        return 0.0
    return sum(p == g for p, g in pairs) / len(pairs)


def macro_f1(pred_labels, gold_labels) -> float:
    """라벨 집합에 대한 Macro-F1(클래스별 F1의 단순 평균)."""
    pairs = [(p, g) for p, g in zip(pred_labels, gold_labels) if g is not None]
    if not pairs:
        return 0.0
    labels = sorted({g for _, g in pairs} | {p for p, _ in pairs})
    f1s = []
    for L in labels:
        tp = sum(p == L and g == L for p, g in pairs)
        fp = sum(p == L and g != L for p, g in pairs)
        fn = sum(p != L and g == L for p, g in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def recall_precision(retrievals, k: int) -> dict:
    """평균 Recall@k/Precision@k. retrievals: (retrieved_urls, gold_urls) 리스트."""
    recs = [r for ru, gu in retrievals if (r := recall_at_k(ru, gu, k)) is not None]
    precs = [p for ru, gu in retrievals if (p := precision_at_k(ru, gu, k)) is not None]
    return {
        f"recall@{k}": sum(recs) / len(recs) if recs else None,
        f"precision@{k}": sum(precs) / len(precs) if precs else None,
        "n_with_gold": len(recs),
    }


def citation_rate(cited_lists) -> dict:
    """유효 인용율 + 무인용(빈 cited) 개수. 무인용 판정 0 확인용."""
    n = len(cited_lists)
    nonempty = sum(1 for c in cited_lists if c)
    return {"rate": nonempty / n if n else 0.0, "uncited": n - nonempty, "n": n}


def evaluate_mode(records: list[dict], k: int = 10) -> dict:
    """단일 조건 레코드의 지표를 산출한다.

    각 record: {pred_label, gold_label, cited, retrieved_urls, gold_urls, retrieval_category}.
    """
    pred = [r["pred_label"] for r in records]
    gold = [r["gold_label"] for r in records]
    retrievals = [(r.get("retrieved_urls") or [], r.get("gold_urls") or []) for r in records]
    breakdown = Counter(
        r["retrieval_category"] for r in records if r.get("retrieval_category")
    )
    return {
        "n": len(records),
        "accuracy": accuracy(pred, gold),
        "macro_f1": macro_f1(pred, gold),
        **recall_precision(retrievals, k),
        "citation": citation_rate([r.get("cited") or [] for r in records]),
        "retrieval_breakdown": dict(breakdown),
    }


def evaluate(records: list[dict], k: int = 10) -> dict[str, dict]:
    """조건(mode)별로 그룹화해 지표 표를 만든다."""
    by_mode: dict[str, list[dict]] = {}
    for r in records:
        by_mode.setdefault(r["mode"], []).append(r)
    return {mode: evaluate_mode(rs, k) for mode, rs in by_mode.items()}

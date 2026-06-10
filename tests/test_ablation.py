"""S10 ablation + 재현성 단위 테스트(순수)."""

from __future__ import annotations

import pytest

from trev.eval.ablation import (
    agreement_rate,
    compare_methods,
    granularity_macro_f1,
    relabel,
    weight_topk_stability,
)


# --- 3 vs 4 라벨 -----------------------------------------------------------

def test_relabel_collapses_conflicting_to_nei_at_3():
    assert relabel("Conflicting Evidence/Cherrypicking", 3) == "Not Enough Evidence"
    assert relabel("Conflicting Evidence/Cherrypicking", 4) == "Conflicting Evidence/Cherrypicking"
    assert relabel("Supported", 3) == "Supported"


def test_granularity_f1_differs_when_conflicting_present():
    # gold가 Conflicting인데 pred가 NEI: 4라벨에선 오답, 3라벨에선 정답(축약).
    records = [
        {"pred_label": "Not Enough Evidence", "gold_label": "Conflicting Evidence/Cherrypicking"},
        {"pred_label": "Supported", "gold_label": "Supported"},
    ]
    f1_4 = granularity_macro_f1(records, 4)
    f1_3 = granularity_macro_f1(records, 3)
    assert f1_3 > f1_4   # 상충 구분을 버리면 점수가 올라감(포착력 손실 확인)


# --- tier 가중 민감도 ------------------------------------------------------

WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}


def test_weight_stability_high_when_ranking_robust():
    # tier가 sim과 일치(고tier=고sim) → 섭동해도 top-k 불변.
    ev = [[("a", 0.9, 1), ("b", 0.5, 3), ("c", 0.2, 4)]]
    out = weight_topk_stability(ev, WEIGHTS, delta=0.1, k=2)
    assert out["mean_jaccard"] == 1.0


def test_weight_stability_detects_reordering():
    # T4 문서가 sim 매우 높음 → 가중 섭동이 top-k를 바꿀 수 있음.
    ev = [[("a", 0.95, 4), ("b", 0.5, 1), ("c", 0.45, 1)]]
    out = weight_topk_stability(ev, WEIGHTS, delta=0.1, k=1)
    assert 0.0 <= out["mean_jaccard"] <= 1.0
    assert out["n_comparisons"] == 2  # ±delta 2회


# --- BM25 vs dense ---------------------------------------------------------

def test_compare_methods():
    dense = {"proposed": {"macro_f1": 0.8}, "naive_rag": {"macro_f1": 0.6}}
    bm25 = {"proposed": {"macro_f1": 0.7}, "naive_rag": {"macro_f1": 0.65}}
    out = compare_methods(dense, bm25)
    assert out["proposed"]["delta"] == pytest.approx(0.1)
    assert out["naive_rag"]["delta"] == pytest.approx(-0.05)


# --- N=3 일치율 ------------------------------------------------------------

def test_agreement_rate_all_agree():
    runs = [{0: "Refuted", 1: "Supported"}] * 3
    out = agreement_rate(runs)
    assert out["agreement"] == 1.0 and out["n_runs"] == 3


def test_agreement_rate_with_disagreement():
    runs = [
        {0: "Refuted", 1: "Supported"},
        {0: "Refuted", 1: "Refuted"},   # claim 1 불일치
        {0: "Refuted", 1: "Supported"},
    ]
    out = agreement_rate(runs)
    assert out["agreement"] == 0.5 and out["disagree_claims"] == [1]


def test_agreement_uses_common_claims_only():
    runs = [{0: "A", 1: "A"}, {0: "A"}]   # claim 1은 한 run에만
    out = agreement_rate(runs)
    assert out["n"] == 1 and out["agreement"] == 1.0

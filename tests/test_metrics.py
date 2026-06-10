"""S7 평가 지표 단위 테스트(순수 — 소규모 예측/gold 픽스처)."""

from __future__ import annotations

from trev.eval.metrics import (
    accuracy,
    citation_rate,
    evaluate,
    macro_f1,
    recall_precision,
)


# --- accuracy / macro_f1 ---------------------------------------------------

def test_accuracy_ignores_missing_gold():
    assert accuracy(["A", "B", "C"], ["A", "B", None]) == 1.0   # None 쌍 제외
    assert accuracy(["A", "B"], ["A", "X"]) == 0.5


def test_macro_f1_perfect_and_partial():
    assert macro_f1(["A", "B"], ["A", "B"]) == 1.0
    # 한 클래스만 맞추면 macro는 1.0 미만(클래스별 평균).
    assert 0.0 < macro_f1(["A", "A"], ["A", "B"]) < 1.0


# --- recall / precision ----------------------------------------------------

def test_recall_precision_aggregate():
    retrievals = [
        (["https://g.com/a", "https://x.com/b"], ["https://g.com/a"]),  # recall 1.0, prec 0.5
        (["https://y.com/c"], ["https://g.com/z"]),                      # recall 0.0
    ]
    out = recall_precision(retrievals, k=10)
    assert out["recall@10"] == 0.5     # (1.0 + 0.0)/2
    assert out["n_with_gold"] == 2


def test_recall_skips_no_gold():
    out = recall_precision([(["https://x.com/a"], [])], k=10)
    assert out["recall@10"] is None and out["n_with_gold"] == 0


# --- citation rate ---------------------------------------------------------

def test_citation_rate_counts_uncited():
    out = citation_rate([["d0"], [], ["d1", "d2"]])
    assert out["rate"] == 2 / 3 and out["uncited"] == 1


# --- evaluate (조건별 표) --------------------------------------------------

def _rec(mode, pred, gold, cited, retrieved, gold_urls, cat):
    return {"mode": mode, "pred_label": pred, "gold_label": gold, "cited": cited,
            "retrieved_urls": retrieved, "gold_urls": gold_urls, "retrieval_category": cat}


def test_evaluate_groups_by_mode():
    records = [
        _rec("proposed", "Refuted", "Refuted", ["d0"], ["g"], ["g"], "retrieved"),
        _rec("proposed", "Supported", "Refuted", ["d1"], [], ["g"], "retrieval_failure"),
        _rec("gpt_only", "Refuted", "Refuted", [], [], [], "true_nei"),
    ]
    table = evaluate(records, k=10)
    assert set(table) == {"proposed", "gpt_only"}
    assert table["proposed"]["accuracy"] == 0.5
    assert table["proposed"]["retrieval_breakdown"] == {
        "retrieved": 1, "retrieval_failure": 1,
    }
    # gpt_only는 무인용(검색 면제) → uncited 카운트에 잡힘.
    assert table["gpt_only"]["citation"]["uncited"] == 1

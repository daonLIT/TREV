"""S6 실험 실행기 + 인덱스 영속화 테스트(가짜 임베더 + 스텁 LLM)."""

from __future__ import annotations

import json

from trev.experiment import (
    DEFAULT_MODES,
    predict_claim,
    predictions_to_records,
    run_experiments,
)
from trev.data.indexing import ClaimIndex
from trev.llm import LLM
from trev.schemas import AveritecLabel, Claim, ClaimType, Passage
from tests.test_indexing import FakeEmbedder
from tests.test_llm import FakeClient

TIER_CFG = {
    "weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1},
    "overrides": {},
    "heuristics": {1: [".gov"], 3: ["reuters"]},
}


def _claim():
    return Claim(claim_id=0, text="economy grew last year", type=ClaimType.NUMERICAL,
                 label=AveritecLabel.SUPPORTED)


def _passages():
    # 3+ 문서라야 BM25 희소 term IDF가 0이 아님(소코퍼스 퇴화 회피).
    return [
        Passage(claim_id=0, url="https://cdc.gov/a", text="economy grew strongly last year",
                source_domain="cdc.gov", ks_type="question"),
        Passage(claim_id=0, url="https://blog.example/b", text="random unrelated text",
                source_domain="blog.example", ks_type="background"),
        Passage(claim_id=0, url="https://other.example/c", text="cooking recipe food tips",
                source_domain="other.example", ks_type="background"),
    ]


# --- 인덱스 영속화 round-trip ----------------------------------------------

def test_claimindex_save_load_roundtrip(tmp_path):
    idx = ClaimIndex.build(_passages(), FakeEmbedder())
    idx.save(tmp_path / "0")
    loaded = ClaimIndex.load(tmp_path / "0")
    assert [p.url for p in loaded.passages] == [p.url for p in idx.passages]
    hits = loaded.search_dense("economy grew", FakeEmbedder(), k=2)
    assert hits and hits[0].passage.source_domain == "cdc.gov"
    # 메타 보존(영속화 후에도 tier 입력 가능)
    assert loaded.passages[0].source_domain == "cdc.gov"


def test_bm25_search_after_load(tmp_path):
    idx = ClaimIndex.build(_passages(), FakeEmbedder())
    idx.save(tmp_path / "0")
    loaded = ClaimIndex.load(tmp_path / "0")
    hits = loaded.search_bm25("economy", k=2)
    assert hits[0].passage.source_domain == "cdc.gov"


# --- 4조건 실행기 ----------------------------------------------------------

def _verifier_json(label="SUPPORT"):
    # run_verifier(스키마 검증) + gpt_only_verdict 둘 다 만족하는 JSON.
    return json.dumps({
        "label": label, "confidence": 0.9, "justification": "j",
        "cited": ["0-0"], "stances": [{"doc_id": "0-0", "stance": "SUPPORT"}],
    })


def test_predict_claim_each_mode_returns_verdict():
    index = ClaimIndex.build(_passages(), FakeEmbedder())
    for mode in DEFAULT_MODES:
        # 모드마다 호출 수가 달라 넉넉히 같은 응답을 반복 제공.
        llm = LLM(client=FakeClient([_verifier_json()] * 4))
        v = predict_claim(_claim(), index, FakeEmbedder(), llm, mode=mode,
                          method="dense", tier_config=TIER_CFG)
        assert v.claim_id == 0
        assert v.averitec_label in set(AveritecLabel)


def test_run_experiments_and_serialize():
    index = ClaimIndex.build(_passages(), FakeEmbedder())
    llm = LLM(client=FakeClient([_verifier_json()] * 40))
    results = run_experiments([_claim()], FakeEmbedder(), llm,
                              index_provider=lambda c: index, tier_config=TIER_CFG,
                              method="dense")
    assert set(results) == set(DEFAULT_MODES)
    # 회수 URL이 RAG 모드엔 동반, gpt_only엔 빈 리스트.
    assert results["proposed"][0]["retrieved_urls"]
    assert results["gpt_only"][0]["retrieved_urls"] == []

    records = predictions_to_records(
        [_claim()], results,
        gold_urls_fn=lambda cid: ["https://cdc.gov/a"],   # 픽스처 gold(회수됨)
        ks_urls_fn=lambda cid: ["https://cdc.gov/a"],
    )
    assert len(records) == len(DEFAULT_MODES)
    blob = json.dumps(records)
    assert '"gold_label": "Supported"' in blob
    assert all("retrieved_urls" in r and "retrieval_category" in r for r in records)
    # proposed는 cdc.gov를 회수 → retrieved 분류.
    proposed = next(r for r in records if r["mode"] == "proposed")
    assert proposed["retrieval_category"] == "retrieved"


def test_bm25_method_runs():
    index = ClaimIndex.build(_passages(), FakeEmbedder())
    llm = LLM(client=FakeClient([_verifier_json("REFUTE")] * 4))
    v = predict_claim(_claim(), index, FakeEmbedder(), llm, mode="proposed",
                      method="bm25", tier_config=TIER_CFG)
    assert v.claim_id == 0

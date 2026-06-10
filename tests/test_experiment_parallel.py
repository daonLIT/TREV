"""run_experiments 병렬(max_workers) 경로 — 순차와 동일 결과 + 순서 보존."""

from __future__ import annotations

from trev.data.indexing import ClaimIndex, LockedEmbedder
from trev.experiment import run_experiments
from trev.llm import LLM
from trev.schemas import Claim, ClaimType, Passage
from tests.test_indexing import FakeEmbedder
from tests.test_llm import FakeClient

TIER_CFG = {"weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1}, "overrides": {},
            "heuristics": {1: [".gov"]}}


def _claims(n):
    return [Claim(claim_id=i, text=f"claim {i} economy", type=ClaimType.NUMERICAL)
            for i in range(n)]


def _index(claim):
    return ClaimIndex.build(
        [Passage(claim_id=claim.claim_id, url=f"https://cdc.gov/{claim.claim_id}",
                 text="economy claim evidence", source_domain="cdc.gov")], FakeEmbedder())


def _verifier_json():
    import json
    return json.dumps({"label": "REFUTE", "confidence": 0.9, "justification": "j",
                       "cited": ["0-0"], "stances": [{"doc_id": "0-0", "stance": "REFUTE"}]})


def _run(claims, workers):
    llm = LLM(client=FakeClient([_verifier_json()] * 200))
    emb = LockedEmbedder(FakeEmbedder()) if workers > 1 else FakeEmbedder()
    return run_experiments(claims, emb, llm, index_provider=_index, tier_config=TIER_CFG,
                           modes=("gpt_only", "proposed"), max_workers=workers, verbose=False)


def test_parallel_matches_sequential_and_preserves_order():
    claims = _claims(6)
    seq = _run(claims, workers=1)
    par = _run(claims, workers=4)
    # claim_id 순서 보존
    assert [p["verdict"].claim_id for p in par["proposed"]] == list(range(6))
    # 두 경로의 예측 라벨 동일
    assert ([p["verdict"].label5 for p in seq["proposed"]]
            == [p["verdict"].label5 for p in par["proposed"]])
    assert len(par["gpt_only"]) == 6


def test_locked_embedder_delegates():
    le = LockedEmbedder(FakeEmbedder(dim=8))
    out = le.encode(["a b", "c"], is_query=True)
    assert out.shape == (2, 8)

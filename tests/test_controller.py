"""S5 controller 단위 테스트 — 가짜 retriever/verifier/gpt_only 주입(LLM·데이터 없이)."""

from __future__ import annotations

import pytest

from trev.controller import ControllerConfig, run
from trev.schemas import (
    AveritecLabel,
    Claim,
    ClaimType,
    Evidence,
    Label5,
    Stance,
    Verdict,
)
from trev.verifier import EvidenceStance, VerifierOutput, to_averitec_label

TIER_CFG = {
    "weights": {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.1},
    "overrides": {},
    "heuristics": {1: [".gov"], 2: ["factcheck"], 3: ["reuters"]},
}


def _claim(checkworthiness=None):
    return Claim(claim_id=0, text="t", type=ClaimType.NUMERICAL,
                 checkworthiness=checkworthiness)


def _ev(doc_id, domain, sim=0.5):
    return Evidence(doc_id=doc_id, snippet="s", source_domain=domain, sim=sim)


def _out(label="SUPPORT", confidence=0.9, cited=("e0",), stances=()):
    return VerifierOutput(
        label=label, confidence=confidence, justification="j", cited=list(cited),
        stances=[EvidenceStance(doc_id=d, stance=s) for d, s in stances],
    )


def _retriever(*evidences):
    """expand=False면 첫 세트, expand=True면 둘째 세트를 반환."""
    sets = list(evidences)

    def fn(claim, expand=False):
        return sets[1] if (expand and len(sets) > 1) else sets[0]
    return fn


def _verifier(*outputs):
    calls = {"n": 0}

    def fn(claim, evidence):
        out = outputs[min(calls["n"], len(outputs) - 1)]
        calls["n"] += 1
        return out
    fn.calls = calls
    return fn


def _gpt_only(label="SUPPORT"):
    def fn(claim):
        l5 = Label5(label)
        return Verdict(claim_id=claim.claim_id, label5=l5,
                       averitec_label=to_averitec_label(l5), confidence=0.8,
                       justification="claim-only", cited=[])
    return fn


def _run(claim, mode, retrieve_fn, verify_fn, gpt_only_fn=None, config=None):
    return run(claim, mode=mode, retrieve_fn=retrieve_fn, verify_fn=verify_fn,
               gpt_only_fn=gpt_only_fn or _gpt_only(), tier_config=TIER_CFG,
               config=config or ControllerConfig())


# --- R1 ---------------------------------------------------------------------

def test_r1_discards_low_checkworthiness():
    v = _run(_claim(checkworthiness=0.1), "proposed",
             _retriever([_ev("e0", "cdc.gov")]), _verifier(_out()))
    assert v.label5 is Label5.NEI and "R1" in v.justification


# --- R2 ---------------------------------------------------------------------

def test_r2_nei_when_no_evidence():
    v = _run(_claim(), "proposed", _retriever([]), _verifier(_out()))
    assert v.label5 is Label5.NEI and "R2" in v.justification


def test_r2_nei_when_only_t4():
    v = _run(_claim(), "proposed", _retriever([_ev("e0", "randomblog.com")]),
             _verifier(_out()))
    assert v.label5 is Label5.NEI and "R2" in v.justification


# --- R3 ---------------------------------------------------------------------

def test_r3_verifies_when_trusted_evidence():
    v = _run(_claim(), "proposed", _retriever([_ev("e0", "cdc.gov")]),
             _verifier(_out(label="REFUTE", cited=["e0"])))
    assert v.label5 is Label5.REFUTE
    assert v.averitec_label is AveritecLabel.REFUTED


# --- R4 ---------------------------------------------------------------------

def test_r4_reretrieves_then_succeeds():
    verifier = _verifier(_out(confidence=0.3), _out(label="SUPPORT", confidence=0.9))
    v = _run(_claim(), "proposed",
             _retriever([_ev("e0", "cdc.gov")], [_ev("e1", "reuters.com")]), verifier)
    assert v.label5 is Label5.SUPPORT
    assert verifier.calls["n"] == 2  # 1회 재검증


def test_r4_nei_when_still_low_confidence():
    verifier = _verifier(_out(confidence=0.3), _out(confidence=0.2))
    v = _run(_claim(), "proposed",
             _retriever([_ev("e0", "cdc.gov")], [_ev("e1", "reuters.com")]), verifier)
    assert v.label5 is Label5.NEI and "R4" in v.justification


# --- R5 (stance CONFLICT) ---------------------------------------------------

def test_r5_conflict_on_high_tier_stance_disagreement():
    evidence = [_ev("e0", "cdc.gov"), _ev("e1", "reuters.com")]  # T1, T3
    # T1(e0)=SUPPORT, T2가 아닌 T3(e1)=REFUTE → 둘 다 T1~T2는 아님? e0 T1, e1 T3.
    # T1~T2 공존이어야 하므로 둘 다 고tier로: cdc.gov(T1)+factcheck(T2).
    evidence = [_ev("e0", "cdc.gov"), _ev("e1", "factcheck.org")]
    out = _out(label="SUPPORT", cited=["e0", "e1"],
               stances=[("e0", Stance.SUPPORT), ("e1", Stance.REFUTE)])
    v = _run(_claim(), "proposed", _retriever(evidence), _verifier(out))
    assert v.label5 is Label5.CONFLICT
    assert v.averitec_label is AveritecLabel.CONFLICTING


def test_no_conflict_when_disagreement_only_in_low_tier():
    evidence = [_ev("e0", "cdc.gov"), _ev("e1", "randomblog.com")]  # T1, T4
    out = _out(label="SUPPORT", cited=["e0"],
               stances=[("e0", Stance.SUPPORT), ("e1", Stance.REFUTE)])  # 충돌은 T4에만
    v = _run(_claim(), "proposed", _retriever(evidence), _verifier(out))
    assert v.label5 is Label5.SUPPORT  # CONFLICT 아님


# --- gpt_only 예외 ----------------------------------------------------------

def test_gpt_only_bypasses_retrieval_and_does_not_collapse_to_nei():
    # 검색이 빈 결과여도(R2라면 NEI) gpt_only는 우회하고 라벨을 직출.
    v = _run(_claim(), "gpt_only", _retriever([]), _verifier(_out()),
             gpt_only_fn=_gpt_only(label="REFUTE"))
    assert v.label5 is Label5.REFUTE
    assert v.cited == []  # cited 면제


# --- baseline 모드 동일 controller 통과 ------------------------------------

@pytest.mark.parametrize("mode", ["naive_rag", "unweighted_rag", "proposed"])
def test_rag_modes_pass_same_controller(mode):
    v = _run(_claim(), mode, _retriever([_ev("e0", "cdc.gov")]),
             _verifier(_out(label="SUPPORT", cited=["e0"])))
    assert v.label5 is Label5.SUPPORT


def test_naive_vs_unweighted_self_source_routing():
    # claim 자기출처 = cdc.gov인 유일 근거. unweighted(dynamic)면 target→T4 강등→R2 NEI.
    # naive(자기출처 미적용)면 cdc.gov=T1 유지→verify.
    claim = Claim(claim_id=0, text="t", type=ClaimType.NUMERICAL,
                  source_domains=["cdc.gov"])
    ev = [_ev("e0", "cdc.gov")]
    v_unw = _run(claim, "unweighted_rag", _retriever(ev), _verifier(_out(cited=["e0"])))
    v_naive = _run(claim, "naive_rag", _retriever(ev), _verifier(_out(cited=["e0"])))
    assert v_unw.label5 is Label5.NEI       # 자기출처 강등 → T4뿐 → R2
    assert v_naive.label5 is Label5.SUPPORT  # 강등 없음 → T1 → verify

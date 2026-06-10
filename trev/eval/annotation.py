"""S9: 사람 주석 하네스(HITL) — 토픽 κ(200건) + Conflicting 5라벨 세분.

- 토픽: 표본 200건을 CSV로 export → 사람 검수 → 취합해 LLM 태깅과의 Cohen's κ 보고.
- 5라벨: 서브셋 Conflicting gold ≥20(D1 게이트)이면 PARTIAL/CONFLICT를 사람이 세분 라벨링
  → 정량 F1(부족하면 정성 사례연구). export/ingest 하니스를 제공한다.

순수 함수(κ 계산·분기) + CSV 입출력 헬퍼. 사람 작업은 CSV 편집으로 수행한다.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from trev.data.dataset import CONFLICTING_QUANT_THRESHOLD
from trev.schemas import AveritecLabel, Claim

VALID_LABEL5 = {"PARTIAL", "CONFLICT"}  # Conflicting 세분 라벨


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """두 라벨열의 Cohen's κ(우연 보정 일치도). 빈 입력은 None."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def export_topic_sample(
    tagged_claims: list[Claim], n: int = 200, *, seed: int = 42
) -> list[dict]:
    """토픽 검수용 표본 n건을 export한다(`human_topic` 빈칸을 사람이 채움)."""
    rng = random.Random(seed)
    sample = rng.sample(tagged_claims, min(n, len(tagged_claims)))
    return [
        {"claim_id": c.claim_id, "claim": c.text, "speaker": c.speaker or "",
         "publisher": c.publisher or "", "llm_topic": c.topic or "", "human_topic": ""}
        for c in sorted(sample, key=lambda c: c.claim_id)
    ]


def topic_agreement(records: list[dict]) -> dict:
    """취합된 토픽 주석에서 일치율 + Cohen's κ를 보고한다(빈 human_topic 제외)."""
    pairs = [(r["llm_topic"], r["human_topic"]) for r in records if r.get("human_topic")]
    llm = [x for x, _ in pairs]
    human = [y for _, y in pairs]
    agreement = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
    return {"n": len(pairs), "agreement": agreement, "cohens_kappa": cohens_kappa(llm, human)}


def export_conflicting_for_refinement(claims: list[Claim]) -> list[dict]:
    """Conflicting gold claim을 PARTIAL/CONFLICT 세분 라벨링용으로 export한다."""
    return [
        {"claim_id": c.claim_id, "claim": c.text, "gold_label": c.label.value,
         "human_label5": ""}
        for c in claims if c.label is AveritecLabel.CONFLICTING
    ]


def refinement_branch(claims: list[Claim]) -> str:
    """Conflicting 개수로 정량(≥20)/정성 분기를 결정한다(D1 게이트와 동일 임계값)."""
    n = sum(1 for c in claims if c.label is AveritecLabel.CONFLICTING)
    return "quantitative" if n >= CONFLICTING_QUANT_THRESHOLD else "qualitative"


def ingest_5label(records: list[dict]) -> dict[int, str]:
    """취합된 5라벨 주석을 {claim_id: PARTIAL|CONFLICT}로 읽는다(유효값만)."""
    out: dict[int, str] = {}
    for r in records:
        label = (r.get("human_label5") or "").strip().upper()
        if label in VALID_LABEL5:
            out[int(r["claim_id"])] = label
    return out


def write_csv(path: str | Path, records: list[dict]) -> None:
    """주석용 CSV를 쓴다(사람이 편집)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: str | Path) -> list[dict]:
    """편집된 주석 CSV를 읽는다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

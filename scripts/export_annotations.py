"""S9: 사람 주석용 CSV export.

- 토픽 검수 표본 200건 → outputs/annotations/topic_sample.csv (human_topic 채우기)
- Conflicting 5라벨 세분 → outputs/annotations/conflicting_5label.csv (PARTIAL/CONFLICT)

사람이 CSV를 채운 뒤 trev.annotation.topic_agreement / ingest_5label로 취합한다.
실행: python -m scripts.export_annotations
"""

from __future__ import annotations

from pathlib import Path

from trev.annotation import (
    export_conflicting_for_refinement,
    export_topic_sample,
    refinement_branch,
    write_csv,
)
from trev.config import load_config
from trev.dataset import load_averitec
from trev.topics import tag_claims

OUT = Path(__file__).resolve().parent.parent / "outputs" / "annotations"


def main() -> None:
    cfg = load_config()
    claims = load_averitec(
        include_quote=cfg.get("data", {}).get("include_quote_verification", False)
    )
    tagged = tag_claims(claims)  # 키워드 태깅(애매=other; LLM 분류는 별도)

    topic_rows = export_topic_sample(tagged, n=200)
    write_csv(OUT / "topic_sample.csv", topic_rows)

    conflicting_rows = export_conflicting_for_refinement(claims)
    write_csv(OUT / "conflicting_5label.csv", conflicting_rows)

    branch = refinement_branch(claims)
    print(f"[topic]       {len(topic_rows)}건 → {OUT/'topic_sample.csv'}")
    print(f"[conflicting] {len(conflicting_rows)}건 → {OUT/'conflicting_5label.csv'}")
    print(f"[branch]      Conflicting {len(conflicting_rows)}건 → {branch} "
          f"({'세분 F1' if branch == 'quantitative' else '정성 사례연구'})")


if __name__ == "__main__":
    main()

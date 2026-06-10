"""G0 게이트: knowledge_store 실구조 점검·문서화.

claim별 문서 배치(위치 인덱스 ↔ claim_id), KS 레코드 필드(url/published_at/source),
아카이브 스냅샷 비율(=published_at 유도 가능성)을 출력하고 docs/ks_structure.md에 기록한다.

실행: python -m scripts.inspect_ks
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from trev.guards import assert_knowledge_store_path

REPO_ROOT = Path(__file__).resolve().parent.parent
KS_DIR = REPO_ROOT / "knowledge_store" / "dev"
OUT_DOC = REPO_ROOT / "docs" / "ks_structure.md"
ARCHIVE_RE = re.compile(r"web\.archive\.org/web/(\d{14})/")


def _iter_records(path: Path):
    """KS 파일(JSONL)의 레코드를 순회한다."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def inspect() -> str:
    assert_knowledge_store_path(KS_DIR)  # dev split만 허용(가드)
    files = sorted(KS_DIR.glob("*.json"), key=lambda p: int(p.stem))
    n_files = len(files)

    record_keys: set[str] = set()
    type_counter: Counter[str] = Counter()
    total_records = 0
    archive_records = 0
    archive_gold = 0  # 아카이브 URL 중 type=gold 비중(=시점유도 가능분이 gold에 집중되는지)
    url2text_lens: list[int] = []
    mapping_mismatches: list[str] = []

    for path in files:
        # claim_id는 문자열("0")이므로 파일 stem과 문자열 비교로 교차검증.
        for rec in _iter_records(path):
            total_records += 1
            record_keys.update(rec.keys())
            rtype = rec.get("type", "<none>")
            type_counter[rtype] += 1
            if str(rec.get("claim_id")) != path.stem:
                mapping_mismatches.append(f"{path.name}: claim_id={rec.get('claim_id')!r}")
            url = rec.get("url") or ""
            if ARCHIVE_RE.search(url):
                archive_records += 1
                if rtype == "gold":
                    archive_gold += 1
            u = rec.get("url2text")
            if isinstance(u, list):
                url2text_lens.append(len(u))

    has_published_at = "published_at" in record_keys
    has_source = "source" in record_keys
    archive_ratio = archive_records / total_records if total_records else 0.0
    avg_passages = sum(url2text_lens) / len(url2text_lens) if url2text_lens else 0.0

    lines = [
        "# Knowledge Store 구조 점검 (G0)",
        "",
        f"- KS 디렉터리: `knowledge_store/dev/`",
        f"- 파일 수: **{n_files}** (이름 `{{index}}.json`, claim 위치 인덱스로 매핑)",
        f"- 총 레코드(JSONL 라인): **{total_records}**",
        f"- 레코드 키: `{sorted(record_keys)}`",
        f"- `type` 분포: `{dict(type_counter)}`",
        f"- `published_at` 필드 존재: **{has_published_at}** "
        f"(없음 → 아카이브 스냅샷 타임스탬프로 유도)",
        f"- `source` 필드 존재: **{has_source}** (없음 → `url` netloc에서 도메인 추출)",
        f"- web.archive.org 스냅샷 비율: **{archive_ratio:.1%}** "
        f"({archive_records}/{total_records}, =published_at 유도 가능 비율)",
        f"- 아카이브 스냅샷 중 type=gold: **{archive_gold}/{archive_records}** "
        f"→ 시점유도 가능 문서가 gold(검색 코퍼스에서 격리 대상)에 집중. "
        f"실제 검색 코퍼스의 published_at 커버리지는 사실상 0%.",
        f"- `url2text` passage 평균 길이: **{avg_passages:.1f}** "
        f"(min {min(url2text_lens) if url2text_lens else 0}, "
        f"max {max(url2text_lens) if url2text_lens else 0})",
        f"- claim_id ↔ 파일 인덱스 불일치: **{len(mapping_mismatches)}건** "
        + ("✅ 전부 일치" if not mapping_mismatches else f"⚠️ {mapping_mismatches[:5]}"),
    ]
    report = "\n".join(lines) + "\n"
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(inspect())
    print(f"\n[written] {OUT_DOC}")

"""D2: knowledge_store per-claim 로더 + passage/메타 추출 + published_at 유도.

claim 위치 인덱스 → `knowledge_store/dev/{index}.json` 매핑으로 후보 풀을 로드한다.
KS 레코드 `claim_id`는 문자열("0")이므로 파일 stem과 문자열 비교로 교차검증한다.
JSONL 레코드 `{claim_id, type, query, url, url2text}`에서 url2text(페이지 라인 리스트)를
Passage로 추출하고, `{source_domain, published_at, ks_type}` 메타를 부착한다.

데이터 위생(누수 차단):
- dev.json gold `questions`는 절대 코퍼스에 넣지 않는다(이 로더는 KS만 읽음).
- KS `type` 14종은 모두 후보 코퍼스에 동등 포함하며, `type=gold`도 일반 후보로만
  취급한다(부스팅·격리 제거 금지) — `ks_type`은 분석용 메타로만 보존.
- url 기준으로 동일 (url, text) passage는 1회만 유지(`question_duplicate` 등 중복 제거).

도메인/시점 유도 헬퍼(`extract_domain`·`derive_published_at`)는 D3(#19) tier 입력도 재사용한다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from trev.guards import assert_knowledge_store_path
from trev.schemas import Passage

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KS_DIR = _REPO_ROOT / "knowledge_store" / "dev"

# web.archive.org/web/<14자리 타임스탬프>/<원본 URL>
_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/(\d{14})/(.*)", re.IGNORECASE)


def recover_archive_url(url: str) -> str:
    """아카이브 스냅샷 URL이면 원본 URL을, 아니면 입력을 그대로 반환한다."""
    m = _ARCHIVE_RE.search(url)
    return m.group(2) if m else url


def extract_domain(url: str) -> str | None:
    """URL의 등록 도메인(netloc, www·스킴 제거)을 반환한다(아카이브면 원본 복원 후)."""
    original = recover_archive_url(url)
    netloc = urlparse(original).netloc
    if not netloc:  # 스킴 없는 경우 보강
        netloc = urlparse("http://" + original).netloc
    netloc = netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[len("www."):]
    return netloc or None


def derive_published_at(url: str) -> str | None:
    """web.archive.org 스냅샷 타임스탬프(14자리)에서 published_at(YYYY-MM-DD)을 유도한다.

    비-아카이브/파싱 실패 → None(시점필터 통과). 스냅샷은 거의 type=gold에만 존재하므로
    실제 검색 코퍼스의 커버리지는 사실상 0%다(G0).
    """
    m = _ARCHIVE_RE.search(url)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").date().isoformat()
    except ValueError:
        return None


def _iter_records(path: Path):
    """KS 파일(JSONL)의 레코드를 순회한다(빈 줄 무시)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_claim_passages(
    claim_id: int, ks_dir: str | Path = DEFAULT_KS_DIR
) -> list[Passage]:
    """claim_id에 매핑된 `{claim_id}.json`의 passage들을 추출한다.

    레코드 claim_id ↔ 파일 인덱스 교차검증(문자열 비교, 불일치 시 ValueError),
    url2text element를 Passage로 펼친 뒤 (url, text) 기준으로 dedup한다.
    """
    ks_dir = Path(ks_dir)
    path = ks_dir / f"{claim_id}.json"
    assert_knowledge_store_path(path)  # dev split만 허용(#16 가드)

    passages: list[Passage] = []
    seen: set[tuple[str, str]] = set()
    for rec in _iter_records(path):
        if str(rec.get("claim_id")) != str(claim_id):
            raise ValueError(
                f"{path.name}: claim_id 불일치 (레코드 {rec.get('claim_id')!r} != {claim_id})"
            )
        url = rec.get("url") or ""
        if not url:
            continue
        domain = extract_domain(url)
        published_at = derive_published_at(url)
        ks_type = rec.get("type")
        for text in rec.get("url2text") or []:
            if not isinstance(text, str) or not text.strip():
                continue
            key = (url, text)
            if key in seen:  # 동일 (url, text) 중복 제거
                continue
            seen.add(key)
            passages.append(
                Passage(
                    claim_id=claim_id,
                    url=url,
                    text=text,
                    source_domain=domain,
                    published_at=published_at,
                    ks_type=ks_type,
                )
            )
    return passages
